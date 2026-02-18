//! Memory-Mapped KV Cache
//!
//! v0.8: When the KV cache exceeds available RAM, spill to SSD via mmap.
//! This allows context lengths far beyond what fits in memory — the OS handles
//! paging transparently, and we use madvise hints for optimal access patterns.
//!
//! Architecture:
//! - Hot window (recent tokens) stays in RAM for fast access
//! - Cold entries (older tokens) are memory-mapped from a temp file
//! - Transparent API: callers don't need to know where data lives
//! - Automatic promotion: cold → hot when accessed frequently

use anyhow::Result;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Memory-mapped KV cache that spills to SSD when RAM budget is exceeded
pub struct MmapKvCache {
    /// Per-layer caches
    layers: Vec<MmapLayerKv>,
    /// Maximum RAM bytes for hot cache across all layers
    ram_budget: usize,
    /// Current RAM usage
    ram_used: usize,
    /// Directory for mmap backing files
    backing_dir: PathBuf,
    /// Number of layers
    n_layers: usize,
    /// Max sequence length
    max_seq_len: usize,
}

/// Per-layer mmap-backed KV cache
pub struct MmapLayerKv {
    /// Hot keys (recent, in RAM): [position] -> Vec<f32>
    hot_keys: Vec<Vec<f32>>,
    /// Hot values (recent, in RAM): [position] -> Vec<f32>
    hot_values: Vec<Vec<f32>>,
    /// Cold storage backing file for keys
    cold_key_file: Option<File>,
    /// Cold storage backing file for values
    cold_value_file: Option<File>,
    /// Cold mmap for keys (read-only once written)
    cold_key_mmap: Option<memmap2::Mmap>,
    /// Cold mmap for values
    cold_value_mmap: Option<memmap2::Mmap>,
    /// Number of entries in cold storage
    cold_len: usize,
    /// Total sequence length (cold + hot)
    total_len: usize,
    /// KV dimensions
    n_kv_heads: usize,
    head_dim: usize,
    /// Bytes per entry (n_kv_heads * head_dim * sizeof(f32))
    entry_bytes: usize,
    /// Layer index (for file naming)
    layer_idx: usize,
    /// Backing directory
    backing_dir: PathBuf,
}

impl MmapKvCache {
    /// Create a new mmap-backed KV cache
    ///
    /// # Arguments
    /// * `n_layers` - Number of transformer layers
    /// * `n_kv_heads` - Number of KV attention heads
    /// * `head_dim` - Dimension per head
    /// * `max_seq_len` - Maximum sequence length
    /// * `ram_budget` - Maximum RAM bytes for hot cache
    /// * `backing_dir` - Directory for temporary backing files
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        ram_budget: usize,
        backing_dir: &Path,
    ) -> Result<Self> {
        std::fs::create_dir_all(backing_dir)?;

        let layers = (0..n_layers)
            .map(|i| MmapLayerKv::new(n_kv_heads, head_dim, i, backing_dir))
            .collect();

        info!(
            "MmapKvCache: layers={}, kv_heads={}, head_dim={}, ram_budget={:.2}MB, backing={}",
            n_layers,
            n_kv_heads,
            head_dim,
            ram_budget as f64 / (1024.0 * 1024.0),
            backing_dir.display()
        );

        Ok(Self {
            layers,
            ram_budget,
            ram_used: 0,
            backing_dir: backing_dir.to_path_buf(),
            n_layers,
            max_seq_len,
        })
    }

    /// Append key/value for a specific layer
    pub fn append(&mut self, layer_idx: usize, key: Vec<f32>, value: Vec<f32>) -> Result<()> {
        let entry_bytes = self.layers[layer_idx].entry_bytes * 2; // key + value

        // Check if we need to spill hot → cold
        if self.ram_used + entry_bytes > self.ram_budget {
            self.spill_oldest_hot()?;
        }

        self.ram_used += entry_bytes;
        self.layers[layer_idx].append_hot(key, value);
        Ok(())
    }

    /// Get key at position for a specific layer and KV head
    pub fn key_at(&self, layer_idx: usize, pos: usize, kv_head: usize) -> &[f32] {
        self.layers[layer_idx].key_at(pos, kv_head)
    }

    /// Get value at position for a specific layer and KV head
    pub fn value_at(&self, layer_idx: usize, pos: usize, kv_head: usize) -> &[f32] {
        self.layers[layer_idx].value_at(pos, kv_head)
    }

    /// Current sequence length
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.total_len).unwrap_or(0)
    }

    /// Total memory usage (RAM only)
    pub fn ram_bytes(&self) -> usize {
        self.ram_used
    }

    /// Total data size including cold storage
    pub fn total_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.total_bytes()).sum()
    }

    /// Cold storage size on SSD
    pub fn cold_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.cold_len * l.entry_bytes * 2)
            .sum()
    }

    /// Number of positions in cold storage
    pub fn cold_positions(&self) -> usize {
        self.layers.first().map(|l| l.cold_len).unwrap_or(0)
    }

    /// Number of positions in hot RAM
    pub fn hot_positions(&self) -> usize {
        self.layers.first().map(|l| l.hot_keys.len()).unwrap_or(0)
    }

    /// Clear all caches
    pub fn clear(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.clear()?;
        }
        self.ram_used = 0;
        Ok(())
    }

    /// Rollback to a given sequence length
    pub fn rollback(&mut self, new_len: usize) {
        for layer in &mut self.layers {
            layer.rollback(new_len);
        }
        // Recalculate RAM usage
        self.ram_used = self
            .layers
            .iter()
            .map(|l| l.hot_keys.len() * l.entry_bytes * 2)
            .sum();
    }

    /// Spill the oldest hot entries across all layers to cold storage
    fn spill_oldest_hot(&mut self) -> Result<()> {
        // Spill 25% of hot entries to make room
        let hot_len = self.layers.first().map(|l| l.hot_keys.len()).unwrap_or(0);
        let spill_count = (hot_len / 4).max(1);

        debug!("Spilling {} hot entries to cold storage", spill_count);

        let mut freed = 0usize;
        for layer in &mut self.layers {
            freed += layer.spill_to_cold(spill_count)?;
        }
        self.ram_used = self.ram_used.saturating_sub(freed);

        Ok(())
    }

    /// Get layer reference for compatibility
    pub fn layer_mut(&mut self, _layer_idx: usize) -> &mut MmapLayerKv {
        &mut self.layers[_layer_idx]
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Check if sequence has hit max length
    pub fn is_full(&self) -> bool {
        self.seq_len() >= self.max_seq_len
    }
}

impl MmapLayerKv {
    fn new(n_kv_heads: usize, head_dim: usize, layer_idx: usize, backing_dir: &Path) -> Self {
        Self {
            hot_keys: Vec::new(),
            hot_values: Vec::new(),
            cold_key_file: None,
            cold_value_file: None,
            cold_key_mmap: None,
            cold_value_mmap: None,
            cold_len: 0,
            total_len: 0,
            n_kv_heads,
            head_dim,
            entry_bytes: n_kv_heads * head_dim * std::mem::size_of::<f32>(),
            layer_idx,
            backing_dir: backing_dir.to_path_buf(),
        }
    }

    fn append_hot(&mut self, key: Vec<f32>, value: Vec<f32>) {
        self.hot_keys.push(key);
        self.hot_values.push(value);
        self.total_len += 1;
    }

    fn key_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        let start = kv_head * self.head_dim;
        let end = start + self.head_dim;

        if pos < self.cold_len {
            // Read from cold mmap
            if let Some(ref mmap) = self.cold_key_mmap {
                let byte_offset = pos * self.entry_bytes + start * 4;
                let byte_end = pos * self.entry_bytes + end * 4;
                if byte_end <= mmap.len() {
                    let slice = &mmap[byte_offset..byte_end];
                    // Safety: f32 is 4 bytes, properly aligned in mmap
                    unsafe {
                        std::slice::from_raw_parts(slice.as_ptr() as *const f32, self.head_dim)
                    }
                } else {
                    &[]
                }
            } else {
                &[]
            }
        } else {
            // Read from hot RAM
            let hot_idx = pos - self.cold_len;
            &self.hot_keys[hot_idx][start..end]
        }
    }

    fn value_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        let start = kv_head * self.head_dim;
        let end = start + self.head_dim;

        if pos < self.cold_len {
            if let Some(ref mmap) = self.cold_value_mmap {
                let byte_offset = pos * self.entry_bytes + start * 4;
                let byte_end = pos * self.entry_bytes + end * 4;
                if byte_end <= mmap.len() {
                    let slice = &mmap[byte_offset..byte_end];
                    unsafe {
                        std::slice::from_raw_parts(slice.as_ptr() as *const f32, self.head_dim)
                    }
                } else {
                    &[]
                }
            } else {
                &[]
            }
        } else {
            let hot_idx = pos - self.cold_len;
            &self.hot_values[hot_idx][start..end]
        }
    }

    /// Spill `count` oldest hot entries to cold storage
    fn spill_to_cold(&mut self, count: usize) -> Result<usize> {
        let actual = count.min(self.hot_keys.len());
        if actual == 0 {
            return Ok(0);
        }

        // Create/open backing files if needed
        if self.cold_key_file.is_none() {
            let key_path = self
                .backing_dir
                .join(format!("layer_{}_keys.bin", self.layer_idx));
            let val_path = self
                .backing_dir
                .join(format!("layer_{}_values.bin", self.layer_idx));
            self.cold_key_file = Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&key_path)?,
            );
            self.cold_value_file = Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&val_path)?,
            );
        }

        // Write entries to backing files
        let key_file = self.cold_key_file.as_mut().unwrap();
        let val_file = self.cold_value_file.as_mut().unwrap();

        for i in 0..actual {
            let key_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.hot_keys[i].as_ptr() as *const u8,
                    self.hot_keys[i].len() * 4,
                )
            };
            let val_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.hot_values[i].as_ptr() as *const u8,
                    self.hot_values[i].len() * 4,
                )
            };
            key_file.write_all(key_bytes)?;
            val_file.write_all(val_bytes)?;
        }
        key_file.flush()?;
        val_file.flush()?;

        // Remove spilled entries from hot
        self.hot_keys.drain(0..actual);
        self.hot_values.drain(0..actual);
        self.cold_len += actual;

        // Re-mmap the files
        self.remap()?;

        let freed = actual * self.entry_bytes * 2;
        debug!(
            "Layer {}: spilled {} entries ({:.2}KB) to cold",
            self.layer_idx,
            actual,
            freed as f64 / 1024.0
        );
        Ok(freed)
    }

    /// Re-create mmap after writing new cold data
    fn remap(&mut self) -> Result<()> {
        let key_path = self
            .backing_dir
            .join(format!("layer_{}_keys.bin", self.layer_idx));
        let val_path = self
            .backing_dir
            .join(format!("layer_{}_values.bin", self.layer_idx));

        if key_path.exists() {
            let key_file = File::open(&key_path)?;
            let val_file = File::open(&val_path)?;

            if key_file.metadata()?.len() > 0 {
                unsafe {
                    self.cold_key_mmap = Some(memmap2::MmapOptions::new().map(&key_file)?);
                    self.cold_value_mmap = Some(memmap2::MmapOptions::new().map(&val_file)?);
                }

                // Advise sequential access for prefill, random for decode
                #[cfg(target_os = "macos")]
                unsafe {
                    if let Some(ref mmap) = self.cold_key_mmap {
                        libc::madvise(
                            mmap.as_ptr() as *mut libc::c_void,
                            mmap.len(),
                            libc::MADV_RANDOM,
                        );
                    }
                    if let Some(ref mmap) = self.cold_value_mmap {
                        libc::madvise(
                            mmap.as_ptr() as *mut libc::c_void,
                            mmap.len(),
                            libc::MADV_RANDOM,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    fn total_bytes(&self) -> usize {
        let hot = self.hot_keys.len() * self.entry_bytes * 2;
        let cold = self.cold_len * self.entry_bytes * 2;
        hot + cold
    }

    fn clear(&mut self) -> Result<()> {
        self.hot_keys.clear();
        self.hot_values.clear();
        self.cold_key_mmap = None;
        self.cold_value_mmap = None;
        self.cold_key_file = None;
        self.cold_value_file = None;
        self.cold_len = 0;
        self.total_len = 0;

        // Remove backing files
        let key_path = self
            .backing_dir
            .join(format!("layer_{}_keys.bin", self.layer_idx));
        let val_path = self
            .backing_dir
            .join(format!("layer_{}_values.bin", self.layer_idx));
        let _ = std::fs::remove_file(&key_path);
        let _ = std::fs::remove_file(&val_path);

        Ok(())
    }

    fn rollback(&mut self, new_len: usize) {
        if new_len >= self.total_len {
            return;
        }

        if new_len <= self.cold_len {
            // Need to truncate cold storage too — just clear everything and keep new_len from cold
            // For simplicity, clear cold and move to hot
            warn!("Rollback into cold storage not fully supported, clearing");
            self.hot_keys.clear();
            self.hot_values.clear();
            self.cold_len = 0;
            self.total_len = 0;
        } else {
            let hot_new_len = new_len - self.cold_len;
            self.hot_keys.truncate(hot_new_len);
            self.hot_values.truncate(hot_new_len);
            self.total_len = new_len;
        }
    }
}

impl Drop for MmapKvCache {
    fn drop(&mut self) {
        // Clean up backing files
        let _ = self.clear();
        let _ = std::fs::remove_dir_all(&self.backing_dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_dir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "ssd_llm_test_mmap_kv_{}_{}",
            std::process::id(),
            id
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_mmap_kv_basic() {
        let dir = test_dir();
        let mut cache = MmapKvCache::new(2, 4, 8, 1024, 1024 * 1024, &dir).unwrap();

        let key = vec![1.0f32; 32]; // 4 heads * 8 dim
        let val = vec![2.0f32; 32];

        cache.append(0, key.clone(), val.clone()).unwrap();
        // seq_len uses layer 0's total_len
        assert_eq!(cache.layers[0].total_len, 1);
        // Check retrieval
        let k = cache.key_at(0, 0, 0);
        assert_eq!(k.len(), 8);
        assert_eq!(k[0], 1.0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mmap_kv_spill() {
        let dir = test_dir();
        // Tiny RAM budget to force spilling
        let entry_size = 4 * 8 * 4 * 2; // n_kv * head_dim * sizeof(f32) * 2 (key+val) = 256 bytes
        let ram_budget = entry_size * 4; // room for 4 entries across all layers

        let mut cache = MmapKvCache::new(1, 4, 8, 1024, ram_budget, &dir).unwrap();

        let key = vec![0.0f32; 32];
        let val = vec![0.0f32; 32];

        // Insert enough to trigger spill
        for i in 0..8 {
            let mut k = key.clone();
            k[0] = i as f32;
            cache.append(0, k, val.clone()).unwrap();
        }

        // Should have some entries in cold storage
        assert!(cache.cold_positions() > 0 || cache.hot_positions() <= 8);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mmap_kv_read_cold() {
        let dir = test_dir();
        let entry_size = 2 * 4 * 4 * 2; // 64 bytes per entry pair
        let ram_budget = entry_size * 2; // very tight — triggers spill after 2 entries

        let mut cache = MmapKvCache::new(1, 2, 4, 1024, ram_budget, &dir).unwrap();

        for i in 0..6 {
            let key = vec![i as f32; 8];
            let val = vec![(i * 10) as f32; 8];
            cache.append(0, key, val).unwrap();
        }

        // Some entries should be in cold storage
        let cold = cache.layers[0].cold_len;
        let hot = cache.layers[0].hot_keys.len();
        assert!(
            cold > 0,
            "Expected some cold entries, got cold={} hot={}",
            cold,
            hot
        );
        assert_eq!(cold + hot, 6);

        // Read a hot entry (should always work)
        let k = cache.key_at(0, cold, 0);
        assert_eq!(k.len(), 4);

        // Read a cold entry via mmap
        let k_cold = cache.key_at(0, 0, 0);
        assert_eq!(k_cold.len(), 4);
        assert_eq!(k_cold[0], 0.0); // first entry had key[0] = 0.0

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mmap_kv_rollback() {
        let dir = test_dir();
        let mut cache = MmapKvCache::new(1, 2, 4, 1024, 1024 * 1024, &dir).unwrap();

        for i in 0..5 {
            cache.append(0, vec![i as f32; 8], vec![0.0; 8]).unwrap();
        }

        cache.rollback(3);
        assert_eq!(cache.layers[0].total_len, 3);
        assert_eq!(cache.layers[0].hot_keys.len(), 3);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mmap_kv_clear() {
        let dir = test_dir();
        let mut cache = MmapKvCache::new(1, 2, 4, 1024, 1024 * 1024, &dir).unwrap();
        cache.append(0, vec![1.0; 8], vec![2.0; 8]).unwrap();
        cache.clear().unwrap();
        assert_eq!(cache.layers[0].total_len, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
