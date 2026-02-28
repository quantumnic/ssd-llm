//! Prompt prefix caching for KV state reuse
//!
//! When multiple requests share a common prompt prefix (e.g., system prompt),
//! the KV cache from prefill can be reused, skipping expensive SSD reads.
//! Uses a hash-trie structure keyed on token sequences.
//!
//! v1.36: Persistent prompt cache — save/load KV cache states to disk so that
//! system prompts survive server restarts. Uses a simple binary format:
//! `[magic:4][version:4][n_entries:4] [entry...]` where each entry is
//! `[hash:8][n_tokens:4][tokens...][n_layers:4][hidden_dim:4][hidden...][layer_data...]`.

use crate::inference::kv_cache::KvCache;
use std::collections::HashMap;
use std::io::{Read as _, Write as _};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info};

/// A cached KV state for a specific token prefix
#[derive(Clone)]
struct CachedKvState {
    /// The token sequence this cache covers
    tokens: Vec<u32>,
    /// Per-layer key caches: [layer][position][kv_dim]
    keys: Vec<Vec<Vec<f32>>>,
    /// Per-layer value caches: [layer][position][kv_dim]
    values: Vec<Vec<Vec<f32>>>,
    /// The hidden state after processing these tokens
    hidden_state: Vec<f32>,
    /// Last access time for eviction
    last_access: Instant,
    /// Size in bytes
    size_bytes: usize,
}

/// Prompt cache with LRU eviction
pub struct PromptCache {
    /// Cache entries keyed by hash of token prefix
    entries: HashMap<u64, CachedKvState>,
    /// Maximum cache size in bytes
    max_bytes: usize,
    /// Current cache size
    used_bytes: usize,
    /// Stats
    pub hits: u64,
    pub misses: u64,
    pub partial_hits: u64,
}

impl PromptCache {
    pub fn new(max_bytes: usize) -> Self {
        info!(
            "Prompt cache initialized: {:.1} MB budget",
            max_bytes as f64 / (1024.0 * 1024.0)
        );
        Self {
            entries: HashMap::new(),
            max_bytes,
            used_bytes: 0,
            hits: 0,
            misses: 0,
            partial_hits: 0,
        }
    }

    /// Try to find a cached KV state matching a token prefix.
    /// Returns (matched_length, hidden_state) and restores the KV cache.
    pub fn lookup(&mut self, tokens: &[u32], kv_cache: &mut KvCache) -> Option<(usize, Vec<f32>)> {
        // Try exact match first
        let hash = hash_tokens(tokens);
        if let Some(entry) = self.entries.get_mut(&hash) {
            if entry.tokens == tokens {
                entry.last_access = Instant::now();
                restore_kv_cache(kv_cache, &entry.keys, &entry.values);
                self.hits += 1;
                info!("Prompt cache: exact hit for {} tokens", tokens.len());
                return Some((tokens.len(), entry.hidden_state.clone()));
            }
        }

        // Try progressively shorter prefixes
        let mut best_match: Option<(usize, &CachedKvState)> = None;
        for len in (1..tokens.len()).rev() {
            let prefix = &tokens[..len];
            let prefix_hash = hash_tokens(prefix);
            if let Some(entry) = self.entries.get(&prefix_hash) {
                if entry.tokens == prefix {
                    best_match = Some((len, entry));
                    break;
                }
            }
        }

        if let Some((len, _entry)) = best_match {
            // Update last_access via hash lookup
            let prefix_hash = hash_tokens(&tokens[..len]);
            if let Some(entry) = self.entries.get_mut(&prefix_hash) {
                entry.last_access = Instant::now();
                restore_kv_cache(kv_cache, &entry.keys, &entry.values);
                self.partial_hits += 1;
                info!(
                    "Prompt cache: partial hit — {} of {} tokens cached",
                    len,
                    tokens.len()
                );
                return Some((len, entry.hidden_state.clone()));
            }
        }

        self.misses += 1;
        debug!("Prompt cache miss for {} tokens", tokens.len());
        None
    }

    /// Store a KV state for a token prefix
    pub fn store(&mut self, tokens: &[u32], kv_cache: &KvCache, hidden_state: &[f32]) {
        let hash = hash_tokens(tokens);

        // Calculate size
        let n_layers = kv_cache.n_layers();
        let mut keys = Vec::with_capacity(n_layers);
        let mut values = Vec::with_capacity(n_layers);
        let mut size_bytes = tokens.len() * 4 + hidden_state.len() * 4; // tokens + hidden

        for i in 0..n_layers {
            let layer = kv_cache.layer(i);
            keys.push(layer.keys.clone());
            values.push(layer.values.clone());
            size_bytes += layer.size_bytes();
        }

        // Evict old entries if needed
        while self.used_bytes + size_bytes > self.max_bytes && !self.entries.is_empty() {
            self.evict_oldest();
        }

        if size_bytes > self.max_bytes {
            debug!(
                "Prompt cache: entry too large ({:.1} MB), skipping",
                size_bytes as f64 / (1024.0 * 1024.0)
            );
            return;
        }

        self.entries.insert(
            hash,
            CachedKvState {
                tokens: tokens.to_vec(),
                keys,
                values,
                hidden_state: hidden_state.to_vec(),
                last_access: Instant::now(),
                size_bytes,
            },
        );
        self.used_bytes += size_bytes;

        debug!(
            "Prompt cache: stored {} tokens ({:.1} MB), total: {:.1}/{:.1} MB",
            tokens.len(),
            size_bytes as f64 / (1024.0 * 1024.0),
            self.used_bytes as f64 / (1024.0 * 1024.0),
            self.max_bytes as f64 / (1024.0 * 1024.0),
        );
    }

    fn evict_oldest(&mut self) {
        let oldest = self
            .entries
            .iter()
            .min_by_key(|(_, v)| v.last_access)
            .map(|(k, v)| (*k, v.size_bytes));

        if let Some((key, size)) = oldest {
            self.entries.remove(&key);
            self.used_bytes -= size;
            debug!(
                "Prompt cache: evicted entry ({:.1} MB)",
                size as f64 / (1024.0 * 1024.0)
            );
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Save all cached entries to a binary file on disk.
    ///
    /// Format: `[magic:4][version:4][n_entries:4] [entry...]`
    /// Each entry: `[hash:8][n_tokens:4][tokens:n_tokens*4][hidden_dim:4][hidden:hidden_dim*4]`
    ///             `[n_layers:4]` then per layer: `[seq_len:4][kv_dim:4][keys:seq_len*kv_dim*4][values:seq_len*kv_dim*4]`
    pub fn save_to_disk(&self, path: &Path) -> Result<usize, String> {
        let mut buf: Vec<u8> = Vec::new();
        // Magic: "PCCH" (Prompt Cache CHeckpoint)
        buf.extend_from_slice(b"PCCH");
        // Version
        buf.extend_from_slice(&1u32.to_le_bytes());
        // Number of entries
        let n_entries = self.entries.len() as u32;
        buf.extend_from_slice(&n_entries.to_le_bytes());

        for (&hash, entry) in &self.entries {
            // Hash
            buf.extend_from_slice(&hash.to_le_bytes());
            // Tokens
            let n_tokens = entry.tokens.len() as u32;
            buf.extend_from_slice(&n_tokens.to_le_bytes());
            for &t in &entry.tokens {
                buf.extend_from_slice(&t.to_le_bytes());
            }
            // Hidden state
            let hidden_dim = entry.hidden_state.len() as u32;
            buf.extend_from_slice(&hidden_dim.to_le_bytes());
            for &v in &entry.hidden_state {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            // Layers
            let n_layers = entry.keys.len() as u32;
            buf.extend_from_slice(&n_layers.to_le_bytes());
            for (keys, values) in entry.keys.iter().zip(entry.values.iter()) {
                let seq_len = keys.len() as u32;
                let kv_dim = if keys.is_empty() {
                    0u32
                } else {
                    keys[0].len() as u32
                };
                buf.extend_from_slice(&seq_len.to_le_bytes());
                buf.extend_from_slice(&kv_dim.to_le_bytes());
                for k in keys {
                    for &v in k {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
                for val in values {
                    for &v in val {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }

        // Write atomically: write to temp file, then rename
        let tmp_path = path.with_extension("tmp");
        let mut file = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("Failed to create cache file: {}", e))?;
        file.write_all(&buf)
            .map_err(|e| format!("Failed to write cache file: {}", e))?;
        file.sync_all()
            .map_err(|e| format!("Failed to sync cache file: {}", e))?;
        drop(file);
        std::fs::rename(&tmp_path, path)
            .map_err(|e| format!("Failed to rename cache file: {}", e))?;

        let saved = n_entries as usize;
        info!(
            "Prompt cache saved: {} entries ({:.1} MB) to {}",
            saved,
            buf.len() as f64 / (1024.0 * 1024.0),
            path.display()
        );
        Ok(saved)
    }

    /// Load cached entries from a binary file on disk.
    /// Merges into the current cache (existing entries are preserved).
    pub fn load_from_disk(&mut self, path: &Path) -> Result<usize, String> {
        if !path.exists() {
            debug!("No prompt cache file at {}", path.display());
            return Ok(0);
        }

        let data = std::fs::read(path).map_err(|e| format!("Failed to read cache file: {}", e))?;

        let mut cursor = &data[..];

        // Read and validate magic
        if cursor.len() < 12 {
            return Err("Cache file too small".to_string());
        }
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != b"PCCH" {
            return Err(format!("Invalid cache magic: {:?}", magic));
        }

        let version = read_u32(&mut cursor)?;
        if version != 1 {
            return Err(format!("Unsupported cache version: {}", version));
        }

        let n_entries = read_u32(&mut cursor)? as usize;
        let mut loaded = 0usize;

        for _ in 0..n_entries {
            let hash = read_u64(&mut cursor)?;
            let n_tokens = read_u32(&mut cursor)? as usize;
            let mut tokens = Vec::with_capacity(n_tokens);
            for _ in 0..n_tokens {
                tokens.push(read_u32(&mut cursor)?);
            }

            let hidden_dim = read_u32(&mut cursor)? as usize;
            let mut hidden_state = Vec::with_capacity(hidden_dim);
            for _ in 0..hidden_dim {
                hidden_state.push(read_f32(&mut cursor)?);
            }

            let n_layers = read_u32(&mut cursor)? as usize;
            let mut keys = Vec::with_capacity(n_layers);
            let mut values = Vec::with_capacity(n_layers);
            let mut size_bytes = n_tokens * 4 + hidden_dim * 4;

            for _ in 0..n_layers {
                let seq_len = read_u32(&mut cursor)? as usize;
                let kv_dim = read_u32(&mut cursor)? as usize;

                let mut layer_keys = Vec::with_capacity(seq_len);
                for _ in 0..seq_len {
                    let mut k = Vec::with_capacity(kv_dim);
                    for _ in 0..kv_dim {
                        k.push(read_f32(&mut cursor)?);
                    }
                    layer_keys.push(k);
                }

                let mut layer_values = Vec::with_capacity(seq_len);
                for _ in 0..seq_len {
                    let mut v = Vec::with_capacity(kv_dim);
                    for _ in 0..kv_dim {
                        v.push(read_f32(&mut cursor)?);
                    }
                    layer_values.push(v);
                }

                size_bytes += seq_len * kv_dim * 4 * 2; // keys + values
                keys.push(layer_keys);
                values.push(layer_values);
            }

            // Skip if already present or would exceed budget
            if self.entries.contains_key(&hash) {
                continue;
            }
            if self.used_bytes + size_bytes > self.max_bytes {
                debug!("Prompt cache: skipping disk entry (would exceed budget)");
                continue;
            }

            self.entries.insert(
                hash,
                CachedKvState {
                    tokens,
                    keys,
                    values,
                    hidden_state,
                    last_access: Instant::now(),
                    size_bytes,
                },
            );
            self.used_bytes += size_bytes;
            loaded += 1;
        }

        info!(
            "Prompt cache loaded: {} of {} entries from {} ({:.1} MB used)",
            loaded,
            n_entries,
            path.display(),
            self.used_bytes as f64 / (1024.0 * 1024.0),
        );
        Ok(loaded)
    }
}

/// Read a u32 from a byte slice cursor
fn read_u32(cursor: &mut &[u8]) -> Result<u32, String> {
    if cursor.len() < 4 {
        return Err("Unexpected end of cache data (u32)".to_string());
    }
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a u64 from a byte slice cursor
fn read_u64(cursor: &mut &[u8]) -> Result<u64, String> {
    if cursor.len() < 8 {
        return Err("Unexpected end of cache data (u64)".to_string());
    }
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(buf))
}

/// Read an f32 from a byte slice cursor
fn read_f32(cursor: &mut &[u8]) -> Result<f32, String> {
    if cursor.len() < 4 {
        return Err("Unexpected end of cache data (f32)".to_string());
    }
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(f32::from_le_bytes(buf))
}

/// Restore KV cache from stored state
fn restore_kv_cache(kv_cache: &mut KvCache, keys: &[Vec<Vec<f32>>], values: &[Vec<Vec<f32>>]) {
    kv_cache.clear();
    for (layer_idx, (k, v)) in keys.iter().zip(values.iter()).enumerate() {
        let layer = kv_cache.layer_mut(layer_idx);
        for (ki, vi) in k.iter().zip(v.iter()) {
            layer.append(ki.clone(), vi.clone());
        }
    }
}

/// FNV-1a hash for token sequences
fn hash_tokens(tokens: &[u32]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &t in tokens {
        let bytes = t.to_le_bytes();
        for &b in &bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::KvCache;

    #[test]
    fn test_hash_consistency() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        assert_eq!(hash_tokens(&tokens), hash_tokens(&tokens));
        assert_ne!(hash_tokens(&tokens), hash_tokens(&[1, 2, 3, 4, 6]));
    }

    #[test]
    fn test_prompt_cache_store_and_lookup() {
        let mut cache = PromptCache::new(100 * 1024 * 1024); // 100MB
        let tokens = vec![1u32, 2, 3];
        let hidden = vec![0.5f32; 64];

        let mut kv = KvCache::new(2, 4, 16, 512);
        // Add some data to KV cache
        kv.layer_mut(0).append(vec![1.0; 64], vec![2.0; 64]);
        kv.layer_mut(0).append(vec![1.0; 64], vec![2.0; 64]);
        kv.layer_mut(1).append(vec![3.0; 64], vec![4.0; 64]);
        kv.layer_mut(1).append(vec![3.0; 64], vec![4.0; 64]);

        cache.store(&tokens, &kv, &hidden);

        // Lookup exact match
        let mut kv2 = KvCache::new(2, 4, 16, 512);
        let result = cache.lookup(&tokens, &mut kv2);
        assert!(result.is_some());
        let (len, h) = result.unwrap();
        assert_eq!(len, 3);
        assert_eq!(h, hidden);
        assert_eq!(kv2.layer(0).seq_len(), 2);
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn test_prompt_cache_partial_hit() {
        let mut cache = PromptCache::new(100 * 1024 * 1024);
        let prefix = vec![1u32, 2, 3];
        let full = vec![1u32, 2, 3, 4, 5];
        let hidden = vec![0.5f32; 64];

        let kv = KvCache::new(1, 2, 8, 512);
        cache.store(&prefix, &kv, &hidden);

        let mut kv2 = KvCache::new(1, 2, 8, 512);
        let result = cache.lookup(&full, &mut kv2);
        assert!(result.is_some());
        let (len, _) = result.unwrap();
        assert_eq!(len, 3); // matched prefix only
        assert_eq!(cache.partial_hits, 1);
    }

    #[test]
    fn test_prompt_cache_save_load_roundtrip() {
        let mut cache = PromptCache::new(100 * 1024 * 1024);
        let tokens1 = vec![1u32, 2, 3];
        let tokens2 = vec![10u32, 20, 30, 40];
        let hidden1 = vec![0.5f32; 64];
        let hidden2 = vec![-1.0f32; 64];

        let mut kv1 = KvCache::new(2, 4, 16, 512);
        kv1.layer_mut(0).append(vec![1.0; 64], vec![2.0; 64]);
        kv1.layer_mut(0).append(vec![3.0; 64], vec![4.0; 64]);
        kv1.layer_mut(1).append(vec![5.0; 64], vec![6.0; 64]);

        let mut kv2 = KvCache::new(2, 4, 16, 512);
        kv2.layer_mut(0).append(vec![7.0; 64], vec![8.0; 64]);
        kv2.layer_mut(1).append(vec![9.0; 64], vec![10.0; 64]);

        cache.store(&tokens1, &kv1, &hidden1);
        cache.store(&tokens2, &kv2, &hidden2);
        assert_eq!(cache.len(), 2);

        // Save to disk
        let tmp_dir = std::env::temp_dir();
        let cache_path = tmp_dir.join("ssd_llm_test_prompt_cache.bin");
        let saved = cache.save_to_disk(&cache_path).unwrap();
        assert_eq!(saved, 2);

        // Load into a fresh cache
        let mut cache2 = PromptCache::new(100 * 1024 * 1024);
        let loaded = cache2.load_from_disk(&cache_path).unwrap();
        assert_eq!(loaded, 2);
        assert_eq!(cache2.len(), 2);

        // Verify lookups work
        let mut kv_out = KvCache::new(2, 4, 16, 512);
        let result = cache2.lookup(&tokens1, &mut kv_out);
        assert!(result.is_some());
        let (len, h) = result.unwrap();
        assert_eq!(len, 3);
        assert_eq!(h, hidden1);
        assert_eq!(kv_out.layer(0).seq_len(), 2);
        assert_eq!(kv_out.layer(1).seq_len(), 1);

        let mut kv_out2 = KvCache::new(2, 4, 16, 512);
        let result2 = cache2.lookup(&tokens2, &mut kv_out2);
        assert!(result2.is_some());
        let (len2, h2) = result2.unwrap();
        assert_eq!(len2, 4);
        assert_eq!(h2, hidden2);

        // Cleanup
        let _ = std::fs::remove_file(&cache_path);
    }

    #[test]
    fn test_prompt_cache_save_empty() {
        let cache = PromptCache::new(1024);
        let tmp_dir = std::env::temp_dir();
        let cache_path = tmp_dir.join("ssd_llm_test_empty_cache.bin");
        let saved = cache.save_to_disk(&cache_path).unwrap();
        assert_eq!(saved, 0);

        let mut cache2 = PromptCache::new(1024);
        let loaded = cache2.load_from_disk(&cache_path).unwrap();
        assert_eq!(loaded, 0);
        assert_eq!(cache2.len(), 0);

        let _ = std::fs::remove_file(&cache_path);
    }

    #[test]
    fn test_prompt_cache_load_nonexistent() {
        let mut cache = PromptCache::new(1024);
        let result = cache.load_from_disk(Path::new("/nonexistent/path/cache.bin"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_prompt_cache_load_invalid_magic() {
        let tmp_dir = std::env::temp_dir();
        let cache_path = tmp_dir.join("ssd_llm_test_bad_magic.bin");
        std::fs::write(&cache_path, b"BADMxxxxxxxx").unwrap();

        let mut cache = PromptCache::new(1024);
        let result = cache.load_from_disk(&cache_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid cache magic"));

        let _ = std::fs::remove_file(&cache_path);
    }

    #[test]
    fn test_prompt_cache_load_budget_respected() {
        // Save a large entry, then load into a tiny-budget cache
        let mut cache = PromptCache::new(100 * 1024 * 1024);
        let tokens = vec![1u32, 2, 3];
        let hidden = vec![0.5f32; 256];

        let mut kv = KvCache::new(2, 4, 16, 512);
        kv.layer_mut(0).append(vec![1.0; 64], vec![2.0; 64]);
        kv.layer_mut(1).append(vec![3.0; 64], vec![4.0; 64]);

        cache.store(&tokens, &kv, &hidden);

        let tmp_dir = std::env::temp_dir();
        let cache_path = tmp_dir.join("ssd_llm_test_budget_cache.bin");
        cache.save_to_disk(&cache_path).unwrap();

        // Load into cache with tiny budget — entry should be skipped
        let mut tiny = PromptCache::new(32); // 32 bytes, way too small
        let loaded = tiny.load_from_disk(&cache_path).unwrap();
        assert_eq!(loaded, 0);
        assert_eq!(tiny.len(), 0);

        let _ = std::fs::remove_file(&cache_path);
    }

    #[test]
    fn test_prompt_cache_eviction() {
        let mut cache = PromptCache::new(1024); // tiny budget
        let tokens1 = vec![1u32, 2];
        let tokens2 = vec![3u32, 4];
        let hidden = vec![0.0f32; 64];

        let kv = KvCache::new(1, 1, 4, 16);
        cache.store(&tokens1, &kv, &hidden);
        cache.store(&tokens2, &kv, &hidden);
        // At least one should have been evicted or stored
        assert!(cache.len() >= 1);
    }
}
