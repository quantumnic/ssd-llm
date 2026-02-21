//! SSD Block Swapping for PagedAttention
//!
//! Enables swapping cold KV cache blocks to SSD when GPU/RAM memory is under pressure,
//! and prefetching them back before they're needed. This is the key integration between
//! PagedAttention's block-based memory management and ssd-llm's SSD offloading thesis.
//!
//! Architecture:
//! ```
//! PagedKvCache ←→ BlockSwapper ←→ SSD (swap file)
//!                     ↑
//!              SwapScheduler (LRU eviction + prefetch)
//! ```
//!
//! Swap file format (per block):
//!   [keys: block_size * kv_dim * f32] [values: block_size * kv_dim * f32] [num_filled: u32]

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use tracing::{debug, info, warn};

/// Quantization mode for SSD block swapping
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SwapQuantMode {
    /// Store blocks as raw f32 (no compression)
    #[default]
    None,
    /// Quantize to INT8 with per-vector absmax scale (4x size reduction)
    Int8,
}

/// Quantize an f32 slice to INT8 with absmax scaling.
/// Returns (quantized_data, scales) where each row of `dim` elements gets one scale.
fn quantize_vectors_int8(data: &[f32], dim: usize) -> (Vec<i8>, Vec<f32>) {
    let n_rows = data.len() / dim;
    let mut quantized = vec![0i8; data.len()];
    let mut scales = vec![0.0f32; n_rows];

    for (row, scale) in scales.iter_mut().enumerate().take(n_rows) {
        let start = row * dim;
        let end = start + dim;
        let row_data = &data[start..end];

        let absmax = row_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if absmax == 0.0 {
            continue;
        }

        let inv_scale = 127.0 / absmax;
        *scale = absmax / 127.0;

        for (i, &v) in row_data.iter().enumerate() {
            quantized[start + i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }
    }

    (quantized, scales)
}

/// Dequantize INT8 data back to f32 using per-vector scales.
fn dequantize_vectors_int8(quantized: &[i8], scales: &[f32], dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; quantized.len()];

    for (row, &scale) in scales.iter().enumerate() {
        let start = row * dim;
        for i in 0..dim {
            output[start + i] = quantized[start + i] as f32 * scale;
        }
    }

    output
}

/// Identity of a block in the paged cache: (layer_index, physical_block_id)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId {
    pub layer_idx: usize,
    pub block_id: usize,
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L{}:B{}", self.layer_idx, self.block_id)
    }
}

/// A block's data extracted for swap operations
#[derive(Clone)]
pub struct BlockData {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub num_filled: usize,
}

/// Slot in the swap file where a block is stored
#[derive(Debug, Clone, Copy)]
struct SwapSlot {
    /// Offset in the swap file (bytes)
    offset: u64,
    /// Size in bytes (keys + values + metadata)
    size: usize,
}

/// Manages the SSD swap file and block I/O
pub struct SwapFile {
    /// Path to the swap file
    path: PathBuf,
    /// Open file handle
    file: File,
    /// Block size (tokens per block)
    block_size: usize,
    /// KV dimension (n_kv_heads * head_dim)
    kv_dim: usize,
    /// Size of one serialized block in bytes
    slot_size: usize,
    /// Map from block ID to swap slot
    slots: HashMap<BlockId, SwapSlot>,
    /// Free slot offsets available for reuse
    free_slots: VecDeque<u64>,
    /// Next offset to allocate if no free slots
    next_offset: u64,
    /// Total bytes written
    total_bytes_written: u64,
    /// Total bytes read
    total_bytes_read: u64,
    /// Quantization mode for SSD storage
    quant_mode: SwapQuantMode,
}

impl SwapFile {
    /// Create or open a swap file at the given path
    pub fn new(swap_dir: &Path, block_size: usize, kv_dim: usize) -> io::Result<Self> {
        Self::with_quant_mode(swap_dir, block_size, kv_dim, SwapQuantMode::None)
    }

    /// Create a swap file with the specified quantization mode
    pub fn with_quant_mode(
        swap_dir: &Path,
        block_size: usize,
        kv_dim: usize,
        quant_mode: SwapQuantMode,
    ) -> io::Result<Self> {
        fs::create_dir_all(swap_dir)?;
        let path = swap_dir.join("paged_kv_swap.bin");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        let float_count = block_size * kv_dim * 2; // keys + values
        let n_rows = block_size * 2; // one scale per row (key row + value row)

        let slot_size = match quant_mode {
            SwapQuantMode::None => {
                // keys (f32) + values (f32) + num_filled (u32)
                float_count * std::mem::size_of::<f32>() + std::mem::size_of::<u32>()
            }
            SwapQuantMode::Int8 => {
                // keys (i8) + values (i8) + scales (f32 per row) + num_filled (u32)
                float_count * std::mem::size_of::<i8>()
                    + n_rows * std::mem::size_of::<f32>()
                    + std::mem::size_of::<u32>()
            }
        };

        let mode_str = match quant_mode {
            SwapQuantMode::None => "f32",
            SwapQuantMode::Int8 => "INT8",
        };
        info!(
            "SSD block swap file: {:?} (slot_size={} bytes, {:.1} KB/block, mode={})",
            path,
            slot_size,
            slot_size as f64 / 1024.0,
            mode_str,
        );

        Ok(Self {
            path,
            file,
            block_size,
            kv_dim,
            slot_size,
            slots: HashMap::new(),
            free_slots: VecDeque::new(),
            next_offset: 0,
            total_bytes_written: 0,
            total_bytes_read: 0,
            quant_mode,
        })
    }

    /// Write a block's data to the swap file
    pub fn write_block(&mut self, id: BlockId, data: &BlockData) -> io::Result<()> {
        let offset = if let Some(free_offset) = self.free_slots.pop_front() {
            free_offset
        } else {
            let o = self.next_offset;
            self.next_offset += self.slot_size as u64;
            o
        };

        self.file.seek(SeekFrom::Start(offset))?;

        match self.quant_mode {
            SwapQuantMode::None => {
                // Write keys as raw f32 bytes
                let key_bytes = unsafe {
                    std::slice::from_raw_parts(
                        data.keys.as_ptr() as *const u8,
                        data.keys.len() * std::mem::size_of::<f32>(),
                    )
                };
                self.file.write_all(key_bytes)?;

                // Write values as raw f32 bytes
                let val_bytes = unsafe {
                    std::slice::from_raw_parts(
                        data.values.as_ptr() as *const u8,
                        data.values.len() * std::mem::size_of::<f32>(),
                    )
                };
                self.file.write_all(val_bytes)?;
            }
            SwapQuantMode::Int8 => {
                // Quantize keys and values to INT8
                let (q_keys, key_scales) = quantize_vectors_int8(&data.keys, self.kv_dim);
                let (q_values, val_scales) = quantize_vectors_int8(&data.values, self.kv_dim);

                // Write quantized keys (i8)
                let key_bytes = unsafe {
                    std::slice::from_raw_parts(q_keys.as_ptr() as *const u8, q_keys.len())
                };
                self.file.write_all(key_bytes)?;

                // Write quantized values (i8)
                let val_bytes = unsafe {
                    std::slice::from_raw_parts(q_values.as_ptr() as *const u8, q_values.len())
                };
                self.file.write_all(val_bytes)?;

                // Write scales (key scales then value scales, f32 each)
                let scale_bytes = unsafe {
                    std::slice::from_raw_parts(
                        key_scales.as_ptr() as *const u8,
                        key_scales.len() * std::mem::size_of::<f32>(),
                    )
                };
                self.file.write_all(scale_bytes)?;
                let scale_bytes = unsafe {
                    std::slice::from_raw_parts(
                        val_scales.as_ptr() as *const u8,
                        val_scales.len() * std::mem::size_of::<f32>(),
                    )
                };
                self.file.write_all(scale_bytes)?;
            }
        }

        // Write num_filled as u32
        self.file
            .write_all(&(data.num_filled as u32).to_le_bytes())?;

        self.slots.insert(
            id,
            SwapSlot {
                offset,
                size: self.slot_size,
            },
        );
        self.total_bytes_written += self.slot_size as u64;

        debug!("Swapped out block {} to offset {}", id, offset);
        Ok(())
    }

    /// Read a block's data back from the swap file
    pub fn read_block(&mut self, id: BlockId) -> io::Result<BlockData> {
        let slot = self.slots.get(&id).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("block {id} not in swap"))
        })?;

        self.file.seek(SeekFrom::Start(slot.offset))?;

        let float_count = self.block_size * self.kv_dim;

        let (keys, values) = match self.quant_mode {
            SwapQuantMode::None => {
                // Read keys
                let mut keys = vec![0.0f32; float_count];
                let key_bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        keys.as_mut_ptr() as *mut u8,
                        float_count * std::mem::size_of::<f32>(),
                    )
                };
                self.file.read_exact(key_bytes)?;

                // Read values
                let mut values = vec![0.0f32; float_count];
                let val_bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        values.as_mut_ptr() as *mut u8,
                        float_count * std::mem::size_of::<f32>(),
                    )
                };
                self.file.read_exact(val_bytes)?;

                (keys, values)
            }
            SwapQuantMode::Int8 => {
                let n_rows = self.block_size;

                // Read quantized keys (i8)
                let mut q_keys = vec![0i8; float_count];
                let key_bytes = unsafe {
                    std::slice::from_raw_parts_mut(q_keys.as_mut_ptr() as *mut u8, float_count)
                };
                self.file.read_exact(key_bytes)?;

                // Read quantized values (i8)
                let mut q_values = vec![0i8; float_count];
                let val_bytes = unsafe {
                    std::slice::from_raw_parts_mut(q_values.as_mut_ptr() as *mut u8, float_count)
                };
                self.file.read_exact(val_bytes)?;

                // Read scales
                let mut key_scales = vec![0.0f32; n_rows];
                let scale_bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        key_scales.as_mut_ptr() as *mut u8,
                        n_rows * std::mem::size_of::<f32>(),
                    )
                };
                self.file.read_exact(scale_bytes)?;

                let mut val_scales = vec![0.0f32; n_rows];
                let scale_bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        val_scales.as_mut_ptr() as *mut u8,
                        n_rows * std::mem::size_of::<f32>(),
                    )
                };
                self.file.read_exact(scale_bytes)?;

                // Dequantize
                let keys = dequantize_vectors_int8(&q_keys, &key_scales, self.kv_dim);
                let values = dequantize_vectors_int8(&q_values, &val_scales, self.kv_dim);

                (keys, values)
            }
        };

        // Read num_filled
        let mut num_filled_bytes = [0u8; 4];
        self.file.read_exact(&mut num_filled_bytes)?;
        let num_filled = u32::from_le_bytes(num_filled_bytes) as usize;

        self.total_bytes_read += self.slot_size as u64;

        debug!("Swapped in block {} from offset {}", id, slot.offset);
        Ok(BlockData {
            keys,
            values,
            num_filled,
        })
    }

    /// Free a swap slot (block was evicted or sequence removed)
    pub fn free_block(&mut self, id: BlockId) {
        if let Some(slot) = self.slots.remove(&id) {
            self.free_slots.push_back(slot.offset);
            debug!("Freed swap slot for block {}", id);
        }
    }

    /// Check if a block is on SSD
    pub fn contains(&self, id: &BlockId) -> bool {
        self.slots.contains_key(id)
    }

    /// Number of blocks currently on SSD
    pub fn num_swapped(&self) -> usize {
        self.slots.len()
    }

    /// Swap file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the quantization mode
    pub fn quant_mode(&self) -> SwapQuantMode {
        self.quant_mode
    }

    /// Total I/O stats
    pub fn stats(&self) -> SwapFileStats {
        SwapFileStats {
            num_swapped_blocks: self.slots.len(),
            file_size_bytes: self.next_offset,
            total_written_bytes: self.total_bytes_written,
            total_read_bytes: self.total_bytes_read,
            free_slots: self.free_slots.len(),
            quant_mode: self.quant_mode,
        }
    }
}

impl Drop for SwapFile {
    fn drop(&mut self) {
        // Clean up swap file
        if let Err(e) = fs::remove_file(&self.path) {
            warn!("Failed to clean up swap file {:?}: {}", self.path, e);
        } else {
            info!("Cleaned up swap file {:?}", self.path);
        }
    }
}

/// LRU tracker for deciding which blocks to swap out
pub struct SwapScheduler {
    /// LRU order: front = least recently used, back = most recently used
    lru_order: VecDeque<BlockId>,
    /// Set for O(1) membership check
    lru_set: HashMap<BlockId, ()>,
    /// High watermark: start swapping when used blocks exceed this fraction
    high_watermark: f64,
    /// Low watermark: swap until used blocks drop below this fraction
    low_watermark: f64,
    /// Minimum number of blocks to keep in RAM (never swap these)
    min_resident: usize,
}

impl SwapScheduler {
    /// Create a new swap scheduler
    ///
    /// `high_watermark` — fraction of total blocks that triggers swap-out (e.g. 0.9)
    /// `low_watermark` — fraction to swap down to (e.g. 0.7)
    /// `min_resident` — always keep at least this many blocks in RAM
    pub fn new(high_watermark: f64, low_watermark: f64, min_resident: usize) -> Self {
        Self {
            lru_order: VecDeque::new(),
            lru_set: HashMap::new(),
            high_watermark,
            low_watermark,
            min_resident,
        }
    }

    /// Record a block access (moves to MRU position)
    pub fn touch(&mut self, id: BlockId) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.lru_set.entry(id) {
            e.insert(());
        } else {
            // Remove from current position and re-add at back
            self.lru_order.retain(|x| *x != id);
        }
        self.lru_order.push_back(id);
    }

    /// Remove a block from tracking (when freed entirely)
    pub fn remove(&mut self, id: &BlockId) {
        if self.lru_set.remove(id).is_some() {
            self.lru_order.retain(|x| x != id);
        }
    }

    /// Determine which blocks should be swapped out given current memory pressure
    ///
    /// `total_blocks` — total capacity
    /// `used_blocks` — currently used (in RAM)
    /// `already_swapped` — blocks already on SSD (not counting against RAM)
    ///
    /// Returns list of block IDs to swap out (LRU order)
    pub fn blocks_to_swap_out(
        &self,
        total_blocks: usize,
        used_blocks: usize,
        already_swapped: &HashMap<BlockId, ()>,
    ) -> Vec<BlockId> {
        let usage_ratio = used_blocks as f64 / total_blocks.max(1) as f64;

        if usage_ratio < self.high_watermark {
            return Vec::new();
        }

        let target_used = (total_blocks as f64 * self.low_watermark) as usize;
        let to_swap = used_blocks.saturating_sub(target_used.max(self.min_resident));

        let mut result = Vec::new();
        for id in &self.lru_order {
            if result.len() >= to_swap {
                break;
            }
            // Don't re-swap blocks that are already on SSD
            if !already_swapped.contains_key(id) {
                result.push(*id);
            }
        }

        if !result.is_empty() {
            debug!(
                "SwapScheduler: usage {:.1}% > {:.1}%, swapping {} blocks to SSD",
                usage_ratio * 100.0,
                self.high_watermark * 100.0,
                result.len()
            );
        }

        result
    }

    /// Predict which swapped blocks will be needed soon for prefetching
    ///
    /// Uses the MRU (most recently used) pattern to guess upcoming blocks.
    /// For sequential decoding, the next blocks in each active sequence are likely.
    pub fn blocks_to_prefetch(
        &self,
        swapped_blocks: &HashMap<BlockId, ()>,
        max_prefetch: usize,
    ) -> Vec<BlockId> {
        // Prefetch the most recently used swapped blocks (they're likely to be
        // accessed again soon, e.g., in iterative decoding patterns)
        let mut result = Vec::new();
        for id in self.lru_order.iter().rev() {
            if result.len() >= max_prefetch {
                break;
            }
            if swapped_blocks.contains_key(id) {
                result.push(*id);
            }
        }
        result
    }

    /// Number of tracked blocks
    pub fn num_tracked(&self) -> usize {
        self.lru_set.len()
    }
}

/// The main block swapper that coordinates swap-out/swap-in
pub struct BlockSwapper {
    /// Swap file for SSD I/O
    swap_file: SwapFile,
    /// LRU-based scheduler
    scheduler: SwapScheduler,
    /// Set of blocks currently on SSD (not in RAM)
    swapped_set: HashMap<BlockId, ()>,
    /// Stats
    total_swap_outs: u64,
    total_swap_ins: u64,
}

impl BlockSwapper {
    /// Create a new block swapper
    pub fn new(
        swap_dir: &Path,
        block_size: usize,
        kv_dim: usize,
        high_watermark: f64,
        low_watermark: f64,
        min_resident: usize,
    ) -> io::Result<Self> {
        Self::with_quant_mode(
            swap_dir,
            block_size,
            kv_dim,
            high_watermark,
            low_watermark,
            min_resident,
            SwapQuantMode::None,
        )
    }

    /// Create a new block swapper with quantized SSD storage
    pub fn with_quant_mode(
        swap_dir: &Path,
        block_size: usize,
        kv_dim: usize,
        high_watermark: f64,
        low_watermark: f64,
        min_resident: usize,
        quant_mode: SwapQuantMode,
    ) -> io::Result<Self> {
        let swap_file = SwapFile::with_quant_mode(swap_dir, block_size, kv_dim, quant_mode)?;
        let scheduler = SwapScheduler::new(high_watermark, low_watermark, min_resident);

        Ok(Self {
            swap_file,
            scheduler,
            swapped_set: HashMap::new(),
            total_swap_outs: 0,
            total_swap_ins: 0,
        })
    }

    /// Record that a block was accessed (for LRU tracking)
    pub fn touch(&mut self, id: BlockId) {
        self.scheduler.touch(id);
    }

    /// Swap a block out to SSD
    pub fn swap_out(&mut self, id: BlockId, data: &BlockData) -> io::Result<()> {
        self.swap_file.write_block(id, data)?;
        self.swapped_set.insert(id, ());
        self.total_swap_outs += 1;
        Ok(())
    }

    /// Swap a block back from SSD to RAM
    pub fn swap_in(&mut self, id: BlockId) -> io::Result<BlockData> {
        let data = self.swap_file.read_block(id)?;
        self.swapped_set.remove(&id);
        self.swap_file.free_block(id);
        self.total_swap_ins += 1;
        Ok(data)
    }

    /// Check if a block is currently swapped out
    pub fn is_swapped(&self, id: &BlockId) -> bool {
        self.swapped_set.contains_key(id)
    }

    /// Get blocks that should be swapped out based on memory pressure
    pub fn get_swap_out_candidates(&self, total_blocks: usize, used_blocks: usize) -> Vec<BlockId> {
        self.scheduler
            .blocks_to_swap_out(total_blocks, used_blocks, &self.swapped_set)
    }

    /// Get blocks that should be prefetched from SSD
    pub fn get_prefetch_candidates(&self, max_prefetch: usize) -> Vec<BlockId> {
        self.scheduler
            .blocks_to_prefetch(&self.swapped_set, max_prefetch)
    }

    /// Remove a block from all tracking (when sequence is removed)
    pub fn remove_block(&mut self, id: &BlockId) {
        self.scheduler.remove(id);
        if self.swapped_set.remove(id).is_some() {
            self.swap_file.free_block(*id);
        }
    }

    /// Get swap statistics
    pub fn stats(&self) -> BlockSwapStats {
        let file_stats = self.swap_file.stats();
        BlockSwapStats {
            blocks_on_ssd: self.swapped_set.len(),
            total_swap_outs: self.total_swap_outs,
            total_swap_ins: self.total_swap_ins,
            file_size_bytes: file_stats.file_size_bytes,
            total_written_bytes: file_stats.total_written_bytes,
            total_read_bytes: file_stats.total_read_bytes,
            tracked_blocks: self.scheduler.num_tracked(),
        }
    }
}

/// Statistics for the block swapper
#[derive(Debug, Clone)]
pub struct BlockSwapStats {
    pub blocks_on_ssd: usize,
    pub total_swap_outs: u64,
    pub total_swap_ins: u64,
    pub file_size_bytes: u64,
    pub total_written_bytes: u64,
    pub total_read_bytes: u64,
    pub tracked_blocks: usize,
}

/// Statistics for the swap file
#[derive(Debug, Clone)]
pub struct SwapFileStats {
    pub num_swapped_blocks: usize,
    pub file_size_bytes: u64,
    pub total_written_bytes: u64,
    pub total_read_bytes: u64,
    pub free_slots: usize,
    pub quant_mode: SwapQuantMode,
}

impl std::fmt::Display for BlockSwapStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SSD Swap: {} blocks on disk, {} out/{} in, {:.1} MB written, {:.1} MB read",
            self.blocks_on_ssd,
            self.total_swap_outs,
            self.total_swap_ins,
            self.total_written_bytes as f64 / (1024.0 * 1024.0),
            self.total_read_bytes as f64 / (1024.0 * 1024.0),
        )
    }
}

/// Compute the compression ratio of INT8 vs f32 swap storage
pub fn swap_compression_ratio(block_size: usize, kv_dim: usize) -> f64 {
    let float_count = block_size * kv_dim * 2;
    let n_rows = block_size * 2;
    let f32_size = float_count * 4 + 4; // f32 data + num_filled
    let int8_size = float_count + n_rows * 4 + 4; // i8 data + scales + num_filled
    f32_size as f64 / int8_size as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    use std::sync::atomic::{AtomicU64, Ordering};
    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_swap_dir() -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = env::temp_dir().join(format!("ssd_llm_swap_test_{}_{id}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_swap_file_write_read() {
        let dir = temp_swap_dir();
        let block_size = 4;
        let kv_dim = 8;
        let mut sf = SwapFile::new(&dir, block_size, kv_dim).unwrap();

        let id = BlockId {
            layer_idx: 0,
            block_id: 5,
        };
        let data = BlockData {
            keys: (0..block_size * kv_dim).map(|i| i as f32 * 0.1).collect(),
            values: (0..block_size * kv_dim).map(|i| i as f32 * 0.2).collect(),
            num_filled: 3,
        };

        sf.write_block(id, &data).unwrap();
        assert!(sf.contains(&id));
        assert_eq!(sf.num_swapped(), 1);

        let recovered = sf.read_block(id).unwrap();
        assert_eq!(recovered.num_filled, 3);
        assert_eq!(recovered.keys.len(), data.keys.len());
        for (a, b) in recovered.keys.iter().zip(data.keys.iter()) {
            assert!((a - b).abs() < 1e-6, "key mismatch: {} vs {}", a, b);
        }
        for (a, b) in recovered.values.iter().zip(data.values.iter()) {
            assert!((a - b).abs() < 1e-6, "value mismatch: {} vs {}", a, b);
        }

        cleanup(&dir);
    }

    #[test]
    fn test_swap_file_slot_reuse() {
        let dir = temp_swap_dir();
        let mut sf = SwapFile::new(&dir, 2, 4).unwrap();

        let id1 = BlockId {
            layer_idx: 0,
            block_id: 0,
        };
        let id2 = BlockId {
            layer_idx: 0,
            block_id: 1,
        };
        let id3 = BlockId {
            layer_idx: 0,
            block_id: 2,
        };

        let data = BlockData {
            keys: vec![1.0; 8],
            values: vec![2.0; 8],
            num_filled: 2,
        };

        sf.write_block(id1, &data).unwrap();
        sf.write_block(id2, &data).unwrap();

        // Free id1, write id3 — should reuse id1's slot
        let offset_before = sf.next_offset;
        sf.free_block(id1);
        sf.write_block(id3, &data).unwrap();
        assert_eq!(sf.next_offset, offset_before); // No new space allocated

        assert!(!sf.contains(&id1));
        assert!(sf.contains(&id2));
        assert!(sf.contains(&id3));

        cleanup(&dir);
    }

    #[test]
    fn test_swap_scheduler_lru() {
        let mut sched = SwapScheduler::new(0.8, 0.5, 2);

        let b0 = BlockId {
            layer_idx: 0,
            block_id: 0,
        };
        let b1 = BlockId {
            layer_idx: 0,
            block_id: 1,
        };
        let b2 = BlockId {
            layer_idx: 0,
            block_id: 2,
        };
        let b3 = BlockId {
            layer_idx: 0,
            block_id: 3,
        };

        // Access order: b0, b1, b2, b3 → LRU = b0
        sched.touch(b0);
        sched.touch(b1);
        sched.touch(b2);
        sched.touch(b3);

        let empty = HashMap::new();
        // 9/10 used = 90% > 80% watermark → should suggest swaps
        let to_swap = sched.blocks_to_swap_out(10, 9, &empty);
        // Target = 50% of 10 = 5, but min_resident = 2
        // to_swap = 9 - 5 = 4 blocks
        assert_eq!(to_swap.len(), 4);
        assert_eq!(to_swap[0], b0); // LRU first
        assert_eq!(to_swap[1], b1);
    }

    #[test]
    fn test_swap_scheduler_no_swap_below_watermark() {
        let sched = SwapScheduler::new(0.8, 0.5, 1);
        let empty = HashMap::new();
        // 5/10 = 50% < 80% → no swap needed
        let to_swap = sched.blocks_to_swap_out(10, 5, &empty);
        assert!(to_swap.is_empty());
    }

    #[test]
    fn test_swap_scheduler_touch_updates_lru() {
        let mut sched = SwapScheduler::new(0.8, 0.5, 0);

        let b0 = BlockId {
            layer_idx: 0,
            block_id: 0,
        };
        let b1 = BlockId {
            layer_idx: 0,
            block_id: 1,
        };

        sched.touch(b0);
        sched.touch(b1);
        // Re-touch b0 → now b1 is LRU
        sched.touch(b0);

        let empty = HashMap::new();
        let to_swap = sched.blocks_to_swap_out(10, 9, &empty);
        assert!(!to_swap.is_empty());
        assert_eq!(to_swap[0], b1); // b1 is now LRU
    }

    #[test]
    fn test_block_swapper_roundtrip() {
        let dir = temp_swap_dir();
        let mut swapper = BlockSwapper::new(&dir, 4, 8, 0.8, 0.5, 1).unwrap();

        let id = BlockId {
            layer_idx: 1,
            block_id: 7,
        };
        let data = BlockData {
            keys: vec![3.14; 32],
            values: vec![2.71; 32],
            num_filled: 4,
        };

        assert!(!swapper.is_swapped(&id));
        swapper.swap_out(id, &data).unwrap();
        assert!(swapper.is_swapped(&id));

        let recovered = swapper.swap_in(id).unwrap();
        assert!(!swapper.is_swapped(&id));
        assert_eq!(recovered.num_filled, 4);
        assert!((recovered.keys[0] - 3.14).abs() < 1e-6);
        assert!((recovered.values[0] - 2.71).abs() < 1e-6);

        let stats = swapper.stats();
        assert_eq!(stats.total_swap_outs, 1);
        assert_eq!(stats.total_swap_ins, 1);
        assert_eq!(stats.blocks_on_ssd, 0);

        cleanup(&dir);
    }

    #[test]
    fn test_block_swapper_remove() {
        let dir = temp_swap_dir();
        let mut swapper = BlockSwapper::new(&dir, 2, 4, 0.8, 0.5, 0).unwrap();

        let id = BlockId {
            layer_idx: 0,
            block_id: 0,
        };
        let data = BlockData {
            keys: vec![1.0; 8],
            values: vec![1.0; 8],
            num_filled: 2,
        };

        swapper.touch(id);
        swapper.swap_out(id, &data).unwrap();
        assert!(swapper.is_swapped(&id));

        swapper.remove_block(&id);
        assert!(!swapper.is_swapped(&id));
        assert_eq!(swapper.stats().blocks_on_ssd, 0);

        cleanup(&dir);
    }

    #[test]
    fn test_prefetch_candidates() {
        let dir = temp_swap_dir();
        let mut swapper = BlockSwapper::new(&dir, 2, 4, 0.8, 0.5, 0).unwrap();

        let data = BlockData {
            keys: vec![1.0; 8],
            values: vec![1.0; 8],
            num_filled: 2,
        };

        let b0 = BlockId {
            layer_idx: 0,
            block_id: 0,
        };
        let b1 = BlockId {
            layer_idx: 0,
            block_id: 1,
        };
        let b2 = BlockId {
            layer_idx: 0,
            block_id: 2,
        };

        // Touch and swap all three
        swapper.touch(b0);
        swapper.touch(b1);
        swapper.touch(b2);
        swapper.swap_out(b0, &data).unwrap();
        swapper.swap_out(b1, &data).unwrap();
        swapper.swap_out(b2, &data).unwrap();

        // Prefetch should prefer MRU → b2, b1
        let prefetch = swapper.get_prefetch_candidates(2);
        assert_eq!(prefetch.len(), 2);
        assert_eq!(prefetch[0], b2); // MRU
        assert_eq!(prefetch[1], b1);

        cleanup(&dir);
    }

    #[test]
    fn test_stats_display() {
        let stats = BlockSwapStats {
            blocks_on_ssd: 42,
            total_swap_outs: 100,
            total_swap_ins: 80,
            file_size_bytes: 10 * 1024 * 1024,
            total_written_bytes: 50 * 1024 * 1024,
            total_read_bytes: 40 * 1024 * 1024,
            tracked_blocks: 200,
        };
        let display = format!("{}", stats);
        assert!(display.contains("42 blocks"));
        assert!(display.contains("100 out"));
        assert!(display.contains("80 in"));
    }

    // ── Quantized block swapping tests ──

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let dim = 64;
        let data: Vec<f32> = (0..dim * 4).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let (quantized, scales) = quantize_vectors_int8(&data, dim);
        let recovered = dequantize_vectors_int8(&quantized, &scales, dim);

        assert_eq!(recovered.len(), data.len());
        for (a, b) in recovered.iter().zip(data.iter()) {
            // INT8 quantization error should be small (< 1% of absmax)
            assert!(
                (a - b).abs() < 0.02,
                "quantization error too large: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_quantize_zeros() {
        let dim = 8;
        let data = vec![0.0f32; dim * 2];
        let (quantized, scales) = quantize_vectors_int8(&data, dim);
        assert!(scales.iter().all(|&s| s == 0.0));
        assert!(quantized.iter().all(|&q| q == 0));

        let recovered = dequantize_vectors_int8(&quantized, &scales, dim);
        assert!(recovered.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_swap_file_int8_write_read() {
        let dir = temp_swap_dir();
        let block_size = 4;
        let kv_dim = 8;
        let mut sf =
            SwapFile::with_quant_mode(&dir, block_size, kv_dim, SwapQuantMode::Int8).unwrap();

        let id = BlockId {
            layer_idx: 0,
            block_id: 3,
        };
        let data = BlockData {
            keys: (0..block_size * kv_dim)
                .map(|i| (i as f32 - 16.0) * 0.1)
                .collect(),
            values: (0..block_size * kv_dim)
                .map(|i| (i as f32 - 16.0) * 0.05)
                .collect(),
            num_filled: 3,
        };

        sf.write_block(id, &data).unwrap();
        assert!(sf.contains(&id));
        assert_eq!(sf.quant_mode(), SwapQuantMode::Int8);

        let recovered = sf.read_block(id).unwrap();
        assert_eq!(recovered.num_filled, 3);
        assert_eq!(recovered.keys.len(), data.keys.len());

        // Allow INT8 quantization error
        for (a, b) in recovered.keys.iter().zip(data.keys.iter()) {
            assert!((a - b).abs() < 0.02, "key quant error: {} vs {}", a, b);
        }
        for (a, b) in recovered.values.iter().zip(data.values.iter()) {
            assert!((a - b).abs() < 0.02, "value quant error: {} vs {}", a, b);
        }

        cleanup(&dir);
    }

    #[test]
    fn test_swap_file_int8_smaller_than_f32() {
        let dir1 = temp_swap_dir();
        let dir2 = temp_swap_dir();
        let block_size = 16;
        let kv_dim = 128;

        let sf_f32 = SwapFile::new(&dir1, block_size, kv_dim).unwrap();
        let sf_int8 =
            SwapFile::with_quant_mode(&dir2, block_size, kv_dim, SwapQuantMode::Int8).unwrap();

        // INT8 slot should be significantly smaller than f32 slot
        assert!(
            sf_int8.slot_size < sf_f32.slot_size,
            "INT8 slot {} should be smaller than f32 slot {}",
            sf_int8.slot_size,
            sf_f32.slot_size
        );

        // Should be roughly 3-4x smaller
        let ratio = sf_f32.slot_size as f64 / sf_int8.slot_size as f64;
        assert!(
            ratio > 2.5 && ratio < 4.5,
            "compression ratio {} not in expected range [2.5, 4.5]",
            ratio
        );

        cleanup(&dir1);
        cleanup(&dir2);
    }

    #[test]
    fn test_block_swapper_int8_roundtrip() {
        let dir = temp_swap_dir();
        let mut swapper =
            BlockSwapper::with_quant_mode(&dir, 4, 8, 0.8, 0.5, 1, SwapQuantMode::Int8).unwrap();

        let id = BlockId {
            layer_idx: 2,
            block_id: 5,
        };
        let data = BlockData {
            keys: vec![1.5; 32],
            values: vec![-0.75; 32],
            num_filled: 4,
        };

        swapper.swap_out(id, &data).unwrap();
        assert!(swapper.is_swapped(&id));

        let recovered = swapper.swap_in(id).unwrap();
        assert!(!swapper.is_swapped(&id));
        assert_eq!(recovered.num_filled, 4);

        // Check values within INT8 quantization tolerance
        for &v in &recovered.keys {
            assert!((v - 1.5).abs() < 0.02, "key value {} != 1.5", v);
        }
        for &v in &recovered.values {
            assert!((v - (-0.75)).abs() < 0.02, "value {} != -0.75", v);
        }

        let stats = swapper.stats();
        assert_eq!(stats.total_swap_outs, 1);
        assert_eq!(stats.total_swap_ins, 1);

        cleanup(&dir);
    }

    #[test]
    fn test_swap_file_int8_slot_reuse() {
        let dir = temp_swap_dir();
        let mut sf = SwapFile::with_quant_mode(&dir, 2, 4, SwapQuantMode::Int8).unwrap();

        let id1 = BlockId {
            layer_idx: 0,
            block_id: 0,
        };
        let id2 = BlockId {
            layer_idx: 0,
            block_id: 1,
        };
        let id3 = BlockId {
            layer_idx: 0,
            block_id: 2,
        };

        let data = BlockData {
            keys: vec![1.0; 8],
            values: vec![2.0; 8],
            num_filled: 2,
        };

        sf.write_block(id1, &data).unwrap();
        sf.write_block(id2, &data).unwrap();

        let offset_before = sf.next_offset;
        sf.free_block(id1);
        sf.write_block(id3, &data).unwrap();
        assert_eq!(sf.next_offset, offset_before); // Slot reused

        assert!(!sf.contains(&id1));
        assert!(sf.contains(&id2));
        assert!(sf.contains(&id3));

        cleanup(&dir);
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = swap_compression_ratio(16, 128);
        // With block_size=16, kv_dim=128:
        // f32: 16*128*2*4 + 4 = 16388 bytes
        // int8: 16*128*2*1 + 32*4 + 4 = 4228 bytes
        // ratio ≈ 3.88
        assert!(ratio > 3.0 && ratio < 4.5, "ratio {} unexpected", ratio);
    }
}
