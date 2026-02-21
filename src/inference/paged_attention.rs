//! PagedAttention — vLLM-style paged KV cache management
//!
//! Instead of allocating contiguous memory for each sequence's full context,
//! PagedAttention divides the KV cache into fixed-size pages (blocks).
//! Pages are allocated on-demand as sequences grow, and freed when sequences complete.
//!
//! Benefits:
//! - Near-zero memory waste (no pre-allocation for max_seq_len)
//! - Efficient memory sharing for parallel sampling (beam search, best-of-N)
//! - Natural SSD offloading: cold pages can be swapped to disk
//! - Better memory utilization for concurrent requests with varying lengths
//!
//! Architecture:
//! ```
//! Sequence → BlockTable → [Page0, Page1, Page2, ...]
//!                              ↓       ↓       ↓
//!                          PhysicalBlock (fixed-size KV storage)
//! ```

use std::collections::{HashMap, VecDeque};
use tracing::{debug, info};

/// Number of token positions stored per page/block
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// A physical block storing KV data for a fixed number of token positions
#[derive(Clone)]
pub struct PhysicalBlock {
    /// Block ID (index in the block pool)
    pub block_id: usize,
    /// Key data: [block_size, n_kv_heads * head_dim] flattened
    pub keys: Vec<f32>,
    /// Value data: [block_size, n_kv_heads * head_dim] flattened
    pub values: Vec<f32>,
    /// Number of positions currently filled (0..=block_size)
    pub num_filled: usize,
    /// Reference count for copy-on-write sharing
    pub ref_count: usize,
    /// Whether this block has been swapped to SSD
    pub swapped: bool,
}

impl PhysicalBlock {
    pub fn new(block_id: usize, block_size: usize, kv_dim: usize) -> Self {
        Self {
            block_id,
            keys: vec![0.0; block_size * kv_dim],
            values: vec![0.0; block_size * kv_dim],
            num_filled: 0,
            ref_count: 1,
            swapped: false,
        }
    }

    /// Check if this block is full
    pub fn is_full(&self, block_size: usize) -> bool {
        self.num_filled >= block_size
    }

    /// Write KV data at the next available slot
    pub fn append(&mut self, key: &[f32], value: &[f32], kv_dim: usize) {
        let offset = self.num_filled * kv_dim;
        self.keys[offset..offset + kv_dim].copy_from_slice(key);
        self.values[offset..offset + kv_dim].copy_from_slice(value);
        self.num_filled += 1;
    }

    /// Read key at a given slot within this block
    pub fn key_at(&self, slot: usize, kv_head: usize, head_dim: usize, kv_dim: usize) -> &[f32] {
        let offset = slot * kv_dim + kv_head * head_dim;
        &self.keys[offset..offset + head_dim]
    }

    /// Read value at a given slot within this block
    pub fn value_at(&self, slot: usize, kv_head: usize, head_dim: usize, kv_dim: usize) -> &[f32] {
        let offset = slot * kv_dim + kv_head * head_dim;
        &self.values[offset..offset + head_dim]
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * std::mem::size_of::<f32>()
    }

    /// Clear this block for reuse
    pub fn clear(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.num_filled = 0;
        self.ref_count = 1;
        self.swapped = false;
    }
}

/// Block table mapping logical blocks to physical blocks for one layer of one sequence
#[derive(Clone, Debug)]
pub struct BlockTable {
    /// Physical block IDs in order: block_ids[i] = physical block for logical block i
    pub block_ids: Vec<usize>,
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            block_ids: Vec::new(),
        }
    }

    /// Get the physical block ID for a given token position
    pub fn physical_block_for_position(
        &self,
        position: usize,
        block_size: usize,
    ) -> Option<(usize, usize)> {
        let logical_block = position / block_size;
        let slot_in_block = position % block_size;
        self.block_ids
            .get(logical_block)
            .map(|&phys_id| (phys_id, slot_in_block))
    }

    /// Number of logical blocks allocated
    pub fn num_blocks(&self) -> usize {
        self.block_ids.len()
    }
}

impl Default for BlockTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-sequence metadata for paged KV cache
pub struct PagedSequence {
    /// Block tables per layer: block_tables[layer_idx]
    pub block_tables: Vec<BlockTable>,
    /// Current sequence length
    pub seq_len: usize,
    /// Number of layers
    pub n_layers: usize,
}

impl PagedSequence {
    pub fn new(n_layers: usize) -> Self {
        Self {
            block_tables: (0..n_layers).map(|_| BlockTable::new()).collect(),
            seq_len: 0,
            n_layers,
        }
    }

    /// Get block table for a layer
    pub fn block_table(&self, layer_idx: usize) -> &BlockTable {
        &self.block_tables[layer_idx]
    }
}

/// The paged block allocator — manages a pool of physical blocks
pub struct BlockAllocator {
    /// All physical blocks (pre-allocated pool)
    blocks: Vec<PhysicalBlock>,
    /// Free block IDs available for allocation
    free_blocks: VecDeque<usize>,
    /// Block size (tokens per block)
    block_size: usize,
    /// KV dimension (n_kv_heads * head_dim)
    kv_dim: usize,
    /// Total number of blocks in pool
    num_blocks: usize,
}

impl BlockAllocator {
    /// Create a new block allocator with the given capacity
    ///
    /// `max_blocks` — total number of physical blocks to pre-allocate
    pub fn new(max_blocks: usize, block_size: usize, kv_dim: usize) -> Self {
        let blocks: Vec<PhysicalBlock> = (0..max_blocks)
            .map(|i| PhysicalBlock::new(i, block_size, kv_dim))
            .collect();
        let free_blocks: VecDeque<usize> = (0..max_blocks).collect();

        let total_mem = max_blocks * block_size * kv_dim * 2 * std::mem::size_of::<f32>();
        info!(
            "PagedAttention block allocator: {} blocks × {} tokens = {} total slots, {:.1} MB",
            max_blocks,
            block_size,
            max_blocks * block_size,
            total_mem as f64 / (1024.0 * 1024.0)
        );

        Self {
            blocks,
            free_blocks,
            block_size,
            kv_dim,
            num_blocks: max_blocks,
        }
    }

    /// Allocate a new block, returns block_id or None if OOM
    pub fn allocate(&mut self) -> Option<usize> {
        let block_id = self.free_blocks.pop_front()?;
        self.blocks[block_id].clear();
        debug!("Allocated block {}", block_id);
        Some(block_id)
    }

    /// Free a block (decrement ref count, return to pool if zero)
    pub fn free(&mut self, block_id: usize) {
        let block = &mut self.blocks[block_id];
        block.ref_count = block.ref_count.saturating_sub(1);
        if block.ref_count == 0 {
            self.free_blocks.push_back(block_id);
            debug!("Freed block {}", block_id);
        }
    }

    /// Get a reference to a physical block
    pub fn block(&self, block_id: usize) -> &PhysicalBlock {
        &self.blocks[block_id]
    }

    /// Get a mutable reference to a physical block
    pub fn block_mut(&mut self, block_id: usize) -> &mut PhysicalBlock {
        &mut self.blocks[block_id]
    }

    /// Number of free blocks available
    pub fn num_free(&self) -> usize {
        self.free_blocks.len()
    }

    /// Number of used blocks
    pub fn num_used(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }

    /// Total capacity in blocks
    pub fn capacity(&self) -> usize {
        self.num_blocks
    }

    /// Block size (tokens per block)
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// KV dimension
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Increment ref count for copy-on-write
    pub fn inc_ref(&mut self, block_id: usize) {
        self.blocks[block_id].ref_count += 1;
    }

    /// Copy-on-write: if block has ref_count > 1, copy to a new block
    pub fn copy_on_write(&mut self, block_id: usize) -> Option<usize> {
        if self.blocks[block_id].ref_count <= 1 {
            return Some(block_id);
        }

        let new_id = self.allocate()?;
        let (src_keys, src_values, num_filled) = {
            let src = &self.blocks[block_id];
            (src.keys.clone(), src.values.clone(), src.num_filled)
        };
        let dst = &mut self.blocks[new_id];
        dst.keys = src_keys;
        dst.values = src_values;
        dst.num_filled = num_filled;

        // Decrement original's ref count
        self.blocks[block_id].ref_count -= 1;

        debug!(
            "Copy-on-write: block {} → {} (original ref_count now {})",
            block_id, new_id, self.blocks[block_id].ref_count
        );
        Some(new_id)
    }
}

/// Sequence ID type
pub type SeqId = u64;

/// The main PagedKvCache that manages all sequences
pub struct PagedKvCache {
    /// Block allocator (shared across all sequences and layers)
    /// One allocator per layer for independent block management
    allocators: Vec<BlockAllocator>,
    /// Active sequences
    sequences: HashMap<SeqId, PagedSequence>,
    /// Configuration
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    /// Next sequence ID
    next_seq_id: SeqId,
}

impl PagedKvCache {
    /// Create a new paged KV cache
    ///
    /// `max_blocks_per_layer` — number of physical blocks per layer
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_blocks_per_layer: usize,
    ) -> Self {
        let kv_dim = n_kv_heads * head_dim;
        let allocators = (0..n_layers)
            .map(|_| BlockAllocator::new(max_blocks_per_layer, block_size, kv_dim))
            .collect();

        info!(
            "PagedKvCache initialized: layers={}, kv_heads={}, head_dim={}, block_size={}, blocks_per_layer={}",
            n_layers, n_kv_heads, head_dim, block_size, max_blocks_per_layer
        );

        Self {
            allocators,
            sequences: HashMap::new(),
            n_layers,
            n_kv_heads,
            head_dim,
            block_size,
            next_seq_id: 0,
        }
    }

    /// Register a new sequence, returns its ID
    pub fn add_sequence(&mut self) -> SeqId {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;
        self.sequences
            .insert(seq_id, PagedSequence::new(self.n_layers));
        debug!("Added paged sequence {}", seq_id);
        seq_id
    }

    /// Append KV data for a token position across one layer
    pub fn append(
        &mut self,
        seq_id: SeqId,
        layer_idx: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), PagedAttentionError> {
        let kv_dim = self.n_kv_heads * self.head_dim;
        let block_size = self.block_size;

        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(PagedAttentionError::UnknownSequence(seq_id))?;
        let block_table = &mut seq.block_tables[layer_idx];

        // Check if we need a new block
        let need_new_block = block_table.block_ids.is_empty()
            || self.allocators[layer_idx]
                .block(
                    *block_table
                        .block_ids
                        .last()
                        .expect("block_ids not empty after is_empty check"),
                )
                .is_full(block_size);

        if need_new_block {
            let block_id = self.allocators[layer_idx]
                .allocate()
                .ok_or(PagedAttentionError::OutOfBlocks)?;
            block_table.block_ids.push(block_id);
        }

        // Get the last block and check for CoW
        let last_block_id = *block_table
            .block_ids
            .last()
            .expect("block_ids must have at least one entry after allocation");

        let actual_block_id = self.allocators[layer_idx]
            .copy_on_write(last_block_id)
            .ok_or(PagedAttentionError::OutOfBlocks)?;

        if actual_block_id != last_block_id {
            *block_table
                .block_ids
                .last_mut()
                .expect("block_ids not empty") = actual_block_id;
        }

        // Append to the block
        self.allocators[layer_idx]
            .block_mut(actual_block_id)
            .append(key, value, kv_dim);

        // Update seq_len only when all layers are done (caller tracks this)
        Ok(())
    }

    /// Increment the sequence length (call after appending to all layers for a position)
    pub fn increment_seq_len(&mut self, seq_id: SeqId) -> Result<(), PagedAttentionError> {
        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(PagedAttentionError::UnknownSequence(seq_id))?;
        seq.seq_len += 1;
        Ok(())
    }

    /// Read key at a given position for a sequence/layer/head
    pub fn key_at(
        &self,
        seq_id: SeqId,
        layer_idx: usize,
        position: usize,
        kv_head: usize,
    ) -> Result<&[f32], PagedAttentionError> {
        let kv_dim = self.n_kv_heads * self.head_dim;
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(PagedAttentionError::UnknownSequence(seq_id))?;
        let (phys_id, slot) = seq.block_tables[layer_idx]
            .physical_block_for_position(position, self.block_size)
            .ok_or(PagedAttentionError::PositionOutOfRange(position))?;
        Ok(self.allocators[layer_idx]
            .block(phys_id)
            .key_at(slot, kv_head, self.head_dim, kv_dim))
    }

    /// Read value at a given position for a sequence/layer/head
    pub fn value_at(
        &self,
        seq_id: SeqId,
        layer_idx: usize,
        position: usize,
        kv_head: usize,
    ) -> Result<&[f32], PagedAttentionError> {
        let kv_dim = self.n_kv_heads * self.head_dim;
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(PagedAttentionError::UnknownSequence(seq_id))?;
        let (phys_id, slot) = seq.block_tables[layer_idx]
            .physical_block_for_position(position, self.block_size)
            .ok_or(PagedAttentionError::PositionOutOfRange(position))?;
        Ok(self.allocators[layer_idx]
            .block(phys_id)
            .value_at(slot, kv_head, self.head_dim, kv_dim))
    }

    /// Get sequence length
    pub fn seq_len(&self, seq_id: SeqId) -> Result<usize, PagedAttentionError> {
        self.sequences
            .get(&seq_id)
            .map(|s| s.seq_len)
            .ok_or(PagedAttentionError::UnknownSequence(seq_id))
    }

    /// Remove a sequence, freeing all its blocks
    pub fn remove_sequence(&mut self, seq_id: SeqId) -> Result<(), PagedAttentionError> {
        let seq = self
            .sequences
            .remove(&seq_id)
            .ok_or(PagedAttentionError::UnknownSequence(seq_id))?;

        for (layer_idx, block_table) in seq.block_tables.iter().enumerate() {
            for &block_id in &block_table.block_ids {
                self.allocators[layer_idx].free(block_id);
            }
        }

        debug!(
            "Removed paged sequence {} (freed {} blocks per layer)",
            seq_id,
            seq.block_tables
                .first()
                .map(|bt| bt.num_blocks())
                .unwrap_or(0)
        );
        Ok(())
    }

    /// Fork a sequence (for beam search / parallel sampling) using copy-on-write
    pub fn fork_sequence(&mut self, src_seq_id: SeqId) -> Result<SeqId, PagedAttentionError> {
        let src = self
            .sequences
            .get(&src_seq_id)
            .ok_or(PagedAttentionError::UnknownSequence(src_seq_id))?;

        let new_seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let mut new_seq = PagedSequence::new(self.n_layers);
        new_seq.seq_len = src.seq_len;

        // Share all blocks via ref counting (copy-on-write)
        for (layer_idx, src_bt) in src.block_tables.iter().enumerate() {
            new_seq.block_tables[layer_idx].block_ids = src_bt.block_ids.clone();
            for &block_id in &src_bt.block_ids {
                self.allocators[layer_idx].inc_ref(block_id);
            }
        }

        self.sequences.insert(new_seq_id, new_seq);
        debug!(
            "Forked paged sequence {} → {} (CoW)",
            src_seq_id, new_seq_id
        );
        Ok(new_seq_id)
    }

    /// Rollback a sequence to a given length
    pub fn rollback(&mut self, seq_id: SeqId, new_len: usize) -> Result<(), PagedAttentionError> {
        let block_size = self.block_size;

        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(PagedAttentionError::UnknownSequence(seq_id))?;

        if new_len >= seq.seq_len {
            return Ok(());
        }

        // Calculate how many blocks to keep
        let blocks_needed = if new_len == 0 {
            0
        } else {
            new_len.div_ceil(block_size)
        };

        // Free excess blocks
        for (layer_idx, block_table) in seq.block_tables.iter_mut().enumerate() {
            while block_table.block_ids.len() > blocks_needed {
                if let Some(block_id) = block_table.block_ids.pop() {
                    self.allocators[layer_idx].free(block_id);
                }
            }

            // Update the fill count of the last block
            if let Some(&last_block_id) = block_table.block_ids.last() {
                let slots_in_last = if new_len.is_multiple_of(block_size) && new_len > 0 {
                    block_size
                } else {
                    new_len % block_size
                };
                self.allocators[layer_idx]
                    .block_mut(last_block_id)
                    .num_filled = slots_in_last;
            }
        }

        seq.seq_len = new_len;
        debug!("Rolled back paged sequence {} to len {}", seq_id, new_len);
        Ok(())
    }

    /// Get memory stats
    pub fn stats(&self) -> PagedCacheStats {
        let (total, used) = if let Some(alloc) = self.allocators.first() {
            (alloc.capacity(), alloc.num_used())
        } else {
            (0, 0)
        };

        PagedCacheStats {
            total_blocks_per_layer: total,
            used_blocks_per_layer: used,
            free_blocks_per_layer: total - used,
            num_sequences: self.sequences.len(),
            block_size: self.block_size,
            n_layers: self.n_layers,
        }
    }

    /// Number of active sequences
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }
}

/// Statistics for the paged cache
#[derive(Debug, Clone)]
pub struct PagedCacheStats {
    pub total_blocks_per_layer: usize,
    pub used_blocks_per_layer: usize,
    pub free_blocks_per_layer: usize,
    pub num_sequences: usize,
    pub block_size: usize,
    pub n_layers: usize,
}

impl std::fmt::Display for PagedCacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let utilization = if self.total_blocks_per_layer > 0 {
            self.used_blocks_per_layer as f64 / self.total_blocks_per_layer as f64 * 100.0
        } else {
            0.0
        };
        write!(
            f,
            "PagedKV: {}/{} blocks/layer ({:.1}%), {} sequences, block_size={}",
            self.used_blocks_per_layer,
            self.total_blocks_per_layer,
            utilization,
            self.num_sequences,
            self.block_size,
        )
    }
}

/// Errors from paged attention operations
#[derive(Debug, thiserror::Error)]
pub enum PagedAttentionError {
    #[error("Unknown sequence ID: {0}")]
    UnknownSequence(SeqId),

    #[error("Out of physical blocks — increase max_blocks or free sequences")]
    OutOfBlocks,

    #[error("Position {0} out of range for sequence")]
    PositionOutOfRange(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_paged_kv_cache() {
        let n_layers = 2;
        let n_kv_heads = 4;
        let head_dim = 8;
        let block_size = 4;
        let max_blocks = 10;

        let mut cache = PagedKvCache::new(n_layers, n_kv_heads, head_dim, block_size, max_blocks);

        let seq_id = cache.add_sequence();
        assert_eq!(cache.seq_len(seq_id).unwrap(), 0);

        // Append a token's KV to all layers
        let kv_dim = n_kv_heads * head_dim;
        let key: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.1).collect();
        let value: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.2).collect();

        for layer in 0..n_layers {
            cache.append(seq_id, layer, &key, &value).unwrap();
        }
        cache.increment_seq_len(seq_id).unwrap();

        assert_eq!(cache.seq_len(seq_id).unwrap(), 1);

        // Read back
        let k = cache.key_at(seq_id, 0, 0, 0).unwrap();
        assert_eq!(k.len(), head_dim);
        assert!((k[0] - 0.0).abs() < 1e-6);
        assert!((k[1] - 0.1).abs() < 1e-6);

        let v = cache.value_at(seq_id, 0, 0, 0).unwrap();
        assert_eq!(v.len(), head_dim);
        assert!((v[0] - 0.0).abs() < 1e-6);
        assert!((v[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_multi_block_allocation() {
        let mut cache = PagedKvCache::new(1, 2, 4, 4, 100);
        let seq_id = cache.add_sequence();
        let kv_dim = 2 * 4;

        // Fill 10 positions (should use 3 blocks: 4+4+2)
        for i in 0..10 {
            let key: Vec<f32> = (0..kv_dim).map(|j| (i * kv_dim + j) as f32).collect();
            let value = key.clone();
            cache.append(seq_id, 0, &key, &value).unwrap();
            cache.increment_seq_len(seq_id).unwrap();
        }

        assert_eq!(cache.seq_len(seq_id).unwrap(), 10);

        let stats = cache.stats();
        assert_eq!(stats.used_blocks_per_layer, 3); // ceil(10/4) = 3

        // Read position 5 (block 1, slot 1)
        let k = cache.key_at(seq_id, 0, 5, 0).unwrap();
        assert_eq!(k.len(), 4);
        assert!((k[0] - (5 * kv_dim) as f32).abs() < 1e-6);
    }

    #[test]
    fn test_fork_and_cow() {
        let mut cache = PagedKvCache::new(1, 2, 4, 4, 100);
        let seq_a = cache.add_sequence();
        let kv_dim = 2 * 4;

        // Add 4 tokens to seq_a
        for i in 0..4 {
            let key: Vec<f32> = vec![i as f32; kv_dim];
            let value = key.clone();
            cache.append(seq_a, 0, &key, &value).unwrap();
            cache.increment_seq_len(seq_a).unwrap();
        }

        // Fork
        let seq_b = cache.fork_sequence(seq_a).unwrap();
        assert_eq!(cache.seq_len(seq_b).unwrap(), 4);
        assert_eq!(cache.num_sequences(), 2);

        // Verify shared data
        let k_a = cache.key_at(seq_a, 0, 2, 0).unwrap();
        let k_b = cache.key_at(seq_b, 0, 2, 0).unwrap();
        assert_eq!(k_a, k_b);

        // Append to seq_b (triggers CoW on new block, not shared block)
        let new_key = vec![99.0; kv_dim];
        let new_val = vec![99.0; kv_dim];
        cache.append(seq_b, 0, &new_key, &new_val).unwrap();
        cache.increment_seq_len(seq_b).unwrap();

        assert_eq!(cache.seq_len(seq_b).unwrap(), 5);
        assert_eq!(cache.seq_len(seq_a).unwrap(), 4);
    }

    #[test]
    fn test_remove_sequence() {
        let mut cache = PagedKvCache::new(1, 2, 4, 4, 10);
        let seq = cache.add_sequence();
        let kv_dim = 2 * 4;

        for _ in 0..8 {
            let key = vec![1.0; kv_dim];
            let value = vec![1.0; kv_dim];
            cache.append(seq, 0, &key, &value).unwrap();
            cache.increment_seq_len(seq).unwrap();
        }

        assert_eq!(cache.stats().used_blocks_per_layer, 2);
        cache.remove_sequence(seq).unwrap();
        assert_eq!(cache.stats().used_blocks_per_layer, 0);
        assert_eq!(cache.num_sequences(), 0);
    }

    #[test]
    fn test_rollback() {
        let mut cache = PagedKvCache::new(1, 2, 4, 4, 100);
        let seq = cache.add_sequence();
        let kv_dim = 2 * 4;

        for i in 0..10 {
            let key = vec![i as f32; kv_dim];
            let value = vec![i as f32; kv_dim];
            cache.append(seq, 0, &key, &value).unwrap();
            cache.increment_seq_len(seq).unwrap();
        }

        assert_eq!(cache.seq_len(seq).unwrap(), 10);
        assert_eq!(cache.stats().used_blocks_per_layer, 3);

        // Rollback to 5 (should keep 2 blocks: ceil(5/4) = 2)
        cache.rollback(seq, 5).unwrap();
        assert_eq!(cache.seq_len(seq).unwrap(), 5);
        assert_eq!(cache.stats().used_blocks_per_layer, 2);

        // Verify data still valid at position 3
        let k = cache.key_at(seq, 0, 3, 0).unwrap();
        assert!((k[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_out_of_blocks() {
        let mut cache = PagedKvCache::new(1, 1, 4, 4, 2); // Only 2 blocks = 8 positions
        let seq = cache.add_sequence();
        let kv_dim = 4;

        for _ in 0..8 {
            cache
                .append(seq, 0, &vec![1.0; kv_dim], &vec![1.0; kv_dim])
                .unwrap();
            cache.increment_seq_len(seq).unwrap();
        }

        // 9th token should fail
        let result = cache.append(seq, 0, &vec![1.0; kv_dim], &vec![1.0; kv_dim]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PagedAttentionError::OutOfBlocks
        ));
    }

    #[test]
    fn test_multiple_sequences() {
        let mut cache = PagedKvCache::new(2, 2, 4, 4, 50);
        let kv_dim = 2 * 4;

        let seq_a = cache.add_sequence();
        let seq_b = cache.add_sequence();

        // Different lengths
        for _ in 0..3 {
            for layer in 0..2 {
                cache
                    .append(seq_a, layer, &vec![1.0; kv_dim], &vec![1.0; kv_dim])
                    .unwrap();
            }
            cache.increment_seq_len(seq_a).unwrap();
        }

        for _ in 0..7 {
            for layer in 0..2 {
                cache
                    .append(seq_b, layer, &vec![2.0; kv_dim], &vec![2.0; kv_dim])
                    .unwrap();
            }
            cache.increment_seq_len(seq_b).unwrap();
        }

        assert_eq!(cache.seq_len(seq_a).unwrap(), 3);
        assert_eq!(cache.seq_len(seq_b).unwrap(), 7);
        assert_eq!(cache.num_sequences(), 2);
    }

    #[test]
    fn test_block_table_position_mapping() {
        let bt = BlockTable {
            block_ids: vec![5, 12, 3],
        };

        // block_size = 4
        assert_eq!(bt.physical_block_for_position(0, 4), Some((5, 0)));
        assert_eq!(bt.physical_block_for_position(3, 4), Some((5, 3)));
        assert_eq!(bt.physical_block_for_position(4, 4), Some((12, 0)));
        assert_eq!(bt.physical_block_for_position(9, 4), Some((3, 1)));
        assert_eq!(bt.physical_block_for_position(12, 4), None); // out of range
    }

    #[test]
    fn test_stats_display() {
        let cache = PagedKvCache::new(32, 8, 128, 16, 1000);
        let stats = cache.stats();
        let display = format!("{}", stats);
        assert!(display.contains("0/1000"));
        assert!(display.contains("0 sequences"));
    }
}
