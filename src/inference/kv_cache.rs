//! Key-Value Cache for transformer attention
//!
//! Stores past key/value projections per layer to avoid recomputation.
//! This is essential for efficient autoregressive generation — without it,
//! each new token would need to recompute attention over the entire sequence.

use tracing::debug;

/// Per-layer KV cache storing key and value vectors for all past positions
#[derive(Clone)]
pub struct LayerKvCache {
    /// Cached keys: [seq_len, n_kv_heads * head_dim]
    pub keys: Vec<Vec<f32>>,
    /// Cached values: [seq_len, n_kv_heads * head_dim]
    pub values: Vec<Vec<f32>>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl LayerKvCache {
    pub fn new(n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            n_kv_heads,
            head_dim,
        }
    }

    /// Append key/value for the current position
    pub fn append(&mut self, key: Vec<f32>, value: Vec<f32>) {
        self.keys.push(key);
        self.values.push(value);
    }

    /// Current sequence length
    pub fn seq_len(&self) -> usize {
        self.keys.len()
    }

    /// Get key vector at position for a specific KV head
    pub fn key_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        let start = kv_head * self.head_dim;
        let end = start + self.head_dim;
        &self.keys[pos][start..end]
    }

    /// Get value vector at position for a specific KV head
    pub fn value_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        let start = kv_head * self.head_dim;
        let end = start + self.head_dim;
        &self.values[pos][start..end]
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        let per_entry = self.n_kv_heads * self.head_dim * std::mem::size_of::<f32>();
        self.keys.len() * per_entry * 2 // keys + values
    }

    /// Clear the cache (e.g. for a new sequence)
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }

    /// Rollback to a given sequence length (discard positions >= new_len)
    /// Used by speculative decoding to reject draft tokens
    pub fn rollback(&mut self, new_len: usize) {
        self.keys.truncate(new_len);
        self.values.truncate(new_len);
    }
}

/// Full KV cache across all layers
pub struct KvCache {
    layers: Vec<LayerKvCache>,
    max_seq_len: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let layers = (0..n_layers)
            .map(|_| LayerKvCache::new(n_kv_heads, head_dim))
            .collect();
        debug!(
            "KV cache initialized: layers={}, kv_heads={}, head_dim={}, max_seq={}",
            n_layers, n_kv_heads, head_dim, max_seq_len
        );
        Self {
            layers,
            max_seq_len,
        }
    }

    /// Get mutable reference to a layer's KV cache
    pub fn layer_mut(&mut self, layer_idx: usize) -> &mut LayerKvCache {
        &mut self.layers[layer_idx]
    }

    /// Get reference to a layer's KV cache
    pub fn layer(&self, layer_idx: usize) -> &LayerKvCache {
        &self.layers[layer_idx]
    }

    /// Current sequence length (same across all layers)
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len()).unwrap_or(0)
    }

    /// Total memory usage
    pub fn size_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.size_bytes()).sum()
    }

    /// Clear all layers
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Rollback all layers to a given sequence length
    /// Used by speculative decoding to discard rejected draft tokens
    pub fn rollback(&mut self, new_len: usize) {
        for layer in &mut self.layers {
            layer.rollback(new_len);
        }
    }

    /// Number of layers
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Check if we've hit the context window limit
    pub fn is_full(&self) -> bool {
        self.seq_len() >= self.max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_basic() {
        let mut cache = KvCache::new(2, 4, 64, 2048);
        assert_eq!(cache.seq_len(), 0);

        // Append to layer 0
        let key = vec![1.0f32; 4 * 64];
        let val = vec![2.0f32; 4 * 64];
        cache.layer_mut(0).append(key, val);
        assert_eq!(cache.layer(0).seq_len(), 1);

        // Check retrieval
        let k = cache.layer(0).key_at(0, 0);
        assert_eq!(k.len(), 64);
        assert_eq!(k[0], 1.0);

        let v = cache.layer(0).value_at(0, 0);
        assert_eq!(v[0], 2.0);
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KvCache::new(1, 2, 32, 512);
        cache.layer_mut(0).append(vec![0.0; 64], vec![0.0; 64]);
        cache.layer_mut(0).append(vec![0.0; 64], vec![0.0; 64]);
        assert_eq!(cache.seq_len(), 2);
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }
}
