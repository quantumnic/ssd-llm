//! Heavy Hitter Oracle (H2O) KV Cache Eviction
//!
//! Based on Zhang et al., 2023 — "H2O: Heavy-Hitter Oracle for Efficient
//! Generative Inference of Large Language Models"
//!
//! Instead of evicting the oldest tokens when the KV cache is full (FIFO),
//! H2O tracks cumulative attention scores and keeps "heavy hitter" tokens
//! that receive the most attention across decoding steps. This preserves
//! semantically important context even when the cache budget is small.
//!
//! Key insight: A small fraction of tokens (5-10%) receive the vast majority
//! of attention mass. Keeping these tokens dramatically improves quality
//! compared to naive sliding-window eviction.
//!
//! Integration: wraps `LayerKvCache` with attention tracking and provides
//! automatic eviction when the budget is exceeded.

use crate::inference::kv_cache::LayerKvCache;

/// Eviction policy for H2O cache
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvictionPolicy {
    /// Keep tokens with highest cumulative attention scores
    CumulativeScore,
    /// Exponentially decay older scores (recency-biased)
    /// Parameter: decay factor per step (e.g., 0.95)
    DecayedScore(f32),
}

/// Per-layer H2O cache that wraps a KV cache with attention-based eviction
pub struct H2oLayerCache {
    /// Underlying KV storage
    pub kv: LayerKvCache,
    /// Cumulative attention score per position (sum across heads and steps)
    scores: Vec<f64>,
    /// Maximum number of tokens to keep (budget)
    budget: usize,
    /// Number of sink tokens to always keep (beginning of sequence)
    sink_tokens: usize,
    /// Eviction policy
    policy: EvictionPolicy,
    /// Original position indices (for tracking after evictions)
    original_positions: Vec<usize>,
    /// Number of evictions performed
    pub eviction_count: usize,
}

impl H2oLayerCache {
    /// Create a new H2O layer cache
    ///
    /// - `n_kv_heads`, `head_dim`: KV cache dimensions
    /// - `budget`: max tokens to retain
    /// - `sink_tokens`: first N tokens always kept (system prompt, BOS)
    /// - `policy`: eviction scoring policy
    pub fn new(
        n_kv_heads: usize,
        head_dim: usize,
        budget: usize,
        sink_tokens: usize,
        policy: EvictionPolicy,
    ) -> Self {
        assert!(budget > sink_tokens, "budget must exceed sink_tokens");
        Self {
            kv: LayerKvCache::new(n_kv_heads, head_dim),
            scores: Vec::new(),
            budget,
            sink_tokens,
            policy,
            original_positions: Vec::new(),
            eviction_count: 0,
        }
    }

    /// Current number of cached tokens
    pub fn len(&self) -> usize {
        self.kv.seq_len()
    }

    /// Whether cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a new KV entry and its initial attention score
    pub fn append(&mut self, key: Vec<f32>, value: Vec<f32>) {
        let pos = self.original_positions.last().map(|p| p + 1).unwrap_or(0);
        self.original_positions.push(pos);
        self.kv.append(key, value);
        self.scores.push(0.0);
    }

    /// Record attention scores from a decoding step.
    ///
    /// `attention_weights`: scores for each cached position, shape [seq_len].
    /// These are typically the softmax outputs summed/averaged across heads.
    pub fn record_attention(&mut self, attention_weights: &[f32]) {
        let len = self.len();
        let n = attention_weights.len().min(len);

        // Apply decay if using decayed scoring
        if let EvictionPolicy::DecayedScore(decay) = self.policy {
            let decay = decay as f64;
            for s in &mut self.scores {
                *s *= decay;
            }
        }

        // Accumulate new attention scores
        for (i, &w) in attention_weights.iter().enumerate().take(n) {
            self.scores[i] += w as f64;
        }
    }

    /// Evict tokens to bring cache size within budget.
    /// Returns the number of tokens evicted.
    pub fn evict_if_needed(&mut self) -> usize {
        if self.len() <= self.budget {
            return 0;
        }

        let to_evict = self.len() - self.budget;
        self.evict_n(to_evict)
    }

    /// Evict exactly `n` lowest-scoring tokens (respecting sink tokens).
    fn evict_n(&mut self, n: usize) -> usize {
        let len = self.len();
        if n == 0 || len <= self.sink_tokens {
            return 0;
        }

        // Find indices of evictable tokens (skip sinks) sorted by score ascending
        let mut evictable: Vec<(usize, f64)> = self
            .scores
            .iter()
            .enumerate()
            .skip(self.sink_tokens)
            .map(|(i, &s)| (i, s))
            .collect();

        evictable.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let actual_evict = n.min(evictable.len());
        if actual_evict == 0 {
            return 0;
        }

        // Collect indices to remove (sorted descending so removal doesn't shift indices)
        let mut to_remove: Vec<usize> = evictable[..actual_evict].iter().map(|(i, _)| *i).collect();
        to_remove.sort_unstable_by(|a, b| b.cmp(a));

        for idx in &to_remove {
            self.kv.keys.remove(*idx);
            self.kv.values.remove(*idx);
            self.scores.remove(*idx);
            self.original_positions.remove(*idx);
        }

        self.eviction_count += actual_evict;
        actual_evict
    }

    /// Get the original position indices of currently cached tokens
    pub fn original_positions(&self) -> &[usize] {
        &self.original_positions
    }

    /// Memory usage in bytes (KV data + score tracking overhead)
    pub fn size_bytes(&self) -> usize {
        self.kv.size_bytes()
            + self.scores.len() * std::mem::size_of::<f64>()
            + self.original_positions.len() * std::mem::size_of::<usize>()
    }

    /// Clear the cache completely
    pub fn clear(&mut self) {
        self.kv.clear();
        self.scores.clear();
        self.original_positions.clear();
        self.eviction_count = 0;
    }

    /// Get scores for diagnostics
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Rollback to a given length
    pub fn rollback(&mut self, new_len: usize) {
        self.kv.rollback(new_len);
        self.scores.truncate(new_len);
        self.original_positions.truncate(new_len);
    }
}

/// Full H2O KV cache across all layers
pub struct H2oCache {
    layers: Vec<H2oLayerCache>,
    budget: usize,
}

impl H2oCache {
    /// Create H2O cache for all layers
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        budget: usize,
        sink_tokens: usize,
        policy: EvictionPolicy,
    ) -> Self {
        let layers = (0..n_layers)
            .map(|_| H2oLayerCache::new(n_kv_heads, head_dim, budget, sink_tokens, policy))
            .collect();
        Self { layers, budget }
    }

    /// Get mutable reference to a layer
    pub fn layer_mut(&mut self, idx: usize) -> &mut H2oLayerCache {
        &mut self.layers[idx]
    }

    /// Get reference to a layer
    pub fn layer(&self, idx: usize) -> &H2oLayerCache {
        &self.layers[idx]
    }

    /// Current sequence length (from first layer)
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.len()).unwrap_or(0)
    }

    /// Total memory usage
    pub fn size_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.size_bytes()).sum()
    }

    /// Evict across all layers
    pub fn evict_all_layers(&mut self) -> usize {
        self.layers.iter_mut().map(|l| l.evict_if_needed()).sum()
    }

    /// Total evictions across all layers
    pub fn total_evictions(&self) -> usize {
        self.layers.iter().map(|l| l.eviction_count).sum()
    }

    /// Number of layers
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Budget per layer
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Clear all layers
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Rollback all layers
    pub fn rollback(&mut self, new_len: usize) {
        for layer in &mut self.layers {
            layer.rollback(new_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kv(val: f32, n_kv_heads: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let size = n_kv_heads * head_dim;
        (vec![val; size], vec![val; size])
    }

    #[test]
    fn test_h2o_basic_append() {
        let mut cache = H2oLayerCache::new(2, 32, 10, 0, EvictionPolicy::CumulativeScore);
        let (k, v) = make_kv(1.0, 2, 32);
        cache.append(k, v);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.original_positions(), &[0]);
    }

    #[test]
    fn test_h2o_no_eviction_within_budget() {
        let mut cache = H2oLayerCache::new(2, 32, 5, 0, EvictionPolicy::CumulativeScore);
        for i in 0..5 {
            let (k, v) = make_kv(i as f32, 2, 32);
            cache.append(k, v);
        }
        let evicted = cache.evict_if_needed();
        assert_eq!(evicted, 0);
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn test_h2o_evicts_lowest_scores() {
        let mut cache = H2oLayerCache::new(2, 32, 4, 0, EvictionPolicy::CumulativeScore);
        // Add 5 tokens
        for i in 0..5 {
            let (k, v) = make_kv(i as f32, 2, 32);
            cache.append(k, v);
        }
        // Token 2 and 4 are "heavy hitters", token 0 is low
        cache.record_attention(&[0.1, 0.05, 0.5, 0.05, 0.3]);

        let evicted = cache.evict_if_needed();
        assert_eq!(evicted, 1);
        assert_eq!(cache.len(), 4);

        // Token at original position 1 (score 0.05) should be evicted
        // Remaining: 0 (0.1), 2 (0.5), 3 (0.05 — tied but came after 1), 4 (0.3)
        // Actually both 1 and 3 have 0.05, evict first one encountered (index 1 after sinks=0)
        let positions = cache.original_positions();
        // Position 1 (lowest score among evictable, first tie) should be gone
        assert!(!positions.contains(&1));
        assert!(positions.contains(&0));
        assert!(positions.contains(&2));
        assert!(positions.contains(&4));
    }

    #[test]
    fn test_h2o_sink_tokens_protected() {
        let mut cache = H2oLayerCache::new(2, 32, 4, 2, EvictionPolicy::CumulativeScore);
        // Add 5 tokens
        for i in 0..5 {
            let (k, v) = make_kv(i as f32, 2, 32);
            cache.append(k, v);
        }
        // Give sink tokens (0, 1) LOW scores — they should still be kept
        cache.record_attention(&[0.01, 0.01, 0.2, 0.5, 0.3]);

        let evicted = cache.evict_if_needed();
        assert_eq!(evicted, 1);
        assert_eq!(cache.len(), 4);

        let positions = cache.original_positions();
        // Sinks (0, 1) must be kept despite low scores
        assert!(positions.contains(&0));
        assert!(positions.contains(&1));
        // Token 2 (score 0.2) is lowest non-sink, should be evicted
        assert!(!positions.contains(&2));
    }

    #[test]
    fn test_h2o_decayed_scoring() {
        let mut cache = H2oLayerCache::new(2, 32, 3, 0, EvictionPolicy::DecayedScore(0.5));
        for i in 0..4 {
            let (k, v) = make_kv(i as f32, 2, 32);
            cache.append(k, v);
        }

        // Step 1: token 0 gets high score
        cache.record_attention(&[1.0, 0.0, 0.0, 0.0]);
        // Step 2: scores decay by 0.5, token 3 gets high score
        cache.record_attention(&[0.0, 0.0, 0.0, 1.0]);

        // After step 2: token 0 has 1.0*0.5 + 0.0 = 0.5, token 3 has 0.0*0.5 + 1.0 = 1.0
        // Token 1 has 0, token 2 has 0 — one should be evicted
        let evicted = cache.evict_if_needed();
        assert_eq!(evicted, 1);
        assert_eq!(cache.len(), 3);

        // Token 1 or 2 evicted (both score 0), token 0 and 3 kept
        let positions = cache.original_positions();
        assert!(positions.contains(&0));
        assert!(positions.contains(&3));
    }

    #[test]
    fn test_h2o_multiple_evictions() {
        let mut cache = H2oLayerCache::new(2, 32, 3, 0, EvictionPolicy::CumulativeScore);
        for i in 0..6 {
            let (k, v) = make_kv(i as f32, 2, 32);
            cache.append(k, v);
        }
        cache.record_attention(&[0.1, 0.0, 0.5, 0.0, 0.3, 0.1]);

        let evicted = cache.evict_if_needed();
        assert_eq!(evicted, 3);
        assert_eq!(cache.len(), 3);

        // Top 3 scores: token 2 (0.5), token 4 (0.3), token 0 or 5 (both 0.1)
        let positions = cache.original_positions();
        assert!(positions.contains(&2));
        assert!(positions.contains(&4));
    }

    #[test]
    fn test_h2o_full_cache_rollback() {
        let mut cache = H2oCache::new(2, 4, 32, 10, 2, EvictionPolicy::CumulativeScore);
        for _ in 0..5 {
            for l in 0..2 {
                let (k, v) = make_kv(1.0, 4, 32);
                cache.layer_mut(l).append(k, v);
            }
        }
        assert_eq!(cache.seq_len(), 5);

        cache.rollback(3);
        assert_eq!(cache.seq_len(), 3);
        assert_eq!(cache.layer(0).original_positions().len(), 3);
    }

    #[test]
    fn test_h2o_clear() {
        let mut cache = H2oLayerCache::new(2, 32, 10, 0, EvictionPolicy::CumulativeScore);
        let (k, v) = make_kv(1.0, 2, 32);
        cache.append(k, v);
        cache.eviction_count = 5;
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.eviction_count, 0);
        assert!(cache.scores().is_empty());
    }

    #[test]
    fn test_h2o_size_bytes() {
        let mut cache = H2oLayerCache::new(2, 32, 10, 0, EvictionPolicy::CumulativeScore);
        let (k, v) = make_kv(1.0, 2, 32);
        cache.append(k, v);
        // KV: 2 * (2*32*4) = 512 bytes, scores: 8 bytes, positions: 8 bytes
        assert_eq!(cache.size_bytes(), 512 + 8 + 8);
    }

    #[test]
    fn test_h2o_multi_layer_eviction() {
        let mut cache = H2oCache::new(2, 4, 32, 3, 1, EvictionPolicy::CumulativeScore);
        // Fill to 4 tokens
        for _ in 0..4 {
            for l in 0..2 {
                let (k, v) = make_kv(1.0, 4, 32);
                cache.layer_mut(l).append(k, v);
            }
        }
        // Give different attention patterns per layer
        cache.layer_mut(0).record_attention(&[0.1, 0.6, 0.2, 0.1]);
        cache.layer_mut(1).record_attention(&[0.1, 0.1, 0.6, 0.2]);

        let total = cache.evict_all_layers();
        assert_eq!(total, 2); // 1 eviction per layer
                              // Each layer keeps its own top-3 non-sink tokens
        assert_eq!(cache.layer(0).len(), 3);
        assert_eq!(cache.layer(1).len(), 3);
    }

    #[test]
    fn test_h2o_cumulative_across_steps() {
        let mut cache = H2oLayerCache::new(2, 32, 3, 0, EvictionPolicy::CumulativeScore);
        for i in 0..4 {
            let (k, v) = make_kv(i as f32, 2, 32);
            cache.append(k, v);
        }

        // Step 1: token 0 is important
        cache.record_attention(&[0.8, 0.1, 0.05, 0.05]);
        // Step 2: token 3 is important
        cache.record_attention(&[0.1, 0.05, 0.05, 0.8]);

        // Cumulative: token 0 = 0.9, token 1 = 0.15, token 2 = 0.1, token 3 = 0.85
        let evicted = cache.evict_if_needed();
        assert_eq!(evicted, 1);

        let positions = cache.original_positions();
        // Token 2 has lowest cumulative score
        assert!(!positions.contains(&2));
        assert!(positions.contains(&0));
        assert!(positions.contains(&3));
    }
}
