//! Sliding Window Attention
//!
//! v0.8: Limits attention to the most recent W tokens instead of the full sequence.
//! This bounds memory usage and compute cost for long sequences while maintaining
//! quality — most useful information is in recent context anyway.
//!
//! Models like Mistral use sliding window attention natively (W=4096).
//! For other models, this provides an optional memory-saving mode.

use crate::inference::kv_cache::LayerKvCache;
use crate::metal::compute::{matvec_f32_simd, softmax_f32_fast};

/// Configuration for sliding window attention
#[derive(Clone, Debug)]
pub struct SlidingWindowConfig {
    /// Window size — number of recent tokens to attend to (0 = unlimited/full attention)
    pub window_size: usize,
    /// Whether to use sink tokens (keep first N tokens always visible)
    /// Useful for system prompts and BOS token
    pub sink_tokens: usize,
}

impl SlidingWindowConfig {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            sink_tokens: 0,
        }
    }

    pub fn with_sinks(window_size: usize, sink_tokens: usize) -> Self {
        Self {
            window_size,
            sink_tokens,
        }
    }

    /// Returns true if sliding window is active
    pub fn is_active(&self) -> bool {
        self.window_size > 0
    }

    /// Compute the effective attention range for a given sequence length
    /// Returns (start_positions, includes_sinks)
    pub fn attention_range(&self, seq_len: usize) -> AttentionRange {
        if !self.is_active() || seq_len <= self.window_size + self.sink_tokens {
            // Full attention — sequence fits within window
            return AttentionRange {
                sink_end: 0,
                window_start: 0,
                window_end: seq_len,
                total_positions: seq_len,
            };
        }

        let window_start = seq_len.saturating_sub(self.window_size);
        if self.sink_tokens > 0 && window_start > self.sink_tokens {
            // Sink tokens + sliding window
            AttentionRange {
                sink_end: self.sink_tokens,
                window_start,
                window_end: seq_len,
                total_positions: self.sink_tokens + self.window_size,
            }
        } else {
            // Just sliding window (or sinks overlap with window)
            AttentionRange {
                sink_end: 0,
                window_start,
                window_end: seq_len,
                total_positions: seq_len - window_start,
            }
        }
    }
}

/// Describes which positions to attend to
#[derive(Debug, Clone)]
pub struct AttentionRange {
    /// End of sink region (exclusive), 0 if no sinks
    pub sink_end: usize,
    /// Start of sliding window region (inclusive)
    pub window_start: usize,
    /// End of sliding window region (exclusive, == seq_len)
    pub window_end: usize,
    /// Total number of positions to attend to
    pub total_positions: usize,
}

impl AttentionRange {
    /// Iterate over all positions that should be attended to
    pub fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        let sink_iter = 0..self.sink_end;
        let window_iter = self.window_start..self.window_end;
        sink_iter.chain(window_iter)
    }
}

/// Compute multi-head attention with sliding window and KV cache
///
/// Same as `multi_head_attention_cached` but only attends to positions
/// within the sliding window (plus optional sink tokens).
pub fn sliding_window_attention(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    position: usize,
    kv_cache: &mut LayerKvCache,
    window_config: &SlidingWindowConfig,
) -> Vec<f32> {
    let n_embd = x.len();
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;

    // Project Q, K, V
    let mut q = matvec_f32_simd(wq, x, q_dim, n_embd);
    let mut k = matvec_f32_simd(wk, x, kv_dim, n_embd);
    let v = matvec_f32_simd(wv, x, kv_dim, n_embd);

    // Apply RoPE
    apply_rope_inplace(&mut q, head_dim, n_head, position);
    apply_rope_inplace(&mut k, head_dim, n_head_kv, position);

    // Append to cache
    kv_cache.append(k, v);

    let seq_len = kv_cache.seq_len();
    let range = window_config.attention_range(seq_len);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_group_size = (n_head / n_head_kv).max(1);

    let mut attn_output = vec![0.0f32; n_embd];

    for h in 0..n_head {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        // Compute attention scores only for positions in the window
        let positions: Vec<usize> = range.positions().collect();
        let mut scores = Vec::with_capacity(positions.len());

        for &pos in &positions {
            let k_head = kv_cache.key_at(pos, kv_h);
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_head[d];
            }
            scores.push(dot * scale);
        }

        softmax_f32_fast(&mut scores);

        // Weighted sum of values
        let out_offset = h * head_dim;
        for d in 0..head_dim {
            let mut weighted = 0.0f32;
            for (i, &pos) in positions.iter().enumerate() {
                let v_head = kv_cache.value_at(pos, kv_h);
                weighted += scores[i] * v_head[d];
            }
            if out_offset + d < attn_output.len() {
                attn_output[out_offset + d] = weighted;
            }
        }
    }

    matvec_f32_simd(wo, &attn_output, n_embd, n_embd)
}

/// Apply RoPE in-place (duplicated to avoid circular deps — could be shared)
fn apply_rope_inplace(x: &mut [f32], head_dim: usize, n_heads: usize, position: usize) {
    let theta_base = 10000.0f32;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / theta_base.powf(i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let (sin_val, cos_val) = angle.sin_cos();
            let idx0 = base + i;
            let idx1 = base + i + 1;
            if idx1 < x.len() {
                let x0 = x[idx0];
                let x1 = x[idx1];
                x[idx0] = x0 * cos_val - x1 * sin_val;
                x[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/// Evict old KV entries outside the sliding window to reclaim memory
/// Returns the number of positions evicted
pub fn evict_outside_window(kv_cache: &mut LayerKvCache, window_config: &SlidingWindowConfig) -> usize {
    if !window_config.is_active() {
        return 0;
    }

    let seq_len = kv_cache.seq_len();
    if seq_len <= window_config.window_size + window_config.sink_tokens {
        return 0;
    }

    let keep_start = seq_len.saturating_sub(window_config.window_size);
    let sink_end = window_config.sink_tokens.min(keep_start);

    // Build new keys/values: sinks + window
    let mut new_keys = Vec::new();
    let mut new_values = Vec::new();

    // Keep sink tokens
    for i in 0..sink_end {
        new_keys.push(kv_cache.keys[i].clone());
        new_values.push(kv_cache.values[i].clone());
    }

    // Keep window tokens
    for i in keep_start..seq_len {
        new_keys.push(kv_cache.keys[i].clone());
        new_values.push(kv_cache.values[i].clone());
    }

    let evicted = seq_len - new_keys.len();
    kv_cache.keys = new_keys;
    kv_cache.values = new_values;
    evicted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::LayerKvCache;

    #[test]
    fn test_attention_range_full() {
        let config = SlidingWindowConfig::new(4096);
        let range = config.attention_range(100);
        assert_eq!(range.total_positions, 100);
        assert_eq!(range.window_start, 0);
    }

    #[test]
    fn test_attention_range_windowed() {
        let config = SlidingWindowConfig::new(512);
        let range = config.attention_range(2000);
        assert_eq!(range.window_start, 1488);
        assert_eq!(range.window_end, 2000);
        assert_eq!(range.total_positions, 512);
    }

    #[test]
    fn test_attention_range_with_sinks() {
        let config = SlidingWindowConfig::with_sinks(512, 4);
        let range = config.attention_range(2000);
        assert_eq!(range.sink_end, 4);
        assert_eq!(range.window_start, 1488);
        assert_eq!(range.total_positions, 516); // 4 sinks + 512 window
    }

    #[test]
    fn test_attention_range_small_sequence() {
        let config = SlidingWindowConfig::with_sinks(512, 4);
        let range = config.attention_range(100);
        // 100 < 512 + 4 → full attention
        assert_eq!(range.total_positions, 100);
        assert_eq!(range.window_start, 0);
    }

    #[test]
    fn test_evict_outside_window() {
        let mut kv = LayerKvCache::new(2, 4);
        for i in 0..10 {
            kv.append(vec![i as f32; 8], vec![(i * 10) as f32; 8]);
        }
        assert_eq!(kv.seq_len(), 10);

        let config = SlidingWindowConfig::new(4);
        let evicted = evict_outside_window(&mut kv, &config);
        assert_eq!(evicted, 6);
        assert_eq!(kv.seq_len(), 4);
        // Remaining should be positions 6,7,8,9
        assert_eq!(kv.keys[0][0], 6.0);
    }

    #[test]
    fn test_evict_with_sinks() {
        let mut kv = LayerKvCache::new(2, 4);
        for i in 0..10 {
            kv.append(vec![i as f32; 8], vec![(i * 10) as f32; 8]);
        }

        let config = SlidingWindowConfig::with_sinks(4, 2);
        let evicted = evict_outside_window(&mut kv, &config);
        assert_eq!(evicted, 4); // keep sinks 0,1 + window 6,7,8,9
        assert_eq!(kv.seq_len(), 6);
        assert_eq!(kv.keys[0][0], 0.0); // sink 0
        assert_eq!(kv.keys[1][0], 1.0); // sink 1
        assert_eq!(kv.keys[2][0], 6.0); // window start
    }

    #[test]
    fn test_sliding_window_attention_basic() {
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let mut kv = LayerKvCache::new(n_head_kv, head_dim);
        let config = SlidingWindowConfig::new(4);

        let x = vec![1.0f32; n_embd];
        let w = vec![0.01f32; n_embd * n_embd];

        // Add 6 tokens — window should only attend to last 4
        for pos in 0..5 {
            let _ = sliding_window_attention(
                &x, &w, &w, &w, &w, n_head, n_head_kv, head_dim, pos, &mut kv, &config,
            );
        }
        assert_eq!(kv.seq_len(), 5);

        let result = sliding_window_attention(
            &x, &w, &w, &w, &w, n_head, n_head_kv, head_dim, 5, &mut kv, &config,
        );
        assert_eq!(result.len(), n_embd);
        assert_eq!(kv.seq_len(), 6);
    }

    #[test]
    fn test_disabled_sliding_window() {
        let config = SlidingWindowConfig::new(0);
        assert!(!config.is_active());
        let range = config.attention_range(1000);
        assert_eq!(range.total_positions, 1000);
    }
}
