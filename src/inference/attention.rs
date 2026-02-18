//! Multi-Head Attention with KV Cache support
//!
//! v0.3: Full KV-cached attention for efficient autoregressive generation.
//! Supports Grouped-Query Attention (GQA) where n_head > n_head_kv.

use crate::inference::kv_cache::LayerKvCache;
use crate::metal::compute::{matvec_f32_simd, softmax_f32_fast};

/// Compute multi-head attention with KV cache
///
/// For each new token position:
///   1. Project input to Q, K, V
///   2. Apply RoPE to Q and K
///   3. Append K, V to the cache
///   4. Compute attention scores against all cached K
///   5. Apply softmax and weight cached V
///   6. Output projection
pub fn multi_head_attention_cached(
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
) -> Vec<f32> {
    let n_embd = x.len();
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;

    // Project: Q = x @ Wq, K = x @ Wk, V = x @ Wv
    let mut q = matvec_f32_simd(wq, x, q_dim, n_embd);
    let mut k = matvec_f32_simd(wk, x, kv_dim, n_embd);
    let v = matvec_f32_simd(wv, x, kv_dim, n_embd);

    // Apply RoPE to Q and K
    apply_rope_inplace(&mut q, head_dim, n_head, position);
    apply_rope_inplace(&mut k, head_dim, n_head_kv, position);

    // Append K, V to cache
    kv_cache.append(k, v);

    let seq_len = kv_cache.seq_len();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_group_size = (n_head / n_head_kv).max(1);

    let mut attn_output = vec![0.0f32; n_embd];

    // Process each query head
    for h in 0..n_head {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        // Compute attention scores against all cached keys
        let mut scores = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let k_head = kv_cache.key_at(pos, kv_h);
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_head[d];
            }
            scores.push(dot * scale);
        }

        // Apply causal mask (not needed since we only have past + current positions)
        // Apply softmax
        softmax_f32_fast(&mut scores);

        // Weighted sum of values
        let out_offset = h * head_dim;
        for d in 0..head_dim {
            let mut weighted = 0.0f32;
            for (pos, &score) in scores.iter().enumerate().take(seq_len) {
                let v_head = kv_cache.value_at(pos, kv_h);
                weighted += score * v_head[d];
            }
            if out_offset + d < attn_output.len() {
                attn_output[out_offset + d] = weighted;
            }
        }
    }

    // Output projection: attn_output @ Wo
    matvec_f32_simd(wo, &attn_output, n_embd, n_embd)
}

/// Legacy single-token attention (no cache, kept for compatibility)
pub fn multi_head_attention(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    position: usize,
) -> Vec<f32> {
    let n_embd = x.len();
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;

    let mut q = matvec_f32_simd(wq, x, q_dim, n_embd);
    let mut k = matvec_f32_simd(wk, x, kv_dim, n_embd);
    let v = matvec_f32_simd(wv, x, kv_dim, n_embd);

    apply_rope_inplace(&mut q, head_dim, n_head, position);
    apply_rope_inplace(&mut k, head_dim, n_head_kv, position);

    let _scale = 1.0 / (head_dim as f32).sqrt();
    let kv_group_size = (n_head / n_head_kv).max(1);

    let mut attn_output = vec![0.0f32; n_embd];

    for h in 0..n_head {
        let kv_h = h / kv_group_size;
        let _q_start = h * head_dim;
        let _k_start = kv_h * head_dim;
        let v_start = kv_h * head_dim;

        // Single-token: softmax of one value = 1.0 → output = V
        for d in 0..head_dim {
            let out_idx = h * head_dim + d;
            if out_idx < attn_output.len() {
                attn_output[out_idx] = v.get(v_start + d).copied().unwrap_or(0.0);
            }
        }
    }

    matvec_f32_simd(wo, &attn_output, n_embd, n_embd)
}

/// Apply RoPE in-place across all heads
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::LayerKvCache;

    #[test]
    fn test_rope_deterministic() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let mut b = a.clone();
        apply_rope_inplace(&mut a, 4, 1, 5);
        apply_rope_inplace(&mut b, 4, 1, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn test_rope_position_zero_identity() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let orig = x.clone();
        apply_rope_inplace(&mut x, 4, 1, 0);
        // At position 0, angle = 0, so cos=1, sin=0 → identity
        for i in 0..4 {
            assert!(
                (x[i] - orig[i]).abs() < 1e-6,
                "RoPE at pos 0 should be identity"
            );
        }
    }

    #[test]
    fn test_cached_attention_grows() {
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let mut kv = LayerKvCache::new(n_head_kv, head_dim);

        let x = vec![1.0f32; n_embd];
        let w = vec![0.01f32; n_embd * n_embd]; // simple weight

        let _ = multi_head_attention_cached(
            &x, &w, &w, &w, &w, n_head, n_head_kv, head_dim, 0, &mut kv,
        );
        assert_eq!(kv.seq_len(), 1);

        let _ = multi_head_attention_cached(
            &x, &w, &w, &w, &w, n_head, n_head_kv, head_dim, 1, &mut kv,
        );
        assert_eq!(kv.seq_len(), 2);
    }
}
