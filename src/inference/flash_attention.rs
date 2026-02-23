//! Flash Attention — memory-efficient fused attention kernel
//!
//! Implements the online softmax algorithm from "FlashAttention: Fast and Memory-Efficient
//! Exact Attention with IO-Awareness" (Dao et al., 2022). Instead of materializing the full
//! N×N attention matrix, we compute attention in a single pass using running max/sum
//! accumulators, achieving O(1) extra memory per head instead of O(N).
//!
//! This is critical for SSD-LLM: with sliding window attention allowing long contexts,
//! the attention matrix can be enormous. Flash attention keeps memory bounded regardless
//! of sequence length.

use crate::inference::kv_cache::LayerKvCache;
use crate::metal::compute::{matvec_f32_simd, MetalCompute};

/// Flash attention configuration
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size for tiled computation (smaller = less memory, more overhead)
    pub block_size: usize,
    /// Whether to use causal masking
    pub causal: bool,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            causal: true,
        }
    }
}

/// Compute multi-head attention using the flash attention algorithm.
///
/// Key difference from standard attention:
/// - Standard: scores = Q @ K^T (full N×N matrix), softmax(scores), output = scores @ V
/// - Flash: iterate over K/V blocks, maintain running softmax numerator/denominator per output element
///
/// Memory: O(head_dim) per head instead of O(seq_len) per head
pub fn flash_attention_cached(
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
    _config: &FlashAttentionConfig,
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

    // Append to KV cache
    kv_cache.append(k, v);

    let seq_len = kv_cache.seq_len();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_group_size = (n_head / n_head_kv).max(1);

    let mut attn_output = vec![0.0f32; n_embd];

    // Flash attention: online softmax per head
    for h in 0..n_head {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        // Online softmax accumulators
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut output_acc = vec![0.0f32; head_dim];

        // Single pass over all KV positions
        for pos in 0..seq_len {
            let k_head = kv_cache.key_at(pos, kv_h);

            // Compute score = Q · K / sqrt(d)
            let mut dot = 0.0f32;
            // 4-wide manual unroll for SIMD-friendly computation
            let chunks = head_dim / 4;
            let remainder = head_dim % 4;
            for c in 0..chunks {
                let base = c * 4;
                dot += q_head[base] * k_head[base]
                    + q_head[base + 1] * k_head[base + 1]
                    + q_head[base + 2] * k_head[base + 2]
                    + q_head[base + 3] * k_head[base + 3];
            }
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                dot += q_head[i] * k_head[i];
            }
            let score = dot * scale;

            // Online softmax update (Milakov & Gimelshein, 2018)
            let v_head = kv_cache.value_at(pos, kv_h);

            if score > running_max {
                // New maximum: rescale existing accumulator
                let correction = (running_max - score).exp();
                running_sum *= correction;
                for acc in output_acc.iter_mut().take(head_dim) {
                    *acc *= correction;
                }
                running_max = score;
            }

            let weight = (score - running_max).exp();
            running_sum += weight;

            // Accumulate weighted value
            for d in 0..head_dim {
                output_acc[d] += weight * v_head[d];
            }
        }

        // Normalize by total softmax denominator
        let out_offset = h * head_dim;
        if running_sum > 0.0 {
            let inv_sum = 1.0 / running_sum;
            for d in 0..head_dim {
                if out_offset + d < attn_output.len() {
                    attn_output[out_offset + d] = output_acc[d] * inv_sum;
                }
            }
        }
    }

    // Output projection
    matvec_f32_simd(wo, &attn_output, n_embd, n_embd)
}

/// Flash attention for a range of positions (compatible with sliding window)
pub fn flash_attention_windowed(
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
    window_start: usize,
    window_end: usize,
    _config: &FlashAttentionConfig,
) -> Vec<f32> {
    let n_embd = x.len();
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;

    let mut q = matvec_f32_simd(wq, x, q_dim, n_embd);
    let mut k = matvec_f32_simd(wk, x, kv_dim, n_embd);
    let v = matvec_f32_simd(wv, x, kv_dim, n_embd);

    apply_rope_inplace(&mut q, head_dim, n_head, position);
    apply_rope_inplace(&mut k, head_dim, n_head_kv, position);

    kv_cache.append(k, v);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_group_size = (n_head / n_head_kv).max(1);
    let effective_end = window_end.min(kv_cache.seq_len());

    let mut attn_output = vec![0.0f32; n_embd];

    for h in 0..n_head {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut output_acc = vec![0.0f32; head_dim];

        for pos in window_start..effective_end {
            let k_head = kv_cache.key_at(pos, kv_h);
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_head[d];
            }
            let score = dot * scale;

            let v_head = kv_cache.value_at(pos, kv_h);

            if score > running_max {
                let correction = (running_max - score).exp();
                running_sum *= correction;
                for acc in output_acc.iter_mut().take(head_dim) {
                    *acc *= correction;
                }
                running_max = score;
            }

            let weight = (score - running_max).exp();
            running_sum += weight;
            for d in 0..head_dim {
                output_acc[d] += weight * v_head[d];
            }
        }

        let out_offset = h * head_dim;
        if running_sum > 0.0 {
            let inv_sum = 1.0 / running_sum;
            for d in 0..head_dim {
                if out_offset + d < attn_output.len() {
                    attn_output[out_offset + d] = output_acc[d] * inv_sum;
                }
            }
        }
    }

    matvec_f32_simd(wo, &attn_output, n_embd, n_embd)
}

/// GPU-accelerated flash attention using Metal compute.
///
/// Same semantics as `flash_attention_cached` but offloads the attention
/// computation (QK scoring + online softmax + V accumulation) to the GPU.
/// Falls back to CPU if Metal is unavailable or sequence is too short.
pub fn flash_attention_gpu(
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
    _config: &FlashAttentionConfig,
    metal: Option<&MetalCompute>,
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

    // Append to KV cache
    kv_cache.append(k, v);

    let seq_len = kv_cache.seq_len();

    // Try GPU path: flatten KV cache into contiguous buffers for Metal
    if let Some(mc) = metal {
        let (k_flat, v_flat) = kv_cache.flatten_kv(n_head_kv, head_dim);
        if let Some(attn_output) = mc.flash_attention_f32(
            &q, &k_flat, &v_flat, n_head, n_head_kv, head_dim, seq_len, 0, seq_len,
        ) {
            // Output projection
            return matvec_f32_simd(wo, &attn_output, n_embd, n_embd);
        }
    }

    // CPU fallback: standard online softmax
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_group_size = (n_head / n_head_kv).max(1);
    let mut attn_output = vec![0.0f32; n_embd];

    for h in 0..n_head {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut output_acc = vec![0.0f32; head_dim];

        for pos in 0..seq_len {
            let k_head = kv_cache.key_at(pos, kv_h);
            let mut dot = 0.0f32;
            let chunks = head_dim / 4;
            let remainder = head_dim % 4;
            for c in 0..chunks {
                let base = c * 4;
                dot += q_head[base] * k_head[base]
                    + q_head[base + 1] * k_head[base + 1]
                    + q_head[base + 2] * k_head[base + 2]
                    + q_head[base + 3] * k_head[base + 3];
            }
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                dot += q_head[i] * k_head[i];
            }
            let score = dot * scale;

            let v_head = kv_cache.value_at(pos, kv_h);
            if score > running_max {
                let correction = (running_max - score).exp();
                running_sum *= correction;
                for acc in output_acc.iter_mut().take(head_dim) {
                    *acc *= correction;
                }
                running_max = score;
            }

            let weight = (score - running_max).exp();
            running_sum += weight;
            for d in 0..head_dim {
                output_acc[d] += weight * v_head[d];
            }
        }

        let out_offset = h * head_dim;
        if running_sum > 0.0 {
            let inv_sum = 1.0 / running_sum;
            for d in 0..head_dim {
                if out_offset + d < attn_output.len() {
                    attn_output[out_offset + d] = output_acc[d] * inv_sum;
                }
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
    fn test_flash_attention_basic() {
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let mut kv = LayerKvCache::new(n_head_kv, head_dim);

        let x = vec![1.0f32; n_embd];
        let w = vec![0.01f32; n_embd * n_embd];
        let config = FlashAttentionConfig::default();

        let output = flash_attention_cached(
            &x, &w, &w, &w, &w, n_head, n_head_kv, head_dim, 0, &mut kv, &config,
        );

        assert_eq!(output.len(), n_embd);
        assert_eq!(kv.seq_len(), 1);
    }

    #[test]
    fn test_flash_attention_multi_token() {
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let mut kv = LayerKvCache::new(n_head_kv, head_dim);

        let x = vec![0.5f32; n_embd];
        let w = vec![0.01f32; n_embd * n_embd];
        let config = FlashAttentionConfig::default();

        // Process 5 tokens
        for pos in 0..5 {
            let _ = flash_attention_cached(
                &x, &w, &w, &w, &w, n_head, n_head_kv, head_dim, pos, &mut kv, &config,
            );
        }

        assert_eq!(kv.seq_len(), 5);
    }

    #[test]
    fn test_flash_vs_standard_single_token() {
        // With a single token, flash attention should give same result as standard
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;

        let x = vec![1.0f32; n_embd];
        let w = vec![0.02f32; n_embd * n_embd];

        let mut kv_flash = LayerKvCache::new(n_head_kv, head_dim);
        let config = FlashAttentionConfig::default();
        let flash_out = flash_attention_cached(
            &x,
            &w,
            &w,
            &w,
            &w,
            n_head,
            n_head_kv,
            head_dim,
            0,
            &mut kv_flash,
            &config,
        );

        let mut kv_std = LayerKvCache::new(n_head_kv, head_dim);
        let std_out = crate::inference::attention::multi_head_attention_cached(
            &x,
            &w,
            &w,
            &w,
            &w,
            n_head,
            n_head_kv,
            head_dim,
            0,
            &mut kv_std,
        );

        // Should be identical for single token (softmax of 1 element = 1.0)
        for i in 0..n_embd {
            assert!(
                (flash_out[i] - std_out[i]).abs() < 1e-5,
                "flash[{}]={} != std[{}]={}",
                i,
                flash_out[i],
                i,
                std_out[i]
            );
        }
    }

    #[test]
    fn test_flash_attention_numerically_stable() {
        // Test with extreme values that would cause overflow in naive softmax
        let n_embd = 4;
        let n_head = 1;
        let n_head_kv = 1;
        let head_dim = 4;
        let mut kv = LayerKvCache::new(n_head_kv, head_dim);

        // Large input values
        let x = vec![100.0f32; n_embd];
        let w = vec![0.1f32; n_embd * n_embd];
        let config = FlashAttentionConfig::default();

        let output = flash_attention_cached(
            &x, &w, &w, &w, &w, n_head, n_head_kv, head_dim, 0, &mut kv, &config,
        );

        // Output should be finite (no NaN/Inf from overflow)
        for i in 0..n_embd {
            assert!(
                output[i].is_finite(),
                "output[{}] = {} is not finite",
                i,
                output[i]
            );
        }
    }

    #[test]
    fn test_flash_attention_windowed() {
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let mut kv = LayerKvCache::new(n_head_kv, head_dim);

        let x = vec![0.5f32; n_embd];
        let w = vec![0.01f32; n_embd * n_embd];
        let config = FlashAttentionConfig::default();

        // Fill cache with 10 tokens using standard flash attention
        for pos in 0..10 {
            let _ = flash_attention_cached(
                &x, &w, &w, &w, &w, n_head, n_head_kv, head_dim, pos, &mut kv, &config,
            );
        }

        // Now do windowed attention on a subset (positions 5..10)
        let mut kv2 = kv.clone();
        let output = flash_attention_windowed(
            &x, &w, &w, &w, &w, n_head, n_head_kv, head_dim, 10, &mut kv2, 5, 11, &config,
        );

        assert_eq!(output.len(), n_embd);
        for i in 0..n_embd {
            assert!(output[i].is_finite());
        }
    }

    #[test]
    fn test_flash_attention_gqa() {
        // Test with GQA (more query heads than KV heads)
        let n_embd = 16;
        let n_head = 4;
        let n_head_kv = 2; // GQA: group_size = 2
        let head_dim = 4;
        let mut kv = LayerKvCache::new(n_head_kv, head_dim);

        let x = vec![1.0f32; n_embd];
        let w_q = vec![0.01f32; n_head * head_dim * n_embd];
        let w_kv = vec![0.01f32; n_head_kv * head_dim * n_embd];
        let w_o = vec![0.01f32; n_embd * n_embd];
        let config = FlashAttentionConfig::default();

        // Process 3 tokens
        for pos in 0..3 {
            let _ = flash_attention_cached(
                &x, &w_q, &w_kv, &w_kv, &w_o, n_head, n_head_kv, head_dim, pos, &mut kv, &config,
            );
        }

        assert_eq!(kv.seq_len(), 3);
    }

    #[test]
    fn test_flash_attention_gpu_basic() {
        // Test GPU-accelerated path (will use CPU fallback if Metal unavailable)
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let mut kv = LayerKvCache::new(n_head_kv, head_dim);

        let x = vec![1.0f32; n_embd];
        let w = vec![0.01f32; n_embd * n_embd];
        let config = FlashAttentionConfig::default();

        let output = flash_attention_gpu(
            &x, &w, &w, &w, &w, n_head, n_head_kv, head_dim, 0, &mut kv, &config, None,
        );

        assert_eq!(output.len(), n_embd);
        assert_eq!(kv.seq_len(), 1);
    }

    #[test]
    fn test_flash_attention_gpu_matches_cpu() {
        // GPU path (with CPU fallback) should match standard flash attention
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let x = vec![0.3f32; n_embd];
        let w = vec![0.05f32; n_embd * n_embd];
        let config = FlashAttentionConfig::default();

        // Build 3-token context with GPU path
        let mut kv_gpu = LayerKvCache::new(n_head_kv, head_dim);
        let mut gpu_out = vec![];
        for pos in 0..3 {
            gpu_out = flash_attention_gpu(
                &x,
                &w,
                &w,
                &w,
                &w,
                n_head,
                n_head_kv,
                head_dim,
                pos,
                &mut kv_gpu,
                &config,
                None,
            );
        }

        // Build 3-token context with standard
        let mut kv_std = LayerKvCache::new(n_head_kv, head_dim);
        let mut std_out = vec![];
        for pos in 0..3 {
            std_out = flash_attention_cached(
                &x,
                &w,
                &w,
                &w,
                &w,
                n_head,
                n_head_kv,
                head_dim,
                pos,
                &mut kv_std,
                &config,
            );
        }

        for i in 0..n_embd {
            assert!(
                (gpu_out[i] - std_out[i]).abs() < 1e-4,
                "Mismatch at {}: gpu={} std={}",
                i,
                gpu_out[i],
                std_out[i]
            );
        }
    }

    #[test]
    fn test_flash_attention_gpu_with_metal() {
        // Try with actual Metal if available
        let metal = MetalCompute::new();
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let x = vec![0.5f32; n_embd];
        let w = vec![0.02f32; n_embd * n_embd];
        let config = FlashAttentionConfig::default();

        let mut kv_gpu = LayerKvCache::new(n_head_kv, head_dim);
        let mut kv_cpu = LayerKvCache::new(n_head_kv, head_dim);

        for pos in 0..5 {
            let gpu_out = flash_attention_gpu(
                &x,
                &w,
                &w,
                &w,
                &w,
                n_head,
                n_head_kv,
                head_dim,
                pos,
                &mut kv_gpu,
                &config,
                metal.as_ref(),
            );
            let cpu_out = flash_attention_cached(
                &x,
                &w,
                &w,
                &w,
                &w,
                n_head,
                n_head_kv,
                head_dim,
                pos,
                &mut kv_cpu,
                &config,
            );

            for i in 0..n_embd {
                assert!(
                    (gpu_out[i] - cpu_out[i]).abs() < 1e-3,
                    "pos={} idx={}: gpu={} cpu={}",
                    pos,
                    i,
                    gpu_out[i],
                    cpu_out[i]
                );
            }
        }
    }

    #[test]
    fn test_flash_attention_gpu_gqa() {
        // Test GQA with GPU path
        let n_embd = 16;
        let n_head = 4;
        let n_head_kv = 2;
        let head_dim = 4;
        let metal = MetalCompute::new();

        let x = vec![1.0f32; n_embd];
        let w_q = vec![0.01f32; n_head * head_dim * n_embd];
        let w_kv = vec![0.01f32; n_head_kv * head_dim * n_embd];
        let w_o = vec![0.01f32; n_embd * n_embd];
        let config = FlashAttentionConfig::default();

        let mut kv_gpu = LayerKvCache::new(n_head_kv, head_dim);
        let mut kv_cpu = LayerKvCache::new(n_head_kv, head_dim);

        for pos in 0..3 {
            let gpu_out = flash_attention_gpu(
                &x,
                &w_q,
                &w_kv,
                &w_kv,
                &w_o,
                n_head,
                n_head_kv,
                head_dim,
                pos,
                &mut kv_gpu,
                &config,
                metal.as_ref(),
            );
            let cpu_out = flash_attention_cached(
                &x,
                &w_q,
                &w_kv,
                &w_kv,
                &w_o,
                n_head,
                n_head_kv,
                head_dim,
                pos,
                &mut kv_cpu,
                &config,
            );

            for i in 0..n_embd {
                assert!(
                    (gpu_out[i] - cpu_out[i]).abs() < 1e-3,
                    "GQA pos={} idx={}: gpu={} cpu={}",
                    pos,
                    i,
                    gpu_out[i],
                    cpu_out[i]
                );
            }
        }
    }

    #[test]
    fn test_kv_cache_flatten() {
        let mut kv = LayerKvCache::new(2, 4);
        kv.append(vec![1.0; 8], vec![2.0; 8]);
        kv.append(vec![3.0; 8], vec![4.0; 8]);

        let (k_flat, v_flat) = kv.flatten_kv(2, 4);
        assert_eq!(k_flat.len(), 16); // 2 positions × 2 heads × 4 dim
        assert_eq!(v_flat.len(), 16);
        assert_eq!(k_flat[0], 1.0);
        assert_eq!(k_flat[8], 3.0);
        assert_eq!(v_flat[0], 2.0);
        assert_eq!(v_flat[8], 4.0);
    }

    #[test]
    fn test_online_softmax_correctness() {
        // Verify online softmax gives same result as standard softmax
        // by checking that flash attention output matches standard attention
        // for multi-token sequence
        let n_embd = 4;
        let n_head = 1;
        let n_head_kv = 1;
        let head_dim = 4;

        let x = vec![0.3f32; n_embd];
        let w = vec![0.05f32; n_embd * n_embd];
        let config = FlashAttentionConfig::default();

        // Build 3-token context with flash
        let mut kv_flash = LayerKvCache::new(n_head_kv, head_dim);
        let mut flash_out = vec![];
        for pos in 0..3 {
            flash_out = flash_attention_cached(
                &x,
                &w,
                &w,
                &w,
                &w,
                n_head,
                n_head_kv,
                head_dim,
                pos,
                &mut kv_flash,
                &config,
            );
        }

        // Build 3-token context with standard
        let mut kv_std = LayerKvCache::new(n_head_kv, head_dim);
        let mut std_out = vec![];
        for pos in 0..3 {
            std_out = crate::inference::attention::multi_head_attention_cached(
                &x,
                &w,
                &w,
                &w,
                &w,
                n_head,
                n_head_kv,
                head_dim,
                pos,
                &mut kv_std,
            );
        }

        // Should produce identical results (same algorithm, just different memory pattern)
        for i in 0..n_embd {
            assert!(
                (flash_out[i] - std_out[i]).abs() < 1e-4,
                "Mismatch at {}: flash={} std={}",
                i,
                flash_out[i],
                std_out[i]
            );
        }
    }
}
