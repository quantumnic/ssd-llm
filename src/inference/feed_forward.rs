//! SwiGLU Feed-Forward Network (as used in LLaMA)
//!
//! FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
//!
//! Supports both CPU SIMD and fused Metal GPU paths.
//! When Metal is available and the intermediate dimension is large enough,
//! the fused SwiGLU kernel computes gate_proj + SiLU + up_proj + element-wise
//! multiply in a single GPU dispatch, halving memory round-trips.

use crate::metal::compute::{matvec_f32_simd, silu_f32};
use crate::metal::gpu::MetalGpu;

/// Minimum intermediate size to justify GPU dispatch overhead
const MIN_GPU_FF_ELEMENTS: usize = 2048;

/// Compute SwiGLU feed-forward block (CPU SIMD path)
pub fn feed_forward(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    n_embd: usize,
) -> Vec<f32> {
    // Infer intermediate size from weight dimensions
    let n_ff = w_gate.len() / n_embd;
    if n_ff == 0 {
        return vec![0.0f32; n_embd];
    }

    // gate = silu(x @ W_gate)
    let mut gate = matvec_f32_simd(w_gate, x, n_ff, n_embd);
    silu_f32(&mut gate);

    // up = x @ W_up
    let up = matvec_f32_simd(w_up, x, n_ff, n_embd);

    // element-wise: gate * up
    for (g, u) in gate.iter_mut().zip(up.iter()) {
        *g *= u;
    }

    // down = (gate * up) @ W_down
    matvec_f32_simd(w_down, &gate, n_embd, n_ff)
}

/// Compute SwiGLU feed-forward with optional Metal GPU acceleration.
/// Falls back to CPU SIMD when GPU is unavailable or tensors are too small.
pub fn feed_forward_gpu(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    n_embd: usize,
    gpu: Option<&MetalGpu>,
) -> Vec<f32> {
    let n_ff = w_gate.len() / n_embd;

    // Use fused GPU kernel for large enough tensors
    #[cfg(target_os = "macos")]
    if let Some(gpu) = gpu {
        if n_ff >= MIN_GPU_FF_ELEMENTS && gpu.is_available() {
            return gpu.feed_forward_f32(x, w_gate, w_up, w_down, n_embd);
        }
    }

    // Suppress unused variable warning on non-macOS
    #[cfg(not(target_os = "macos"))]
    let _ = gpu;

    feed_forward(x, w_gate, w_up, w_down, n_embd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward_shapes() {
        let n_embd = 4;
        let n_ff = 8;
        let x = vec![1.0f32; n_embd];
        let w_gate = vec![0.1f32; n_ff * n_embd];
        let w_up = vec![0.1f32; n_ff * n_embd];
        let w_down = vec![0.1f32; n_embd * n_ff];

        let out = feed_forward(&x, &w_gate, &w_up, &w_down, n_embd);
        assert_eq!(out.len(), n_embd);
    }

    #[test]
    fn test_feed_forward_gpu_fallback() {
        // Without GPU, should produce same result as CPU path
        let n_embd = 4;
        let n_ff = 8;
        let x = vec![1.0f32; n_embd];
        let w_gate = vec![0.1f32; n_ff * n_embd];
        let w_up = vec![0.1f32; n_ff * n_embd];
        let w_down = vec![0.1f32; n_embd * n_ff];

        let cpu_out = feed_forward(&x, &w_gate, &w_up, &w_down, n_embd);
        let gpu_out = feed_forward_gpu(&x, &w_gate, &w_up, &w_down, n_embd, None);
        assert_eq!(cpu_out, gpu_out);
    }

    #[test]
    fn test_feed_forward_gpu_with_metal() {
        let gpu = MetalGpu::new();
        let n_embd = 64;
        let n_ff = 2048; // meets MIN_GPU_FF_ELEMENTS threshold
        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32) * 0.01).collect();
        let w_gate: Vec<f32> = (0..n_ff * n_embd)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.001)
            .collect();
        let w_up: Vec<f32> = (0..n_ff * n_embd)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.001)
            .collect();
        let w_down: Vec<f32> = (0..n_embd * n_ff)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.001)
            .collect();

        let cpu_out = feed_forward(&x, &w_gate, &w_up, &w_down, n_embd);
        let gpu_out = feed_forward_gpu(&x, &w_gate, &w_up, &w_down, n_embd, gpu.as_ref());
        assert_eq!(cpu_out.len(), gpu_out.len());

        // Allow small floating-point divergence between CPU SIMD and GPU
        for (c, g) in cpu_out.iter().zip(gpu_out.iter()) {
            let diff = (c - g).abs();
            assert!(diff < 0.01, "CPU={c} GPU={g} diff={diff} exceeds tolerance");
        }
    }

    #[test]
    fn test_feed_forward_gpu_small_tensor_uses_cpu() {
        // Small tensors below threshold should use CPU path (exact match)
        let gpu = MetalGpu::new();
        let n_embd = 4;
        let n_ff = 8; // well below MIN_GPU_FF_ELEMENTS
        let x = vec![0.5f32; n_embd];
        let w_gate = vec![0.2f32; n_ff * n_embd];
        let w_up = vec![0.3f32; n_ff * n_embd];
        let w_down = vec![0.1f32; n_embd * n_ff];

        let cpu_out = feed_forward(&x, &w_gate, &w_up, &w_down, n_embd);
        let gpu_out = feed_forward_gpu(&x, &w_gate, &w_up, &w_down, n_embd, gpu.as_ref());
        // Should be identical since GPU path isn't used for small tensors
        assert_eq!(cpu_out, gpu_out);
    }

    #[test]
    fn test_feed_forward_zero_intermediate() {
        let n_embd = 4;
        let x = vec![1.0f32; n_embd];
        // Empty weights => n_ff = 0
        let out = feed_forward_gpu(&x, &[], &[], &[], n_embd, None);
        assert_eq!(out, vec![0.0f32; n_embd]);
    }

    #[test]
    fn test_feed_forward_known_values() {
        // Verify numerical correctness with known inputs
        let n_embd = 2;
        let n_ff = 2;
        // Identity-like gate and up weights
        let w_gate = vec![1.0, 0.0, 0.0, 1.0]; // [n_ff, n_embd]
        let w_up = vec![1.0, 0.0, 0.0, 1.0];
        let w_down = vec![1.0, 0.0, 0.0, 1.0]; // [n_embd, n_ff]
        let x = vec![2.0, 3.0];

        let out = feed_forward(&x, &w_gate, &w_up, &w_down, n_embd);
        // gate_proj: [2.0, 3.0]
        // silu(2.0) = 2.0 / (1 + exp(-2)) ≈ 1.7616
        // silu(3.0) = 3.0 / (1 + exp(-3)) ≈ 2.8577
        // up_proj: [2.0, 3.0]
        // intermediate: [1.7616 * 2.0, 2.8577 * 3.0] ≈ [3.5232, 8.5731]
        // down_proj (identity): [3.5232, 8.5731]
        assert!((out[0] - 3.523).abs() < 0.01);
        assert!((out[1] - 8.573).abs() < 0.01);
    }
}
