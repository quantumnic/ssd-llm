//! ARM NEON SIMD intrinsics for Apple Silicon CPU acceleration
//!
//! v1.27: Replaces scalar loop-unrolled "simd" with actual NEON vector instructions.
//! Provides 2-4x speedup for CPU-bound matvec, rmsnorm, softmax, and dot product
//! operations on aarch64 (Apple M1/M2/M3/M4).

// NEON intrinsics require index-based loops for unsafe pointer arithmetic.
#![cfg_attr(not(doc), allow(clippy::needless_range_loop))]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ─── f32 matrix-vector multiply ────────────────────────────────────────────

/// NEON-accelerated f32 matrix-vector multiply.
/// Processes 16 elements per iteration using 4 accumulators.
#[cfg(target_arch = "aarch64")]
pub fn matvec_f32_neon(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];

    for (i, y_val) in y.iter_mut().enumerate() {
        let row_start = i * in_dim;
        if row_start + in_dim > w.len() {
            continue;
        }
        let row = &w[row_start..row_start + in_dim];

        unsafe {
            *y_val = dot_f32_neon_inner(row, x);
        }
    }

    y
}

/// Scalar fallback for non-aarch64 platforms.
#[cfg(not(target_arch = "aarch64"))]
pub fn matvec_f32_neon(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    super::compute::matvec_f32_simd_scalar(w, x, out_dim, in_dim)
}

// ─── f32 dot product ──────────────────────────────────────────────────────

/// NEON-accelerated dot product of two f32 slices.
#[cfg(target_arch = "aarch64")]
pub fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe { dot_f32_neon_inner(a, b) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_f32_neon_inner(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks16 = n / 16;
    let remaining = n % 16;

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for c in 0..chunks16 {
        let base = c * 16;
        let a0 = vld1q_f32(a_ptr.add(base));
        let a1 = vld1q_f32(a_ptr.add(base + 4));
        let a2 = vld1q_f32(a_ptr.add(base + 8));
        let a3 = vld1q_f32(a_ptr.add(base + 12));

        let b0 = vld1q_f32(b_ptr.add(base));
        let b1 = vld1q_f32(b_ptr.add(base + 4));
        let b2 = vld1q_f32(b_ptr.add(base + 8));
        let b3 = vld1q_f32(b_ptr.add(base + 12));

        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        acc2 = vfmaq_f32(acc2, a2, b2);
        acc3 = vfmaq_f32(acc3, a3, b3);
    }

    // Combine accumulators
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    let mut sum = vaddvq_f32(acc0);

    // Handle remaining elements
    let base = chunks16 * 16;
    // Process remaining 4-element chunks
    let rem_chunks4 = remaining / 4;
    for c in 0..rem_chunks4 {
        let off = base + c * 4;
        let va = vld1q_f32(a_ptr.add(off));
        let vb = vld1q_f32(b_ptr.add(off));
        sum += vaddvq_f32(vmulq_f32(va, vb));
    }

    // Scalar tail
    let scalar_start = base + rem_chunks4 * 4;
    for j in scalar_start..n {
        sum += a[j] * b[j];
    }

    sum
}

#[cfg(not(target_arch = "aarch64"))]
pub fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

// ─── RMS normalization ────────────────────────────────────────────────────

/// NEON-accelerated RMS normalization with fused weight multiply.
#[cfg(target_arch = "aarch64")]
pub fn rmsnorm_f32_neon(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }

    // Compute sum of squares using NEON
    let sum_sq = unsafe {
        let chunks = n / 16;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let ptr = x.as_ptr();

        for c in 0..chunks {
            let base = c * 16;
            let v0 = vld1q_f32(ptr.add(base));
            let v1 = vld1q_f32(ptr.add(base + 4));
            let v2 = vld1q_f32(ptr.add(base + 8));
            let v3 = vld1q_f32(ptr.add(base + 12));
            acc0 = vfmaq_f32(acc0, v0, v0);
            acc1 = vfmaq_f32(acc1, v1, v1);
            acc2 = vfmaq_f32(acc2, v2, v2);
            acc3 = vfmaq_f32(acc3, v3, v3);
        }

        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);
        let mut ss = vaddvq_f32(acc0);

        let base = chunks * 16;
        for j in base..n {
            ss += x[j] * x[j];
        }
        ss
    };

    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    // Fused multiply: x[i] = x[i] * inv_rms * weight[i]
    unsafe {
        let chunks = n / 4;
        let inv_rms_v = vdupq_n_f32(inv_rms);
        let x_ptr = x.as_mut_ptr();
        let w_ptr = weight.as_ptr();

        for c in 0..chunks {
            let off = c * 4;
            if off + 4 <= n && off + 4 <= weight.len() {
                let xv = vld1q_f32(x_ptr.add(off));
                let wv = vld1q_f32(w_ptr.add(off));
                let result = vmulq_f32(vmulq_f32(xv, inv_rms_v), wv);
                vst1q_f32(x_ptr.add(off), result);
            }
        }

        let base = chunks * 4;
        for j in base..n {
            x[j] *= inv_rms * weight.get(j).copied().unwrap_or(1.0);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn rmsnorm_f32_neon(x: &mut [f32], weight: &[f32], eps: f32) {
    super::compute::rmsnorm_f32_fast_scalar(x, weight, eps);
}

// ─── Softmax ──────────────────────────────────────────────────────────────

/// NEON-accelerated softmax.
#[cfg(target_arch = "aarch64")]
pub fn softmax_f32_neon(x: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }

    unsafe {
        // Find max using NEON
        let chunks = n / 4;
        let ptr = x.as_mut_ptr();

        let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
        for c in 0..chunks {
            let v = vld1q_f32(ptr.add(c * 4));
            max_v = vmaxq_f32(max_v, v);
        }
        let mut max_val = vmaxvq_f32(max_v);
        for j in (chunks * 4)..n {
            max_val = max_val.max(x[j]);
        }

        // exp(x - max) and sum
        let mut sum = 0.0f32;
        for j in 0..n {
            x[j] = (x[j] - max_val).exp();
            sum += x[j];
        }

        // Normalize
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            let inv_v = vdupq_n_f32(inv_sum);
            for c in 0..chunks {
                let off = c * 4;
                let v = vld1q_f32(ptr.add(off));
                vst1q_f32(ptr.add(off), vmulq_f32(v, inv_v));
            }
            for j in (chunks * 4)..n {
                x[j] *= inv_sum;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn softmax_f32_neon(x: &mut [f32]) {
    super::compute::softmax_f32_fast_scalar(x);
}

// ─── SGEMV with quantized f16 input ──────────────────────────────────────

/// NEON-accelerated f16→f32 matrix-vector multiply.
/// Converts f16 weights to f32 via the `half` crate, then uses NEON FMA.
#[cfg(target_arch = "aarch64")]
pub fn matvec_f16_neon(w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    let bytes_per_row = in_dim * 2; // f16 = 2 bytes

    // Pre-convert entire row to f32, then use NEON dot product
    let mut row_f32 = vec![0.0f32; in_dim];

    for i in 0..out_dim {
        let row_start = i * bytes_per_row;
        if row_start + bytes_per_row > w.len() {
            continue;
        }

        // Convert f16 → f32
        for j in 0..in_dim {
            let off = row_start + j * 2;
            let bits = u16::from_le_bytes([w[off], w[off + 1]]);
            row_f32[j] = half::f16::from_bits(bits).to_f32();
        }

        unsafe {
            y[i] = dot_f32_neon_inner(&row_f32, x);
        }
    }

    y
}

#[cfg(not(target_arch = "aarch64"))]
pub fn matvec_f16_neon(w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    super::compute::matvec_f16_cpu(w, x, out_dim, in_dim)
}

// ─── Element-wise vector operations ──────────────────────────────────────

/// NEON-accelerated element-wise multiply (Hadamard product).
#[cfg(target_arch = "aarch64")]
pub fn vec_mul_f32_neon(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len().min(b.len());
    let mut out = vec![0.0f32; n];

    unsafe {
        let chunks = n / 4;
        for c in 0..chunks {
            let off = c * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vmulq_f32(va, vb));
        }
        for j in (chunks * 4)..n {
            out[j] = a[j] * b[j];
        }
    }

    out
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_mul_f32_neon(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).collect()
}

/// NEON-accelerated vector addition.
#[cfg(target_arch = "aarch64")]
pub fn vec_add_f32_neon(a: &mut [f32], b: &[f32]) {
    let n = a.len().min(b.len());

    unsafe {
        let chunks = n / 4;
        for c in 0..chunks {
            let off = c * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            vst1q_f32(a.as_mut_ptr().add(off), vaddq_f32(va, vb));
        }
        for j in (chunks * 4)..n {
            a[j] += b[j];
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_add_f32_neon(a: &mut [f32], b: &[f32]) {
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a += *b;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f32_neon_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = dot_f32_neon(&a, &b);
        assert!((result - 70.0).abs() < 1e-4, "dot={result}, expected 70.0");
    }

    #[test]
    fn test_dot_f32_neon_large() {
        let n = 4096;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.0001).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
        let result = dot_f32_neon(&a, &b);
        let rel_err = (result - expected).abs() / expected.abs().max(1e-8);
        assert!(
            rel_err < 1e-4,
            "dot={result}, expected={expected}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_dot_f32_neon_odd_length() {
        // Length not divisible by 16
        let a = vec![1.0; 37];
        let b = vec![2.0; 37];
        let result = dot_f32_neon(&a, &b);
        assert!((result - 74.0).abs() < 1e-4, "dot={result}, expected 74.0");
    }

    #[test]
    fn test_matvec_f32_neon_identity() {
        // 3x3 identity
        let w = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
        ];
        let x = vec![5.0, 7.0, 11.0];
        let y = matvec_f32_neon(&w, &x, 3, 3);
        assert!((y[0] - 5.0).abs() < 1e-5);
        assert!((y[1] - 7.0).abs() < 1e-5);
        assert!((y[2] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_matvec_f32_neon_2x2() {
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let y = matvec_f32_neon(&w, &x, 2, 2);
        assert!((y[0] - 3.0).abs() < 1e-5);
        assert!((y[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_matvec_f32_neon_large() {
        let out_dim = 128;
        let in_dim = 512;
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 0.001).collect();
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();

        let y_neon = matvec_f32_neon(&w, &x, out_dim, in_dim);
        let y_scalar = super::super::compute::matvec_f32_simd_scalar(&w, &x, out_dim, in_dim);

        for i in 0..out_dim {
            let rel_err = (y_neon[i] - y_scalar[i]).abs() / y_scalar[i].abs().max(1e-8);
            assert!(
                rel_err < 1e-3,
                "row {i}: neon={}, scalar={}, rel_err={rel_err}",
                y_neon[i],
                y_scalar[i]
            );
        }
    }

    #[test]
    fn test_rmsnorm_f32_neon() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut x_ref = x.clone();

        rmsnorm_f32_neon(&mut x, &weight, 1e-5);
        super::super::compute::rmsnorm_f32_fast_scalar(&mut x_ref, &weight, 1e-5);

        for i in 0..4 {
            assert!(
                (x[i] - x_ref[i]).abs() < 1e-5,
                "idx {i}: neon={}, ref={}",
                x[i],
                x_ref[i]
            );
        }
    }

    #[test]
    fn test_softmax_f32_neon() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x_ref = x.clone();

        softmax_f32_neon(&mut x);
        super::super::compute::softmax_f32_fast_scalar(&mut x_ref);

        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {sum}"
        );

        for i in 0..8 {
            assert!(
                (x[i] - x_ref[i]).abs() < 1e-5,
                "idx {i}: neon={}, ref={}",
                x[i],
                x_ref[i]
            );
        }
    }

    #[test]
    fn test_vec_mul_f32_neon() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let c = vec_mul_f32_neon(&a, &b);
        assert!((c[0] - 2.0).abs() < 1e-5);
        assert!((c[1] - 6.0).abs() < 1e-5);
        assert!((c[4] - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec_add_f32_neon() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        vec_add_f32_neon(&mut a, &b);
        assert!((a[0] - 11.0).abs() < 1e-5);
        assert!((a[4] - 55.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_f32_neon_zeros() {
        let a = vec![0.0; 256];
        let b = vec![1.0; 256];
        assert!((dot_f32_neon(&a, &b)).abs() < 1e-8);
    }

    #[test]
    fn test_rmsnorm_f32_neon_large() {
        let n = 4096;
        let mut x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
        let mut x_ref = x.clone();

        rmsnorm_f32_neon(&mut x, &weight, 1e-5);
        super::super::compute::rmsnorm_f32_fast_scalar(&mut x_ref, &weight, 1e-5);

        for i in 0..n {
            let abs_err = (x[i] - x_ref[i]).abs();
            assert!(
                abs_err < 1e-3,
                "idx {i}: neon={}, ref={}, err={abs_err}",
                x[i],
                x_ref[i]
            );
        }
    }

    #[test]
    fn test_matvec_f32_neon_matches_scalar_exact() {
        // Small enough for exact comparison
        let w = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ];
        let x = vec![1.0, 0.5, 0.25, 0.125];
        let y = matvec_f32_neon(&w, &x, 3, 4);
        // row0: 1 + 1 + 0.75 + 0.5 = 3.25
        // row1: 5 + 3 + 1.75 + 1.0 = 10.75
        // row2: 9 + 5 + 2.75 + 1.5 = 18.25
        assert!((y[0] - 3.25).abs() < 1e-5);
        assert!((y[1] - 10.75).abs() < 1e-5);
        assert!((y[2] - 18.25).abs() < 1e-5);
    }
}
