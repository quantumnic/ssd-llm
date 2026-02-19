//! Metal Compute Pipeline for Apple Silicon GPU acceleration
//!
//! Provides GPU-accelerated matrix operations using Metal compute shaders.
//! Falls back to optimized CPU SIMD paths when Metal is unavailable or for small tensors.
//!
//! v0.3: Added Metal shader compilation, GPU dispatch for matvec/rmsnorm/softmax,
//!       and quantized Q4_0/Q8_0 on-GPU dequant+matmul kernels.

use crate::metal::gpu::MetalGpu;
use crate::model::gguf::GgmlType;
use tracing::{debug, info, warn};

/// Threshold below which CPU is faster than GPU dispatch overhead
const GPU_DISPATCH_THRESHOLD: usize = 4096;

/// Metal compute context for GPU-accelerated operations
pub struct MetalCompute {
    available: bool,
    /// Pre-compiled shader library path
    shader_lib: Option<String>,
    /// Real Metal GPU context (v0.4)
    gpu: Option<MetalGpu>,
}

unsafe impl Send for MetalCompute {}
unsafe impl Sync for MetalCompute {}

impl MetalCompute {
    /// Check if Metal is available on this system
    pub fn is_available() -> bool {
        #[cfg(target_os = "macos")]
        {
            std::process::Command::new("system_profiler")
                .arg("SPDisplaysDataType")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).contains("Metal"))
                .unwrap_or(false)
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// Create a new Metal compute context, compiling shaders if possible
    pub fn new() -> Option<Self> {
        if !Self::is_available() {
            warn!("Metal not available, falling back to CPU");
            return None;
        }

        // Try to compile Metal shaders
        let shader_lib = Self::compile_shaders();

        // Try to initialize real GPU dispatch via metal-rs
        let gpu = MetalGpu::new();
        if gpu.is_some() {
            info!("Metal GPU acceleration ready (metal-rs pipelines compiled)");
        } else if shader_lib.is_some() {
            info!("Metal GPU acceleration ready (shaders compiled, CPU dispatch)");
        } else {
            info!("Metal GPU available (using CPU SIMD)");
        }

        Some(Self {
            available: true,
            shader_lib,
            gpu,
        })
    }

    /// Compile Metal shaders to a .metallib
    fn compile_shaders() -> Option<String> {
        let shader_src = include_str!("shaders/kernels.metal");
        let tmp_dir = std::env::temp_dir().join("ssd-llm-metal");
        let _ = std::fs::create_dir_all(&tmp_dir);

        let metal_path = tmp_dir.join("kernels.metal");
        let air_path = tmp_dir.join("kernels.air");
        let lib_path = tmp_dir.join("kernels.metallib");

        // Write shader source
        if std::fs::write(&metal_path, shader_src).is_err() {
            return None;
        }

        // Compile to .air (Metal IR)
        let status = std::process::Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                metal_path.to_str()?,
                "-o",
                air_path.to_str()?,
            ])
            .output();

        match status {
            Ok(output) if output.status.success() => {
                debug!("Metal shaders compiled to AIR");
            }
            Ok(output) => {
                warn!(
                    "Metal shader compilation failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                return None;
            }
            Err(e) => {
                warn!("xcrun not found: {}", e);
                return None;
            }
        }

        // Link to .metallib
        let status = std::process::Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metallib",
                air_path.to_str()?,
                "-o",
                lib_path.to_str()?,
            ])
            .output();

        match status {
            Ok(output) if output.status.success() => {
                info!("Metal shader library built: {}", lib_path.display());
                Some(lib_path.to_string_lossy().to_string())
            }
            _ => None,
        }
    }

    /// Check if GPU dispatch is beneficial for this operation size
    pub fn should_use_gpu(&self, elements: usize) -> bool {
        self.available && self.shader_lib.is_some() && elements >= GPU_DISPATCH_THRESHOLD
    }

    /// GPU-accelerated matrix-vector multiply (falls back to SIMD CPU)
    pub fn matvec_f32(&self, w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        if let Some(ref gpu) = self.gpu {
            if gpu.should_dispatch(out_dim * in_dim) {
                return gpu.matvec_f32(w, x, out_dim, in_dim);
            }
        }
        matvec_f32_simd(w, x, out_dim, in_dim)
    }

    /// Quantized matrix-vector multiply with automatic dispatch by type
    /// Supports Q4_0, Q4_K, Q6_K, Q8_0 on GPU; falls back to CPU for others
    pub fn matvec_quantized(
        &self,
        w_raw: &[u8],
        x: &[f32],
        out_dim: usize,
        in_dim: usize,
        dtype: GgmlType,
    ) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        if let Some(ref gpu) = self.gpu {
            if gpu.should_dispatch(out_dim * in_dim) {
                match dtype {
                    GgmlType::Q4_0 => return gpu.matvec_q4_0(w_raw, x, out_dim, in_dim),
                    GgmlType::Q4K => return gpu.matvec_q4_k(w_raw, x, out_dim, in_dim),
                    GgmlType::Q6K => return gpu.matvec_q6_k(w_raw, x, out_dim, in_dim),
                    GgmlType::Q8_0 => return gpu.matvec_q8_0(w_raw, x, out_dim, in_dim),
                    _ => {}
                }
            }
        }
        // CPU fallback
        matvec_quantized_cpu(w_raw, x, out_dim, in_dim, dtype)
    }

    pub fn rmsnorm_f32(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        #[cfg(target_os = "macos")]
        if let Some(ref gpu) = self.gpu {
            if gpu.should_dispatch(x.len()) {
                gpu.rmsnorm_f32(x, weight, eps);
                return;
            }
        }
        rmsnorm_f32_fast(x, weight, eps);
    }

    pub fn softmax_f32(&self, x: &mut [f32]) {
        #[cfg(target_os = "macos")]
        if let Some(ref gpu) = self.gpu {
            if gpu.should_dispatch(x.len()) {
                gpu.softmax_f32(x);
                return;
            }
        }
        softmax_f32_fast(x);
    }

    /// Whether GPU dispatch is active
    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Get path to compiled shader library
    pub fn shader_library_path(&self) -> Option<&str> {
        self.shader_lib.as_deref()
    }
}

/// SIMD-friendly matrix-vector multiply using 4-wide accumulation
/// On Apple Silicon, the compiler auto-vectorizes this to NEON instructions
pub fn matvec_f32_simd(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    let chunks = in_dim / 4;
    let remainder = in_dim % 4;

    for (i, y_val) in y.iter_mut().enumerate().take(out_dim) {
        let row_start = i * in_dim;
        if row_start + in_dim > w.len() {
            continue;
        }
        let row = &w[row_start..row_start + in_dim];

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        for c in 0..chunks {
            let j = c * 4;
            sum0 += row[j] * x[j];
            sum1 += row[j + 1] * x[j + 1];
            sum2 += row[j + 2] * x[j + 2];
            sum3 += row[j + 3] * x[j + 3];
        }

        let base = chunks * 4;
        for j in 0..remainder {
            sum0 += row[base + j] * x[base + j];
        }

        *y_val = sum0 + sum1 + sum2 + sum3;
    }

    y
}

/// Fast RMS normalization with fused multiply
pub fn rmsnorm_f32_fast(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    for (i, val) in x.iter_mut().enumerate() {
        *val *= inv_rms * weight.get(i).copied().unwrap_or(1.0);
    }
}

/// Numerically stable softmax
pub fn softmax_f32_fast(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Apply RoPE (Rotary Position Embedding) in-place
pub fn rope_f32_fast(x: &mut [f32], head_dim: usize, position: usize, theta_base: f32) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let freq = 1.0 / theta_base.powf(2.0 * i as f32 / head_dim as f32);
        let angle = position as f32 * freq;
        let (sin_val, cos_val) = angle.sin_cos();
        let idx0 = i * 2;
        let idx1 = idx0 + 1;
        if idx1 < x.len() {
            let x0 = x[idx0];
            let x1 = x[idx1];
            x[idx0] = x0 * cos_val - x1 * sin_val;
            x[idx1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

/// SiLU activation: x * sigmoid(x)
pub fn silu_f32(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// CPU quantized matvec dispatch by type
pub fn matvec_quantized_cpu(
    w_raw: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    dtype: GgmlType,
) -> Vec<f32> {
    match dtype {
        GgmlType::Q4_0 => matvec_q4_0_cpu(w_raw, x, out_dim, in_dim),
        GgmlType::Q4K => matvec_q4_k_cpu(w_raw, x, out_dim, in_dim),
        GgmlType::Q6K => matvec_q6_k_cpu(w_raw, x, out_dim, in_dim),
        GgmlType::Q8_0 => matvec_q8_0_cpu(w_raw, x, out_dim, in_dim),
        _ => {
            debug!("Unsupported quant type {:?}, returning zeros", dtype);
            vec![0.0; out_dim]
        }
    }
}

/// CPU Q4_0 dequant matvec
fn matvec_q4_0_cpu(w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let block_size = 32usize;
    let block_bytes = 18usize;
    let blocks_per_row = in_dim / block_size;
    let mut y = vec![0.0f32; out_dim];

    for (row, y_val) in y.iter_mut().enumerate() {
        let row_off = row * blocks_per_row * block_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let boff = row_off + b * block_bytes;
            let scale = f16_to_f32(w[boff], w[boff + 1]);
            let x_base = b * block_size;
            for j in 0..16 {
                let byte = w[boff + 2 + j];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                sum += scale * lo as f32 * x[x_base + j * 2];
                sum += scale * hi as f32 * x[x_base + j * 2 + 1];
            }
        }
        *y_val = sum;
    }
    y
}

/// CPU Q4_K dequant matvec
fn matvec_q4_k_cpu(w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let block_size = 256usize;
    let block_bytes = 144usize;
    let blocks_per_row = in_dim / block_size;
    let mut y = vec![0.0f32; out_dim];

    for (row, y_val) in y.iter_mut().enumerate() {
        let row_off = row * blocks_per_row * block_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let boff = row_off + b * block_bytes;
            let d = f16_to_f32(w[boff], w[boff + 1]);
            let dmin = f16_to_f32(w[boff + 2], w[boff + 3]);
            let sc = &w[boff + 4..boff + 16];
            let qs = &w[boff + 16..boff + 144];
            let x_base = b * block_size;

            for sb in 0..8u8 {
                let (sc_low, m_low) = if sb < 4 {
                    (sc[sb as usize] & 0x3F, sc[sb as usize + 4] & 0x3F)
                } else {
                    let si = (sb - 4) as usize;
                    (
                        (sc[si] >> 6) | ((sc[sb as usize + 4] & 0x0F) << 2),
                        (sc[sb as usize] >> 6) | ((sc[sb as usize + 4] >> 4) << 2),
                    )
                };
                let scale = d * sc_low as f32;
                let min_val = dmin * m_low as f32;

                let qs_off = sb as usize * 16;
                for j in 0..16 {
                    let byte_val = qs[qs_off + j];
                    let v0 = scale * (byte_val & 0x0F) as f32 - min_val;
                    let v1 = scale * ((byte_val >> 4) & 0x0F) as f32 - min_val;
                    let xi = x_base + sb as usize * 32 + j * 2;
                    sum += v0 * x[xi] + v1 * x[xi + 1];
                }
            }
        }
        *y_val = sum;
    }
    y
}

/// CPU Q6_K dequant matvec
fn matvec_q6_k_cpu(w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let block_size = 256usize;
    let block_bytes = 210usize;
    let blocks_per_row = in_dim / block_size;
    let mut y = vec![0.0f32; out_dim];

    for (row, y_val) in y.iter_mut().enumerate() {
        let row_off = row * blocks_per_row * block_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let boff = row_off + b * block_bytes;
            let ql = &w[boff..boff + 128];
            let qh = &w[boff + 128..boff + 192];
            let scales = &w[boff + 192..boff + 208];
            let d = f16_to_f32(w[boff + 208], w[boff + 209]);
            let x_base = b * block_size;

            for (sb, &scale_byte) in scales.iter().enumerate().take(16) {
                let sc = d * (scale_byte as i8) as f32;
                for j in 0..16usize {
                    let idx = sb * 16 + j;
                    let ql_byte = ql[idx / 2];
                    let low4 = if idx & 1 != 0 {
                        (ql_byte >> 4) & 0x0F
                    } else {
                        ql_byte & 0x0F
                    };
                    let qh_byte = qh[idx / 4];
                    let shift = (idx % 4) * 2;
                    let high2 = (qh_byte >> shift) & 0x03;
                    let q = ((high2 << 4) | low4) as i32 - 32;
                    sum += sc * q as f32 * x[x_base + idx];
                }
            }
        }
        *y_val = sum;
    }
    y
}

/// CPU Q8_0 dequant matvec
fn matvec_q8_0_cpu(w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let block_size = 32usize;
    let block_bytes = 34usize;
    let blocks_per_row = in_dim / block_size;
    let mut y = vec![0.0f32; out_dim];

    for (row, y_val) in y.iter_mut().enumerate() {
        let row_off = row * blocks_per_row * block_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let boff = row_off + b * block_bytes;
            let scale = f16_to_f32(w[boff], w[boff + 1]);
            let x_base = b * block_size;
            for j in 0..32 {
                let val = w[boff + 2 + j] as i8;
                sum += scale * val as f32 * x[x_base + j];
            }
        }
        *y_val = sum;
    }
    y
}

/// Convert two bytes (little-endian) to f32 via f16
#[inline]
fn f16_to_f32(lo: u8, hi: u8) -> f32 {
    half::f16::from_bits(lo as u16 | ((hi as u16) << 8)).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matvec_identity() {
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![3.0, 4.0];
        let y = matvec_f32_simd(&w, &x, 2, 2);
        assert!((y[0] - 3.0).abs() < 1e-6);
        assert!((y[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_3x2() {
        // [[1,2],[3,4],[5,6]] × [1,1] = [3, 7, 11]
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0];
        let y = matvec_f32_simd(&w, &x, 3, 2);
        assert!((y[0] - 3.0).abs() < 1e-6);
        assert!((y[1] - 7.0).abs() < 1e-6);
        assert!((y[2] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_f32_fast(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn test_softmax_single() {
        let mut x = vec![42.0];
        softmax_f32_fast(&mut x);
        assert!((x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rmsnorm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        rmsnorm_f32_fast(&mut x, &w, 1e-5);
        let rms: f32 = (1.0 + 4.0 + 9.0 + 16.0) / 4.0 + 1e-5;
        let inv_rms = 1.0 / rms.sqrt();
        assert!((x[0] - inv_rms).abs() < 1e-4);
    }

    #[test]
    fn test_silu() {
        let mut x = vec![0.0, 1.0, -1.0];
        silu_f32(&mut x);
        assert!((x[0] - 0.0).abs() < 1e-6); // silu(0) = 0
        assert!(x[1] > 0.5); // silu(1) ≈ 0.731
        assert!(x[2] < 0.0 && x[2] > -0.5); // silu(-1) ≈ -0.269
    }

    #[test]
    fn test_matvec_q4_0_cpu_basic() {
        // 1 row, 32 elements (1 block)
        // Block: f16 scale = 1.0, then 16 bytes of nibbles
        let scale_bits = half::f16::from_f32(1.0).to_bits();
        let mut block = vec![scale_bits as u8, (scale_bits >> 8) as u8];
        // All nibbles = 8 (so value = 8-8 = 0), except first byte = 0x19 (lo=9-8=1, hi=1-8=-7... wait)
        // Simpler: all zero nibbles => val = 0-8 = -8 each
        // With x = [1.0; 32], sum = scale * sum_of_dequantized * x
        // Let's use all 0x88 nibbles: lo = 8, hi = 8, dequant = 0
        for _ in 0..16 {
            block.push(0x88);
        }
        let x = vec![1.0f32; 32];
        let y = matvec_q4_0_cpu(&block, &x, 1, 32);
        assert!(
            (y[0] - 0.0).abs() < 1e-5,
            "All zeros should give 0, got {}",
            y[0]
        );
    }

    #[test]
    fn test_matvec_q8_0_cpu_basic() {
        // 1 row, 32 elements (1 block)
        let scale_bits = half::f16::from_f32(1.0).to_bits();
        let mut block = vec![scale_bits as u8, (scale_bits >> 8) as u8];
        // 32 int8 values, all = 1
        for _ in 0..32 {
            block.push(1u8);
        }
        let x = vec![1.0f32; 32];
        let y = matvec_q8_0_cpu(&block, &x, 1, 32);
        // Each val = 1.0 * 1 = 1.0, dot with x[j]=1.0 => sum = 32.0
        assert!((y[0] - 32.0).abs() < 1e-3, "Expected 32.0, got {}", y[0]);
    }

    #[test]
    fn test_f16_to_f32_roundtrip() {
        let bits = half::f16::from_f32(3.14).to_bits();
        let result = f16_to_f32(bits as u8, (bits >> 8) as u8);
        assert!((result - 3.14).abs() < 0.01);
    }

    #[test]
    fn test_metal_shader_compilation() {
        // Only on macOS
        if MetalCompute::is_available() {
            let metal = MetalCompute::new().unwrap();
            assert!(
                metal.shader_library_path().is_some(),
                "Metal shaders should compile on macOS"
            );
        }
    }
}
