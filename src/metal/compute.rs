//! Metal Compute Pipeline for Apple Silicon GPU acceleration
//!
//! Provides GPU-accelerated matrix operations using Metal compute shaders.
//! Falls back to optimized CPU SIMD paths when Metal is unavailable or for small tensors.
//!
//! v1.14: Added Q3_K and Q5_K dequantization (CPU + Metal GPU shaders).
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
    /// Supports Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, Q4_0, Q8_0 on GPU; falls back to CPU for others
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
                    GgmlType::Q3K => return gpu.matvec_q3_k(w_raw, x, out_dim, in_dim),
                    GgmlType::Q4K => return gpu.matvec_q4_k(w_raw, x, out_dim, in_dim),
                    GgmlType::Q5K => return gpu.matvec_q5_k(w_raw, x, out_dim, in_dim),
                    GgmlType::Q6K => return gpu.matvec_q6_k(w_raw, x, out_dim, in_dim),
                    GgmlType::Q8_0 => return gpu.matvec_q8_0(w_raw, x, out_dim, in_dim),
                    GgmlType::Q2K => return gpu.matvec_q2_k(w_raw, x, out_dim, in_dim),
                    GgmlType::Q8K => return gpu.matvec_q8_k(w_raw, x, out_dim, in_dim),
                    GgmlType::IQ4NL => return gpu.matvec_iq4_nl(w_raw, x, out_dim, in_dim),
                    GgmlType::IQ4XS => return gpu.matvec_iq4_xs(w_raw, x, out_dim, in_dim),
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
        GgmlType::Q3K => matvec_q3_k_cpu(w_raw, x, out_dim, in_dim),
        GgmlType::Q4K => matvec_q4_k_cpu(w_raw, x, out_dim, in_dim),
        GgmlType::Q5K => matvec_q5_k_cpu(w_raw, x, out_dim, in_dim),
        GgmlType::Q6K => matvec_q6_k_cpu(w_raw, x, out_dim, in_dim),
        GgmlType::Q8_0 => matvec_q8_0_cpu(w_raw, x, out_dim, in_dim),
        GgmlType::Q2K => matvec_q2_k_cpu(w_raw, x, out_dim, in_dim),
        GgmlType::Q8K => matvec_q8_k_cpu(w_raw, x, out_dim, in_dim),
        GgmlType::IQ4NL => matvec_iq4_nl_cpu(w_raw, x, out_dim, in_dim),
        GgmlType::IQ4XS => matvec_iq4_xs_cpu(w_raw, x, out_dim, in_dim),
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

/// CPU Q3_K dequant matvec
/// Block layout (256 elements, 110 bytes):
///   32B hmask (high bit) + 64B qs (low 2 bits, 4 per byte) + 12B scales + 2B f16 d
fn matvec_q3_k_cpu(w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let block_size = 256usize;
    let block_bytes = 110usize;
    let blocks_per_row = in_dim / block_size;
    let mut y = vec![0.0f32; out_dim];

    for (row, y_val) in y.iter_mut().enumerate() {
        let row_off = row * blocks_per_row * block_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let boff = row_off + b * block_bytes;
            let hmask = &w[boff..boff + 32];
            let qs = &w[boff + 32..boff + 96];
            let sc = &w[boff + 96..boff + 108];
            let d = f16_to_f32(w[boff + 108], w[boff + 109]);
            let x_base = b * block_size;

            for sb in 0..16usize {
                // Extract 6-bit signed scale
                let low4 = (sc[sb / 2] >> ((sb & 1) * 4)) & 0x0F;
                let hi2 = (sc[8 + sb / 4] >> ((sb % 4) * 2)) & 0x03;
                let mut scale_val = (low4 | (hi2 << 4)) as i32;
                if scale_val >= 32 {
                    scale_val -= 64;
                }
                let sc_f = d * scale_val as f32;

                for j in 0..16usize {
                    let idx = sb * 16 + j;
                    // Low 2 bits from qs (4 per byte)
                    let qs_byte = qs[idx / 4];
                    let shift = (idx % 4) * 2;
                    let low2 = (qs_byte >> shift) & 0x03;
                    // High bit from hmask
                    let hbit = (hmask[idx / 8] >> (idx % 8)) & 1;
                    // 3-bit value centered at 4
                    let q = (low2 | (hbit << 2)) as i32 - 4;
                    sum += sc_f * q as f32 * x[x_base + idx];
                }
            }
        }
        *y_val = sum;
    }
    y
}

/// CPU Q5_K dequant matvec
/// Block layout (256 elements, 176 bytes):
///   f16 d (2B) + f16 dmin (2B) + 12B scales + 32B qh (high bit) + 128B ql (nibbles)
fn matvec_q5_k_cpu(w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let block_size = 256usize;
    let block_bytes = 176usize;
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
            let qh = &w[boff + 16..boff + 48];
            let ql = &w[boff + 48..boff + 176];
            let x_base = b * block_size;

            for sb in 0..8u8 {
                // Extract 6-bit scale and min (same packing as Q4_K)
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
                for j in 0..16usize {
                    let idx0 = sb as usize * 32 + j * 2;
                    let idx1 = idx0 + 1;

                    let byte_val = ql[qs_off + j];
                    let low0 = byte_val & 0x0F;
                    let low1 = (byte_val >> 4) & 0x0F;

                    // High bit from qh
                    let hbit0 = (qh[idx0 / 8] >> (idx0 % 8)) & 1;
                    let hbit1 = (qh[idx1 / 8] >> (idx1 % 8)) & 1;

                    // 5-bit value
                    let v0 = scale * (low0 | (hbit0 << 4)) as f32 - min_val;
                    let v1 = scale * (low1 | (hbit1 << 4)) as f32 - min_val;

                    sum += v0 * x[x_base + idx0] + v1 * x[x_base + idx1];
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

/// CPU Q2_K dequant matvec
/// Block layout (256 elements, 84 bytes):
///   f16 d (2B) + f16 dmin (2B) + 16B scales/mins (4-bit packed) + 64B qs (2-bit quants)
fn matvec_q2_k_cpu(w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let block_size = 256usize;
    let block_bytes = 84usize;
    let blocks_per_row = in_dim / block_size;
    let mut y = vec![0.0f32; out_dim];

    for (row, y_val) in y.iter_mut().enumerate() {
        let row_off = row * blocks_per_row * block_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let boff = row_off + b * block_bytes;
            let d = f16_to_f32(w[boff], w[boff + 1]);
            let dmin = f16_to_f32(w[boff + 2], w[boff + 3]);
            let sc = &w[boff + 4..boff + 20]; // 16 bytes: scales and mins
            let qs = &w[boff + 20..boff + 84]; // 64 bytes: 2-bit quants (4 per byte)
            let x_base = b * block_size;

            // 16 sub-blocks of 16 elements each
            for (sb, &sc_byte) in sc.iter().enumerate() {
                // Each scale byte packs: low 4 bits = scale, high 4 bits = min
                let scale = d * (sc_byte & 0x0F) as f32;
                let min_val = dmin * ((sc_byte >> 4) & 0x0F) as f32;

                for j in 0..16usize {
                    let idx = sb * 16 + j;
                    // 2-bit quant, 4 per byte
                    let qs_byte = qs[idx / 4];
                    let shift = (idx % 4) * 2;
                    let q = ((qs_byte >> shift) & 0x03) as f32;
                    sum += (scale * q - min_val) * x[x_base + idx];
                }
            }
        }
        *y_val = sum;
    }
    y
}

/// CPU Q8_K dequant matvec
/// Block layout (256 elements, 292 bytes):
///   f32 d (4B) + 256B qs (int8) + 32B bsums (16 × int16, unused for matvec)
fn matvec_q8_k_cpu(w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let block_size = 256usize;
    let block_bytes = 292usize;
    let blocks_per_row = in_dim / block_size;
    let mut y = vec![0.0f32; out_dim];

    for (row, y_val) in y.iter_mut().enumerate() {
        let row_off = row * blocks_per_row * block_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let boff = row_off + b * block_bytes;
            // f32 scale (little-endian)
            let d = f32::from_le_bytes([w[boff], w[boff + 1], w[boff + 2], w[boff + 3]]);
            let x_base = b * block_size;
            for j in 0..256 {
                let val = w[boff + 4 + j] as i8;
                sum += d * val as f32 * x[x_base + j];
            }
        }
        *y_val = sum;
    }
    y
}

/// IQ4_NL non-linear lookup table (from llama.cpp ggml-quants.c)
/// Maps 4-bit indices (0..15) to non-linearly spaced dequantized values.
const IQ4_NL_QUANTS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// CPU IQ4_NL dequant matvec
/// Block layout (32 elements, 18 bytes):
///   f16 d (2B) + 16B qs (4-bit indices into IQ4_NL_QUANTS)
fn matvec_iq4_nl_cpu(w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let block_size = 32usize;
    let block_bytes = 18usize;
    let blocks_per_row = in_dim / block_size;
    let mut y = vec![0.0f32; out_dim];

    for (row, y_val) in y.iter_mut().enumerate() {
        let row_off = row * blocks_per_row * block_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let boff = row_off + b * block_bytes;
            let d = f16_to_f32(w[boff], w[boff + 1]);
            let x_base = b * block_size;
            for j in 0..16 {
                let byte = w[boff + 2 + j];
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;
                sum += d * IQ4_NL_QUANTS[lo] as f32 * x[x_base + 2 * j];
                sum += d * IQ4_NL_QUANTS[hi] as f32 * x[x_base + 2 * j + 1];
            }
        }
        *y_val = sum;
    }
    y
}

/// CPU IQ4_XS dequant matvec
/// Block layout (256 elements, 148 bytes):
///   f16 d (2B) + u16 scales_h (2B) + 8×u16 scales_l (16B) + 128B qs
/// Each sub-block of 32 elements has a 6-bit scale (low 4 bits from scales_l, high 2 bits from scales_h).
fn matvec_iq4_xs_cpu(w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let block_size = 256usize;
    let block_bytes = 148usize;
    let blocks_per_row = in_dim / block_size;
    let mut y = vec![0.0f32; out_dim];

    for (row, y_val) in y.iter_mut().enumerate() {
        let row_off = row * blocks_per_row * block_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let boff = row_off + b * block_bytes;
            let d = f16_to_f32(w[boff], w[boff + 1]);
            let scales_h = u16::from_le_bytes([w[boff + 2], w[boff + 3]]);

            // 8 sub-blocks of 32 elements each
            for sb in 0..8 {
                // Low 4 bits of scale from scales_l (packed as nibbles in u16s)
                let sl_idx = sb / 2;
                let sl_word =
                    u16::from_le_bytes([w[boff + 4 + sl_idx * 2], w[boff + 5 + sl_idx * 2]]);
                let sl = if sb % 2 == 0 {
                    (sl_word & 0x0F) as i32
                } else {
                    ((sl_word >> 4) & 0x0F) as i32
                };
                // High 2 bits from scales_h
                let sh = ((scales_h >> (2 * sb)) & 0x03) as i32;
                let scale = (sl | (sh << 4)) as f32 - 32.0;

                let qs_off = boff + 20 + sb * 16; // 20 = 2 + 2 + 16 header
                let x_base = b * block_size + sb * 32;
                for j in 0..16 {
                    let byte = w[qs_off + j];
                    let lo = (byte & 0x0F) as usize;
                    let hi = ((byte >> 4) & 0x0F) as usize;
                    sum += d * scale * IQ4_NL_QUANTS[lo] as f32 * x[x_base + 2 * j];
                    sum += d * scale * IQ4_NL_QUANTS[hi] as f32 * x[x_base + 2 * j + 1];
                }
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
    fn test_matvec_q3_k_cpu_basic() {
        // 1 row, 256 elements (1 block = 110 bytes)
        // Block: 32B hmask + 64B qs + 12B scales + 2B f16 d
        let mut block = vec![0u8; 110];
        // Set d = 1.0
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[108] = d_bits as u8;
        block[109] = (d_bits >> 8) as u8;
        // All qs = 0, hmask = 0, scales = 0 => all quant values = 0-4 = -4, but scale=0 => sum=0
        // Actually scale bytes are all 0 so sc_f = 0 for all sub-blocks
        let x = vec![1.0f32; 256];
        let y = matvec_q3_k_cpu(&block, &x, 1, 256);
        assert!(y[0].abs() < 1e-5, "Zero scales should give 0, got {}", y[0]);
    }

    #[test]
    fn test_matvec_q5_k_cpu_basic() {
        // 1 row, 256 elements (1 block = 176 bytes)
        // Block: 2B d + 2B dmin + 12B scales + 32B qh + 128B ql
        let mut block = vec![0u8; 176];
        // Set d = 1.0, dmin = 0.0
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        // dmin = 0 (already 0)
        // All scales = 0, ql = 0, qh = 0 => scale=0, min=0, all values = 0
        let x = vec![1.0f32; 256];
        let y = matvec_q5_k_cpu(&block, &x, 1, 256);
        assert!(y[0].abs() < 1e-5, "Zero scales should give 0, got {}", y[0]);
    }

    #[test]
    fn test_matvec_q3_k_cpu_nonzero() {
        // Test with nonzero scale to verify dequantization produces something
        let mut block = vec![0u8; 110];
        let d_bits = half::f16::from_f32(0.5).to_bits();
        block[108] = d_bits as u8;
        block[109] = (d_bits >> 8) as u8;
        // Set scale for sub-block 0: low4=1, hi2=0 => scale_val=1, sc_f=0.5*1=0.5
        block[96] = 0x01; // low nibble for sb=0
                          // hmask and qs all zero => each quant = (0|0) - 4 = -4
                          // sub-block 0 has 16 values each = -4, sc_f=0.5 => contribution = 0.5 * (-4) * 1.0 * 16 = -32
        let x = vec![1.0f32; 256];
        let y = matvec_q3_k_cpu(&block, &x, 1, 256);
        assert!(
            (y[0] - (-32.0)).abs() < 1e-3,
            "Expected -32.0, got {}",
            y[0]
        );
    }

    #[test]
    fn test_matvec_q5_k_cpu_nonzero() {
        // Test with scale=1, dmin=0, ql all 0x11 (lo=1, hi=1), qh=0
        // sub-block 0 scale_low=1 => scale=1.0*1=1.0, min=0
        let mut block = vec![0u8; 176];
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        // scale for sub-block 0: sc[0] low 6 bits = 1
        block[4] = 0x01;
        // ql for sub-block 0 (16 bytes at offset 48): each = 0x11 => low=1, high=1
        for j in 0..16 {
            block[48 + j] = 0x11;
        }
        // qh all 0 => hbit=0
        // v0 = 1.0 * (1 | 0) - 0 = 1.0, v1 = 1.0 * (1 | 0) - 0 = 1.0
        // 32 values * 1.0 * x[i]=1.0 = 32.0
        let x = vec![1.0f32; 256];
        let y = matvec_q5_k_cpu(&block, &x, 1, 256);
        assert!((y[0] - 32.0).abs() < 1e-3, "Expected 32.0, got {}", y[0]);
    }

    #[test]
    fn test_matvec_q2_k_cpu_basic() {
        // 1 row, 256 elements (1 block = 84 bytes)
        // Block: f16 d + f16 dmin + 16B scales + 64B qs
        let mut block = vec![0u8; 84];
        // d = 1.0, dmin = 0.0
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        // dmin = 0, all scales = 0, all qs = 0 => scale=0, min=0 => sum=0
        let x = vec![1.0f32; 256];
        let y = matvec_q2_k_cpu(&block, &x, 1, 256);
        assert!(y[0].abs() < 1e-5, "Zero scales should give 0, got {}", y[0]);
    }

    #[test]
    fn test_matvec_q2_k_cpu_nonzero() {
        // Test with scale=1, dmin=0, sub-block 0 scale_nibble=2
        // qs for sub-block 0: all bits = 01 => q=1
        let mut block = vec![0u8; 84];
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        // dmin = 0
        // sc[0] low nibble = 2 (scale=2.0), high nibble = 0 (min=0)
        block[4] = 0x02;
        // qs for sub-block 0 elements [0..16] → indices 0..15 in qs (bytes 0..3)
        // Each byte holds 4 quants at 2 bits. Set all to 01 = 0x55
        for j in 0..4 {
            block[20 + j] = 0x55; // 01 01 01 01
        }
        // Sub-block 0: 16 elements, each q=1, scale=2.0, min=0
        // contribution = 2.0 * 1 * 1.0 * 16 = 32.0
        let x = vec![1.0f32; 256];
        let y = matvec_q2_k_cpu(&block, &x, 1, 256);
        assert!((y[0] - 32.0).abs() < 1e-3, "Expected 32.0, got {}", y[0]);
    }

    #[test]
    fn test_matvec_q2_k_cpu_with_min() {
        // Test that dmin subtracts correctly
        let mut block = vec![0u8; 84];
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        let dmin_bits = half::f16::from_f32(0.5).to_bits();
        block[2] = dmin_bits as u8;
        block[3] = (dmin_bits >> 8) as u8;
        // sc[0]: low nibble = 0 (scale=0), high nibble = 2 (min=0.5*2=1.0)
        block[4] = 0x20;
        // qs all 0 => q=0 for everything
        // Sub-block 0: 16 elements, each = (0*0 - 1.0) = -1.0
        // contribution = -1.0 * 1.0 * 16 = -16.0
        let x = vec![1.0f32; 256];
        let y = matvec_q2_k_cpu(&block, &x, 1, 256);
        assert!(
            (y[0] - (-16.0)).abs() < 1e-3,
            "Expected -16.0, got {}",
            y[0]
        );
    }

    #[test]
    fn test_matvec_q8_k_cpu_basic() {
        // 1 row, 256 elements (1 block = 292 bytes)
        // Block: f32 d (4B) + 256B qs (int8) + 32B bsums
        let mut block = vec![0u8; 292];
        // d = 1.0 (f32 little-endian)
        let d_bytes = 1.0f32.to_le_bytes();
        block[0..4].copy_from_slice(&d_bytes);
        // All qs = 1 (int8)
        for j in 0..256 {
            block[4 + j] = 1u8;
        }
        let x = vec![1.0f32; 256];
        let y = matvec_q8_k_cpu(&block, &x, 1, 256);
        // Each val = 1.0 * 1 = 1.0, sum = 256.0
        assert!((y[0] - 256.0).abs() < 1e-3, "Expected 256.0, got {}", y[0]);
    }

    #[test]
    fn test_matvec_q8_k_cpu_negative() {
        // Test with negative values
        let mut block = vec![0u8; 292];
        let d_bytes = 0.5f32.to_le_bytes();
        block[0..4].copy_from_slice(&d_bytes);
        // All qs = -2 (0xFE as i8)
        for j in 0..256 {
            block[4 + j] = 0xFE; // -2 as i8
        }
        let x = vec![1.0f32; 256];
        let y = matvec_q8_k_cpu(&block, &x, 1, 256);
        // Each val = 0.5 * (-2) = -1.0, sum = -256.0
        assert!(
            (y[0] - (-256.0)).abs() < 1e-3,
            "Expected -256.0, got {}",
            y[0]
        );
    }

    #[test]
    fn test_f16_to_f32_roundtrip() {
        let bits = half::f16::from_f32(3.14).to_bits();
        let result = f16_to_f32(bits as u8, (bits >> 8) as u8);
        assert!((result - 3.14).abs() < 0.01);
    }

    #[test]
    fn test_matvec_iq4_nl_cpu_basic() {
        // 1 row, 32 elements = 1 block, 18 bytes
        let mut block = vec![0u8; 18];
        // d = 1.0 as f16
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        // All qs = 0x00 → lo=0, hi=0 → both map to IQ4_NL_QUANTS[0] = -127
        for j in 0..16 {
            block[2 + j] = 0x00;
        }
        let x = vec![1.0f32; 32];
        let y = matvec_iq4_nl_cpu(&block, &x, 1, 32);
        let expected = 1.0 * -127.0 * 32.0;
        assert!(
            (y[0] - expected).abs() < 1.0,
            "Expected {}, got {}",
            expected,
            y[0]
        );
    }

    #[test]
    fn test_matvec_iq4_nl_cpu_lut_values() {
        // Verify lookup table mapping: index 8 → +1, index 15 → +113
        let mut block = vec![0u8; 18];
        let d_bits = half::f16::from_f32(0.5).to_bits();
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        // First byte: lo=8 (val=1), hi=15 (val=113)
        block[2] = 0xF8; // hi=15, lo=8
                         // Rest are 0
        let mut x = vec![0.0f32; 32];
        x[0] = 1.0; // multiplied by LUT[8]=1
        x[1] = 1.0; // multiplied by LUT[15]=113
        let y = matvec_iq4_nl_cpu(&block, &x, 1, 32);
        let expected = 0.5 * (1.0 + 113.0);
        assert!(
            (y[0] - expected).abs() < 0.5,
            "Expected {}, got {}",
            expected,
            y[0]
        );
    }

    #[test]
    fn test_matvec_iq4_nl_cpu_two_rows() {
        // 2 rows × 32 elements = 2 blocks
        let mut data = vec![0u8; 36]; // 2 × 18 bytes
        let d1 = half::f16::from_f32(1.0).to_bits();
        let d2 = half::f16::from_f32(2.0).to_bits();
        data[0] = d1 as u8;
        data[1] = (d1 >> 8) as u8;
        // index 8 = +1 for all
        for j in 0..16 {
            data[2 + j] = 0x88;
        }
        data[18] = d2 as u8;
        data[19] = (d2 >> 8) as u8;
        for j in 0..16 {
            data[20 + j] = 0x88;
        }
        let x = vec![1.0f32; 32];
        let y = matvec_iq4_nl_cpu(&data, &x, 2, 32);
        // IQ4_NL_QUANTS[8] = 1, so row0 = 1.0 * 1 * 32 = 32, row1 = 2.0 * 1 * 32 = 64
        assert!(
            (y[0] - 32.0).abs() < 1.0,
            "Row 0: expected 32.0, got {}",
            y[0]
        );
        assert!(
            (y[1] - 64.0).abs() < 1.0,
            "Row 1: expected 64.0, got {}",
            y[1]
        );
    }

    #[test]
    fn test_matvec_iq4_xs_cpu_basic() {
        // 1 row, 256 elements = 1 block, 148 bytes
        let mut block = vec![0u8; 148];
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        // scales_h = 0 (all high bits zero)
        block[2] = 0;
        block[3] = 0;
        // scales_l: we want scale = (sl | sh<<4) - 32 = sl - 32
        // Set sl = 33 (nibble = 1 for even sub-blocks) so scale = 33 - 32 = 1
        // Actually for simplicity set all scales_l words so each nibble = 0
        // Then scale = (0 | 0) - 32 = -32
        // Set all qs to index 8 (val=1): byte = 0x88
        for j in 0..16 {
            block[4 + j] = 0x00; // scales_l all 0
        }
        for j in 0..128 {
            block[20 + j] = 0x88; // all index 8 → IQ4_NL_QUANTS[8] = 1
        }
        let x = vec![1.0f32; 256];
        let y = matvec_iq4_xs_cpu(&block, &x, 1, 256);
        // Each element: d * scale * 1 = 1.0 * (-32) * 1 = -32, total = -32 * 256
        let expected = 1.0 * -32.0 * 256.0;
        assert!(
            (y[0] - expected).abs() < 10.0,
            "Expected {}, got {}",
            expected,
            y[0]
        );
    }

    #[test]
    fn test_matvec_iq4_xs_cpu_positive_scale() {
        // Set scales so scale = 1.0 (sl=1, sh=2 → 1 | 8 = 33 - 32 = 1)
        let mut block = vec![0u8; 148];
        let d_bits = half::f16::from_f32(0.01).to_bits();
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        // scales_h: sh=2 for all sub-blocks → bit pairs = 10
        // Sub-block 0: bits 1:0 = 2, sub-block 1: bits 3:2 = 2, etc.
        // = 0b10_10_10_10_10_10_10_10 = 0xAAAA
        block[2] = 0xAA;
        block[3] = 0xAA;
        // scales_l: sl=1 for all nibbles
        // Each u16 word: nibble0=1, nibble1=1 → 0x0011
        for i in 0..8 {
            block[4 + i * 2] = 0x11;
            block[5 + i * 2] = 0x00;
        }
        // qs: all index 8 → val=1
        for j in 0..128 {
            block[20 + j] = 0x88;
        }
        let x = vec![1.0f32; 256];
        let y = matvec_iq4_xs_cpu(&block, &x, 1, 256);
        // scale = (1 | (2<<4)) - 32 = 33 - 32 = 1
        // Each element: 0.01 * 1 * 1 = 0.01, total ≈ 0.01 * 256 = 2.56
        assert!((y[0] - 2.56).abs() < 0.5, "Expected ~2.56, got {}", y[0]);
    }

    #[test]
    fn test_iq4_nl_lut_symmetry() {
        // LUT has 16 entries from -127 to +113, roughly symmetric around 0
        assert_eq!(IQ4_NL_QUANTS[0], -127);
        assert_eq!(IQ4_NL_QUANTS[15], 113);
        assert_eq!(IQ4_NL_QUANTS[8], 1); // close to 0 crossing
    }

    #[test]
    fn test_matvec_iq4_nl_cpu_dispatch() {
        // Test through the dispatch function
        let mut block = vec![0u8; 18];
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        for j in 0..16 {
            block[2 + j] = 0x88; // all index 8 → val=1
        }
        let x = vec![1.0f32; 32];
        let y = matvec_quantized_cpu(&block, &x, 1, 32, GgmlType::IQ4NL);
        assert!((y[0] - 32.0).abs() < 1.0, "Expected ~32.0, got {}", y[0]);
    }

    #[test]
    fn test_matvec_iq4_xs_cpu_dispatch() {
        let mut block = vec![0u8; 148];
        let d_bits = half::f16::from_f32(0.01).to_bits();
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        block[2] = 0xAA;
        block[3] = 0xAA;
        for i in 0..8 {
            block[4 + i * 2] = 0x11;
            block[5 + i * 2] = 0x00;
        }
        for j in 0..128 {
            block[20 + j] = 0x88;
        }
        let x = vec![1.0f32; 256];
        let y = matvec_quantized_cpu(&block, &x, 1, 256, GgmlType::IQ4XS);
        assert!((y[0] - 2.56).abs() < 0.5, "Expected ~2.56, got {}", y[0]);
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
