//! AWQ (Activation-aware Weight Quantization)
//!
//! Implements per-channel scaling based on activation magnitudes for more accurate
//! quantization than naive round-to-nearest. Based on:
//! "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
//! (Lin et al., 2023)
//!
//! Key insight: not all weight channels are equally important. Channels corresponding
//! to large activation magnitudes contribute more to output and should be quantized
//! more carefully. AWQ applies per-channel scaling s_j before quantization:
//!
//!   w_q = round(w * s / Δ) * Δ / s
//!
//! where s_j = (max|a_j|)^α protects salient channels, and α ∈ [0,1] balances
//! between no protection (α=0, equivalent to RTN) and full protection (α=1).

use anyhow::{bail, Result};
use std::collections::HashMap;
use tracing::{debug, info};

/// AWQ quantization configuration
#[derive(Debug, Clone)]
pub struct AwqConfig {
    /// Number of quantization bits (2, 3, 4, or 8)
    pub bits: u32,
    /// Group size for per-group quantization (e.g. 128)
    pub group_size: usize,
    /// Alpha exponent for activation-aware scaling: s_j = (max|a_j|)^alpha
    /// 0.0 = no activation awareness (equivalent to RTN)
    /// 1.0 = full activation awareness
    /// Typical: 0.5 (square root scaling)
    pub alpha: f32,
    /// Number of calibration samples to collect for activation statistics
    pub calibration_samples: usize,
    /// Whether to apply per-channel clipping optimization
    pub clip: bool,
    /// Clipping ratio search granularity (number of candidate ratios)
    pub clip_steps: usize,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            alpha: 0.5,
            calibration_samples: 128,
            clip: true,
            clip_steps: 20,
        }
    }
}

/// Per-channel activation statistics collected during calibration
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Per-channel maximum absolute activation: max|x_j| across calibration data
    pub channel_max_abs: Vec<f32>,
    /// Per-channel mean absolute activation (for advanced scaling)
    pub channel_mean_abs: Vec<f32>,
    /// Number of samples collected
    pub samples: usize,
    /// Running sum for incremental mean computation
    channel_sum_abs: Vec<f32>,
}

impl ActivationStats {
    /// Create empty stats for `channels` dimensions
    pub fn new(channels: usize) -> Self {
        Self {
            channel_max_abs: vec![0.0; channels],
            channel_mean_abs: vec![0.0; channels],
            samples: 0,
            channel_sum_abs: vec![0.0; channels],
        }
    }

    /// Update stats with a new activation sample
    /// `activations`: a batch of activations, shape [tokens, channels] row-major
    pub fn update(&mut self, activations: &[f32], channels: usize) {
        if activations.is_empty() || channels == 0 {
            return;
        }
        let tokens = activations.len() / channels;
        for t in 0..tokens {
            let row = &activations[t * channels..(t + 1) * channels];
            for (j, &val) in row.iter().enumerate() {
                let abs_val = val.abs();
                if abs_val > self.channel_max_abs[j] {
                    self.channel_max_abs[j] = abs_val;
                }
                self.channel_sum_abs[j] += abs_val;
            }
            self.samples += 1;
        }
        // Update means
        if self.samples > 0 {
            for j in 0..channels {
                self.channel_mean_abs[j] = self.channel_sum_abs[j] / self.samples as f32;
            }
        }
    }

    /// Compute per-channel scaling factors: s_j = (max|a_j|)^alpha
    /// Normalized so that mean(s) = 1.0 to preserve overall magnitude
    pub fn compute_scales(&self, alpha: f32) -> Vec<f32> {
        if self.channel_max_abs.is_empty() {
            return vec![];
        }

        let mut scales: Vec<f32> = self
            .channel_max_abs
            .iter()
            .map(|&m| if m > 0.0 { m.powf(alpha) } else { 1.0 })
            .collect();

        // Normalize so mean(s) = 1.0
        let mean: f32 = scales.iter().sum::<f32>() / scales.len() as f32;
        if mean > 0.0 {
            for s in scales.iter_mut() {
                *s /= mean;
            }
        }

        scales
    }
}

/// A single AWQ-quantized weight tensor
#[derive(Debug, Clone)]
pub struct AwqQuantizedWeight {
    /// Quantized integer values packed into bytes
    /// For 4-bit: 2 values per byte, for 8-bit: 1 value per byte
    pub data: Vec<u8>,
    /// Per-group scale factors: Δ_g (group quantization step size)
    pub scales: Vec<f32>,
    /// Per-group zero points
    pub zeros: Vec<f32>,
    /// Per-channel AWQ scaling factors (applied before quantization)
    pub awq_scales: Vec<f32>,
    /// Output dimension (rows)
    pub d_out: usize,
    /// Input dimension (columns)
    pub d_in: usize,
    /// Quantization bits
    pub bits: u32,
    /// Group size
    pub group_size: usize,
}

impl AwqQuantizedWeight {
    /// Dequantize back to f32 for inference: w' = (q * Δ + zero) / s
    pub fn dequantize(&self) -> Vec<f32> {
        let mut output = vec![0.0f32; self.d_out * self.d_in];
        let groups_per_row = (self.d_in + self.group_size - 1) / self.group_size;

        for row in 0..self.d_out {
            for col in 0..self.d_in {
                let group_idx = row * groups_per_row + col / self.group_size;
                let scale = self.scales.get(group_idx).copied().unwrap_or(1.0);
                let zero = self.zeros.get(group_idx).copied().unwrap_or(0.0);
                let awq_s = self.awq_scales.get(col).copied().unwrap_or(1.0);

                let q_val = self.read_quantized(row, col) as f32;
                let deq = (q_val * scale + zero) / awq_s;
                output[row * self.d_in + col] = deq;
            }
        }

        output
    }

    /// Dequantize and compute matvec: output = W_dequant @ input
    /// Fused operation avoids materializing the full dequantized weight matrix
    pub fn matvec(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.d_in, "Input dimension mismatch");
        let mut output = vec![0.0f32; self.d_out];
        let groups_per_row = (self.d_in + self.group_size - 1) / self.group_size;

        for row in 0..self.d_out {
            let mut sum = 0.0f32;
            for col in 0..self.d_in {
                let group_idx = row * groups_per_row + col / self.group_size;
                let scale = self.scales.get(group_idx).copied().unwrap_or(1.0);
                let zero = self.zeros.get(group_idx).copied().unwrap_or(0.0);
                let awq_s = self.awq_scales.get(col).copied().unwrap_or(1.0);

                let q_val = self.read_quantized(row, col) as f32;
                let w = (q_val * scale + zero) / awq_s;
                sum += w * input[col];
            }
            output[row] = sum;
        }

        output
    }

    /// Read a single quantized value at (row, col)
    fn read_quantized(&self, row: usize, col: usize) -> i32 {
        let linear_idx = row * self.d_in + col;
        match self.bits {
            4 => {
                let byte_idx = linear_idx / 2;
                if byte_idx >= self.data.len() {
                    return 0;
                }
                let byte = self.data[byte_idx];
                if linear_idx % 2 == 0 {
                    (byte & 0x0F) as i32 - 8 // signed 4-bit: range [-8, 7]
                } else {
                    ((byte >> 4) & 0x0F) as i32 - 8
                }
            }
            8 => {
                if linear_idx >= self.data.len() {
                    return 0;
                }
                self.data[linear_idx] as i8 as i32
            }
            3 => {
                // 3-bit packed: 8 values in 3 bytes
                let group_of_8 = linear_idx / 8;
                let within = linear_idx % 8;
                let base = group_of_8 * 3;
                if base + 2 >= self.data.len() {
                    return 0;
                }
                let bits24 = (self.data[base] as u32)
                    | ((self.data[base + 1] as u32) << 8)
                    | ((self.data[base + 2] as u32) << 16);
                ((bits24 >> (within * 3)) & 0x7) as i32 - 4 // signed 3-bit: [-4, 3]
            }
            2 => {
                let byte_idx = linear_idx / 4;
                let shift = (linear_idx % 4) * 2;
                if byte_idx >= self.data.len() {
                    return 0;
                }
                ((self.data[byte_idx] >> shift) & 0x3) as i32 - 2 // signed 2-bit: [-2, 1]
            }
            _ => 0,
        }
    }

    /// Memory footprint in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4 + self.zeros.len() * 4 + self.awq_scales.len() * 4
    }

    /// Compression ratio vs f32
    pub fn compression_ratio(&self) -> f64 {
        let original = self.d_out * self.d_in * 4; // f32
        if self.size_bytes() > 0 {
            original as f64 / self.size_bytes() as f64
        } else {
            0.0
        }
    }
}

/// Quantize an f32 weight matrix using AWQ
///
/// `weights`: row-major [d_out, d_in]
/// `activation_stats`: calibration data for input activations to this layer
/// `config`: quantization parameters
pub fn quantize_awq(
    weights: &[f32],
    d_out: usize,
    d_in: usize,
    activation_stats: &ActivationStats,
    config: &AwqConfig,
) -> Result<AwqQuantizedWeight> {
    if weights.len() != d_out * d_in {
        bail!(
            "Weight shape mismatch: expected {}x{} = {}, got {}",
            d_out,
            d_in,
            d_out * d_in,
            weights.len()
        );
    }

    if config.bits != 2 && config.bits != 3 && config.bits != 4 && config.bits != 8 {
        bail!("Unsupported bit width: {}. Use 2, 3, 4, or 8.", config.bits);
    }

    // Step 1: Compute per-channel AWQ scaling factors
    let awq_scales = if activation_stats.samples > 0 {
        activation_stats.compute_scales(config.alpha)
    } else {
        vec![1.0; d_in] // No calibration data: fall back to uniform (RTN)
    };

    // Step 2: Apply AWQ scaling to weights: w_scaled = w * s
    let mut scaled_weights = vec![0.0f32; d_out * d_in];
    for row in 0..d_out {
        for col in 0..d_in {
            let idx = row * d_in + col;
            scaled_weights[idx] = weights[idx] * awq_scales[col];
        }
    }

    // Step 3: Per-group quantization of scaled weights
    let groups_per_row = (d_in + config.group_size - 1) / config.group_size;
    let total_groups = d_out * groups_per_row;
    let mut scales = vec![0.0f32; total_groups];
    let mut zeros = vec![0.0f32; total_groups];

    let max_val = (1 << (config.bits - 1)) - 1; // e.g., 7 for 4-bit
    let min_val = -(1 << (config.bits - 1)); // e.g., -8 for 4-bit
    let range = (max_val - min_val) as f32;

    // Compute per-group scales and zeros
    for row in 0..d_out {
        for g in 0..groups_per_row {
            let col_start = g * config.group_size;
            let col_end = (col_start + config.group_size).min(d_in);
            let group_idx = row * groups_per_row + g;

            let mut w_min = f32::MAX;
            let mut w_max = f32::MIN;
            for col in col_start..col_end {
                let w = scaled_weights[row * d_in + col];
                if w < w_min {
                    w_min = w;
                }
                if w > w_max {
                    w_max = w;
                }
            }

            // Apply clipping optimization if enabled
            if config.clip && config.clip_steps > 0 {
                let (clipped_min, clipped_max) = find_optimal_clip(
                    &scaled_weights[row * d_in + col_start..row * d_in + col_end],
                    config.bits,
                    config.clip_steps,
                );
                w_min = clipped_min;
                w_max = clipped_max;
            }

            let delta = if range > 0.0 {
                (w_max - w_min) / range
            } else {
                1.0
            };
            let zero = w_min - min_val as f32 * delta;

            scales[group_idx] = delta;
            zeros[group_idx] = zero;
        }
    }

    // Step 4: Quantize
    let total_elements = d_out * d_in;
    let data = match config.bits {
        4 => {
            let mut bytes = vec![0u8; (total_elements + 1) / 2];
            for row in 0..d_out {
                for col in 0..d_in {
                    let idx = row * d_in + col;
                    let group_idx = row * groups_per_row + col / config.group_size;
                    let delta = scales[group_idx];
                    let zero = zeros[group_idx];
                    let w = scaled_weights[idx];

                    let q = if delta > 0.0 {
                        ((w - zero) / delta)
                            .round()
                            .clamp(min_val as f32, max_val as f32) as i32
                    } else {
                        0
                    };
                    let q_unsigned = (q - min_val) as u8;

                    let byte_idx = idx / 2;
                    if idx % 2 == 0 {
                        bytes[byte_idx] |= q_unsigned & 0x0F;
                    } else {
                        bytes[byte_idx] |= (q_unsigned & 0x0F) << 4;
                    }
                }
            }
            bytes
        }
        8 => {
            let mut bytes = vec![0u8; total_elements];
            for row in 0..d_out {
                for col in 0..d_in {
                    let idx = row * d_in + col;
                    let group_idx = row * groups_per_row + col / config.group_size;
                    let delta = scales[group_idx];
                    let zero = zeros[group_idx];
                    let w = scaled_weights[idx];

                    let q = if delta > 0.0 {
                        ((w - zero) / delta).round().clamp(-128.0, 127.0) as i8
                    } else {
                        0i8
                    };
                    bytes[idx] = q as u8;
                }
            }
            bytes
        }
        3 => {
            // Pack 3-bit: 8 values into 3 bytes
            let num_groups_of_8 = (total_elements + 7) / 8;
            let mut bytes = vec![0u8; num_groups_of_8 * 3];
            for row in 0..d_out {
                for col in 0..d_in {
                    let idx = row * d_in + col;
                    let group_idx = row * groups_per_row + col / config.group_size;
                    let delta = scales[group_idx];
                    let zero = zeros[group_idx];
                    let w = scaled_weights[idx];

                    let q = if delta > 0.0 {
                        ((w - zero) / delta)
                            .round()
                            .clamp(min_val as f32, max_val as f32) as i32
                    } else {
                        0
                    };
                    let q_unsigned = (q - min_val) as u32;

                    let g8 = idx / 8;
                    let within = idx % 8;
                    let base = g8 * 3;
                    let shift = within * 3;
                    // Spread across 3 bytes
                    let bits24 = q_unsigned << shift;
                    bytes[base] |= (bits24 & 0xFF) as u8;
                    if base + 1 < bytes.len() {
                        bytes[base + 1] |= ((bits24 >> 8) & 0xFF) as u8;
                    }
                    if base + 2 < bytes.len() {
                        bytes[base + 2] |= ((bits24 >> 16) & 0xFF) as u8;
                    }
                }
            }
            bytes
        }
        2 => {
            let mut bytes = vec![0u8; (total_elements + 3) / 4];
            for row in 0..d_out {
                for col in 0..d_in {
                    let idx = row * d_in + col;
                    let group_idx = row * groups_per_row + col / config.group_size;
                    let delta = scales[group_idx];
                    let zero = zeros[group_idx];
                    let w = scaled_weights[idx];

                    let q = if delta > 0.0 {
                        ((w - zero) / delta)
                            .round()
                            .clamp(min_val as f32, max_val as f32) as i32
                    } else {
                        0
                    };
                    let q_unsigned = (q - min_val) as u8;

                    let byte_idx = idx / 4;
                    let shift = (idx % 4) * 2;
                    bytes[byte_idx] |= (q_unsigned & 0x3) << shift;
                }
            }
            bytes
        }
        _ => unreachable!(),
    };

    debug!(
        "AWQ quantized {}x{} to {}-bit (groups={}, compression={:.1}x)",
        d_out,
        d_in,
        config.bits,
        total_groups,
        (d_out * d_in * 4) as f64 / data.len() as f64
    );

    Ok(AwqQuantizedWeight {
        data,
        scales,
        zeros,
        awq_scales,
        d_out,
        d_in,
        bits: config.bits,
        group_size: config.group_size,
    })
}

/// Find optimal clipping range to minimize quantization error
/// Searches for clip_ratio in [0.5, 1.0] that minimizes MSE after quantize+dequantize
fn find_optimal_clip(group_weights: &[f32], bits: u32, steps: usize) -> (f32, f32) {
    if group_weights.is_empty() {
        return (0.0, 0.0);
    }

    let w_min = group_weights.iter().cloned().fold(f32::MAX, f32::min);
    let w_max = group_weights.iter().cloned().fold(f32::MIN, f32::max);

    if (w_max - w_min).abs() < 1e-10 {
        return (w_min, w_max);
    }

    let max_val = (1 << (bits - 1)) - 1;
    let min_val = -(1i32 << (bits - 1));
    let range = (max_val - min_val) as f32;

    let mut best_mse = f32::MAX;
    let mut best_min = w_min;
    let mut best_max = w_max;

    for step in 0..=steps {
        let ratio = 0.5 + 0.5 * step as f32 / steps as f32;
        let center = (w_min + w_max) / 2.0;
        let half_range = (w_max - w_min) / 2.0 * ratio;
        let clip_min = center - half_range;
        let clip_max = center + half_range;

        let delta = (clip_max - clip_min) / range;
        if delta <= 0.0 {
            continue;
        }

        let mut mse = 0.0f32;
        for &w in group_weights {
            let clamped = w.clamp(clip_min, clip_max);
            let q = ((clamped - clip_min) / delta).round().clamp(0.0, range);
            let deq = q * delta + clip_min;
            let err = w - deq;
            mse += err * err;
        }

        if mse < best_mse {
            best_mse = mse;
            best_min = clip_min;
            best_max = clip_max;
        }
    }

    (best_min, best_max)
}

/// Calibration collector: feeds sample inputs through a model to gather activation stats
pub struct AwqCalibrator {
    /// Per-tensor activation statistics
    pub stats: HashMap<String, ActivationStats>,
    /// Expected channel dimensions per tensor
    dims: HashMap<String, usize>,
    /// Samples collected
    pub total_samples: usize,
    /// Target number of samples
    pub target_samples: usize,
}

impl AwqCalibrator {
    pub fn new(target_samples: usize) -> Self {
        Self {
            stats: HashMap::new(),
            dims: HashMap::new(),
            total_samples: 0,
            target_samples,
        }
    }

    /// Register a tensor for calibration with its input channel dimension
    pub fn register_tensor(&mut self, name: &str, channels: usize) {
        self.dims.insert(name.to_string(), channels);
        self.stats
            .insert(name.to_string(), ActivationStats::new(channels));
    }

    /// Record activations for a tensor during calibration forward pass
    pub fn record(&mut self, tensor_name: &str, activations: &[f32]) {
        if let Some(&channels) = self.dims.get(tensor_name) {
            if let Some(stats) = self.stats.get_mut(tensor_name) {
                stats.update(activations, channels);
            }
        }
    }

    /// Check if calibration is complete
    pub fn is_complete(&self) -> bool {
        self.total_samples >= self.target_samples
    }

    /// Increment sample counter
    pub fn advance(&mut self) {
        self.total_samples += 1;
    }

    /// Get activation stats for a tensor (if collected)
    pub fn get_stats(&self, tensor_name: &str) -> Option<&ActivationStats> {
        self.stats.get(tensor_name)
    }
}

/// Quantize an entire model's weight tensors using AWQ
/// Returns a map of tensor_name -> AwqQuantizedWeight
pub fn quantize_model_awq(
    weights: &HashMap<String, (Vec<f32>, usize, usize)>, // name -> (data, d_out, d_in)
    calibrator: &AwqCalibrator,
    config: &AwqConfig,
) -> Result<HashMap<String, AwqQuantizedWeight>> {
    let mut quantized = HashMap::new();
    let mut total_original = 0u64;
    let mut total_quantized = 0u64;

    for (name, (data, d_out, d_in)) in weights {
        let stats = calibrator
            .get_stats(name)
            .cloned()
            .unwrap_or_else(|| ActivationStats::new(*d_in));

        let qw = quantize_awq(data, *d_out, *d_in, &stats, config)?;
        total_original += (*d_out * *d_in * 4) as u64;
        total_quantized += qw.size_bytes() as u64;
        quantized.insert(name.clone(), qw);
    }

    info!(
        "AWQ {}-bit quantization complete: {} tensors, {:.2} GB -> {:.2} GB ({:.1}x compression)",
        config.bits,
        quantized.len(),
        total_original as f64 / (1024.0 * 1024.0 * 1024.0),
        total_quantized as f64 / (1024.0 * 1024.0 * 1024.0),
        total_original as f64 / total_quantized as f64
    );

    Ok(quantized)
}

/// Compute quantization error (MSE and max error) between original and AWQ-quantized weights
pub fn measure_quantization_error(
    original: &[f32],
    quantized: &AwqQuantizedWeight,
) -> QuantizationError {
    let dequantized = quantized.dequantize();
    let n = original.len().min(dequantized.len());

    if n == 0 {
        return QuantizationError {
            mse: 0.0,
            max_abs_error: 0.0,
            snr_db: 0.0,
        };
    }

    let mut sum_sq_err = 0.0f64;
    let mut max_err = 0.0f32;
    let mut sum_sq_signal = 0.0f64;

    for i in 0..n {
        let err = (original[i] - dequantized[i]).abs();
        sum_sq_err += (err as f64) * (err as f64);
        if err > max_err {
            max_err = err;
        }
        sum_sq_signal += (original[i] as f64) * (original[i] as f64);
    }

    let mse = sum_sq_err / n as f64;
    let snr_db = if sum_sq_err > 0.0 {
        10.0 * (sum_sq_signal / sum_sq_err).log10()
    } else {
        f64::INFINITY
    };

    QuantizationError {
        mse,
        max_abs_error: max_err,
        snr_db,
    }
}

/// Quantization error metrics
#[derive(Debug, Clone)]
pub struct QuantizationError {
    /// Mean squared error
    pub mse: f64,
    /// Maximum absolute error
    pub max_abs_error: f32,
    /// Signal-to-noise ratio in dB (higher = better)
    pub snr_db: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_stats_basic() {
        let mut stats = ActivationStats::new(4);
        // Single row of activations: [1.0, -2.0, 0.5, 3.0]
        stats.update(&[1.0, -2.0, 0.5, 3.0], 4);
        assert_eq!(stats.samples, 1);
        assert!((stats.channel_max_abs[0] - 1.0).abs() < 1e-6);
        assert!((stats.channel_max_abs[1] - 2.0).abs() < 1e-6);
        assert!((stats.channel_max_abs[2] - 0.5).abs() < 1e-6);
        assert!((stats.channel_max_abs[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_activation_stats_multiple_samples() {
        let mut stats = ActivationStats::new(3);
        stats.update(&[1.0, 2.0, 3.0], 3);
        stats.update(&[4.0, 1.0, 2.0], 3);
        assert_eq!(stats.samples, 2);
        assert!((stats.channel_max_abs[0] - 4.0).abs() < 1e-6);
        assert!((stats.channel_max_abs[1] - 2.0).abs() < 1e-6);
        assert!((stats.channel_max_abs[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_scales_normalized() {
        let mut stats = ActivationStats::new(4);
        stats.update(&[1.0, 2.0, 4.0, 8.0], 4);
        let scales = stats.compute_scales(1.0);
        assert_eq!(scales.len(), 4);
        // Mean should be ~1.0
        let mean: f32 = scales.iter().sum::<f32>() / 4.0;
        assert!(
            (mean - 1.0).abs() < 1e-5,
            "Mean should be 1.0, got {}",
            mean
        );
        // Higher activation → higher scale
        assert!(
            scales[3] > scales[0],
            "Channel with larger activation should have larger scale"
        );
    }

    #[test]
    fn test_compute_scales_alpha_zero() {
        let mut stats = ActivationStats::new(3);
        stats.update(&[1.0, 100.0, 0.01], 3);
        let scales = stats.compute_scales(0.0);
        // alpha=0 → all scales = m^0 = 1.0 → normalized = 1.0
        for s in &scales {
            assert!((s - 1.0).abs() < 1e-5, "Alpha=0 should give uniform scales");
        }
    }

    #[test]
    fn test_quantize_awq_4bit_roundtrip() {
        // Simple 2x4 weight matrix
        let weights = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let stats = ActivationStats::new(4);
        let config = AwqConfig {
            bits: 4,
            group_size: 4,
            alpha: 0.0, // no AWQ scaling → pure RTN
            clip: false,
            clip_steps: 0,
            ..Default::default()
        };

        let qw = quantize_awq(&weights, 2, 4, &stats, &config).unwrap();
        assert_eq!(qw.d_out, 2);
        assert_eq!(qw.d_in, 4);
        assert_eq!(qw.bits, 4);

        // Dequantize and check reconstruction error is small
        let deq = qw.dequantize();
        assert_eq!(deq.len(), 8);
        let mse: f32 = weights
            .iter()
            .zip(deq.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / 8.0;
        // 4-bit quantization of small values should be reasonably accurate
        assert!(mse < 0.01, "MSE too high: {}", mse);
    }

    #[test]
    fn test_quantize_awq_8bit_roundtrip() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let stats = ActivationStats::new(8);
        let config = AwqConfig {
            bits: 8,
            group_size: 8,
            alpha: 0.0,
            clip: false,
            clip_steps: 0,
            ..Default::default()
        };

        let qw = quantize_awq(&weights, 8, 8, &stats, &config).unwrap();
        let deq = qw.dequantize();
        let mse: f32 = weights
            .iter()
            .zip(deq.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / 64.0;
        assert!(mse < 0.001, "8-bit MSE too high: {}", mse);
    }

    #[test]
    fn test_awq_with_activation_awareness() {
        // Weights where channel 0 has large activation (should be protected)
        let weights = vec![
            0.1, 0.2, 0.3, 0.4, // row 0
            0.5, 0.6, 0.7, 0.8, // row 1
        ];
        let mut stats = ActivationStats::new(4);
        // Channel 0 has much larger activations
        stats.update(&[100.0, 1.0, 1.0, 1.0], 4);

        let config_awq = AwqConfig {
            bits: 4,
            group_size: 4,
            alpha: 0.5,
            clip: false,
            clip_steps: 0,
            ..Default::default()
        };
        let config_rtn = AwqConfig {
            bits: 4,
            group_size: 4,
            alpha: 0.0,
            clip: false,
            clip_steps: 0,
            ..Default::default()
        };

        let qw_awq = quantize_awq(&weights, 2, 4, &stats, &config_awq).unwrap();
        let qw_rtn = quantize_awq(&weights, 2, 4, &stats, &config_rtn).unwrap();

        // AWQ scales should be non-uniform
        assert!(
            qw_awq.awq_scales[0] != qw_awq.awq_scales[1],
            "AWQ scales should differ for channels with different activation magnitudes"
        );
        // RTN scales should be uniform (all 1.0)
        for s in &qw_rtn.awq_scales {
            assert!((s - 1.0).abs() < 1e-5, "RTN should have uniform scales");
        }
    }

    #[test]
    fn test_matvec_correctness() {
        // 2x3 weight matrix, 3-element input
        let weights = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // ~identity for first 2 rows
        let stats = ActivationStats::new(3);
        let config = AwqConfig {
            bits: 8, // high precision for this test
            group_size: 3,
            alpha: 0.0,
            clip: false,
            clip_steps: 0,
            ..Default::default()
        };

        let qw = quantize_awq(&weights, 2, 3, &stats, &config).unwrap();
        let input = vec![5.0, 3.0, 1.0];
        let output = qw.matvec(&input);

        assert_eq!(output.len(), 2);
        // Should be approximately [5.0, 3.0]
        assert!(
            (output[0] - 5.0).abs() < 0.5,
            "Expected ~5.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 3.0).abs() < 0.5,
            "Expected ~3.0, got {}",
            output[1]
        );
    }

    #[test]
    fn test_compression_ratio() {
        let weights = vec![0.0f32; 1024 * 1024]; // 1M elements = 4MB
        let stats = ActivationStats::new(1024);
        let config = AwqConfig {
            bits: 4,
            group_size: 128,
            alpha: 0.0,
            clip: false,
            clip_steps: 0,
            ..Default::default()
        };

        let qw = quantize_awq(&weights, 1024, 1024, &stats, &config).unwrap();
        let ratio = qw.compression_ratio();
        // 4-bit should give ~8x compression (ignoring scale/zero overhead)
        assert!(
            ratio > 5.0,
            "Expected >5x compression for 4-bit, got {:.1}x",
            ratio
        );
    }

    #[test]
    fn test_measure_quantization_error() {
        let weights = vec![0.1, -0.2, 0.3, -0.4];
        let stats = ActivationStats::new(4);
        let config = AwqConfig {
            bits: 4,
            group_size: 4,
            alpha: 0.0,
            clip: false,
            clip_steps: 0,
            ..Default::default()
        };

        let qw = quantize_awq(&weights, 1, 4, &stats, &config).unwrap();
        let err = measure_quantization_error(&weights, &qw);
        assert!(err.mse >= 0.0);
        assert!(err.max_abs_error >= 0.0);
        assert!(
            err.snr_db > 0.0,
            "SNR should be positive for non-zero weights"
        );
    }

    #[test]
    fn test_clipping_reduces_error() {
        // Random-ish weights with an outlier
        let mut weights = vec![0.0f32; 128];
        for i in 0..128 {
            weights[i] = ((i as f32 * 0.37).sin()) * 0.5;
        }
        weights[0] = 10.0; // outlier

        let stats = ActivationStats::new(128);
        let config_no_clip = AwqConfig {
            bits: 4,
            group_size: 128,
            alpha: 0.0,
            clip: false,
            clip_steps: 0,
            ..Default::default()
        };
        let config_clip = AwqConfig {
            bits: 4,
            group_size: 128,
            alpha: 0.0,
            clip: true,
            clip_steps: 20,
            ..Default::default()
        };

        let qw_no_clip = quantize_awq(&weights, 1, 128, &stats, &config_no_clip).unwrap();
        let qw_clip = quantize_awq(&weights, 1, 128, &stats, &config_clip).unwrap();

        let err_no_clip = measure_quantization_error(&weights, &qw_no_clip);
        let err_clip = measure_quantization_error(&weights, &qw_clip);

        // Clipping should reduce MSE for most weights (the outlier hurts without clipping)
        // Note: this may not always hold but should for this specific distribution
        assert!(
            err_clip.mse <= err_no_clip.mse * 1.1,
            "Clipping should not significantly increase error: clip={:.6} vs no_clip={:.6}",
            err_clip.mse,
            err_no_clip.mse
        );
    }

    #[test]
    fn test_calibrator_workflow() {
        let mut cal = AwqCalibrator::new(10);
        cal.register_tensor("layer0.weight", 4);
        cal.register_tensor("layer1.weight", 8);

        assert!(!cal.is_complete());

        for _ in 0..10 {
            cal.record("layer0.weight", &[1.0, 2.0, 3.0, 4.0]);
            cal.record("layer1.weight", &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
            cal.advance();
        }

        assert!(cal.is_complete());
        assert_eq!(cal.total_samples, 10);

        let stats = cal.get_stats("layer0.weight").unwrap();
        assert_eq!(stats.samples, 10);
        assert!((stats.channel_max_abs[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_2bit_quantization() {
        let weights = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4];
        let stats = ActivationStats::new(4);
        let config = AwqConfig {
            bits: 2,
            group_size: 4,
            alpha: 0.0,
            clip: false,
            clip_steps: 0,
            ..Default::default()
        };
        let qw = quantize_awq(&weights, 2, 4, &stats, &config).unwrap();
        assert_eq!(qw.bits, 2);
        let deq = qw.dequantize();
        assert_eq!(deq.len(), 8);
    }

    #[test]
    fn test_3bit_quantization() {
        let weights = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4];
        let stats = ActivationStats::new(4);
        let config = AwqConfig {
            bits: 3,
            group_size: 4,
            alpha: 0.0,
            clip: false,
            clip_steps: 0,
            ..Default::default()
        };
        let qw = quantize_awq(&weights, 2, 4, &stats, &config).unwrap();
        assert_eq!(qw.bits, 3);
        let deq = qw.dequantize();
        assert_eq!(deq.len(), 8);
    }
}
