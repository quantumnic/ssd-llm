//! RoPE Scaling for Extended Context Windows
//!
//! When a model is trained with context length C but you want to use it at length C' > C,
//! the RoPE frequencies must be adjusted. This module implements three scaling strategies:
//!
//! - **Linear**: Scale all frequencies by factor = C_original / C_target (CodeLlama style)
//! - **NTK-aware**: Adjust the base theta to spread frequencies (no fine-tuning needed)
//! - **YaRN**: Yet another RoPE extensioN — interpolates between NTK and linear with
//!   an attention scaling factor for best quality
//!
//! Reference: <https://arxiv.org/abs/2309.00071> (YaRN paper)

use std::f32::consts::PI;
use tracing::debug;

/// RoPE scaling method
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RopeScalingMethod {
    /// No scaling — use original frequencies
    #[default]
    None,
    /// Linear frequency interpolation: freq *= original_ctx / target_ctx
    Linear {
        /// Scaling factor = original_max_position / target_max_position
        factor: f32,
    },
    /// NTK-aware scaling: adjust theta_base instead of frequencies
    /// Works without fine-tuning by modifying the base of the frequency computation
    NtkAware {
        /// Scaling factor (typically target_ctx / original_ctx)
        factor: f32,
    },
    /// YaRN: combines NTK interpolation with attention scaling
    YaRn {
        /// Context extension factor
        factor: f32,
        /// Original context length the model was trained with
        original_max_position: usize,
        /// Attention temperature correction factor (computed automatically if None)
        attn_factor: Option<f32>,
        /// Beta fast parameter (default 32.0)
        beta_fast: f32,
        /// Beta slow parameter (default 1.0)
        beta_slow: f32,
    },
}

/// Configuration for RoPE with optional scaling
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Base theta (default 10000.0, some models use 500000.0)
    pub theta_base: f32,
    /// Head dimension
    pub head_dim: usize,
    /// Scaling method
    pub scaling: RopeScalingMethod,
}

impl RopeConfig {
    pub fn new(theta_base: f32, head_dim: usize) -> Self {
        Self {
            theta_base,
            head_dim,
            scaling: RopeScalingMethod::None,
        }
    }

    pub fn with_scaling(mut self, scaling: RopeScalingMethod) -> Self {
        self.scaling = scaling;
        self
    }

    /// Compute the effective theta base after NTK scaling
    pub fn effective_theta_base(&self) -> f32 {
        match &self.scaling {
            RopeScalingMethod::NtkAware { factor } => {
                // θ' = θ * factor^(dim / (dim - 2))
                let exponent = self.head_dim as f32 / (self.head_dim as f32 - 2.0);
                let scaled = self.theta_base * factor.powf(exponent);
                debug!(
                    "NTK-aware RoPE: base {} -> {}, factor={}",
                    self.theta_base, scaled, factor
                );
                scaled
            }
            _ => self.theta_base,
        }
    }

    /// Compute the frequency for a given dimension pair index
    /// Returns the frequency to multiply by position to get the angle
    pub fn frequency(&self, dim_pair_index: usize) -> f32 {
        let i = (dim_pair_index * 2) as f32;
        let theta = self.effective_theta_base();

        let base_freq = 1.0 / theta.powf(i / self.head_dim as f32);

        match &self.scaling {
            RopeScalingMethod::Linear { factor } => base_freq / factor,
            RopeScalingMethod::YaRn {
                factor,
                original_max_position,
                beta_fast,
                beta_slow,
                ..
            } => {
                let low = yarn_find_correction_dim(
                    *beta_fast,
                    self.head_dim,
                    self.theta_base,
                    *original_max_position,
                );
                let high = yarn_find_correction_dim(
                    *beta_slow,
                    self.head_dim,
                    self.theta_base,
                    *original_max_position,
                );

                let dim = dim_pair_index as f32;

                if dim < low {
                    // High frequency: no interpolation (keep original)
                    base_freq
                } else if dim > high {
                    // Low frequency: full linear interpolation
                    base_freq / factor
                } else {
                    // Ramp region: smooth interpolation
                    let ramp = if (high - low).abs() < 1e-6 {
                        0.5
                    } else {
                        (dim - low) / (high - low)
                    };
                    let interpolated = base_freq / factor;
                    base_freq * (1.0 - ramp) + interpolated * ramp
                }
            }
            _ => base_freq,
        }
    }

    /// Get YaRN attention scaling factor (sqrt(1 + ln(factor) * 0.1))
    pub fn attention_factor(&self) -> f32 {
        match &self.scaling {
            RopeScalingMethod::YaRn {
                factor,
                attn_factor,
                ..
            } => attn_factor.unwrap_or_else(|| {
                // Default: sqrt(0.1 * ln(factor) + 1)
                (0.1 * factor.ln() + 1.0).sqrt()
            }),
            _ => 1.0,
        }
    }

    /// Precompute all frequencies for the head dimension
    pub fn precompute_frequencies(&self) -> Vec<f32> {
        let n_pairs = self.head_dim / 2;
        (0..n_pairs).map(|i| self.frequency(i)).collect()
    }
}

/// Apply RoPE with scaling in-place across all heads
pub fn apply_rope_scaled_inplace(
    x: &mut [f32],
    config: &RopeConfig,
    n_heads: usize,
    position: usize,
) {
    let head_dim = config.head_dim;
    let freqs = config.precompute_frequencies();

    for h in 0..n_heads {
        let base = h * head_dim;
        for (i, &freq) in freqs.iter().enumerate() {
            let angle = position as f32 * freq;
            let (sin_val, cos_val) = angle.sin_cos();

            let idx0 = base + i * 2;
            let idx1 = idx0 + 1;
            if idx1 < x.len() {
                let x0 = x[idx0];
                let x1 = x[idx1];
                x[idx0] = x0 * cos_val - x1 * sin_val;
                x[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/// Find the correction dimension for YaRN ramp boundaries
/// Returns the dimension pair index where the RoPE wavelength equals
/// the boundary defined by beta and max_position.
///
/// Formula: dim = (n_dims * ln(max_pos / (beta * 2π))) / (2 * ln(base))
fn yarn_find_correction_dim(
    beta: f32,
    head_dim: usize,
    theta_base: f32,
    original_max_position: usize,
) -> f32 {
    let wavelength_ratio = original_max_position as f32 / (beta * 2.0 * PI);
    if wavelength_ratio <= 0.0 {
        return 0.0;
    }
    let ln_base = theta_base.ln();
    if ln_base.abs() < 1e-12 {
        return 0.0;
    }
    ((head_dim as f32 * wavelength_ratio.ln()) / (2.0 * ln_base)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_scaling_matches_original() {
        let config = RopeConfig::new(10000.0, 128);
        let freqs = config.precompute_frequencies();

        // Compare with original formula
        for i in 0..64 {
            let expected = 1.0 / 10000.0f32.powf((i * 2) as f32 / 128.0);
            assert!(
                (freqs[i] - expected).abs() < 1e-6,
                "pair {}: got {} expected {}",
                i,
                freqs[i],
                expected
            );
        }
    }

    #[test]
    fn test_linear_scaling_divides_frequency() {
        let config =
            RopeConfig::new(10000.0, 128).with_scaling(RopeScalingMethod::Linear { factor: 2.0 });
        let unscaled = RopeConfig::new(10000.0, 128).precompute_frequencies();
        let scaled = config.precompute_frequencies();

        for i in 0..64 {
            assert!(
                (scaled[i] - unscaled[i] / 2.0).abs() < 1e-6,
                "Linear scaling should halve frequencies"
            );
        }
    }

    #[test]
    fn test_ntk_aware_increases_theta() {
        let config =
            RopeConfig::new(10000.0, 128).with_scaling(RopeScalingMethod::NtkAware { factor: 2.0 });
        let theta = config.effective_theta_base();
        assert!(
            theta > 10000.0,
            "NTK-aware should increase theta base: {}",
            theta
        );
    }

    #[test]
    fn test_yarn_high_freq_unscaled() {
        // High-frequency dimensions should remain unscaled in YaRN
        let config = RopeConfig::new(10000.0, 128).with_scaling(RopeScalingMethod::YaRn {
            factor: 4.0,
            original_max_position: 4096,
            attn_factor: None,
            beta_fast: 32.0,
            beta_slow: 1.0,
        });
        let unscaled = RopeConfig::new(10000.0, 128).precompute_frequencies();
        let yarn_freqs = config.precompute_frequencies();

        // First few pairs (high frequency) should be very close to unscaled
        assert!(
            (yarn_freqs[0] - unscaled[0]).abs() < 1e-6,
            "Highest freq dim should be unscaled in YaRN"
        );
    }

    #[test]
    fn test_yarn_low_freq_scaled() {
        // Use a large factor and long original context so that the ramp region
        // ends well before the last dimension pair
        let factor = 16.0;
        let config = RopeConfig::new(10000.0, 128).with_scaling(RopeScalingMethod::YaRn {
            factor,
            original_max_position: 8192,
            attn_factor: None,
            beta_fast: 32.0,
            beta_slow: 1.0,
        });
        let unscaled = RopeConfig::new(10000.0, 128).precompute_frequencies();
        let yarn_freqs = config.precompute_frequencies();

        // The last pair (lowest frequency) should be fully interpolated
        let last = 63;
        let expected = unscaled[last] / factor;
        assert!(
            (yarn_freqs[last] - expected).abs() < 1e-6,
            "Lowest freq dim should be fully interpolated: got {} expected {}",
            yarn_freqs[last],
            expected
        );

        // Mid-range frequencies should be somewhere between unscaled and fully scaled
        // (in the ramp region or fully interpolated)
        for i in 0..64 {
            let f = yarn_freqs[i];
            let lo = unscaled[i] / factor;
            let hi = unscaled[i];
            assert!(
                f >= lo - 1e-6 && f <= hi + 1e-6,
                "pair {}: freq {} should be in [{}, {}]",
                i,
                f,
                lo,
                hi
            );
        }
    }

    #[test]
    fn test_yarn_attention_factor() {
        let config = RopeConfig::new(10000.0, 128).with_scaling(RopeScalingMethod::YaRn {
            factor: 4.0,
            original_max_position: 4096,
            attn_factor: None,
            beta_fast: 32.0,
            beta_slow: 1.0,
        });
        let af = config.attention_factor();
        assert!(af > 1.0, "YaRN attn factor should be > 1 for factor > 1");
        assert!(af < 2.0, "YaRN attn factor should be reasonable");
    }

    #[test]
    fn test_yarn_custom_attn_factor() {
        let config = RopeConfig::new(10000.0, 128).with_scaling(RopeScalingMethod::YaRn {
            factor: 4.0,
            original_max_position: 4096,
            attn_factor: Some(1.5),
            beta_fast: 32.0,
            beta_slow: 1.0,
        });
        assert!((config.attention_factor() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_rope_scaled_position_zero_identity() {
        let config = RopeConfig::new(10000.0, 4);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let orig = x.clone();
        apply_rope_scaled_inplace(&mut x, &config, 1, 0);
        for i in 0..4 {
            assert!(
                (x[i] - orig[i]).abs() < 1e-6,
                "Position 0 should be identity"
            );
        }
    }

    #[test]
    fn test_apply_rope_scaled_matches_unscaled() {
        // With no scaling, should match the original RoPE
        let config = RopeConfig::new(10000.0, 4);
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut b = a.clone();

        apply_rope_scaled_inplace(&mut a, &config, 2, 7);

        // Manual original RoPE
        let theta_base = 10000.0f32;
        for h in 0..2 {
            let base = h * 4;
            for i in (0..4).step_by(2) {
                let freq = 1.0 / theta_base.powf(i as f32 / 4.0);
                let angle = 7.0 * freq;
                let (sin_val, cos_val) = angle.sin_cos();
                let x0 = b[base + i];
                let x1 = b[base + i + 1];
                b[base + i] = x0 * cos_val - x1 * sin_val;
                b[base + i + 1] = x0 * sin_val + x1 * cos_val;
            }
        }

        for i in 0..8 {
            assert!(
                (a[i] - b[i]).abs() < 1e-5,
                "idx {}: {} vs {}",
                i,
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_ntk_position_consistency() {
        // NTK-scaled RoPE at position 0 should still be identity
        let config =
            RopeConfig::new(10000.0, 4).with_scaling(RopeScalingMethod::NtkAware { factor: 4.0 });
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let orig = x.clone();
        apply_rope_scaled_inplace(&mut x, &config, 1, 0);
        for i in 0..4 {
            assert!(
                (x[i] - orig[i]).abs() < 1e-6,
                "NTK at pos 0 should be identity"
            );
        }
    }

    #[test]
    fn test_no_scaling_attention_factor_is_one() {
        let config = RopeConfig::new(10000.0, 128);
        assert!((config.attention_factor() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_attention_factor_is_one() {
        let config =
            RopeConfig::new(10000.0, 128).with_scaling(RopeScalingMethod::Linear { factor: 4.0 });
        assert!((config.attention_factor() - 1.0).abs() < 1e-6);
    }
}
