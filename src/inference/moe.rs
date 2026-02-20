//! Mixture of Experts (MoE) — sparse expert routing for models like Mixtral
//!
//! MoE replaces the dense FFN with a gating network that selects K experts (out of N)
//! per token. This is ideal for SSD offloading: only activated experts need to be in memory.
//!
//! Architecture:
//!   gate(x) → top-K expert indices + weights
//!   output = Σ weight_i * Expert_i(x)   for i in top-K
//!
//! Each expert is a standard SwiGLU FFN block. The gating network is a linear projection
//! from hidden dim to n_experts, followed by softmax and top-k selection.

use crate::inference::feed_forward::feed_forward;
use crate::metal::compute::matvec_f32_simd;
use tracing::debug;

/// Configuration for a Mixture of Experts layer
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Total number of experts per layer
    pub n_experts: usize,
    /// Number of experts activated per token (top-K)
    pub n_experts_used: usize,
    /// Whether to normalize gating weights (default: true)
    pub norm_weights: bool,
}

impl MoeConfig {
    pub fn new(n_experts: usize, n_experts_used: usize) -> Self {
        Self {
            n_experts,
            n_experts_used,
            norm_weights: true,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> bool {
        self.n_experts > 0 && self.n_experts_used > 0 && self.n_experts_used <= self.n_experts
    }
}

/// Result of gating: which experts to activate and with what weights
#[derive(Debug, Clone)]
pub struct GatingResult {
    /// Indices of selected experts, sorted by weight descending
    pub expert_indices: Vec<usize>,
    /// Corresponding normalized weights
    pub expert_weights: Vec<f32>,
}

/// Compute top-K gating from gate logits
///
/// gate_weights: linear projection matrix (n_experts × n_embd)
/// x: input hidden state (n_embd)
/// Returns the top-K experts with normalized softmax weights
pub fn compute_gating(
    x: &[f32],
    gate_weights: &[f32],
    config: &MoeConfig,
    n_embd: usize,
) -> GatingResult {
    // Linear projection: gate_logits = gate_weights @ x → (n_experts,)
    let gate_logits = matvec_f32_simd(gate_weights, x, config.n_experts, n_embd);

    // Softmax over all experts
    let max_logit = gate_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exp_logits: Vec<f32> = gate_logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    if sum > 0.0 {
        for e in &mut exp_logits {
            *e /= sum;
        }
    }

    // Top-K selection
    let mut indexed: Vec<(usize, f32)> = exp_logits.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(config.n_experts_used);

    let mut expert_indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
    let mut expert_weights: Vec<f32> = indexed.iter().map(|(_, w)| *w).collect();

    // Re-normalize selected weights so they sum to 1.0
    if config.norm_weights {
        let wsum: f32 = expert_weights.iter().sum();
        if wsum > 0.0 {
            for w in &mut expert_weights {
                *w /= wsum;
            }
        }
    }

    // Sort by index for deterministic ordering (helpful for caching)
    let mut pairs: Vec<(usize, f32)> = expert_indices.into_iter().zip(expert_weights).collect();
    pairs.sort_by_key(|(idx, _)| *idx);
    expert_indices = pairs.iter().map(|(i, _)| *i).collect();
    expert_weights = pairs.iter().map(|(_, w)| *w).collect();

    debug!(
        "MoE gating: selected experts {:?} with weights {:?}",
        expert_indices, expert_weights
    );

    GatingResult {
        expert_indices,
        expert_weights,
    }
}

/// Expert weight set for a single expert's SwiGLU FFN
pub struct ExpertWeights<'a> {
    pub w_gate: &'a [f32],
    pub w_up: &'a [f32],
    pub w_down: &'a [f32],
}

/// Run MoE forward pass: gate → select experts → weighted sum of expert outputs
///
/// This is the main entry point for MoE layers, replacing the standard FFN.
/// Only the selected experts' weights are accessed, making this SSD-friendly:
/// inactive experts' memory pages remain on disk.
pub fn moe_forward(
    x: &[f32],
    gate_weights: &[f32],
    experts: &[ExpertWeights<'_>],
    config: &MoeConfig,
    n_embd: usize,
) -> Vec<f32> {
    assert!(config.validate(), "Invalid MoE configuration");
    assert_eq!(experts.len(), config.n_experts, "Expert count mismatch");

    // 1. Compute gating
    let gating = compute_gating(x, gate_weights, config, n_embd);

    // 2. Run selected experts and accumulate weighted output
    let mut output = vec![0.0f32; n_embd];

    for (&expert_idx, &weight) in gating
        .expert_indices
        .iter()
        .zip(gating.expert_weights.iter())
    {
        let expert = &experts[expert_idx];
        let expert_output = feed_forward(x, expert.w_gate, expert.w_up, expert.w_down, n_embd);

        // Weighted accumulation
        for (o, e) in output.iter_mut().zip(expert_output.iter()) {
            *o += weight * e;
        }
    }

    output
}

/// Determine which expert indices are needed for a batch of tokens.
/// Used for prefetching: we can precompute gating for all tokens in a batch,
/// then only load the union of needed experts from SSD.
pub fn batch_expert_selection(
    hidden_states: &[Vec<f32>],
    gate_weights: &[f32],
    config: &MoeConfig,
    n_embd: usize,
) -> Vec<usize> {
    let mut needed: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

    for x in hidden_states {
        let gating = compute_gating(x, gate_weights, config, n_embd);
        for idx in gating.expert_indices {
            needed.insert(idx);
        }
    }

    let result: Vec<usize> = needed.into_iter().collect();
    debug!(
        "Batch expert selection: {}/{} experts needed for {} tokens",
        result.len(),
        config.n_experts,
        hidden_states.len()
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_config_validation() {
        assert!(MoeConfig::new(8, 2).validate());
        assert!(MoeConfig::new(1, 1).validate());
        assert!(!MoeConfig::new(0, 0).validate());
        assert!(!MoeConfig::new(4, 5).validate());
    }

    #[test]
    fn test_gating_selects_top_k() {
        let n_embd = 4;
        let n_experts = 8;
        let config = MoeConfig::new(n_experts, 2);

        // Craft gate weights so expert 3 and 5 get highest scores
        let x = vec![1.0f32; n_embd];
        let mut gate_weights = vec![0.0f32; n_experts * n_embd];
        // Expert 3: high weight
        for j in 0..n_embd {
            gate_weights[3 * n_embd + j] = 2.0;
        }
        // Expert 5: second highest
        for j in 0..n_embd {
            gate_weights[5 * n_embd + j] = 1.5;
        }

        let result = compute_gating(&x, &gate_weights, &config, n_embd);
        assert_eq!(result.expert_indices.len(), 2);
        assert!(result.expert_indices.contains(&3));
        assert!(result.expert_indices.contains(&5));

        // Weights should be normalized to sum to 1.0
        let wsum: f32 = result.expert_weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-5, "Weights sum to {}", wsum);
    }

    #[test]
    fn test_gating_weights_normalized() {
        let n_embd = 4;
        let config = MoeConfig::new(4, 2);
        let x = vec![1.0f32; n_embd];
        let gate_weights = vec![0.1f32; 4 * n_embd];

        let result = compute_gating(&x, &gate_weights, &config, n_embd);
        let wsum: f32 = result.expert_weights.iter().sum();
        assert!(
            (wsum - 1.0).abs() < 1e-5,
            "Normalized weights should sum to 1.0, got {}",
            wsum
        );
    }

    #[test]
    fn test_moe_forward_shape() {
        let n_embd = 4;
        let n_ff = 8;
        let n_experts = 4;
        let config = MoeConfig::new(n_experts, 2);

        let x = vec![1.0f32; n_embd];
        let gate_weights = vec![0.1f32; n_experts * n_embd];

        let w_gate_data = vec![0.1f32; n_ff * n_embd];
        let w_up_data = vec![0.1f32; n_ff * n_embd];
        let w_down_data = vec![0.1f32; n_embd * n_ff];
        let experts: Vec<ExpertWeights> = (0..n_experts)
            .map(|_| ExpertWeights {
                w_gate: &w_gate_data,
                w_up: &w_up_data,
                w_down: &w_down_data,
            })
            .collect();

        let output = moe_forward(&x, &gate_weights, &experts, &config, n_embd);
        assert_eq!(output.len(), n_embd);
    }

    #[test]
    fn test_moe_forward_weighted_combination() {
        // With uniform experts, output should be same as single expert
        let n_embd = 4;
        let n_ff = 8;
        let n_experts = 2;
        let config = MoeConfig::new(n_experts, 2);

        let x = vec![1.0f32; n_embd];

        // Equal gate weights → each expert gets 0.5
        let gate_weights = vec![0.1f32; n_experts * n_embd];
        let w_gate = vec![0.1f32; n_ff * n_embd];
        let w_up = vec![0.1f32; n_ff * n_embd];
        let w_down = vec![0.1f32; n_embd * n_ff];

        let experts: Vec<ExpertWeights> = (0..n_experts)
            .map(|_| ExpertWeights {
                w_gate: &w_gate,
                w_up: &w_up,
                w_down: &w_down,
            })
            .collect();

        let moe_output = moe_forward(&x, &gate_weights, &experts, &config, n_embd);
        let single_output = feed_forward(&x, &w_gate, &w_up, &w_down, n_embd);

        // With 2 identical experts each weighted ~0.5, output ≈ single expert output
        for (m, s) in moe_output.iter().zip(single_output.iter()) {
            assert!(
                (m - s).abs() < 1e-4,
                "MoE output {} != single FFN output {}",
                m,
                s
            );
        }
    }

    #[test]
    fn test_batch_expert_selection() {
        let n_embd = 4;
        let n_experts = 8;
        let config = MoeConfig::new(n_experts, 2);

        let hidden_states = vec![
            vec![1.0f32; n_embd],
            vec![0.5f32; n_embd],
            vec![0.0f32; n_embd],
        ];
        let gate_weights = vec![0.1f32; n_experts * n_embd];

        let needed = batch_expert_selection(&hidden_states, &gate_weights, &config, n_embd);
        // Should have at most n_experts_used * n_tokens unique experts, capped at n_experts
        assert!(!needed.is_empty());
        assert!(needed.len() <= n_experts);
        // All indices should be valid
        for &idx in &needed {
            assert!(idx < n_experts);
        }
    }
}
