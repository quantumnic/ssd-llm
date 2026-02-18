//! Tensor parallelism for splitting matmul across multiple GPU command queues
//!
//! Apple Silicon has a unified GPU with multiple shader cores. By dispatching
//! independent matmul shards to separate command queues, we can utilize more
//! of the GPU simultaneously. This is especially beneficial for large vocab
//! projections and attention weight multiplications.
//!
//! Strategy: Column-parallel for output projection, row-parallel for FFN.
//! Each shard computes a portion of the output, results are concatenated.

use tracing::{debug, info};

/// Configuration for tensor parallelism
#[derive(Clone, Debug)]
pub struct TensorParallelConfig {
    /// Number of parallel shards (should match available GPU compute units)
    pub n_shards: usize,
    /// Minimum matrix dimension to trigger parallel dispatch
    pub min_parallel_dim: usize,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            n_shards: 2,
            min_parallel_dim: 2048,
        }
    }
}

/// Parallel matrix-vector multiply: y = W × x
/// Splits W into n_shards row-groups, computes each independently, concatenates.
///
/// W shape: (out_dim, in_dim), x shape: (in_dim,), y shape: (out_dim,)
pub fn parallel_matvec(
    w: &[f32],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    config: &TensorParallelConfig,
) -> Vec<f32> {
    // Don't parallelize small matrices
    if out_dim < config.min_parallel_dim || config.n_shards <= 1 {
        return crate::metal::compute::matvec_f32_simd(w, x, out_dim, in_dim);
    }

    let n_shards = config.n_shards.min(out_dim);
    let shard_size = out_dim / n_shards;
    let remainder = out_dim % n_shards;

    // Compute shards — in a real implementation these would dispatch to
    // separate Metal command queues. For now, we use Rust's rayon-style
    // parallelism via std threads.
    let mut handles = Vec::with_capacity(n_shards);
    let x_arc = std::sync::Arc::new(x.to_vec());
    let w_arc = std::sync::Arc::new(w.to_vec());

    let mut offset = 0;
    for shard_idx in 0..n_shards {
        let rows = if shard_idx < remainder { shard_size + 1 } else { shard_size };
        let start_row = offset;
        offset += rows;

        let x_ref = x_arc.clone();
        let w_ref = w_arc.clone();

        handles.push(std::thread::spawn(move || {
            let mut result = Vec::with_capacity(rows);
            for r in start_row..(start_row + rows) {
                let w_offset = r * in_dim;
                let mut sum = 0.0f32;

                // 4-wide SIMD-style accumulation
                let chunks = in_dim / 4;
                let mut acc0 = 0.0f32;
                let mut acc1 = 0.0f32;
                let mut acc2 = 0.0f32;
                let mut acc3 = 0.0f32;

                for c in 0..chunks {
                    let i = c * 4;
                    acc0 += w_ref[w_offset + i] * x_ref[i];
                    acc1 += w_ref[w_offset + i + 1] * x_ref[i + 1];
                    acc2 += w_ref[w_offset + i + 2] * x_ref[i + 2];
                    acc3 += w_ref[w_offset + i + 3] * x_ref[i + 3];
                }
                sum = acc0 + acc1 + acc2 + acc3;

                for i in (chunks * 4)..in_dim {
                    sum += w_ref[w_offset + i] * x_ref[i];
                }
                result.push(sum);
            }
            (start_row, result)
        }));
    }

    // Collect and concatenate
    let mut output = vec![0.0f32; out_dim];
    for handle in handles {
        let (start, shard_result) = handle.join().unwrap();
        output[start..start + shard_result.len()].copy_from_slice(&shard_result);
    }

    output
}

/// Parallel FFN: splits the gate/up projection across shards
/// For SwiGLU FFN: out = down @ (silu(gate @ x) * (up @ x))
pub fn parallel_ffn(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    n_embd: usize,
    config: &TensorParallelConfig,
) -> Vec<f32> {
    let hidden_dim = w_gate.len() / n_embd;

    if hidden_dim < config.min_parallel_dim || config.n_shards <= 1 {
        return crate::inference::feed_forward::feed_forward(x, w_gate, w_up, w_down, n_embd);
    }

    // Parallel gate and up projections
    let gate_proj = parallel_matvec(w_gate, x, hidden_dim, n_embd, config);
    let up_proj = parallel_matvec(w_up, x, hidden_dim, n_embd, config);

    // SiLU(gate) * up — element-wise, single thread is fine
    let mut intermediate: Vec<f32> = gate_proj.iter()
        .zip(up_proj.iter())
        .map(|(&g, &u)| {
            let silu_g = g * (1.0 / (1.0 + (-g).exp()));
            silu_g * u
        })
        .collect();

    // Down projection (hidden_dim -> n_embd)
    parallel_matvec(w_down, &intermediate, n_embd, hidden_dim, config)
}

/// Estimate optimal shard count based on model dimensions
pub fn auto_detect_shards(n_embd: usize) -> usize {
    // Heuristic: larger models benefit from more shards
    // Apple M1/M2 has 7-10 GPU cores, M1 Pro/Max has 14-32
    if n_embd >= 8192 {
        4 // 70B+ models
    } else if n_embd >= 4096 {
        2 // 7B-13B models
    } else {
        1 // Small models, overhead not worth it
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_matvec_correctness() {
        // 4x3 matrix × 3-vector
        let w = vec![
            1.0, 2.0, 3.0,  // row 0
            4.0, 5.0, 6.0,  // row 1
            7.0, 8.0, 9.0,  // row 2
            10.0, 11.0, 12.0, // row 3
        ];
        let x = vec![1.0, 1.0, 1.0];

        let config = TensorParallelConfig { n_shards: 2, min_parallel_dim: 1 };
        let result = parallel_matvec(&w, &x, 4, 3, &config);

        assert_eq!(result.len(), 4);
        assert!((result[0] - 6.0).abs() < 1e-5);
        assert!((result[1] - 15.0).abs() < 1e-5);
        assert!((result[2] - 24.0).abs() < 1e-5);
        assert!((result[3] - 33.0).abs() < 1e-5);
    }

    #[test]
    fn test_parallel_matvec_single_shard_fallback() {
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![3.0, 7.0];
        let config = TensorParallelConfig { n_shards: 1, min_parallel_dim: 1 };
        let result = parallel_matvec(&w, &x, 2, 2, &config);
        assert!((result[0] - 3.0).abs() < 1e-5);
        assert!((result[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_auto_detect_shards() {
        assert_eq!(auto_detect_shards(512), 1);
        assert_eq!(auto_detect_shards(4096), 2);
        assert_eq!(auto_detect_shards(8192), 4);
    }

    #[test]
    fn test_parallel_matvec_large() {
        // Test with larger matrix to actually exercise parallelism
        let dim = 64;
        let w: Vec<f32> = (0..dim * dim).map(|i| (i % 7) as f32 * 0.1).collect();
        let x: Vec<f32> = (0..dim).map(|i| (i % 5) as f32 * 0.2).collect();

        let config_1 = TensorParallelConfig { n_shards: 1, min_parallel_dim: 1 };
        let config_4 = TensorParallelConfig { n_shards: 4, min_parallel_dim: 1 };

        let result_1 = parallel_matvec(&w, &x, dim, dim, &config_1);
        let result_4 = parallel_matvec(&w, &x, dim, dim, &config_4);

        for i in 0..dim {
            assert!((result_1[i] - result_4[i]).abs() < 1e-3,
                "Mismatch at {}: {} vs {}", i, result_1[i], result_4[i]);
        }
    }
}
