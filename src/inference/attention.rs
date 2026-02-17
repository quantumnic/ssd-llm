//! Multi-Head Attention implementation (CPU, f32)

/// Compute multi-head attention (simplified, single-token)
/// 
/// This is a basic implementation for v0.1. It computes:
///   Q = x @ Wq, K = x @ Wk, V = x @ Wv
///   attn = softmax(Q @ K^T / sqrt(d_k)) @ V
///   output = attn @ Wo
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

    // Q = x @ Wq
    let q = matmul_vec(x, wq, n_head * head_dim);
    // K = x @ Wk (potentially fewer KV heads with GQA)
    let k = matmul_vec(x, wk, n_head_kv * head_dim);
    // V = x @ Wv
    let v = matmul_vec(x, wv, n_head_kv * head_dim);

    // Apply RoPE (Rotary Position Embedding)
    let q = apply_rope(&q, head_dim, position);
    let k = apply_rope(&k, head_dim, position);

    // For single-token inference (no KV cache yet in v0.1),
    // attention simplifies: attn_score = softmax(q·k / sqrt(d)) → just 1 value per head
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut attn_output = vec![0.0f32; n_embd];

    // Process each head
    let kv_group_size = n_head / n_head_kv.max(1);
    for h in 0..n_head {
        let kv_h = h / kv_group_size.max(1);
        let q_start = h * head_dim;
        let k_start = kv_h * head_dim;
        let v_start = kv_h * head_dim;

        // dot(q, k) * scale
        let mut score = 0.0f32;
        for d in 0..head_dim {
            score += q.get(q_start + d).copied().unwrap_or(0.0)
                   * k.get(k_start + d).copied().unwrap_or(0.0);
        }
        score *= scale;

        // softmax of single value = 1.0, so attn_weight = 1.0
        // output for this head = V
        for d in 0..head_dim {
            let out_idx = h * head_dim + d;
            if out_idx < attn_output.len() {
                attn_output[out_idx] = v.get(v_start + d).copied().unwrap_or(0.0);
            }
        }
    }

    // Output projection: attn_output @ Wo
    matmul_vec(&attn_output, wo, n_embd)
}

/// Simple matrix-vector multiply: x (input_dim) @ W (output_dim × input_dim) → out (output_dim)
fn matmul_vec(x: &[f32], w: &[f32], output_dim: usize) -> Vec<f32> {
    let input_dim = x.len();
    let mut output = vec![0.0f32; output_dim];

    for i in 0..output_dim.min(w.len() / input_dim.max(1)) {
        let mut sum = 0.0f32;
        let row_start = i * input_dim;
        for j in 0..input_dim {
            if row_start + j < w.len() {
                sum += w[row_start + j] * x[j];
            }
        }
        output[i] = sum;
    }

    output
}

/// Apply Rotary Position Embedding (RoPE)
fn apply_rope(x: &[f32], head_dim: usize, position: usize) -> Vec<f32> {
    let mut output = x.to_vec();
    let n_heads = x.len() / head_dim;

    for h in 0..n_heads {
        let base = h * head_dim;
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / 10000.0f32.powf(i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let idx0 = base + i;
            let idx1 = base + i + 1;
            if idx1 < output.len() {
                let x0 = output[idx0];
                let x1 = output[idx1];
                output[idx0] = x0 * cos_val - x1 * sin_val;
                output[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    output
}
