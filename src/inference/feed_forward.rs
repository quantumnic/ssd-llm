//! Feed-Forward Network (SwiGLU variant used in LLaMA)

/// SwiGLU Feed-Forward Network
/// 
/// output = down_proj(silu(gate_proj(x)) * up_proj(x))
pub fn feed_forward(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    n_embd: usize,
) -> Vec<f32> {
    // Infer hidden dim from weight size
    let hidden_dim = w_gate.len() / n_embd.max(1);
    if hidden_dim == 0 {
        return x.to_vec();
    }

    // gate = x @ W_gate
    let gate = matmul_vec(x, w_gate, hidden_dim);
    // up = x @ W_up
    let up = matmul_vec(x, w_up, hidden_dim);

    // SwiGLU activation: silu(gate) * up
    let mut hidden = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        let silu = gate[i] * sigmoid(gate[i]); // SiLU = x * sigmoid(x)
        hidden[i] = silu * up[i];
    }

    // down = hidden @ W_down
    matmul_vec(&hidden, w_down, n_embd)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

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
