//! SwiGLU Feed-Forward Network (as used in LLaMA)
//!
//! FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))

use crate::metal::compute::{matvec_f32_simd, silu_f32};

/// Compute SwiGLU feed-forward block
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
}
