//
// moe_gate.metal — Top-K gating kernel for Mixture of Experts
//
// Computes gate_logits = gate_weights @ x, then finds top-K experts
// with softmax-normalized weights.
//
// This runs entirely on GPU to avoid a round-trip for gating decisions.
//

#include <metal_stdlib>
using namespace metal;

/// Compute gate logits via matrix-vector multiply: out[i] = dot(gate[i*n_embd..], x)
kernel void moe_gate_logits(
    device const float* gate_weights [[buffer(0)]],  // n_experts × n_embd
    device const float* x            [[buffer(1)]],   // n_embd
    device float* logits             [[buffer(2)]],   // n_experts
    constant uint& n_embd            [[buffer(3)]],
    constant uint& n_experts         [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]])
{
    if (tid >= n_experts) return;

    float sum = 0.0f;
    uint base = tid * n_embd;
    for (uint j = 0; j < n_embd; j += 4) {
        // Process 4 elements at a time for better throughput
        if (j + 4 <= n_embd) {
            float4 g = float4(gate_weights[base + j],
                              gate_weights[base + j + 1],
                              gate_weights[base + j + 2],
                              gate_weights[base + j + 3]);
            float4 v = float4(x[j], x[j + 1], x[j + 2], x[j + 3]);
            sum += dot(g, v);
        } else {
            for (uint k = j; k < n_embd; k++) {
                sum += gate_weights[base + k] * x[k];
            }
        }
    }
    logits[tid] = sum;
}

/// Softmax + top-K selection (single-threaded kernel for small n_experts)
/// Writes top-K indices and normalized weights to output buffers.
kernel void moe_topk_softmax(
    device float* logits             [[buffer(0)]],   // n_experts (in-place)
    device uint* top_indices         [[buffer(1)]],   // n_experts_used
    device float* top_weights        [[buffer(2)]],   // n_experts_used
    constant uint& n_experts         [[buffer(3)]],
    constant uint& n_experts_used    [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]])
{
    if (tid != 0) return;  // single-thread kernel

    // Find max for numerical stability
    float max_val = logits[0];
    for (uint i = 1; i < n_experts; i++) {
        max_val = max(max_val, logits[i]);
    }

    // Softmax
    float sum = 0.0f;
    for (uint i = 0; i < n_experts; i++) {
        logits[i] = exp(logits[i] - max_val);
        sum += logits[i];
    }
    for (uint i = 0; i < n_experts; i++) {
        logits[i] /= sum;
    }

    // Greedy top-K selection
    for (uint k = 0; k < n_experts_used; k++) {
        float best = -1.0f;
        uint best_idx = 0;
        for (uint i = 0; i < n_experts; i++) {
            // Check if already selected
            bool used = false;
            for (uint j = 0; j < k; j++) {
                if (top_indices[j] == i) { used = true; break; }
            }
            if (!used && logits[i] > best) {
                best = logits[i];
                best_idx = i;
            }
        }
        top_indices[k] = best_idx;
        top_weights[k] = best;
    }

    // Re-normalize selected weights
    float wsum = 0.0f;
    for (uint k = 0; k < n_experts_used; k++) {
        wsum += top_weights[k];
    }
    if (wsum > 0.0f) {
        for (uint k = 0; k < n_experts_used; k++) {
            top_weights[k] /= wsum;
        }
    }
}
