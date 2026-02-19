// Rotary Position Embedding (RoPE) Metal Compute Shaders
#include <metal_stdlib>
using namespace metal;

// Standard RoPE: apply rotary embeddings to Q/K tensors
// For each pair (x[2i], x[2i+1]):
//   x'[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
//   x'[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
// where θ = position * (1 / theta_base^(2i/d))
kernel void rope_f32(
    device float* x [[buffer(0)]],
    constant uint& head_dim [[buffer(1)]],
    constant uint& position [[buffer(2)]],
    constant float& theta_base [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint pair_idx = gid; // each thread handles one pair
    uint i = pair_idx * 2;

    if (i + 1 >= head_dim) return;

    float freq = 1.0f / pow(theta_base, float(i) / float(head_dim));
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    float x0 = x[i];
    float x1 = x[i + 1];

    x[i]     = x0 * cos_val - x1 * sin_val;
    x[i + 1] = x0 * sin_val + x1 * cos_val;
}

// RoPE with precomputed frequencies (supports all scaling methods)
// The host computes per-pair frequencies accounting for Linear/NTK/YaRN scaling,
// then passes them in a buffer. This is the most flexible approach.
kernel void rope_scaled_f32(
    device float* x [[buffer(0)]],
    constant uint& head_dim [[buffer(1)]],
    constant uint& position [[buffer(2)]],
    constant float* frequencies [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint pair_idx = gid;
    uint i = pair_idx * 2;

    if (i + 1 >= head_dim) return;

    float freq = frequencies[pair_idx];
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    float x0 = x[i];
    float x1 = x[i + 1];

    x[i]     = x0 * cos_val - x1 * sin_val;
    x[i + 1] = x0 * sin_val + x1 * cos_val;
}

// Batched RoPE: apply to multiple heads in a single dispatch
// x layout: [n_heads, head_dim] (contiguous)
kernel void rope_scaled_batch_f32(
    device float* x [[buffer(0)]],
    constant uint& head_dim [[buffer(1)]],
    constant uint& position [[buffer(2)]],
    constant float* frequencies [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_pairs = n_heads * (head_dim / 2);
    if (gid >= total_pairs) return;

    uint pair_in_head = gid % (head_dim / 2);
    uint head_idx = gid / (head_dim / 2);

    uint base = head_idx * head_dim;
    uint i = base + pair_in_head * 2;

    float freq = frequencies[pair_in_head];
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    float x0 = x[i];
    float x1 = x[i + 1];

    x[i]     = x0 * cos_val - x1 * sin_val;
    x[i + 1] = x0 * sin_val + x1 * cos_val;
}
