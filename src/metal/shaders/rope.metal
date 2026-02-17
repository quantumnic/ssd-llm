// Rotary Position Embedding (RoPE) Metal Compute Shader
#include <metal_stdlib>
using namespace metal;

// Apply RoPE to Q and K tensors
// For each pair (x[2i], x[2i+1]):
//   x'[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
//   x'[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
// where θ = position * (1 / 10000^(2i/d))
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
