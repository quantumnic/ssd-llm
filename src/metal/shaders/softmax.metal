// Softmax Metal Compute Shader
#include <metal_stdlib>
using namespace metal;

// Per-row softmax: output[i] = exp(input[i] - max) / sum(exp(input - max))
kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (uint i = 0; i < n; i++) {
        max_val = max(max_val, input[i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < n; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < n; i++) {
        output[i] *= inv_sum;
    }
}
