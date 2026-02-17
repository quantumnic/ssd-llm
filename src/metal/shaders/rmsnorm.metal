// RMS Normalization Metal Compute Shader
#include <metal_stdlib>
using namespace metal;

// RMS Norm: output[i] = (input[i] / rms) * weight[i]
// rms = sqrt(mean(x^2) + eps)
kernel void rmsnorm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    // Calculate sum of squares (single thread for now, TODO: parallel reduction)
    float sum_sq = 0.0f;
    for (uint i = 0; i < n; i++) {
        sum_sq += input[i] * input[i];
    }

    float rms = sqrt(sum_sq / float(n) + eps);
    float inv_rms = 1.0f / rms;

    output[gid] = input[gid] * inv_rms * weight[gid];
}
