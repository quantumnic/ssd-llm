#include <metal_stdlib>
using namespace metal;

/// CLIP Vision patch embedding — 2D convolution with stride=patch_size
///
/// Converts image patches into embeddings via a learned convolution kernel.
/// Input: image pixels in CHW format [3, image_size, image_size]
/// Output: patch embeddings [num_patches, hidden_size]
///
/// Each threadgroup processes one output patch.
/// Each thread within the group computes one output channel.
kernel void vision_patch_embed(
    device const float* image        [[buffer(0)]],  // [3, H, W] CHW
    device const float* weight       [[buffer(1)]],  // [out_ch, 3, ps, ps]
    device const float* bias         [[buffer(2)]],  // [out_ch]
    device float* output             [[buffer(3)]],  // [num_patches, out_ch]
    constant uint& image_size        [[buffer(4)]],
    constant uint& patch_size        [[buffer(5)]],
    constant uint& hidden_size       [[buffer(6)]],
    uint2 gid                        [[thread_position_in_grid]]
)
{
    uint patch_idx = gid.y;  // which patch
    uint out_ch = gid.x;     // which output channel

    if (out_ch >= hidden_size) return;

    uint patches_per_side = image_size / patch_size;
    uint total_patches = patches_per_side * patches_per_side;
    if (patch_idx >= total_patches) return;

    uint py = patch_idx / patches_per_side;
    uint px = patch_idx % patches_per_side;
    uint ps = patch_size;

    float sum = bias[out_ch];

    for (uint in_c = 0; in_c < 3; in_c++) {
        for (uint ky = 0; ky < ps; ky++) {
            for (uint kx = 0; kx < ps; kx++) {
                uint iy = py * ps + ky;
                uint ix = px * ps + kx;
                float pixel = image[in_c * image_size * image_size + iy * image_size + ix];
                uint kernel_idx = out_ch * 3 * ps * ps + in_c * ps * ps + ky * ps + kx;
                sum += pixel * weight[kernel_idx];
            }
        }
    }

    output[patch_idx * hidden_size + out_ch] = sum;
}

/// Vision LayerNorm — standard layer normalization with weight and bias
kernel void vision_layer_norm(
    device float* x                  [[buffer(0)]],  // [num_tokens, hidden_size] (in-place)
    device const float* weight       [[buffer(1)]],  // [hidden_size]
    device const float* bias_buf     [[buffer(2)]],  // [hidden_size]
    constant uint& hidden_size       [[buffer(3)]],
    constant float& eps              [[buffer(4)]],
    uint gid                         [[thread_position_in_grid]]  // token index
)
{
    uint offset = gid * hidden_size;

    // Compute mean
    float mean = 0.0;
    for (uint i = 0; i < hidden_size; i++) {
        mean += x[offset + i];
    }
    mean /= float(hidden_size);

    // Compute variance
    float var = 0.0;
    for (uint i = 0; i < hidden_size; i++) {
        float d = x[offset + i] - mean;
        var += d * d;
    }
    var /= float(hidden_size);

    float inv_std = 1.0 / sqrt(var + eps);

    // Normalize
    for (uint i = 0; i < hidden_size; i++) {
        x[offset + i] = (x[offset + i] - mean) * inv_std * weight[i] + bias_buf[i];
    }
}
