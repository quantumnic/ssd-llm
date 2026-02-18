//  Metal Compute Shaders for ssd-llm
//  GPU-accelerated kernels for transformer inference on Apple Silicon

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Matrix-Vector Multiply: y = W × x
// W: (out_dim, in_dim), x: (in_dim,), y: (out_dim,)
// Each thread computes one output element
// ============================================================
kernel void matvec_f32(
    device const float* W       [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float* y             [[buffer(2)]],
    constant uint& out_dim      [[buffer(3)]],
    constant uint& in_dim       [[buffer(4)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    float sum = 0.0;
    uint row_offset = tid * in_dim;

    // Process 4 elements at a time for better throughput
    uint i = 0;
    for (; i + 3 < in_dim; i += 4) {
        sum += W[row_offset + i]     * x[i]
             + W[row_offset + i + 1] * x[i + 1]
             + W[row_offset + i + 2] * x[i + 2]
             + W[row_offset + i + 3] * x[i + 3];
    }
    for (; i < in_dim; i++) {
        sum += W[row_offset + i] * x[i];
    }

    y[tid] = sum;
}

// ============================================================
// RMS Normalization
// Phase 1: Compute sum of squares (parallel reduction)
// Phase 2: Normalize with weight
// ============================================================
kernel void rmsnorm_sumsq(
    device const float* x       [[buffer(0)]],
    device float* partial_sums  [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]],
    uint tcount                 [[threads_per_grid]]
) {
    float sum = 0.0;
    for (uint i = tid; i < n; i += tcount) {
        sum += x[i] * x[i];
    }
    partial_sums[tid] = sum;
}

kernel void rmsnorm_normalize(
    device float* x             [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    constant float& inv_rms     [[buffer(2)]],
    constant uint& n            [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    x[tid] = x[tid] * inv_rms * weight[tid];
}

// ============================================================
// Softmax (numerically stable)
// Phase 1: Find max (reduction)
// Phase 2: Compute exp(x - max) and sum
// Phase 3: Normalize by sum
// ============================================================
kernel void softmax_exp(
    device float* x             [[buffer(0)]],
    constant float& max_val     [[buffer(1)]],
    device float* partial_sums  [[buffer(2)]],
    constant uint& n            [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]],
    uint tcount                 [[threads_per_grid]]
) {
    float sum = 0.0;
    for (uint i = tid; i < n; i += tcount) {
        float val = exp(x[i] - max_val);
        x[i] = val;
        sum += val;
    }
    partial_sums[tid] = sum;
}

kernel void softmax_normalize(
    device float* x             [[buffer(0)]],
    constant float& inv_sum     [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    x[tid] *= inv_sum;
}

// ============================================================
// RoPE (Rotary Position Embedding)
// Applied in-place to pairs of elements
// ============================================================
kernel void rope_f32(
    device float* x             [[buffer(0)]],
    constant uint& head_dim     [[buffer(1)]],
    constant uint& n_heads      [[buffer(2)]],
    constant uint& position     [[buffer(3)]],
    constant float& theta_base  [[buffer(4)]],
    uint tid                    [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint total_pairs = n_heads * half_dim;
    if (tid >= total_pairs) return;

    uint head = tid / half_dim;
    uint pair = tid % half_dim;
    uint base_idx = head * head_dim + pair * 2;

    float freq = 1.0 / pow(theta_base, float(pair * 2) / float(head_dim));
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    float x0 = x[base_idx];
    float x1 = x[base_idx + 1];
    x[base_idx]     = x0 * cos_val - x1 * sin_val;
    x[base_idx + 1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================
// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================
kernel void silu_f32(
    device float* x             [[buffer(0)]],
    constant uint& n            [[buffer(1)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float val = x[tid];
    x[tid] = val / (1.0 + exp(-val));
}

// ============================================================
// Element-wise multiply: out = a * b
// ============================================================
kernel void elementwise_mul_f32(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float* out           [[buffer(2)]],
    constant uint& n            [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    out[tid] = a[tid] * b[tid];
}

// ============================================================
// Quantized Q4_0 matrix-vector multiply (dequantize on-the-fly)
// Block layout: f16 scale (2 bytes) + 16 bytes (32 nibbles)
// Each thread processes one output row
// ============================================================
kernel void matvec_q4_0(
    device const uchar* W_quantized  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device float* y                  [[buffer(2)]],
    constant uint& out_dim           [[buffer(3)]],
    constant uint& in_dim            [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint block_size = 32;
    uint block_bytes = 18; // 2 (f16 scale) + 16 (nibbles)
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        // Read f16 scale
        ushort scale_bits = ushort(W_quantized[boff]) | (ushort(W_quantized[boff + 1]) << 8);
        float scale = float(as_type<half>(scale_bits));

        // Dequantize 32 values and dot product with x
        uint x_base = b * block_size;
        for (uint j = 0; j < 16; j++) {
            uchar byte = W_quantized[boff + 2 + j];
            float lo = float(int(byte & 0x0F) - 8) * scale;
            float hi = float(int((byte >> 4) & 0x0F) - 8) * scale;
            sum += lo * x[x_base + j * 2] + hi * x[x_base + j * 2 + 1];
        }
    }

    y[tid] = sum;
}

// ============================================================
// Quantized Q8_0 matrix-vector multiply (dequantize on-the-fly)
// Block layout: f16 scale (2 bytes) + 32 x int8
// ============================================================
kernel void matvec_q8_0(
    device const uchar* W_quantized  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device float* y                  [[buffer(2)]],
    constant uint& out_dim           [[buffer(3)]],
    constant uint& in_dim            [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint block_size = 32;
    uint block_bytes = 34; // 2 (f16 scale) + 32 (int8)
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        ushort scale_bits = ushort(W_quantized[boff]) | (ushort(W_quantized[boff + 1]) << 8);
        float scale = float(as_type<half>(scale_bits));

        uint x_base = b * block_size;
        for (uint j = 0; j < 32; j++) {
            float val = float(as_type<char>(W_quantized[boff + 2 + j])) * scale;
            sum += val * x[x_base + j];
        }
    }

    y[tid] = sum;
}
