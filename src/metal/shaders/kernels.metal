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
// ============================================================
// Quantized Q4_K matrix-vector multiply (dequantize on-the-fly)
// K-quant block (256 elements):
//   f16 d (2B) + f16 dmin (2B) + 12B scales/mins + 128B nibbles = 144B
// Super-block: 256 values split into 8 sub-blocks of 32
// Each sub-block has a 6-bit scale and 6-bit min packed in 12 bytes
// ============================================================
kernel void matvec_q4_k(
    device const uchar* W_quantized  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device float* y                  [[buffer(2)]],
    constant uint& out_dim           [[buffer(3)]],
    constant uint& in_dim            [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint block_size = 256;
    uint block_bytes = 144; // 2+2+12+128
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        // Read super-block scale and min (f16)
        ushort d_bits = ushort(W_quantized[boff]) | (ushort(W_quantized[boff + 1]) << 8);
        ushort dmin_bits = ushort(W_quantized[boff + 2]) | (ushort(W_quantized[boff + 3]) << 8);
        float d = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));

        // 12 bytes of packed scales and mins for 8 sub-blocks
        // First 4 bytes: low 6 bits of scales (sub-blocks 0-7, packed into nibble-pairs)
        // Layout: scales[0..3] hold low 6 bits, scales[4..7] hold low 6 bits of mins
        // scales[8..11] hold high 2 bits
        device const uchar* sc = W_quantized + boff + 4;
        device const uchar* qs = W_quantized + boff + 16; // 4+12=16

        uint x_base = b * block_size;

        for (uint sb = 0; sb < 8; sb++) {
            // Extract 6-bit scale and min for this sub-block
            uchar sc_low, m_low;
            if (sb < 4) {
                sc_low = sc[sb] & 0x3F;
                m_low  = sc[sb + 4] & 0x3F;
            } else {
                sc_low = (sc[sb - 4] >> 6) | ((sc[sb + 4] & 0xF) << 2);
                m_low  = (sc[sb] >> 6)     | ((sc[sb + 4] >> 4) << 2);
            }

            float scale = d * float(sc_low);
            float min_val = dmin * float(m_low);

            // 32 values packed as 16 bytes of nibbles
            uint qs_off = sb * 16;
            for (uint j = 0; j < 16; j++) {
                uchar byte_val = qs[qs_off + j];
                float v0 = scale * float(byte_val & 0x0F) - min_val;
                float v1 = scale * float((byte_val >> 4) & 0x0F) - min_val;
                uint xi = x_base + sb * 32 + j * 2;
                sum += v0 * x[xi] + v1 * x[xi + 1];
            }
        }
    }

    y[tid] = sum;
}

// ============================================================
// Quantized Q6_K matrix-vector multiply (dequantize on-the-fly)
// K-quant block (256 elements):
//   128B ql (low 4 bits) + 64B qh (high 2 bits) + 16B scales (int8) + f16 d = 210B
// ============================================================
kernel void matvec_q6_k(
    device const uchar* W_quantized  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device float* y                  [[buffer(2)]],
    constant uint& out_dim           [[buffer(3)]],
    constant uint& in_dim            [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint block_size = 256;
    uint block_bytes = 210; // 128+64+16+2
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        device const uchar* ql = W_quantized + boff;           // 128 bytes: low 4 bits
        device const uchar* qh = W_quantized + boff + 128;     // 64 bytes: high 2 bits
        device const char* scales = (device const char*)(W_quantized + boff + 192); // 16 x int8
        ushort d_bits = ushort(W_quantized[boff + 208]) | (ushort(W_quantized[boff + 209]) << 8);
        float d = float(as_type<half>(d_bits));

        uint x_base = b * block_size;

        // 16 sub-blocks of 16 values each
        for (uint sb = 0; sb < 16; sb++) {
            float sc = d * float(scales[sb]);
            for (uint j = 0; j < 16; j++) {
                uint idx = sb * 16 + j;
                // Low 4 bits from ql (packed as nibble pairs in 128 bytes)
                uchar ql_byte = ql[idx / 2];
                uint low4 = (idx & 1) ? ((ql_byte >> 4) & 0x0F) : (ql_byte & 0x0F);
                // High 2 bits from qh (packed 4 per byte in 64 bytes)
                uchar qh_byte = qh[idx / 4];
                uint shift = (idx % 4) * 2;
                uint high2 = (qh_byte >> shift) & 0x03;
                // 6-bit value, centered at 32
                int q = int((high2 << 4) | low4) - 32;
                sum += sc * float(q) * x[x_base + idx];
            }
        }
    }

    y[tid] = sum;
}

// ============================================================
// Quantized Q3_K matrix-vector multiply (dequantize on-the-fly)
// K-quant block (256 elements):
//   32B hmask (high bit) + 64B qs (2-bit quants) + 12B scales + f16 d = 110B
// ============================================================
kernel void matvec_q3_k(
    device const uchar* W_quantized  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device float* y                  [[buffer(2)]],
    constant uint& out_dim           [[buffer(3)]],
    constant uint& in_dim            [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint block_size = 256;
    uint block_bytes = 110; // 32+64+12+2
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        device const uchar* hmask = W_quantized + boff;         // 32 bytes: high bit mask
        device const uchar* qs = W_quantized + boff + 32;       // 64 bytes: low 2 bits packed
        device const uchar* sc = W_quantized + boff + 96;       // 12 bytes: scales
        ushort d_bits = ushort(W_quantized[boff + 108]) | (ushort(W_quantized[boff + 109]) << 8);
        float d = float(as_type<half>(d_bits));

        uint x_base = b * block_size;

        // 16 sub-blocks of 16 values each
        for (uint sb = 0; sb < 16; sb++) {
            // Extract 6-bit signed scale
            int scale_val;
            if (sb < 8) {
                scale_val = int(sc[sb / 2] >> ((sb & 1) * 4)) & 0x0F;
                // High 2 bits from bytes 8-11
                uint hi_idx = sb / 2;
                uint hi_shift = (sb & 1) * 4;
                int hi = (int(sc[8 + sb / 4] >> ((sb % 4) * 2)) & 0x03) << 4;
                scale_val = scale_val | hi;
            } else {
                uint si = sb - 8;
                scale_val = (int(sc[si / 2] >> ((si & 1) * 4 + 0)) >> 0) & 0x0F;
                int hi = (int(sc[8 + sb / 4] >> ((sb % 4) * 2)) & 0x03) << 4;
                scale_val = scale_val | hi;
            }
            // Sign-extend from 6 bits
            if (scale_val >= 32) scale_val -= 64;
            float sc_f = d * float(scale_val);

            for (uint j = 0; j < 16; j++) {
                uint idx = sb * 16 + j;
                // Low 2 bits from qs (4 per byte)
                uchar qs_byte = qs[idx / 4];
                uint shift = (idx % 4) * 2;
                uint low2 = (qs_byte >> shift) & 0x03;
                // High bit from hmask
                uint hbit = (hmask[idx / 8] >> (idx % 8)) & 1;
                // 3-bit value centered at 4
                int q = int(low2 | (hbit << 2)) - 4;
                sum += sc_f * float(q) * x[x_base + idx];
            }
        }
    }

    y[tid] = sum;
}

// ============================================================
// Quantized Q5_K matrix-vector multiply (dequantize on-the-fly)
// K-quant block (256 elements):
//   f16 d (2B) + f16 dmin (2B) + 12B scales + 32B qh (high bit) + 128B ql (low 4 bits) = 176B
// ============================================================
kernel void matvec_q5_k(
    device const uchar* W_quantized  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device float* y                  [[buffer(2)]],
    constant uint& out_dim           [[buffer(3)]],
    constant uint& in_dim            [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint block_size = 256;
    uint block_bytes = 176; // 2+2+12+32+128
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        ushort d_bits = ushort(W_quantized[boff]) | (ushort(W_quantized[boff + 1]) << 8);
        ushort dmin_bits = ushort(W_quantized[boff + 2]) | (ushort(W_quantized[boff + 3]) << 8);
        float d_val = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));

        device const uchar* sc = W_quantized + boff + 4;   // 12 bytes: packed scales/mins
        device const uchar* qh = W_quantized + boff + 16;  // 32 bytes: high bit
        device const uchar* ql = W_quantized + boff + 48;  // 128 bytes: low 4 bits as nibbles

        uint x_base = b * block_size;

        for (uint sb = 0; sb < 8; sb++) {
            // Extract 6-bit scale and min (same packing as Q4_K)
            uchar sc_low, m_low;
            if (sb < 4) {
                sc_low = sc[sb] & 0x3F;
                m_low  = sc[sb + 4] & 0x3F;
            } else {
                sc_low = (sc[sb - 4] >> 6) | ((sc[sb + 4] & 0xF) << 2);
                m_low  = (sc[sb] >> 6)     | ((sc[sb + 4] >> 4) << 2);
            }

            float scale = d_val * float(sc_low);
            float min_val = dmin * float(m_low);

            // 32 values: low 4 bits in ql (16 bytes of nibbles), high bit in qh
            uint qs_off = sb * 16;
            for (uint j = 0; j < 16; j++) {
                uint idx0 = sb * 32 + j * 2;
                uint idx1 = idx0 + 1;

                uchar byte_val = ql[qs_off + j];
                uint low0 = byte_val & 0x0F;
                uint low1 = (byte_val >> 4) & 0x0F;

                // High bit from qh
                uint hbit0 = (qh[idx0 / 8] >> (idx0 % 8)) & 1;
                uint hbit1 = (qh[idx1 / 8] >> (idx1 % 8)) & 1;

                // 5-bit value
                float v0 = scale * float(low0 | (hbit0 << 4)) - min_val;
                float v1 = scale * float(low1 | (hbit1 << 4)) - min_val;

                sum += v0 * x[x_base + idx0] + v1 * x[x_base + idx1];
            }
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
