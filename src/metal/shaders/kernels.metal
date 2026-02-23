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
// Quantized Q2_K matrix-vector multiply (dequantize on-the-fly)
// K-quant block (256 elements):
//   f16 d (2B) + f16 dmin (2B) + 16B scales/mins + 64B qs (2-bit quants) = 84B
// ============================================================
kernel void matvec_q2_k(
    device const uchar* W_quantized  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device float* y                  [[buffer(2)]],
    constant uint& out_dim           [[buffer(3)]],
    constant uint& in_dim            [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint block_size = 256;
    uint block_bytes = 84; // 2+2+16+64
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        ushort d_bits = ushort(W_quantized[boff]) | (ushort(W_quantized[boff + 1]) << 8);
        ushort dmin_bits = ushort(W_quantized[boff + 2]) | (ushort(W_quantized[boff + 3]) << 8);
        float d = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));

        device const uchar* sc = W_quantized + boff + 4;   // 16 bytes: scales/mins
        device const uchar* qs = W_quantized + boff + 20;  // 64 bytes: 2-bit quants

        uint x_base = b * block_size;

        // 16 sub-blocks of 16 elements each
        for (uint sb = 0; sb < 16; sb++) {
            float scale = d * float(sc[sb] & 0x0F);
            float min_val = dmin * float((sc[sb] >> 4) & 0x0F);

            for (uint j = 0; j < 16; j++) {
                uint idx = sb * 16 + j;
                uchar qs_byte = qs[idx / 4];
                uint shift = (idx % 4) * 2;
                float q = float((qs_byte >> shift) & 0x03);
                sum += (scale * q - min_val) * x[x_base + idx];
            }
        }
    }

    y[tid] = sum;
}

// ============================================================
// Quantized Q8_K matrix-vector multiply (dequantize on-the-fly)
// K-quant block (256 elements):
//   f32 d (4B) + 256B qs (int8) + 32B bsums = 292B
// ============================================================
kernel void matvec_q8_k(
    device const uchar* W_quantized  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device float* y                  [[buffer(2)]],
    constant uint& out_dim           [[buffer(3)]],
    constant uint& in_dim            [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint block_size = 256;
    uint block_bytes = 292; // 4+256+32
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        // f32 scale (little-endian)
        uint d_bits = uint(W_quantized[boff]) | (uint(W_quantized[boff + 1]) << 8) |
                      (uint(W_quantized[boff + 2]) << 16) | (uint(W_quantized[boff + 3]) << 24);
        float d = as_type<float>(d_bits);

        uint x_base = b * block_size;
        for (uint j = 0; j < 256; j++) {
            float val = float(as_type<char>(W_quantized[boff + 4 + j])) * d;
            sum += val * x[x_base + j];
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

// IQ4_NL non-linear lookup table
constant char iq4nl_lut[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

// IQ4_NL dequant matvec: 32 elements/block, 18 bytes (f16 d + 16B qs)
kernel void matvec_iq4_nl(
    device const uchar* W_quantized [[buffer(0)]],
    device const float* x           [[buffer(1)]],
    device float* y                 [[buffer(2)]],
    device const uint* out_dim_buf  [[buffer(3)]],
    device const uint* in_dim_buf   [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint out_dim = out_dim_buf[0];
    uint in_dim  = in_dim_buf[0];
    if (tid >= out_dim) return;

    uint block_size = 32;
    uint block_bytes = 18;
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        ushort d_bits = ushort(W_quantized[boff]) | (ushort(W_quantized[boff + 1]) << 8);
        float d = float(as_type<half>(d_bits));

        uint x_base = b * block_size;
        for (uint j = 0; j < 16; j++) {
            uchar byte = W_quantized[boff + 2 + j];
            uchar lo = byte & 0x0F;
            uchar hi = (byte >> 4) & 0x0F;
            sum += d * float(iq4nl_lut[lo]) * x[x_base + 2 * j];
            sum += d * float(iq4nl_lut[hi]) * x[x_base + 2 * j + 1];
        }
    }

    y[tid] = sum;
}

// IQ4_XS dequant matvec: 256 elements/block, 148 bytes
// Layout: f16 d (2B) + u16 scales_h (2B) + 8x u16 scales_l (16B) + 128B qs
kernel void matvec_iq4_xs(
    device const uchar* W_quantized [[buffer(0)]],
    device const float* x           [[buffer(1)]],
    device float* y                 [[buffer(2)]],
    device const uint* out_dim_buf  [[buffer(3)]],
    device const uint* in_dim_buf   [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint out_dim = out_dim_buf[0];
    uint in_dim  = in_dim_buf[0];
    if (tid >= out_dim) return;

    uint block_size = 256;
    uint block_bytes = 148;
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        ushort d_bits = ushort(W_quantized[boff]) | (ushort(W_quantized[boff + 1]) << 8);
        float d = float(as_type<half>(d_bits));

        ushort scales_h = ushort(W_quantized[boff + 2]) | (ushort(W_quantized[boff + 3]) << 8);

        for (uint sb = 0; sb < 8; sb++) {
            // Extract 6-bit scale: 4 low bits from scales_l nibbles, 2 high bits from scales_h
            uint sl_idx = sb / 2;
            ushort sl_word = ushort(W_quantized[boff + 4 + sl_idx * 2]) | (ushort(W_quantized[boff + 5 + sl_idx * 2]) << 8);
            int sl;
            if (sb % 2 == 0) {
                sl = int(sl_word & 0x0F);
            } else {
                sl = int((sl_word >> 4) & 0x0F);
            }
            int sh = int((scales_h >> (2 * sb)) & 0x03);
            float scale = float((sl | (sh << 4)) - 32);

            uint qs_off = boff + 20 + sb * 16;
            uint x_base = b * block_size + sb * 32;

            for (uint j = 0; j < 16; j++) {
                uchar byte = W_quantized[qs_off + j];
                uchar lo = byte & 0x0F;
                uchar hi = (byte >> 4) & 0x0F;
                sum += d * scale * float(iq4nl_lut[lo]) * x[x_base + 2 * j];
                sum += d * scale * float(iq4nl_lut[hi]) * x[x_base + 2 * j + 1];
            }
        }
    }

    y[tid] = sum;
}

// ============================================================
// IQ3_XXS grid lookup table (256 entries)
// ============================================================
constant uint iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04
};

// Sign lookup table for IQ2/IQ3 formats
constant uchar ksigns_iq2xs_tbl[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
};

constant uchar kmask_iq2xs_tbl[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };

// IQ3_XXS dequant matvec: 256 elements/block, 98 bytes
// Layout: f16 d (2B) + qs[96] (64B grid indices + 32B scales_and_signs)
kernel void matvec_iq3_xxs(
    device const uchar* W_quantized [[buffer(0)]],
    device const float* x           [[buffer(1)]],
    device float* y                 [[buffer(2)]],
    device const uint* out_dim_buf  [[buffer(3)]],
    device const uint* in_dim_buf   [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint out_dim = out_dim_buf[0];
    uint in_dim  = in_dim_buf[0];
    if (tid >= out_dim) return;

    uint block_size = 256;
    uint block_bytes = 98;
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        ushort d_bits = ushort(W_quantized[boff]) | (ushort(W_quantized[boff + 1]) << 8);
        float d = float(as_type<half>(d_bits));

        uint qs_off = boff + 2;
        uint ss_off = qs_off + 64;

        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint aux32 = uint(W_quantized[ss_off + ib32*4])
                       | (uint(W_quantized[ss_off + ib32*4+1]) << 8)
                       | (uint(W_quantized[ss_off + ib32*4+2]) << 16)
                       | (uint(W_quantized[ss_off + ib32*4+3]) << 24);

            float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
            uint x_base = b * block_size + ib32 * 32;

            for (uint l = 0; l < 4; l++) {
                uchar signs = ksigns_iq2xs_tbl[(aux32 >> (7*l)) & 127];
                uint grid_idx0 = W_quantized[qs_off + ib32*8 + 2*l];
                uint grid_idx1 = W_quantized[qs_off + ib32*8 + 2*l + 1];
                uint g1 = iq3xxs_grid[grid_idx0];
                uint g2 = iq3xxs_grid[grid_idx1];

                for (uint j = 0; j < 4; j++) {
                    float v1 = float((g1 >> (8*j)) & 0xFF);
                    float v2 = float((g2 >> (8*j)) & 0xFF);
                    float s1 = (signs & kmask_iq2xs_tbl[j]) ? -1.0f : 1.0f;
                    float s2 = (signs & kmask_iq2xs_tbl[j+4]) ? -1.0f : 1.0f;
                    sum += db * v1 * s1 * x[x_base + l*8 + j];
                    sum += db * v2 * s2 * x[x_base + l*8 + j + 4];
                }
            }
        }
    }

    y[tid] = sum;
}

// ============================================================
// IQ3_S grid lookup table (512 entries)
// ============================================================
constant uint iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
};

// IQ3_S dequant matvec: 256 elements/block, 110 bytes
// Layout: f16 d (2B) + qs[64] + qh[8] + signs[32] + scales[4]
kernel void matvec_iq3_s(
    device const uchar* W_quantized [[buffer(0)]],
    device const float* x           [[buffer(1)]],
    device float* y                 [[buffer(2)]],
    device const uint* out_dim_buf  [[buffer(3)]],
    device const uint* in_dim_buf   [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint out_dim = out_dim_buf[0];
    uint in_dim  = in_dim_buf[0];
    if (tid >= out_dim) return;

    uint block_size = 256;
    uint block_bytes = 110;
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        ushort d_bits = ushort(W_quantized[boff]) | (ushort(W_quantized[boff + 1]) << 8);
        float d = float(as_type<half>(d_bits));

        uint qs_base = boff + 2;
        uint qh_base = qs_base + 64;
        uint signs_base = qh_base + 8;
        uint scales_base = signs_base + 32;

        for (uint ib32 = 0; ib32 < 8; ib32 += 2) {
            uchar scale_byte = W_quantized[scales_base + ib32/2];
            float db1 = d * float(1 + 2 * int(scale_byte & 0x0f));
            float db2 = d * float(1 + 2 * int(scale_byte >> 4));

            // First group of 32
            uchar qh0 = W_quantized[qh_base + ib32];
            for (uint l = 0; l < 4; l++) {
                uint gi0 = uint(W_quantized[qs_base + ib32*8 + 2*l])     | ((uint(qh0) << (8 - 2*l)) & 256u);
                uint gi1 = uint(W_quantized[qs_base + ib32*8 + 2*l + 1]) | ((uint(qh0) << (7 - 2*l)) & 256u);
                uint g1 = iq3s_grid[gi0];
                uint g2 = iq3s_grid[gi1];
                uchar sign_byte = W_quantized[signs_base + ib32*4 + l];
                uint x_off = b * block_size + ib32 * 32 + l * 8;
                for (uint j = 0; j < 4; j++) {
                    float v1 = float((g1 >> (8*j)) & 0xFF);
                    float v2 = float((g2 >> (8*j)) & 0xFF);
                    float s1 = (sign_byte & kmask_iq2xs_tbl[j]) ? -1.0f : 1.0f;
                    float s2 = (sign_byte & kmask_iq2xs_tbl[j+4]) ? -1.0f : 1.0f;
                    sum += db1 * v1 * s1 * x[x_off + j];
                    sum += db1 * v2 * s2 * x[x_off + j + 4];
                }
            }

            // Second group of 32
            uchar qh1 = W_quantized[qh_base + ib32 + 1];
            for (uint l = 0; l < 4; l++) {
                uint gi0 = uint(W_quantized[qs_base + (ib32+1)*8 + 2*l])     | ((uint(qh1) << (8 - 2*l)) & 256u);
                uint gi1 = uint(W_quantized[qs_base + (ib32+1)*8 + 2*l + 1]) | ((uint(qh1) << (7 - 2*l)) & 256u);
                uint g1 = iq3s_grid[gi0];
                uint g2 = iq3s_grid[gi1];
                uchar sign_byte = W_quantized[signs_base + (ib32+1)*4 + l];
                uint x_off = b * block_size + (ib32 + 1) * 32 + l * 8;
                for (uint j = 0; j < 4; j++) {
                    float v1 = float((g1 >> (8*j)) & 0xFF);
                    float v2 = float((g2 >> (8*j)) & 0xFF);
                    float s1 = (sign_byte & kmask_iq2xs_tbl[j]) ? -1.0f : 1.0f;
                    float s2 = (sign_byte & kmask_iq2xs_tbl[j+4]) ? -1.0f : 1.0f;
                    sum += db2 * v1 * s1 * x[x_off + j];
                    sum += db2 * v2 * s2 * x[x_off + j + 4];
                }
            }
        }
    }

    y[tid] = sum;
}

// ============================================================
// IQ2_XXS grid lookup table (256 entries of uint64)
// Each entry encodes 8 weight values. Values are {0x08, 0x19, 0x2b}.
// ============================================================
constant ulong iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908
};

// IQ2_XXS dequant matvec: 256 elements/block, 66 bytes
// Layout: f16 d (2B) + uint16_t qs[32] (64B)
kernel void matvec_iq2_xxs(
    device const uchar* W_quantized [[buffer(0)]],
    device const float* x           [[buffer(1)]],
    device float* y                 [[buffer(2)]],
    device const uint* out_dim_buf  [[buffer(3)]],
    device const uint* in_dim_buf   [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint out_dim = out_dim_buf[0];
    uint in_dim  = in_dim_buf[0];
    if (tid >= out_dim) return;

    uint block_size = 256;
    uint block_bytes = 66;
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        ushort d_bits = ushort(W_quantized[boff]) | (ushort(W_quantized[boff + 1]) << 8);
        float d = float(as_type<half>(d_bits));

        uint qs_off = boff + 2;

        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint g_off = qs_off + ib32 * 8;
            // Read 2 x u32
            uint aux32_0 = uint(W_quantized[g_off])
                         | (uint(W_quantized[g_off+1]) << 8)
                         | (uint(W_quantized[g_off+2]) << 16)
                         | (uint(W_quantized[g_off+3]) << 24);
            uint aux32_1 = uint(W_quantized[g_off+4])
                         | (uint(W_quantized[g_off+5]) << 8)
                         | (uint(W_quantized[g_off+6]) << 16)
                         | (uint(W_quantized[g_off+7]) << 24);

            float db = d * (0.5f + float(aux32_1 >> 28)) * 0.25f;
            uint x_base = b * block_size + ib32 * 32;

            for (uint l = 0; l < 4; l++) {
                uint grid_idx = (aux32_0 >> (8*l)) & 0xFF;
                ulong grid = iq2xxs_grid[grid_idx];
                uchar signs = ksigns_iq2xs_tbl[(aux32_1 >> (7*l)) & 127];

                for (uint j = 0; j < 8; j++) {
                    float v = float((grid >> (8*j)) & 0xFF);
                    float s = (signs & kmask_iq2xs_tbl[j]) ? -1.0f : 1.0f;
                    sum += db * v * s * x[x_base + l*8 + j];
                }
            }
        }
    }

    y[tid] = sum;
}

// ============================================================
// IQ2_XS grid lookup table (512 entries of uint64)
// ============================================================
constant ulong iq2xs_grid[512] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b
};

// IQ2_XS dequant matvec: 256 elements/block, 74 bytes
// Layout: f16 d (2B) + uint16_t qs[32] (64B) + uint8_t scales[8] (8B)
kernel void matvec_iq2_xs(
    device const uchar* W_quantized [[buffer(0)]],
    device const float* x           [[buffer(1)]],
    device float* y                 [[buffer(2)]],
    device const uint* out_dim_buf  [[buffer(3)]],
    device const uint* in_dim_buf   [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint out_dim = out_dim_buf[0];
    uint in_dim  = in_dim_buf[0];
    if (tid >= out_dim) return;

    uint block_size = 256;
    uint block_bytes = 74;
    uint blocks_per_row = in_dim / block_size;
    uint row_offset = tid * blocks_per_row * block_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint boff = row_offset + b * block_bytes;

        ushort d_bits = ushort(W_quantized[boff]) | (ushort(W_quantized[boff + 1]) << 8);
        float d = float(as_type<half>(d_bits));

        uint qs_off = boff + 2;
        uint scales_off = qs_off + 64;

        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uchar scale_byte = W_quantized[scales_off + ib32 / 2];
            uint scale_nibble = (ib32 % 2 == 0) ? (scale_byte & 0x0f) : (scale_byte >> 4);
            float db = d * (0.5f + float(scale_nibble)) * 0.25f;
            uint x_base = b * block_size + ib32 * 32;

            for (uint l = 0; l < 4; l++) {
                uint q_off = qs_off + (ib32 * 4 + l) * 2;
                ushort qs_val = ushort(W_quantized[q_off]) | (ushort(W_quantized[q_off+1]) << 8);
                uint grid_idx = qs_val & 511;
                uint sign_idx = qs_val >> 9;
                ulong grid = iq2xs_grid[grid_idx];
                uchar signs = ksigns_iq2xs_tbl[sign_idx];

                for (uint j = 0; j < 8; j++) {
                    float v = float((grid >> (8*j)) & 0xFF);
                    float s = (signs & kmask_iq2xs_tbl[j]) ? -1.0f : 1.0f;
                    sum += db * v * s * x[x_base + l*8 + j];
                }
            }
        }
    }

    y[tid] = sum;
}

// BF16 matvec: each weight is 2 bytes (brain float 16)
// BF16 = upper 16 bits of f32, so we shift left by 16 to get f32
kernel void matvec_bf16(
    device const uchar* W_quantized  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device float* y                  [[buffer(2)]],
    constant uint& out_dim           [[buffer(3)]],
    constant uint& in_dim            [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint row_offset = tid * in_dim * 2;
    float sum = 0.0;

    for (uint col = 0; col < in_dim; col++) {
        uint off = row_offset + col * 2;
        ushort bits = ushort(W_quantized[off]) | (ushort(W_quantized[off + 1]) << 8);
        float w = as_type<float>(uint(bits) << 16);
        sum += w * x[col];
    }

    y[tid] = sum;
}

// F16 matvec: each weight is 2 bytes (IEEE 754 half precision)
kernel void matvec_f16(
    device const uchar* W_quantized  [[buffer(0)]],
    device const float* x            [[buffer(1)]],
    device float* y                  [[buffer(2)]],
    constant uint& out_dim           [[buffer(3)]],
    constant uint& in_dim            [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint row_offset = tid * in_dim * 2;
    float sum = 0.0;

    for (uint col = 0; col < in_dim; col++) {
        uint off = row_offset + col * 2;
        ushort bits = ushort(W_quantized[off]) | (ushort(W_quantized[off + 1]) << 8);
        float w = float(as_type<half>(bits));
        sum += w * x[col];
    }

    y[tid] = sum;
}

// ============================================================
// Flash Attention — Fused QK scoring + online softmax + V accumulation
// One thread per query head. Iterates over all KV positions in a single
// pass using the online softmax algorithm (Milakov & Gimelshein, 2018).
//
// Memory layout:
//   q_heads:   [n_head × head_dim]  — pre-projected, RoPE'd query vectors
//   k_cache:   [seq_len × n_head_kv × head_dim] — key cache (row-major)
//   v_cache:   [seq_len × n_head_kv × head_dim] — value cache (row-major)
//   output:    [n_head × head_dim]  — attention output per head
//   params:    {n_head, n_head_kv, head_dim, seq_len, window_start, window_end}
//
// Supports GQA: multiple Q heads share one KV head (kv_group_size = n_head/n_head_kv).
// head_dim must be ≤ 256 (covers all practical LLMs: 64, 80, 96, 128).
// ============================================================
kernel void flash_attention_f32(
    device const float* q_heads  [[buffer(0)]],
    device const float* k_cache  [[buffer(1)]],
    device const float* v_cache  [[buffer(2)]],
    device float* output         [[buffer(3)]],
    constant uint* params        [[buffer(4)]],
    uint tid                     [[thread_position_in_grid]]
) {
    const uint n_head      = params[0];
    const uint n_head_kv   = params[1];
    const uint head_dim    = params[2];
    const uint seq_len     = params[3];
    const uint window_start = params[4];
    const uint window_end  = params[5];

    if (tid >= n_head) return;

    const uint h = tid;
    const uint kv_group_size = max(n_head / n_head_kv, 1u);
    const uint kv_h = h / kv_group_size;
    const float scale = rsqrt(float(head_dim));

    float running_max = -INFINITY;
    float running_sum = 0.0;

    // Per-head output accumulator (head_dim ≤ 256)
    float acc[256];
    for (uint d = 0; d < head_dim && d < 256; d++) {
        acc[d] = 0.0;
    }

    const uint q_off = h * head_dim;
    const uint eff_end = min(window_end, seq_len);
    const uint hd4 = (head_dim / 4) * 4;

    for (uint pos = window_start; pos < eff_end; pos++) {
        // Q · K dot product with 4-wide unrolling
        float dot = 0.0;
        const uint k_off = pos * n_head_kv * head_dim + kv_h * head_dim;
        for (uint d = 0; d < hd4; d += 4) {
            dot += q_heads[q_off + d]     * k_cache[k_off + d]
                 + q_heads[q_off + d + 1] * k_cache[k_off + d + 1]
                 + q_heads[q_off + d + 2] * k_cache[k_off + d + 2]
                 + q_heads[q_off + d + 3] * k_cache[k_off + d + 3];
        }
        for (uint d = hd4; d < head_dim; d++) {
            dot += q_heads[q_off + d] * k_cache[k_off + d];
        }
        const float score = dot * scale;

        // Online softmax update
        const uint v_off = pos * n_head_kv * head_dim + kv_h * head_dim;
        if (score > running_max) {
            const float correction = exp(running_max - score);
            running_sum *= correction;
            for (uint d = 0; d < head_dim && d < 256; d++) {
                acc[d] *= correction;
            }
            running_max = score;
        }

        const float weight = exp(score - running_max);
        running_sum += weight;

        for (uint d = 0; d < head_dim && d < 256; d++) {
            acc[d] += weight * v_cache[v_off + d];
        }
    }

    // Normalize and write output
    const uint out_off = h * head_dim;
    if (running_sum > 0.0) {
        const float inv_sum = 1.0 / running_sum;
        for (uint d = 0; d < head_dim && d < 256; d++) {
            output[out_off + d] = acc[d] * inv_sum;
        }
    } else {
        for (uint d = 0; d < head_dim && d < 256; d++) {
            output[out_off + d] = 0.0;
        }
    }
}

// ============================================================
// Fused SwiGLU: out[row] = silu(dot(w_gate[row], x)) * dot(w_up[row], x)
// Each thread computes one output element of the intermediate vector.
// Fuses gate_proj matvec + silu + up_proj matvec + element-wise multiply
// into a single dispatch, halving memory round-trips.
// ============================================================
kernel void fused_swiglu_f32(
    device const float* w_gate   [[buffer(0)]],  // [n_ff, n_embd]
    device const float* w_up     [[buffer(1)]],  // [n_ff, n_embd]
    device const float* x        [[buffer(2)]],  // [n_embd]
    device float* out            [[buffer(3)]],  // [n_ff]
    constant uint& n_ff          [[buffer(4)]],
    constant uint& n_embd        [[buffer(5)]],
    uint tid                     [[thread_position_in_grid]]
) {
    if (tid >= n_ff) return;

    const uint row_off = tid * n_embd;
    float gate_dot = 0.0;
    float up_dot = 0.0;

    // Fused dot products: read x once, accumulate both projections
    for (uint i = 0; i < n_embd; i++) {
        float xi = x[i];
        gate_dot += w_gate[row_off + i] * xi;
        up_dot   += w_up[row_off + i]   * xi;
    }

    // SiLU(gate) * up
    float silu_gate = gate_dot / (1.0 + exp(-gate_dot));
    out[tid] = silu_gate * up_dot;
}
