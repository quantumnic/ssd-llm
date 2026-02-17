// Matrix Multiplication Metal Compute Shader
// ssd-llm v0.2 — GPU-accelerated matmul for Apple Silicon

#include <metal_stdlib>
using namespace metal;

// Tiled matrix multiplication: C = A × B
// A: (M, K), B: (K, N), C: (M, N)
kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Matrix-vector multiplication: y = W × x
// W: (out_dim, in_dim), x: (in_dim), y: (out_dim)
kernel void matvec_f32(
    device const float* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& out_dim [[buffer(3)]],
    constant uint& in_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= out_dim) return;

    float sum = 0.0f;
    for (uint j = 0; j < in_dim; j++) {
        sum += W[gid * in_dim + j] * x[j];
    }
    y[gid] = sum;
}

// Tiled matmul with shared memory (optimized version)
constant uint TILE_SIZE = 16;

kernel void matmul_f32_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float sum = 0.0f;

    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        uint ak = t * TILE_SIZE + tid.x;
        uint bk = t * TILE_SIZE + tid.y;

        As[tid.y][tid.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        Bs[tid.y][tid.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
