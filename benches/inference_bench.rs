//! Criterion benchmarks for ssd-llm core operations
//!
//! Run with: cargo bench
//! Note: These benchmark pure computation; they don't require a model file.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Benchmark softmax computation (mirrors metal::compute softmax)
fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for size in [64, 256, 1024, 4096] {
        let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 2.0).collect();

        group.bench_with_input(BenchmarkId::new("cpu", size), &input, |b, input| {
            b.iter(|| {
                let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut output: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
                let sum: f32 = output.iter().sum();
                for x in &mut output {
                    *x /= sum;
                }
                black_box(&output);
            });
        });
    }

    group.finish();
}

/// Benchmark RMS normalization
fn bench_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm");

    for dim in [512, 2048, 4096, 8192] {
        let input: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.1).sin()).collect();
        let weights: Vec<f32> = vec![1.0; dim];
        let eps = 1e-5f32;

        group.bench_with_input(BenchmarkId::new("cpu", dim), &dim, |b, _| {
            b.iter(|| {
                let ss: f32 = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
                let rms = (ss + eps).sqrt();
                let output: Vec<f32> = input
                    .iter()
                    .zip(weights.iter())
                    .map(|(x, w)| (x / rms) * w)
                    .collect();
                black_box(&output);
            });
        });
    }

    group.finish();
}

/// Benchmark matrix-vector multiplication (core of transformer forward pass)
fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec");

    for (rows, cols) in [(512, 512), (2048, 512), (4096, 4096)] {
        let matrix: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let vector: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.01).cos()).collect();

        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    let mut output = vec![0.0f32; rows];
                    for r in 0..rows {
                        let mut sum = 0.0f32;
                        let row_start = r * cols;
                        for c in 0..cols {
                            sum += matrix[row_start + c] * vector[c];
                        }
                        output[r] = sum;
                    }
                    black_box(&output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RoPE (Rotary Position Embedding) computation
fn bench_rope(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope");

    for head_dim in [64, 128] {
        let mut q: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
        let position = 42usize;
        let base = 10000.0f32;

        group.bench_with_input(BenchmarkId::new("cpu", head_dim), &head_dim, |b, _| {
            b.iter(|| {
                let mut result = q.clone();
                for i in (0..head_dim).step_by(2) {
                    let freq = 1.0 / base.powf(i as f32 / head_dim as f32) * position as f32;
                    let cos = freq.cos();
                    let sin = freq.sin();
                    let x0 = result[i];
                    let x1 = result[i + 1];
                    result[i] = x0 * cos - x1 * sin;
                    result[i + 1] = x0 * sin + x1 * cos;
                }
                black_box(&result);
            });
        });
    }

    group.finish();
}

/// Benchmark SiLU activation (used in SwiGLU FFN)
fn bench_silu(c: &mut Criterion) {
    let mut group = c.benchmark_group("silu");

    for size in [1024, 4096, 11008] {
        let input: Vec<f32> = (0..size).map(|i| ((i as f32) * 0.01) - 5.0).collect();

        group.bench_with_input(BenchmarkId::new("cpu", size), &size, |b, _| {
            b.iter(|| {
                let output: Vec<f32> = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
                black_box(&output);
            });
        });
    }

    group.finish();
}

/// Benchmark dot product (used in attention score computation)
fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for dim in [64, 128, 256] {
        let a: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.1).sin()).collect();
        let b_vec: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.1).cos()).collect();

        group.bench_with_input(BenchmarkId::new("naive", dim), &dim, |b, _| {
            b.iter(|| {
                let dot: f32 = a.iter().zip(b_vec.iter()).map(|(x, y)| x * y).sum();
                black_box(dot);
            });
        });

        group.bench_with_input(BenchmarkId::new("4wide", dim), &dim, |b, _| {
            b.iter(|| {
                let mut acc = [0.0f32; 4];
                let chunks = dim / 4;
                for i in 0..chunks {
                    let base = i * 4;
                    acc[0] += a[base] * b_vec[base];
                    acc[1] += a[base + 1] * b_vec[base + 1];
                    acc[2] += a[base + 2] * b_vec[base + 2];
                    acc[3] += a[base + 3] * b_vec[base + 3];
                }
                let mut dot = acc[0] + acc[1] + acc[2] + acc[3];
                for i in (chunks * 4)..dim {
                    dot += a[i] * b_vec[i];
                }
                black_box(dot);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_softmax,
    bench_rmsnorm,
    bench_matvec,
    bench_rope,
    bench_silu,
    bench_dot_product,
);
criterion_main!(benches);
