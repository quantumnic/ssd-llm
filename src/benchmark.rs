//! Benchmarking SSD streaming and inference throughput

use crate::model::cache::LayerCache;
use crate::model::gguf::GgufFile;
use crate::ssd::streamer::SsdStreamer;
use anyhow::Result;
use std::path::Path;
use std::time::Instant;

pub fn run_benchmark(model_path: &Path, budget: usize) -> Result<()> {
    println!("=== ssd-llm Benchmark ===\n");

    // 1. GGUF parsing speed
    let start = Instant::now();
    let gguf = GgufFile::open(model_path)?;
    let parse_time = start.elapsed();
    println!("GGUF Parse: {:.2}ms", parse_time.as_secs_f64() * 1000.0);
    println!("  Tensors: {}", gguf.tensors.len());
    println!("  Layers: {}", gguf.n_layers());
    println!("  Embedding dim: {}", gguf.n_embd());
    println!();

    // 2. SSD streaming speed
    let streamer = SsdStreamer::new(model_path, budget)?;
    let n_layers = gguf.n_layers();

    if n_layers > 0 {
        // Benchmark: load first layer
        let start = Instant::now();
        let layer = streamer.load_layer(&gguf, 0)?;
        let first_load = start.elapsed();
        let layer_size_mb = layer.size_bytes as f64 / (1024.0 * 1024.0);
        let bandwidth = layer_size_mb / first_load.as_secs_f64();
        println!("Layer Load (cold, layer 0):");
        println!("  Size: {:.2} MB", layer_size_mb);
        println!("  Time: {:.2}ms", first_load.as_secs_f64() * 1000.0);
        println!("  Bandwidth: {:.0} MB/s", bandwidth);
        println!();

        // Benchmark: sequential layer loading with prefetch
        let mut cache = LayerCache::new(budget);
        let layers_to_bench = n_layers.min(10);
        let start = Instant::now();
        let mut total_bytes = 0u64;

        for i in 0..layers_to_bench {
            // Prefetch next
            if i + 1 < n_layers {
                streamer.prefetch_layer(&gguf, i + 1);
            }
            let layer = streamer.load_layer(&gguf, i)?;
            total_bytes += layer.size_bytes as u64;
            cache.insert(i, layer);
        }

        let seq_time = start.elapsed();
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        let avg_bandwidth = total_mb / seq_time.as_secs_f64();
        let avg_per_layer = seq_time.as_secs_f64() * 1000.0 / layers_to_bench as f64;

        println!("Sequential Load ({} layers with prefetch):", layers_to_bench);
        println!("  Total: {:.2} MB in {:.2}ms", total_mb, seq_time.as_secs_f64() * 1000.0);
        println!("  Avg per layer: {:.2}ms", avg_per_layer);
        println!("  Avg bandwidth: {:.0} MB/s", avg_bandwidth);
        println!();

        // Cache stats
        println!("Cache:");
        println!("  Budget: {:.2} GB", budget as f64 / (1024.0 * 1024.0 * 1024.0));
        println!("  Used: {:.2} MB", cache.used_bytes() as f64 / (1024.0 * 1024.0));
        println!("  Hits: {} | Misses: {} | Evictions: {}",
            cache.stats().hits, cache.stats().misses, cache.stats().evictions);
        println!();

        // Estimate full model throughput
        let total_model_mb = gguf.tensors.iter().map(|t| t.size_bytes).sum::<u64>() as f64 / (1024.0 * 1024.0);
        let est_full_time = total_model_mb / avg_bandwidth;
        println!("Estimated full forward pass (SSD → RAM):");
        println!("  Total model: {:.2} GB", total_model_mb / 1024.0);
        println!("  Est. time: {:.2}s at {:.0} MB/s", est_full_time, avg_bandwidth);
        println!("  Est. throughput: ~{:.1} tok/s (I/O bound estimate)", 1.0 / est_full_time);
    }

    println!("\n=== Benchmark Complete ===");
    Ok(())
}
