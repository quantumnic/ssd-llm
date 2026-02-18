mod model;
mod inference;
mod ssd;
mod api;
mod benchmark;
mod metal;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "ssd-llm", version, about = "Run 70B+ LLMs on Apple Silicon via SSD streaming")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a GGUF model
    Run {
        /// Path to GGUF model file
        model: PathBuf,

        /// Memory budget for layer cache (e.g. "8G", "4G", "512M")
        #[arg(long, default_value = "8G")]
        memory_budget: String,

        /// Prompt text
        #[arg(long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,

        /// Temperature for sampling
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,

        /// Top-K sampling
        #[arg(long, default_value_t = 40)]
        top_k: usize,

        /// Top-P (nucleus) sampling
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        /// Path to draft model for speculative decoding (smaller GGUF)
        #[arg(long)]
        draft_model: Option<PathBuf>,

        /// Number of tokens to draft per speculation step
        #[arg(long, default_value_t = 5)]
        draft_ahead: usize,
    },
    /// Show model info from GGUF file
    Info {
        /// Path to GGUF model file
        model: PathBuf,
    },
    /// Benchmark SSD streaming and inference throughput
    Bench {
        /// Path to GGUF model file
        model: PathBuf,

        /// Memory budget
        #[arg(long, default_value = "8G")]
        memory_budget: String,
    },
    /// Start Ollama-compatible API server
    Serve {
        /// Path to GGUF model file
        model: PathBuf,

        /// Memory budget for layer cache
        #[arg(long, default_value = "8G")]
        memory_budget: String,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(long, default_value_t = 11434)]
        port: u16,

        /// Path to draft model for speculative decoding
        #[arg(long)]
        draft_model: Option<PathBuf>,

        /// Number of tokens to draft per speculation step
        #[arg(long, default_value_t = 5)]
        draft_ahead: usize,
    },
}

fn parse_memory_budget(s: &str) -> Result<usize> {
    let s = s.trim().to_uppercase();
    if let Some(num) = s.strip_suffix('G') {
        Ok(num.parse::<usize>()? * 1024 * 1024 * 1024)
    } else if let Some(num) = s.strip_suffix('M') {
        Ok(num.parse::<usize>()? * 1024 * 1024)
    } else {
        Ok(s.parse::<usize>()?)
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    // Check Metal availability on startup
    if metal::compute::MetalCompute::is_available() {
        if let Some(mc) = metal::compute::MetalCompute::new() {
            if mc.has_gpu() {
                println!("✓ Metal GPU acceleration active (metal-rs)");
            } else {
                println!("✓ Metal GPU available (CPU SIMD)");
            }
        } else {
            println!("✓ Metal GPU available (CPU SIMD fallback)");
        }
    } else {
        println!("⚠ Metal not available, using CPU only");
    }

    match cli.command {
        Commands::Info { model } => {
            let gguf = model::gguf::GgufFile::open(&model)?;
            println!("=== GGUF Model Info ===");
            println!("Magic: GGUF v{}", gguf.header.version);
            println!("Architecture: {}", gguf.architecture());
            println!("Tensors: {}", gguf.header.tensor_count);
            println!("Metadata entries: {}", gguf.header.metadata_kv_count);
            println!("File size: {:.2} GB", gguf.file_size as f64 / (1024.0 * 1024.0 * 1024.0));
            println!();

            println!("=== Architecture ===");
            println!("  Layers: {}", gguf.n_layers());
            println!("  Embedding dim: {}", gguf.n_embd());
            println!("  Attention heads: {}", gguf.n_head());
            println!("  KV heads: {}", gguf.n_head_kv());
            println!("  Context length: {}", gguf.n_ctx());
            println!("  Vocab size: {}", gguf.vocab_size());
            println!();

            println!("=== Metadata ===");
            for (key, value) in &gguf.metadata {
                println!("  {}: {}", key, value);
            }
            println!();

            println!("=== Tensor Summary ===");
            let total_bytes: u64 = gguf.tensors.iter().map(|t| t.size_bytes).sum();
            println!("  Total tensors: {}", gguf.tensors.len());
            println!("  Total tensor data: {:.2} GB", total_bytes as f64 / (1024.0 * 1024.0 * 1024.0));

            if gguf.tensors.len() <= 20 {
                for t in &gguf.tensors {
                    println!("  {} — {:?} — {:.2} MB", t.name, t.dimensions, t.size_bytes as f64 / (1024.0 * 1024.0));
                }
            } else {
                for t in gguf.tensors.iter().take(10) {
                    println!("  {} — {:?} — {:.2} MB", t.name, t.dimensions, t.size_bytes as f64 / (1024.0 * 1024.0));
                }
                println!("  ... ({} more tensors)", gguf.tensors.len() - 10);
            }
        }

        Commands::Run {
            model,
            memory_budget,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            draft_model,
            draft_ahead,
        } => {
            let budget = parse_memory_budget(&memory_budget)?;
            info!("Loading model: {}", model.display());
            info!("Memory budget: {} bytes ({:.2} GB)", budget, budget as f64 / (1024.0 * 1024.0 * 1024.0));

            let gguf = model::gguf::GgufFile::open(&model)?;
            println!("Model loaded: {} tensors, {:.2} GB",
                gguf.tensors.len(),
                gguf.file_size as f64 / (1024.0 * 1024.0 * 1024.0)
            );

            // Initialize the mmap loader with prefetching
            let loader = ssd::streamer::SsdStreamer::new(&model, budget)?;
            info!("SSD streamer initialized with {:.2} GB budget", budget as f64 / (1024.0 * 1024.0 * 1024.0));

            // Initialize layer cache
            let mut cache = model::cache::LayerCache::new(budget);

            // Run inference
            let config = inference::transformer::InferenceConfig {
                temperature,
                top_k,
                top_p,
                max_tokens,
            };

            println!("\nPrompt: {}", prompt);
            println!("---");

            if let Some(draft_path) = draft_model {
                // Speculative decoding mode
                println!("🎯 Speculative decoding with draft model: {}", draft_path.display());
                let draft_gguf = model::gguf::GgufFile::open(&draft_path)?;
                let draft_loader = ssd::streamer::SsdStreamer::new(&draft_path, budget / 4)?;
                let mut draft_cache = model::cache::LayerCache::new(budget / 4);

                let spec_config = inference::speculative::SpeculativeConfig {
                    temperature,
                    top_k,
                    top_p,
                    max_tokens,
                    draft_ahead,
                };

                let start = Instant::now();
                let result = inference::speculative::generate_speculative(
                    &gguf, &loader, &mut cache,
                    &draft_gguf, &draft_loader, &mut draft_cache,
                    &prompt, &spec_config,
                )?;

                let elapsed = start.elapsed();
                println!("{}", result.text);
                println!("---");
                println!("Prompt tokens: {} | Generated: {} | Time: {:.2}s | Speed: {:.2} tok/s",
                    result.prompt_tokens, result.token_count, elapsed.as_secs_f64(), result.tokens_per_sec);
                println!("Draft: {}/{} accepted ({:.1}%) | Target passes: {} (saved {:.0}%)",
                    result.draft_tokens_accepted, result.draft_tokens_total,
                    result.acceptance_rate * 100.0, result.target_forward_passes,
                    if result.token_count > 0 {
                        (1.0 - result.target_forward_passes as f64 / result.token_count as f64) * 100.0
                    } else { 0.0 });
                println!("KV cache: {:.2} MB", result.kv_cache_bytes as f64 / (1024.0 * 1024.0));
            } else {
                // Standard decoding
                let start = Instant::now();
                let result = inference::transformer::generate(
                    &gguf,
                    &loader,
                    &mut cache,
                    &prompt,
                    &config,
                )?;

                let elapsed = start.elapsed();
                let tokens_generated = result.token_count;
                let tokens_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

                println!("{}", result.text);
                println!("---");
                println!("Prompt tokens: {} | Generated: {} | Time: {:.2}s | Speed: {:.2} tok/s",
                    result.prompt_tokens, tokens_generated, elapsed.as_secs_f64(), tokens_per_sec);
                println!("KV cache: {:.2} MB | Layer cache hits: {} | misses: {} | prefetch: {}",
                    result.kv_cache_bytes as f64 / (1024.0 * 1024.0),
                    cache.stats().hits, cache.stats().misses, cache.stats().prefetch_hits);
            }
        }

        Commands::Bench { model, memory_budget } => {
            let budget = parse_memory_budget(&memory_budget)?;
            benchmark::run_benchmark(&model, budget)?;
        }

        Commands::Serve { model, memory_budget, host, port, draft_model, draft_ahead } => {
            let budget = parse_memory_budget(&memory_budget)?;
            let server = api::server::ApiServer::new(api::server::ServerConfig {
                host,
                port,
                model_path: model,
                memory_budget: budget,
                draft_model_path: draft_model,
                draft_ahead,
            });
            server.run()?;
        }
    }

    Ok(())
}
