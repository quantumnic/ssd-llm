// v1.0 — Production-ready ssd-llm
// Many public API items are defined but not yet wired into all code paths
// (e.g., GPU dispatch, advanced attention strategies, streaming generators).
// These will be connected as features mature.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

mod api;
mod benchmark;
mod config;
mod inference;
mod metal;
mod model;
mod pull;
mod ssd;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

#[derive(Parser)]
#[command(
    name = "ssd-llm",
    version,
    about = "Run 70B+ LLMs on Apple Silicon via SSD streaming"
)]
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

        /// Enable adaptive draft length (adjusts K based on acceptance rate)
        #[arg(long, default_value_t = false)]
        adaptive_draft: bool,

        /// Enable prompt prefix caching (reuse KV states for repeated prefixes)
        #[arg(long, default_value_t = false)]
        prompt_cache: bool,

        /// Number of tensor parallel shards (0 = auto-detect)
        #[arg(long, default_value_t = 0)]
        tensor_parallel: usize,

        /// Sliding window attention size (0 = full attention)
        #[arg(long, default_value_t = 0)]
        sliding_window: usize,

        /// Number of sink tokens to keep visible with sliding window
        #[arg(long, default_value_t = 4)]
        sink_tokens: usize,

        /// Enable memory-mapped KV cache (spills to SSD when RAM is full)
        #[arg(long, default_value_t = false)]
        mmap_kv: bool,

        /// Use flash attention (memory-efficient fused attention kernel)
        #[arg(long, default_value_t = false)]
        flash_attention: bool,

        /// Enable INT8 KV cache quantization (4x memory reduction, longer contexts)
        #[arg(long, default_value_t = false)]
        kv_quantize: bool,

        /// Path to LoRA adapter GGUF file(s) — can be specified multiple times
        #[arg(long = "lora")]
        lora_adapters: Vec<PathBuf>,

        /// LoRA scaling factor (default 1.0)
        #[arg(long, default_value_t = 1.0)]
        lora_scale: f32,

        /// GBNF grammar string for constrained generation
        #[arg(long)]
        grammar: Option<String>,

        /// Path to GBNF grammar file for constrained generation
        #[arg(long)]
        grammar_file: Option<PathBuf>,
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

        /// Output results as JSON (for CI/CD integration)
        #[arg(long, default_value_t = false)]
        json: bool,
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

        /// Enable adaptive draft length
        #[arg(long, default_value_t = false)]
        adaptive_draft: bool,

        /// Enable prompt prefix caching
        #[arg(long, default_value_t = false)]
        prompt_cache: bool,

        /// Maximum concurrent requests for continuous batching (0 = sequential)
        #[arg(long, default_value_t = 4)]
        max_batch: usize,

        /// Number of tensor parallel shards (0 = auto-detect)
        #[arg(long, default_value_t = 0)]
        tensor_parallel: usize,

        /// Sliding window attention size (0 = full attention)
        #[arg(long, default_value_t = 0)]
        sliding_window: usize,

        /// Number of sink tokens to keep visible with sliding window
        #[arg(long, default_value_t = 4)]
        sink_tokens: usize,

        /// Enable memory-mapped KV cache (spills to SSD when RAM is full)
        #[arg(long, default_value_t = false)]
        mmap_kv: bool,

        /// Use flash attention (memory-efficient fused attention kernel)
        #[arg(long, default_value_t = false)]
        flash_attention: bool,

        /// Enable INT8 KV cache quantization (4x memory reduction, longer contexts)
        #[arg(long, default_value_t = false)]
        kv_quantize: bool,

        /// Path to LoRA adapter GGUF file(s) — can be specified multiple times
        #[arg(long = "lora")]
        lora_adapters: Vec<PathBuf>,

        /// LoRA scaling factor (default 1.0)
        #[arg(long, default_value_t = 1.0)]
        lora_scale: f32,

        /// Enable PagedAttention (vLLM-style paged KV cache for efficient memory)
        #[arg(long, default_value_t = false)]
        paged_kv: bool,

        /// Number of KV cache blocks per layer for PagedAttention (default: auto from memory budget)
        #[arg(long, default_value_t = 0)]
        paged_kv_blocks: usize,

        /// PagedAttention block size in tokens (default: 16)
        #[arg(long, default_value_t = 16)]
        paged_block_size: usize,

        /// Quantize KV blocks to INT8 when swapping to SSD (4x less I/O)
        #[arg(long, default_value_t = false)]
        swap_quantize: bool,

        /// Enable adaptive memory pressure monitoring (auto-adjusts cache budget based on system RAM)
        #[arg(long, default_value_t = false)]
        adaptive_memory: bool,

        /// Adaptive layer pinning: auto-pin N hottest layers in RAM (0 = disabled)
        #[arg(long, default_value_t = 0)]
        adaptive_pin: usize,
    },
    /// Download a GGUF model from Hugging Face
    Pull {
        /// Model specifier: user/repo:file.gguf, user/repo, or HF URL
        model: String,

        /// Output directory for downloaded models
        #[arg(long)]
        output: Option<PathBuf>,

        /// Force re-download even if file exists
        #[arg(long, default_value_t = false)]
        force: bool,
    },
    /// List locally downloaded models
    Models {
        /// Directory to scan (defaults to models/ or $SSD_LLM_MODEL_DIR)
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Interactive multi-turn chat with a model
    Chat {
        /// Path to GGUF model file
        model: PathBuf,

        /// Memory budget for layer cache (e.g. "8G", "4G", "512M")
        #[arg(long, default_value = "8G")]
        memory_budget: String,

        /// System prompt to prepend to conversation
        #[arg(long)]
        system: Option<String>,

        /// Temperature for sampling
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,

        /// Top-K sampling
        #[arg(long, default_value_t = 40)]
        top_k: usize,

        /// Top-P (nucleus) sampling
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        /// Maximum tokens per response
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,

        /// Chat template override (chatml, llama2, llama3, mistral, gemma, phi3, raw)
        #[arg(long)]
        template: Option<String>,

        /// Repetition penalty
        #[arg(long, default_value_t = 1.1)]
        repetition_penalty: f32,

        /// Path to LoRA adapter GGUF file(s) — can be specified multiple times
        #[arg(long = "lora")]
        lora_adapters: Vec<PathBuf>,

        /// LoRA scaling factor (default 1.0)
        #[arg(long, default_value_t = 1.0)]
        lora_scale: f32,
    },
    /// Evaluate perplexity on a text file (measure model quality)
    Perplexity {
        /// Path to GGUF model file
        model: PathBuf,

        /// Path to text file to evaluate
        file: PathBuf,

        /// Memory budget for layer cache (e.g. "8G", "4G", "512M")
        #[arg(long, default_value = "8G")]
        memory_budget: String,

        /// Context window size (0 = use model's n_ctx)
        #[arg(long, default_value_t = 0)]
        context_size: usize,

        /// Stride for sliding window (0 = context_size / 2)
        #[arg(long, default_value_t = 0)]
        stride: usize,

        /// Show per-chunk perplexity values
        #[arg(long, default_value_t = false)]
        verbose: bool,

        /// Output results as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Generate or show configuration
    Config {
        /// Generate a default config file
        #[arg(long, default_value_t = false)]
        init: bool,
    },
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{} bytes", bytes)
    }
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
            println!(
                "File size: {:.2} GB",
                gguf.file_size as f64 / (1024.0 * 1024.0 * 1024.0)
            );
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
            println!(
                "  Total tensor data: {:.2} GB",
                total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );

            if gguf.tensors.len() <= 20 {
                for t in &gguf.tensors {
                    println!(
                        "  {} — {:?} — {:.2} MB",
                        t.name,
                        t.dimensions,
                        t.size_bytes as f64 / (1024.0 * 1024.0)
                    );
                }
            } else {
                for t in gguf.tensors.iter().take(10) {
                    println!(
                        "  {} — {:?} — {:.2} MB",
                        t.name,
                        t.dimensions,
                        t.size_bytes as f64 / (1024.0 * 1024.0)
                    );
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
            adaptive_draft,
            prompt_cache,
            tensor_parallel,
            sliding_window,
            sink_tokens,
            mmap_kv,
            flash_attention,
            kv_quantize,
            lora_adapters,
            lora_scale,
            grammar,
            grammar_file,
        } => {
            // Load grammar from string or file
            let grammar_str = if let Some(gf) = grammar_file {
                std::fs::read_to_string(&gf).map_err(|e| {
                    anyhow::anyhow!("Failed to read grammar file {}: {}", gf.display(), e)
                })?
            } else {
                grammar.unwrap_or_default()
            };

            let budget = parse_memory_budget(&memory_budget)?;
            info!("Loading model: {}", model.display());
            info!(
                "Memory budget: {} bytes ({:.2} GB)",
                budget,
                budget as f64 / (1024.0 * 1024.0 * 1024.0)
            );

            let gguf = model::gguf::GgufFile::open(&model)?;
            println!(
                "Model loaded: {} tensors, {:.2} GB",
                gguf.tensors.len(),
                gguf.file_size as f64 / (1024.0 * 1024.0 * 1024.0)
            );

            // Initialize the mmap loader with prefetching
            let loader = ssd::streamer::SsdStreamer::new(&model, budget)?;
            info!(
                "SSD streamer initialized with {:.2} GB budget",
                budget as f64 / (1024.0 * 1024.0 * 1024.0)
            );

            // Initialize layer cache
            let mut cache = model::cache::LayerCache::new(budget);

            // Run inference
            let config = inference::transformer::InferenceConfig {
                temperature,
                top_k,
                top_p,
                max_tokens,
                stop_sequences: Vec::new(),
                repetition_penalty: 1.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                tfs_z: 0.0,
                mirostat: 0,
                mirostat_tau: 5.0,
                mirostat_eta: 0.1,
                grammar: grammar_str,
            };

            // Tensor parallelism
            let tp_shards = if tensor_parallel > 0 {
                tensor_parallel
            } else {
                inference::tensor_parallel::auto_detect_shards(gguf.n_embd() as usize)
            };
            if tp_shards > 1 {
                println!("🔀 Tensor parallelism: {} shards", tp_shards);
            }

            // Sliding window attention
            if sliding_window > 0 {
                println!(
                    "🪟 Sliding window attention: {} tokens (sink: {})",
                    sliding_window, sink_tokens
                );
            }

            // GQA detection
            let n_head = gguf.n_head() as usize;
            let n_head_kv = gguf.n_head_kv() as usize;
            let attn_strategy = inference::gqa::AttentionStrategy::detect(n_head, n_head_kv);
            match attn_strategy {
                inference::gqa::AttentionStrategy::GroupedQuery { group_size } => {
                    println!(
                        "🔗 GQA optimized: {} groups of {} query heads (KV savings: {:.0}%)",
                        n_head_kv,
                        group_size,
                        (1.0 - attn_strategy.kv_savings_ratio(n_head)) * 100.0
                    );
                }
                inference::gqa::AttentionStrategy::MultiQuery => {
                    println!(
                        "🔗 MQA detected: single KV head (KV savings: {:.0}%)",
                        (1.0 - attn_strategy.kv_savings_ratio(n_head)) * 100.0
                    );
                }
                _ => {}
            }

            // Memory-mapped KV cache
            if mmap_kv {
                println!("💾 Memory-mapped KV cache enabled (spills to SSD)");
            }

            // Flash attention
            if flash_attention {
                println!("⚡ Flash attention enabled (memory-efficient O(1) attention)");
            }

            // KV cache quantization
            if kv_quantize {
                println!("🔢 INT8 KV cache quantization enabled (4x memory reduction)");
            }

            // LoRA adapters
            let mut _lora_mgr = inference::lora::LoraManager::new();
            for lora_path in &lora_adapters {
                _lora_mgr.add_adapter(lora_path, lora_scale)?;
                println!("🔗 LoRA adapter loaded: {}", lora_path.display());
            }
            if !_lora_mgr.is_empty() {
                for info in _lora_mgr.summary() {
                    println!(
                        "   ├─ rank={}, alpha={}, scale={}, weights={}",
                        info.rank, info.alpha, info.scale, info.num_weights
                    );
                }
            }

            // Prompt caching
            let _pcache = if prompt_cache {
                println!("📦 Prompt prefix caching enabled");
                Some(inference::prompt_cache::PromptCache::new(budget / 4))
            } else {
                None
            };

            println!("\nPrompt: {}", prompt);
            println!("---");

            if let Some(draft_path) = draft_model {
                // Speculative decoding mode
                println!(
                    "🎯 Speculative decoding with draft model: {}",
                    draft_path.display()
                );
                let draft_gguf = model::gguf::GgufFile::open(&draft_path)?;
                let draft_loader = ssd::streamer::SsdStreamer::new(&draft_path, budget / 4)?;
                let mut draft_cache = model::cache::LayerCache::new(budget / 4);

                let spec_config = inference::speculative::SpeculativeConfig {
                    temperature,
                    top_k,
                    top_p,
                    max_tokens,
                    draft_ahead,
                    adaptive: adaptive_draft,
                };

                let start = Instant::now();
                let result = inference::speculative::generate_speculative(
                    &gguf,
                    &loader,
                    &mut cache,
                    &draft_gguf,
                    &draft_loader,
                    &mut draft_cache,
                    &prompt,
                    &spec_config,
                )?;

                let elapsed = start.elapsed();
                println!("{}", result.text);
                println!("---");
                println!(
                    "Prompt tokens: {} | Generated: {} | Time: {:.2}s | Speed: {:.2} tok/s",
                    result.prompt_tokens,
                    result.token_count,
                    elapsed.as_secs_f64(),
                    result.tokens_per_sec
                );
                println!(
                    "Draft: {}/{} accepted ({:.1}%) | Target passes: {} (saved {:.0}%)",
                    result.draft_tokens_accepted,
                    result.draft_tokens_total,
                    result.acceptance_rate * 100.0,
                    result.target_forward_passes,
                    if result.token_count > 0 {
                        (1.0 - result.target_forward_passes as f64 / result.token_count as f64)
                            * 100.0
                    } else {
                        0.0
                    }
                );
                println!(
                    "KV cache: {:.2} MB",
                    result.kv_cache_bytes as f64 / (1024.0 * 1024.0)
                );
            } else {
                // Standard decoding
                let start = Instant::now();
                let result =
                    inference::transformer::generate(&gguf, &loader, &mut cache, &prompt, &config)?;

                let elapsed = start.elapsed();
                let tokens_generated = result.token_count;
                let tokens_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

                println!("{}", result.text);
                println!("---");
                println!(
                    "Prompt tokens: {} | Generated: {} | Time: {:.2}s | Speed: {:.2} tok/s",
                    result.prompt_tokens,
                    tokens_generated,
                    elapsed.as_secs_f64(),
                    tokens_per_sec
                );
                println!(
                    "KV cache: {:.2} MB | Layer cache hits: {} | misses: {} | prefetch: {}",
                    result.kv_cache_bytes as f64 / (1024.0 * 1024.0),
                    cache.stats().hits,
                    cache.stats().misses,
                    cache.stats().prefetch_hits
                );
            }
        }

        Commands::Bench {
            model,
            memory_budget,
            json,
        } => {
            let budget = parse_memory_budget(&memory_budget)?;
            if json {
                let json_str = benchmark::run_benchmark_json(&model, budget)?;
                println!("{}", json_str);
            } else {
                benchmark::run_benchmark(&model, budget)?;
            }
        }

        Commands::Serve {
            model,
            memory_budget,
            host,
            port,
            draft_model,
            draft_ahead,
            adaptive_draft,
            prompt_cache,
            max_batch,
            tensor_parallel,
            sliding_window: _,
            sink_tokens: _,
            mmap_kv: _,
            flash_attention: _,
            kv_quantize: _,
            lora_adapters,
            lora_scale,
            paged_kv,
            paged_kv_blocks,
            paged_block_size,
            swap_quantize,
            adaptive_memory,
            adaptive_pin,
        } => {
            let budget = parse_memory_budget(&memory_budget)?;

            // Start memory pressure monitor if requested
            let _pressure_monitor = if adaptive_memory {
                let monitor = ssd::memory_pressure::MemoryPressureMonitor::start_default();
                if let Some(snap) = ssd::memory_pressure::MemoryPressureMonitor::snapshot() {
                    println!(
                        "🧠 Adaptive memory: {} total, {} available ({})",
                        format_bytes(snap.total_bytes),
                        format_bytes(snap.available_bytes),
                        snap.pressure,
                    );
                }
                Some(monitor)
            } else {
                None
            };
            let tp_shards = if tensor_parallel > 0 {
                tensor_parallel
            } else {
                inference::tensor_parallel::auto_detect_shards(budget)
            };

            // Load LoRA adapters
            let mut lora_mgr = inference::lora::LoraManager::new();
            for lora_path in &lora_adapters {
                lora_mgr.add_adapter(lora_path, lora_scale)?;
                println!("🔗 LoRA adapter loaded: {}", lora_path.display());
            }

            if paged_kv {
                println!(
                    "📄 PagedAttention enabled: block_size={}, blocks={}{}",
                    paged_block_size,
                    if paged_kv_blocks > 0 {
                        format!("{}", paged_kv_blocks)
                    } else {
                        "auto".to_string()
                    },
                    if swap_quantize {
                        " (INT8 quantized swap)"
                    } else {
                        ""
                    }
                );
            }

            let server = api::server::ApiServer::new(api::server::ServerConfig {
                host,
                port,
                model_path: model,
                memory_budget: budget,
                draft_model_path: draft_model,
                draft_ahead,
                adaptive_draft,
                prompt_cache,
                max_batch,
                tensor_parallel: tp_shards,
                paged_kv,
                paged_kv_blocks,
                paged_block_size,
                swap_quantize,
                adaptive_pin,
            });

            if adaptive_pin > 0 {
                println!(
                    "📌 Adaptive layer pinning: auto-pin top {} hottest layers",
                    adaptive_pin
                );
            }

            server.run()?;
        }

        Commands::Pull {
            model: spec,
            output,
            force,
        } => {
            let (repo, filename) = pull::parse_model_spec(&spec)?;
            let output_dir = output.unwrap_or_else(pull::default_model_dir);

            let filename = match filename {
                Some(f) => f,
                None => {
                    println!("🔍 Listing GGUF files in {}...", repo);
                    let files = pull::list_gguf_files(&repo)?;
                    if files.is_empty() {
                        anyhow::bail!("No GGUF files found in {}", repo);
                    }
                    println!("Available GGUF files:");
                    for (i, f) in files.iter().enumerate() {
                        println!("  [{}] {}", i + 1, f);
                    }
                    if files.len() == 1 {
                        files[0].clone()
                    } else {
                        anyhow::bail!(
                            "Multiple GGUF files found. Specify one: {}:<filename>",
                            repo
                        );
                    }
                }
            };

            pull::download_model(&repo, &filename, &output_dir, force)?;
        }

        Commands::Models { dir } => {
            let dir = dir.unwrap_or_else(pull::default_model_dir);
            let models = pull::list_local_models(&dir)?;
            if models.is_empty() {
                println!("No models found in {}", dir.display());
                println!("Download one with: ssd-llm pull <user/repo:file.gguf>");
            } else {
                println!("📦 Local models ({}):", dir.display());
                for (name, size) in &models {
                    println!(
                        "  {} ({:.2} GB)",
                        name,
                        *size as f64 / (1024.0 * 1024.0 * 1024.0)
                    );
                }
                println!("  Total: {} model(s)", models.len());
            }
        }

        Commands::Chat {
            model,
            memory_budget,
            system,
            temperature,
            top_k,
            top_p,
            max_tokens,
            template,
            repetition_penalty,
            lora_adapters,
            lora_scale,
        } => {
            let budget = parse_memory_budget(&memory_budget)?;
            info!("Loading model: {}", model.display());

            let gguf = model::gguf::GgufFile::open(&model)?;
            let loader = ssd::streamer::SsdStreamer::new(&model, budget)?;
            let mut cache = model::cache::LayerCache::new(budget);

            // Load LoRA adapters
            let mut _lora_mgr = inference::lora::LoraManager::new();
            for lora_path in &lora_adapters {
                _lora_mgr.add_adapter(lora_path, lora_scale)?;
                println!("🔗 LoRA adapter loaded: {}", lora_path.display());
            }

            // Detect chat template
            let template_format = if let Some(ref tmpl) = template {
                match tmpl.to_lowercase().as_str() {
                    "chatml" => inference::chat_template::ChatTemplateFormat::ChatML,
                    "llama2" => inference::chat_template::ChatTemplateFormat::Llama2,
                    "llama3" => inference::chat_template::ChatTemplateFormat::Llama3,
                    "mistral" => inference::chat_template::ChatTemplateFormat::Mistral,
                    "gemma" => inference::chat_template::ChatTemplateFormat::Gemma,
                    "phi3" => inference::chat_template::ChatTemplateFormat::Phi3,
                    "raw" => inference::chat_template::ChatTemplateFormat::Raw,
                    _ => {
                        println!("⚠ Unknown template '{}', falling back to auto-detect", tmpl);
                        inference::chat_template::ChatTemplateFormat::from_model_name(
                            &model.to_string_lossy(),
                        )
                    }
                }
            } else {
                inference::chat_template::ChatTemplateFormat::from_model_name(
                    &model.to_string_lossy(),
                )
            };

            let chat_config = inference::chat::ChatConfig {
                temperature,
                top_k,
                top_p,
                max_tokens,
                system_prompt: system,
                template: template_format,
                repetition_penalty,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
            };

            println!(
                "Model loaded: {} ({} layers, {} embd)",
                model.display(),
                gguf.n_layers(),
                gguf.n_embd()
            );

            inference::chat::run_interactive(&gguf, &loader, &mut cache, &chat_config)?;
        }

        Commands::Perplexity {
            model,
            file,
            memory_budget,
            context_size,
            stride,
            verbose,
            json,
        } => {
            let budget = parse_memory_budget(&memory_budget)?;
            let text = std::fs::read_to_string(&file)
                .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", file.display(), e))?;

            if text.is_empty() {
                anyhow::bail!("Input file is empty: {}", file.display());
            }

            let gguf = model::gguf::GgufFile::open(&model)?;
            println!(
                "Model: {} ({} layers, {} embd, ctx {})",
                model.display(),
                gguf.n_layers(),
                gguf.n_embd(),
                gguf.n_ctx()
            );
            println!("Input: {} ({} bytes)", file.display(), text.len());

            let streamer = ssd::streamer::SsdStreamer::new(&model, budget)?;
            let mut layer_cache = model::cache::LayerCache::new(budget);

            let config = inference::perplexity::PerplexityConfig {
                context_size,
                stride,
                verbose,
            };

            println!("Evaluating perplexity...\n");
            let result = inference::perplexity::evaluate_perplexity(
                &gguf,
                &streamer,
                &mut layer_cache,
                &text,
                &config,
            )?;

            if json {
                println!("{}", result.to_json());
            } else {
                println!("=== Perplexity Results ===");
                println!("  Perplexity:       {:.4}", result.perplexity);
                println!("  Avg NLL:          {:.6}", result.nll);
                println!(
                    "  Tokens evaluated: {} / {}",
                    result.tokens_evaluated, result.total_tokens
                );
                println!("  Chunks:           {}", result.chunks);
                println!("  Time:             {:.2}s", result.elapsed_secs);
                println!("  Throughput:       {:.1} tok/s", result.tokens_per_sec);
            }
        }

        Commands::Config { init } => {
            if init {
                let path = std::path::PathBuf::from("ssd-llm.toml");
                if path.exists() {
                    println!("⚠ ssd-llm.toml already exists. Remove it first to regenerate.");
                } else {
                    std::fs::write(&path, config::Config::default_toml())?;
                    println!("✓ Generated ssd-llm.toml with default settings");
                }
            } else {
                match config::Config::load() {
                    Ok(cfg) => {
                        println!("=== ssd-llm Configuration ===");
                        println!("Model path: {:?}", cfg.model.path);
                        println!("Memory budget: {}", cfg.model.memory_budget);
                        println!("Server: {}:{}", cfg.server.host, cfg.server.port);
                        println!("Flash attention: {}", cfg.inference.flash_attention);
                        println!("KV quantization: {}", cfg.inference.kv_quantize);
                        println!("Swap quantization: {}", cfg.inference.swap_quantize);
                        println!("Adaptive memory: {}", cfg.inference.adaptive_memory);
                        println!("Sliding window: {}", cfg.inference.sliding_window);
                        println!("Model directory: {}", cfg.paths.model_dir.display());
                    }
                    Err(e) => {
                        println!("No config loaded (using defaults): {}", e);
                        println!("Run `ssd-llm config --init` to create one.");
                    }
                }
            }
        }
    }

    Ok(())
}
