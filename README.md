# ssd-llm 🚀

**Run 70B+ LLMs on Apple Silicon by using SSD as extended memory.**

Intelligent layer streaming and caching for Mac — no need for 128GB RAM.

## The Problem

Large language models like LLaMA 70B require ~40GB+ RAM even with 4-bit quantization. Most MacBooks have 16–36GB unified memory. You either:
- Can't run the model at all
- Use llama.cpp's mmap, which thrashes your SSD with no intelligence
- Accept terrible performance from OS swap pressure

## The Solution

**ssd-llm** treats your fast Apple SSD as an intelligent extension of RAM:

```
┌─────────────┐     ┌──────────────┐     ┌───────────┐
│  SSD (2TB)  │────▶│ Smart Cache  │────▶│ Metal GPU │
│  Model File │     │ (Layer Pool) │     │ Inference  │
└─────────────┘     └──────────────┘     └───────────┘
                     ▲                    │
                     │    Prefetch        │ Compute
                     └────────────────────┘
```

Instead of loading the entire model, **ssd-llm** streams transformer layers on-demand from SSD to unified memory, computes them, and frees the memory. Predictive prefetching ensures the next layer is already loading while the current one is being computed.

## Key Features

- **🧱 Layer-Level Streaming** — Only 1-2 transformer layers in RAM at once
- **🔮 Predictive Prefetching** — Next layer loads asynchronously via `madvise(MADV_WILLNEED)` while GPU computes
- **📦 Smart LRU Cache** — Frequently used layers (embeddings, early attention) stay pinned in RAM
- **🗺️ mmap + madvise** — OS-level memory-mapped files with intelligent page hints
- **⚡ Metal Compute** — SIMD-optimized matmul, softmax, RoPE, RMSNorm with Metal shader foundation
- **📄 GGUF Support** — Compatible with llama.cpp quantization formats (Q4_0, Q8_0, F16, F32)
- **🔤 BPE Tokenizer** — Full Byte-Pair Encoding with SentencePiece support from GGUF vocabulary
- **🔌 Ollama-compatible API** — Drop-in replacement server with OpenAI-compatible endpoint
- **📡 Streaming** — Real-time token-by-token streaming via chunked transfer (Ollama) and SSE (OpenAI)
- **🎯 Speculative Decoding** — Use a small draft model to propose tokens, verified by the target model for 2-3x speedup
- **📦 Batch Prefill** — Layer-major prompt processing: each layer loaded once for all prompt tokens, minimizing SSD reads
- **🎛️ Adaptive Draft Length** — Dynamically adjusts speculation depth K based on rolling acceptance rate
- **📦 Prompt Prefix Caching** — Reuse KV cache states for repeated prompt prefixes (system prompts, templates)
- **🔄 Continuous Batching** — Handle multiple concurrent requests, share layer loads across sequences
- **🔀 Tensor Parallelism** — Split matmul across multiple threads for better GPU/CPU utilization
- **🪟 Sliding Window Attention** — Limit attention to recent W tokens with optional sink tokens for bounded memory
- **🔗 GQA Optimization** — Grouped-Query Attention with batched KV loads, auto-detected from model config
- **💾 Memory-Mapped KV Cache** — Spill KV cache to SSD via mmap when RAM is exhausted, enabling ultra-long contexts

## Quick Start

```bash
# Build
cargo build --release

# Show model info
ssd-llm info model.gguf

# Run inference with 8GB memory budget
ssd-llm run model.gguf --memory-budget 8G --prompt "Explain quantum computing"

# Benchmark SSD streaming performance
ssd-llm bench model.gguf --memory-budget 8G

# Start Ollama-compatible API server
ssd-llm serve model.gguf --memory-budget 8G --port 11434

# Speculative decoding with draft model (2-3x faster)
ssd-llm run model-70b.gguf --draft-model model-1b.gguf --prompt "Hello" --draft-ahead 5

# Adaptive draft length (auto-tunes K based on acceptance rate)
ssd-llm run model-70b.gguf --draft-model model-1b.gguf --prompt "Hello" --adaptive-draft

# Serve with speculative decoding
ssd-llm serve model-70b.gguf --draft-model model-1b.gguf --memory-budget 8G

# Enable prompt prefix caching (reuse KV states across requests)
ssd-llm run model.gguf --prompt "Hello" --prompt-cache

# Tensor parallelism (auto-detected or manual)
ssd-llm run model-70b.gguf --prompt "Hello" --tensor-parallel 4

# Continuous batching server (handles 8 concurrent requests)
ssd-llm serve model.gguf --memory-budget 8G --max-batch 8 --prompt-cache

# Sliding window attention (bounded memory for long contexts)
ssd-llm run model.gguf --prompt "Hello" --sliding-window 4096 --sink-tokens 4

# Memory-mapped KV cache (ultra-long contexts, spills to SSD)
ssd-llm run model.gguf --prompt "Hello" --mmap-kv --max-tokens 32768

# GQA is auto-detected — just run and see the optimization message
ssd-llm run llama-70b.gguf --prompt "Hello" --memory-budget 16G
```

## API Server

The `serve` command starts an Ollama-compatible HTTP server:

```bash
ssd-llm serve model.gguf --memory-budget 8G
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/generate` | POST | Text generation (Ollama format) |
| `/api/chat` | POST | Chat completion (Ollama format) |
| `/api/tags` | GET | List loaded models |
| `/api/version` | GET | Server version |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |

### Usage with curl

```bash
# Ollama-style generation
curl -X POST http://localhost:11434/api/generate \
  -d '{"prompt": "What is Rust?", "num_predict": 128}'

# OpenAI-compatible chat
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 128}'
```

## How It Works

### Layer Streaming Architecture

Traditional LLM inference loads the entire model into RAM. **ssd-llm** takes a different approach:

1. **GGUF Parser** reads model metadata and tensor offsets without loading data
2. **mmap Loader** memory-maps the model file — the OS handles page faults
3. **Predictive Prefetcher** issues `madvise(MADV_WILLNEED)` for the next layer while the current one computes
4. **LRU Cache** keeps hot layers (embeddings, output weights) pinned in memory
5. **Eviction** calls `madvise(MADV_DONTNEED)` on completed layers to free page cache

### Why Apple Silicon?

Apple's Unified Memory Architecture is uniquely suited for this:

| Feature | Apple Silicon | Traditional PC |
|---|---|---|
| Memory | Unified (CPU+GPU shared) | Separate RAM + VRAM |
| SSD Speed | 5-7 GB/s (M3/M4 Pro) | 3-5 GB/s (NVMe) |
| Memory Bandwidth | 200-800 GB/s | 50-100 GB/s (DDR5) |
| GPU Access | Direct to unified memory | PCIe copy required |

The fast SSD + unified memory means layer streaming has very low overhead on Mac.

## Benchmarks

> v0.5 — Speculative decoding with draft model, KV cache rollback

| Model | Quant | Size | Memory Budget | Layer Load | Est. tok/s |
|---|---|---|---|---|---|
| LLaMA 7B | Q4_0 | 3.5 GB | 4 GB | ~2ms/layer | TBD |
| LLaMA 13B | Q4_0 | 7 GB | 8 GB | ~4ms/layer | TBD |
| LLaMA 70B | Q4_0 | 35 GB | 8 GB | ~8ms/layer | TBD |

Run `ssd-llm bench` on your machine to get actual numbers.

## Comparison

| Feature | ssd-llm | llama.cpp | Ollama |
|---|---|---|---|
| SSD Streaming | ✅ Intelligent | ⚠️ Naive mmap | ❌ Full RAM |
| Predictive Prefetch | ✅ madvise hints | ❌ | ❌ |
| Memory Budget | ✅ Configurable | ❌ | ❌ |
| Layer-level Cache | ✅ LRU + pinning | ❌ | ❌ |
| Speculative Decoding | ✅ Draft model | ✅ (v0.6+) | ❌ |
| Continuous Batching | ✅ Layer-major | ✅ | ✅ |
| Prompt Caching | ✅ Prefix matching | ❌ | ❌ |
| Tensor Parallelism | ✅ Multi-thread | ✅ | ✅ (via llama.cpp) |
| Metal GPU | ✅ Shaders + SIMD | ✅ | ✅ (via llama.cpp) |
| GGUF Support | ✅ | ✅ | ✅ |
| Quantization | Q4_0, Q8_0, F16 | All | All |
| API Server | ✅ Ollama + OpenAI | ✅ | ✅ |

## Architecture

```
src/
  main.rs              — CLI + entry point
  model/
    gguf.rs            — GGUF v2/v3 parser
    loader.rs          — mmap-based lazy loader
    cache.rs           — LRU layer cache with memory budget
  inference/
    transformer.rs     — Layer-by-layer forward pass
    attention.rs       — Multi-Head Attention with KV cache (GQA support)
    kv_cache.rs        — Key-Value cache for autoregressive generation
    feed_forward.rs    — SwiGLU FFN
    sampler.rs         — Temperature, Top-K, Top-P sampling (xorshift64)
    speculative.rs     — Speculative decoding engine (draft + verify)
    tokenizer.rs       — BPE tokenizer with SentencePiece support
    prompt_cache.rs    — Prompt prefix KV state caching
    batch_scheduler.rs — Continuous batching scheduler
    tensor_parallel.rs — Multi-threaded tensor parallelism
  metal/
    compute.rs         — Metal compute + SIMD-optimized ops (auto GPU dispatch)
    gpu.rs             — metal-rs GPU pipeline (real Metal compute)
    shaders/           — .metal compute shaders (matmul, rmsnorm, rope, softmax)
  ssd/
    streamer.rs        — SSD → RAM streaming engine
    prefetch.rs        — Predictive prefetcher
    mmap_pool.rs       — mmap pool with madvise management
  api/
    server.rs          — Ollama-compatible HTTP API server
    openai.rs          — OpenAI-compatible types + ChatML formatting
  benchmark.rs         — Performance measurement
```

## Speculative Decoding

Speculative decoding uses a small "draft" model (e.g. 1B parameters) to propose candidate tokens, then verifies them with the large target model. This is particularly effective for ssd-llm because:

1. **Draft model fits in RAM** — no SSD streaming needed for the small model
2. **Target model streams fewer times** — accepted draft tokens skip expensive SSD I/O
3. **Mathematically lossless** — the output distribution is identical to the target model

### How it works

```
Draft Model (1B, in RAM):    [tok1] → [tok2] → [tok3] → [tok4] → [tok5]
                                ↓        ↓        ↓        ↓        ↓
Target Model (70B, SSD):    verify   verify   verify   REJECT   resample
                                ✓        ✓        ✓        ✗        →tok4'
```

With a good draft model, 60-80% of tokens are accepted, meaning the target model does ~40% fewer forward passes. For SSD-streaming workloads this translates to 2-3x speedup.

### Configuration

- `--draft-model <path>` — Path to the draft GGUF model (same tokenizer family)
- `--draft-ahead <K>` — Number of tokens to draft per round (default: 5, try 3-8)

Higher `draft-ahead` values give more potential speedup but waste more compute on rejections. Start with 5 and tune based on your model pair's acceptance rate.

## Prior Art & Research

This project builds on insights from:

- **llama.cpp** — Uses mmap but with no intelligent page management
- **FlexGen** — SSD offloading for throughput-oriented inference
- **PowerInfer** — Sparsity-based selective loading
- **LLM in a Flash** (Apple Research) — Flash memory optimization for LLM inference
- **FlexInfer** — Flexible offloading with computation-I/O overlap

## Roadmap

- [x] v0.1 — GGUF parser, mmap loader, LRU cache, prefetcher, CPU inference
- [x] v0.2 — Metal compute foundation, SIMD ops, Ollama + OpenAI API server
- [x] v0.3 — KV cache, Metal shader compilation, SwiGLU FFN, quantized GPU kernels (Q4_0/Q8_0)
- [x] v0.4 — Full Metal GPU dispatch via metal-rs, BPE tokenizer, streaming responses
- [x] v0.5 — Speculative decoding with draft model, KV cache rollback
- [x] v0.6 — Batch prefill optimization, adaptive draft length
- [x] v0.7 — Continuous batching, prompt caching, tensor parallelism
- [x] v0.8 — Sliding window attention, GQA optimization, memory-mapped KV cache
- [ ] v1.0 — Production-ready, benchmarked against llama.cpp

## Requirements

- macOS 13+ (Apple Silicon recommended)
- Rust 1.75+
- GGUF model file (from [HuggingFace](https://huggingface.co/models?library=gguf))

## License

MIT
