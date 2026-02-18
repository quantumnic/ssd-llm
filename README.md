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
- **🔌 Ollama-compatible API** — Drop-in replacement server with OpenAI-compatible endpoint

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

> v0.3 — KV cache, Metal shaders compiled, SwiGLU FFN, quantized GPU kernels

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
    tokenizer.rs       — Basic tokenizer from GGUF vocab
  metal/
    compute.rs         — Metal compute + SIMD-optimized ops
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
- [ ] v0.4 — Full Metal GPU dispatch via metal-rs, BPE tokenizer, streaming responses
- [ ] v0.5 — Speculative decoding with draft model
- [ ] v1.0 — Production-ready, benchmarked against llama.cpp

## Requirements

- macOS 13+ (Apple Silicon recommended)
- Rust 1.75+
- GGUF model file (from [HuggingFace](https://huggingface.co/models?library=gguf))

## License

MIT
