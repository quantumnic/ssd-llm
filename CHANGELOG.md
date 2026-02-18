# Changelog

## v0.2.0 — Metal Compute + API Server (2026-02-18)

### Added
- **Ollama-compatible API server** (`ssd-llm serve`) with endpoints:
  - `POST /api/generate` — text generation
  - `POST /api/chat` — chat completion
  - `GET /api/tags` — list models
  - `GET /api/version` — server version
- **OpenAI-compatible endpoint** (`POST /v1/chat/completions`)
- **Metal compute module** with SIMD-optimized operations:
  - 4-wide accumulator matrix-vector multiply (`matvec_f32_simd`)
  - Fast RMS normalization (`rmsnorm_f32_fast`)
  - Numerically stable softmax (`softmax_f32_fast`)
  - RoPE (Rotary Position Embedding) (`rope_f32_fast`)
- **OpenAI API types** with ChatML prompt formatting
- **Unit tests** for Metal compute operations (matvec, rmsnorm, softmax)
- `serve` CLI subcommand with `--host`, `--port` options
- Metal availability detection on startup

### Changed
- Inference pipeline now uses optimized compute functions from `metal::compute`
- `info` command shows architecture details (layers, heads, KV heads, context, vocab)
- Updated README with API documentation, revised comparison table, updated roadmap

### Metal Shaders (foundation for v0.3)
- `matmul.metal` — tiled matrix multiplication with shared memory
- `rmsnorm.metal` — RMS normalization
- `rope.metal` — Rotary position embeddings
- `softmax.metal` — numerically stable softmax

## v0.1.0 — Initial Release (2026-02-17)

### Added
- GGUF file format parser (v2/v3)
- Memory-mapped model loader with madvise hints
- LRU layer cache with configurable memory budget
- Predictive prefetcher (look-ahead strategy)
- CPU inference engine (transformer forward pass)
- Multi-Head Attention with GQA support
- SwiGLU Feed-Forward Network
- Temperature, Top-K, Top-P sampling
- Basic tokenizer from GGUF vocabulary
- CLI: `run`, `info`, `bench` subcommands
- SSD streaming benchmark
