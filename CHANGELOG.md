# Changelog

## v0.4.0 — Metal GPU Dispatch + BPE Tokenizer + Streaming (2026-02-18)

### Added
- **Full Metal GPU dispatch via metal-rs** — real GPU compute pipelines for:
  - `matvec_f32` — matrix-vector multiply on GPU
  - `rmsnorm_f32` — RMS normalization (two-phase GPU reduction)
  - `softmax_f32` — numerically stable softmax on GPU
  - `rope_f32` — Rotary Position Embeddings on GPU
  - `silu_f32` — SiLU activation on GPU
  - Automatic CPU/GPU dispatch based on tensor size threshold (4096 elements)
- **BPE Tokenizer** with full Byte-Pair Encoding support:
  - Standard BPE with explicit merge rules from GGUF metadata
  - SentencePiece-style BPE using token scores for merge priority
  - Byte-level fallback (`<0xNN>`) for unknown characters
  - `decode_token()` for single-token streaming decode
  - Greedy longest-match fallback for models without merge data
- **Streaming API responses** for all endpoints:
  - Ollama `/api/generate` and `/api/chat`: chunked transfer encoding (NDJSON)
  - OpenAI `/v1/chat/completions`: Server-Sent Events (SSE)
  - `stream` parameter (default: true for Ollama, false for OpenAI)
  - Token-by-token delivery as they're generated
- **`StreamingGenerator`** — iterator-style token generation for the inference engine

### Changed
- `MetalCompute` now auto-dispatches to real GPU pipelines when metal-rs is available
- Tokenizer upgraded from simple greedy to full BPE (backward compatible)
- API server version bumped to 0.4.0
- Updated README: roadmap, architecture docs, feature list

### Dependencies
- Added `metal` 0.29, `objc` 0.2, `block` 0.1 (macOS only)
- Added `serde` 1.0, `serde_json` 1.0

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
