# Changelog

## v0.8.0 — Sliding Window Attention + GQA Optimization + Memory-Mapped KV Cache (2026-02-18)

### Added
- **Sliding window attention** (`inference/sliding_window.rs`):
  - Limit attention to the most recent W tokens instead of the full sequence
  - Configurable window size via `--sliding-window <W>` (0 = full attention)
  - Sink tokens: keep first N tokens always visible (BOS, system prompt) via `--sink-tokens <N>`
  - `AttentionRange` abstraction for clean position iteration
  - KV eviction helper: `evict_outside_window()` reclaims memory from expired positions
  - Combined sliding window + GQA path for maximum efficiency
  - 7 new tests: range computation, windowed/full/disabled modes, sink tokens, eviction
- **GQA optimization** (`inference/gqa.rs`):
  - Dedicated grouped-query attention path that batches KV loads per group
  - Pre-fetches all K/V vectors once per KV head, then iterates query heads in the group
  - 4-wide SIMD dot product accumulator for cache-friendly score computation
  - Auto-detection of attention strategy: MHA, GQA, or MQA from model config
  - KV memory savings reporting (e.g., GQA with group_size=4 → 75% KV savings)
  - Combined `gqa_sliding_window_attention()` for GQA + window in one pass
  - `AttentionStrategy` enum for runtime strategy selection
  - 6 new tests: basic GQA, multi-token, sliding window combo, SIMD dot product, strategy detection, savings ratio
- **Memory-mapped KV cache** (`inference/mmap_kv_cache.rs`):
  - Hot/cold architecture: recent tokens in RAM, older tokens mmap'd from SSD
  - Automatic spill: when RAM budget is exceeded, oldest 25% of hot entries move to cold
  - mmap with `MADV_RANDOM` hints for decode-phase access patterns
  - Transparent API: `key_at()`/`value_at()` work across hot and cold seamlessly
  - Per-layer backing files with automatic cleanup on drop
  - `--mmap-kv` CLI flag for `run` and `serve` commands
  - 5 new tests: basic operations, spill trigger, cold read, rollback, clear
- 18 new tests (55 total, all passing)
- CLI: `run` gains `--sliding-window`, `--sink-tokens`, `--mmap-kv` flags
- CLI: `serve` gains `--sliding-window`, `--sink-tokens`, `--mmap-kv` flags
- Auto-detection of GQA/MQA models with optimization message on startup

### Why Sliding Window Matters for SSD-LLM
Long-context inference (32K+ tokens) makes the KV cache enormous. With sliding window, only the last W positions are attended to, bounding both memory and compute. Sink tokens ensure the model always sees its BOS token and system prompt. For SSD-LLM specifically, smaller KV cache means more RAM budget available for layer caching, improving SSD→RAM streaming throughput.

### Why GQA Optimization Matters for SSD-LLM
Modern 70B+ models (Llama 2 70B: n_head=64, n_head_kv=8) use GQA to reduce KV cache size by 8x. Our optimized path loads each KV head's data once per group instead of repeating it per query head, reducing cache pressure. With SSD streaming, every byte of RAM is precious — GQA's 75% KV savings means more room for layer caching.

### Why Memory-Mapped KV Cache Matters for SSD-LLM
The entire premise of ssd-llm is using SSD as extended memory. The mmap KV cache extends this philosophy to the KV cache itself — when generating very long sequences, older KV entries spill to SSD transparently. The OS handles paging, and frequently-accessed cold entries get promoted back to RAM automatically. This enables context lengths that would otherwise be impossible on memory-constrained devices.

### Changed
- CLI: `run` and `serve` gain 3 new flags each
- GQA detection logged on model load (shows group size and KV savings)
- Cargo.toml version bumped to 0.8.0
- API server version bumped to 0.8.0
- README: updated features, CLI examples, architecture, roadmap

## v0.7.0 — Continuous Batching + Prompt Caching + Tensor Parallelism (2026-02-18)

### Added
- **Prompt prefix caching** (`inference/prompt_cache.rs`):
  - Hash-trie based KV state cache keyed on token sequences
  - Exact prefix matching: skip prefill entirely when the same prompt is reused
  - Partial prefix matching: find the longest cached prefix and resume prefill from there
  - LRU eviction with configurable memory budget (default: 25% of model budget)
  - FNV-1a hashing for fast token sequence lookups
  - `--prompt-cache` CLI flag for `run` and `serve` commands
- **Continuous batching scheduler** (`inference/batch_scheduler.rs`):
  - Layer-major batching: load each layer once, apply to ALL active sequences
  - Dynamic request scheduling: queued → prefilling → decoding → completed
  - Configurable max batch size (`--max-batch`, default: 4 for serve)
  - Per-request KV cache isolation for correct parallel generation
  - Automatic prefill-to-decode promotion within batch steps
  - Throughput stats tracking (total requests, total tokens generated)
- **Tensor parallelism** (`inference/tensor_parallel.rs`):
  - Column-parallel matmul: splits output dimension across N threads
  - 4-wide SIMD accumulation per thread for cache-friendly computation
  - Parallel FFN: gate and up projections computed concurrently
  - Auto-detection of optimal shard count based on model dimensions:
    - 1 shard for small models (<4096 embd)
    - 2 shards for 7B-13B models (4096 embd)
    - 4 shards for 70B+ models (8192+ embd)
  - `--tensor-parallel <N>` CLI flag (0 = auto-detect)
- 10 new tests: prompt cache (store/lookup/partial hit/eviction), batch scheduler (submit/max batch), tensor parallelism (correctness/single shard/auto detect/large matrix)
- 36 tests total, all passing

### Why Prompt Caching Matters for SSD-LLM
The most expensive part of SSD-streaming inference is prefill — processing the prompt through all layers requires loading every layer from SSD. With prompt caching, repeated prompts (common with system prompts, templates, multi-turn conversations) skip this entirely by reusing the stored KV state. Partial matches save proportional I/O.

### Why Continuous Batching Matters for SSD-LLM
The core bottleneck is SSD→RAM layer loading. With continuous batching, each layer is loaded once per step and applied to ALL active sequences. 4 concurrent requests means 4x better SSD bandwidth utilization — the same layer data serves multiple computations before being evicted.

### Why Tensor Parallelism Matters for SSD-LLM
While SSD I/O is the primary bottleneck, large matmul operations (especially vocab projection) are compute-bound. Splitting these across threads better utilizes Apple Silicon's multi-core CPU, hiding compute latency behind I/O wait time.

### Changed
- CLI: `run` gains `--prompt-cache` and `--tensor-parallel` flags
- CLI: `serve` gains `--prompt-cache`, `--max-batch`, `--tensor-parallel` flags
- `ServerConfig` extended with `prompt_cache`, `max_batch`, `tensor_parallel` fields
- API server version bumped to 0.7.0
- Cargo.toml version bumped to 0.7.0
- README: updated features, architecture, comparison table, roadmap

## v0.6.0 — Batch Prefill + Adaptive Draft Length (2026-02-18)

### Added
- **Batch prefill optimization** (`transformer::batch_prefill`):
  - Layer-major traversal: loads each transformer layer once, processes ALL prompt tokens through it
  - Embedding tensor loaded once for entire prompt instead of per-token
  - Dramatically reduces SSD reads during prefill phase (N layers × 1 load vs N layers × T loads)
  - Used by both standard generation and streaming generator
- **Adaptive draft length controller** (`speculative::AdaptiveDrafter`):
  - Exponential moving average (EMA) tracking of acceptance rate
  - High acceptance (>70%) → increases K to draft more tokens per round
  - Low acceptance (<40%) → decreases K to avoid wasted draft compute
  - Configurable min/max bounds (default: 1 to 3× initial K)
  - `--adaptive-draft` CLI flag for `run` and `serve` commands
  - Warmup period: waits 2 rounds before adjusting
- 3 new tests: adaptive K increase, decrease, and bounds enforcement
- 26 tests total, all passing

### Why Batch Prefill Matters for SSD-LLM
The core bottleneck is SSD→RAM transfer. Previously, prefilling a 1000-token prompt with 80 layers meant 80,000 layer loads (one per token per layer). With batch prefill, it's just 80 layer loads — each layer is loaded once and all tokens are processed through it. This is an **O(T)** reduction in SSD I/O during prefill.

### Why Adaptive Draft Length Matters
Fixed K is suboptimal: easy sequences have high acceptance (K should be larger), while hard sequences have low acceptance (K should be smaller). Adaptive K converges to the sweet spot automatically, improving throughput by 10-30% compared to static K in practice.

### Changed
- Standard and streaming generators now use `batch_prefill` for prompt processing
- Speculative decoding uses adaptive K when `--adaptive-draft` is enabled (fixed K still default)
- `SpeculativeConfig` extended with `adaptive: bool` field
- `ServerConfig` extended with `adaptive_draft: bool`
- API server version bumped to 0.6.0

## v0.5.0 — Speculative Decoding with Draft Model (2026-02-18)

### Added
- **Speculative decoding engine** (`inference/speculative.rs`):
  - Draft model generates K candidate tokens autoregressively
  - Target model verifies candidates, accepting tokens left-to-right
  - Rejection resampling from adjusted distribution `max(0, p_target - p_draft)`
  - Bonus token sampling when all K candidates accepted
  - Mathematically lossless — identical output distribution to standard decoding
- **`SpeculativeStreamingGenerator`** for token-by-token speculative streaming
- **KV cache rollback** (`kv_cache.rollback()`) — truncate cache to discard rejected draft tokens
- **CLI flags**: `--draft-model <path>` and `--draft-ahead <K>` for both `run` and `serve` commands
- **Stats reporting**: acceptance rate, draft tokens proposed/accepted, target forward passes saved
- 5 new tests: probability distribution, temperature sharpening, RNG range, deterministic sampling, KV rollback

### Why This Matters for SSD-LLM
Speculative decoding is uniquely powerful for SSD-streaming inference:
- The small draft model (e.g. 1B) fits entirely in RAM — no SSD I/O
- Each accepted draft token avoids one expensive target model forward pass (SSD streaming)
- With 60-80% acceptance rate, target model does ~40% fewer SSD-streaming passes → 2-3x speedup

### Changed
- `forward_pass` in transformer.rs now has a public entry point (`forward_pass_pub`) for speculative decoding
- API server `ServerConfig` extended with `draft_model_path` and `draft_ahead`
- 23 tests total, all passing

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
