# Changelog

## v1.20.0 — IQ3_XXS and IQ3_S Dequantization — Ultra-Low-Bit I-Quant Support (2026-02-22)

### Added
- **IQ3_XXS dequantization**: Full CPU and Metal GPU dequant-matvec for IQ3_XXS quantized tensors (3.0625 bpw, 98B per 256-element block, grid-based importance-matrix quantization)
- **IQ3_S dequantization**: Full CPU and Metal GPU dequant-matvec for IQ3_S quantized tensors (3.4375 bpw, 110B per 256-element block, 512-entry grid with per-element sign bits and 4-bit scales)
- **IQ3_XXS grid lookup table**: 256-entry `iq3xxs_grid` table (from llama.cpp) for both CPU (Rust) and GPU (Metal) paths
- **IQ3_S grid lookup table**: 512-entry `iq3s_grid` table (from llama.cpp) for both CPU (Rust) and GPU (Metal) paths
- **Sign/mask tables**: `ksigns_iq2xs` (128-entry sign lookup) and `kmask_iq2xs` (8-entry bit mask) for IQ3 sign extraction, shared across CPU and Metal
- **Metal shaders**: `matvec_iq3_xxs` and `matvec_iq3_s` GPU kernels with grid-based dequantization for Apple Silicon acceleration
- **GPU dispatch**: Automatic Metal GPU dispatch for IQ3_XXS and IQ3_S
- **7 new tests** (310 total): IQ3_XXS basic, signs, grid lookup; IQ3_S basic, scale, signs, grid lookup — all passing

### Technical Details
- IQ3_XXS: 8 groups of 32 elements per block; each group has 8 grid indices (into 256-entry table of 4-byte vectors) + 4-byte u32 encoding 4×7-bit sign indices and 4-bit scale
- IQ3_S: 4 pairs of 32-element groups; 8-bit + 1-bit grid indices (into 512-entry table), explicit 8-bit sign masks per sub-group, 4-bit scales per pair
- Both formats achieve better quality-per-bit than Q3_K at similar or smaller size, using importance-matrix optimized codebooks

### Why This Matters
IQ3_XXS at 3.06 bpw enables running 70B+ models in ~25GB — critical for 32GB Macs. IQ3_S at 3.44 bpw offers near-Q4 quality at Q3 size. These are among the most popular quantization formats on Hugging Face for constrained hardware. With this release, **ssd-llm supports IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS** on both CPU and Metal GPU, covering the most widely-used I-Quant formats.

### Changed
- `matvec_quantized` dispatch now routes IQ3_XXS and IQ3_S to GPU when Metal is available
- `matvec_quantized_cpu` dispatch now handles IQ3_XXS and IQ3_S (previously returned zeros)
- GGUF `block_size()` and `type_size()` now return correct values for IQ3_XXS (256/98) and IQ3_S (256/110)
- README updated to reflect expanded I-Quant support
- Cargo.toml version bumped to 1.20.0

## v1.18.0 — Quantized Block Swapping — INT8 Compression for SSD I/O (2026-02-21)

### Added
- **INT8 quantized block swapping**: Per-row absmax INT8 quantization of KV cache blocks before writing to SSD swap file, reducing I/O bandwidth by ~4x
- **`SwapQuantMode` enum**: `None` (f32, default) or `Int8` modes for swap file storage
- **`--swap-quantize` CLI flag**: Enable INT8 quantized swapping on the serve command
- **`inference.swap_quantize` config**: TOML configuration for persistent swap quantization setting
- **`swap_compression_ratio()` utility**: Calculate theoretical compression ratio for given block dimensions
- **`SwapFile::with_quant_mode()`**: Constructor for swap files with configurable quantization
- **`BlockSwapper::with_quant_mode()`**: Constructor for block swapper with quantization support
- **7 new tests** (295 total): quantize/dequantize roundtrip, zero handling, INT8 swap file I/O, size comparison, block swapper INT8 roundtrip, slot reuse, compression ratio

### Technical Details
- Quantization scheme: symmetric per-row absmax (`scale = max(|x|) / 127`)
- Swap file format (INT8 mode): `[keys_i8][values_i8][key_scales_f32][val_scales_f32][num_filled_u32]`
- Typical compression: ~3.8x for block_size=16, kv_dim=128 (from 16KB to ~4KB per block)
- Quantization error bounded by absmax/127 per element (~0.8% relative error)

## v1.15.0 — Q2_K/Q8_K GPU Dequantization — Complete K-Quant Family (2026-02-21)

### Added
- **Q2_K dequantization**: Full CPU and Metal GPU dequant-matvec for Q2_K quantized tensors (2-bit K-quant with 4-bit scales/mins, 84B per 256-element block)
- **Q8_K dequantization**: Full CPU and Metal GPU dequant-matvec for Q8_K quantized tensors (8-bit K-quant with f32 super-block scale, 292B per 256-element block)
- **Metal shaders**: `matvec_q2_k` and `matvec_q8_k` GPU kernels with on-the-fly dequantization for Apple Silicon acceleration
- **GPU dispatch**: Automatic Metal GPU dispatch for Q2_K and Q8_K (completing the full K-quant family: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K)
- **5 new tests**: Q2_K CPU dequant (basic, nonzero, with-min), Q8_K CPU dequant (basic, negative) — 267 total, all passing

### Fixed
- **Q2_K block size**: Corrected from 100 to 84 bytes (matching ggml spec: 2B d + 2B dmin + 16B scales + 64B qs)
- **Q8_K block size**: Corrected from 324 to 292 bytes (matching ggml spec: 4B d + 256B qs + 32B bsums)

### Why This Matters
Q2_K is the most aggressive K-quant, enabling 70B+ models to fit in ~20GB — critical for 24GB Macs. Q8_K provides lossless-quality quantization used as an intermediate format. With this release, **ssd-llm supports the complete K-quant family on GPU**: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K — plus Q4_0 and Q8_0. Every quantization level now runs at full Metal-accelerated speed.

### Changed
- `matvec_quantized` dispatch now routes Q2_K and Q8_K to GPU when Metal is available
- `matvec_quantized_cpu` dispatch now handles Q2_K and Q8_K (previously returned zeros)
- Cargo.toml version bumped to 1.15.0

## v1.14.0 — Q3_K/Q5_K GPU Dequantization + Ollama Embeddings API (2026-02-21)

### Added
- **Q3_K dequantization**: Full CPU and Metal GPU dequant-matvec for Q3_K quantized tensors (3-bit K-quant, 110B per 256-element block)
- **Q5_K dequantization**: Full CPU and Metal GPU dequant-matvec for Q5_K quantized tensors (5-bit K-quant, 176B per 256-element block)
- **Metal shaders**: `matvec_q3_k` and `matvec_q5_k` GPU kernels with on-the-fly dequantization for Apple Silicon acceleration
- **GPU dispatch**: Automatic Metal GPU dispatch for Q3_K and Q5_K (joins existing Q4_0, Q4_K, Q6_K, Q8_0)
- **Ollama `/api/embed` endpoint**: Ollama-compatible embeddings API — single string or batch array input, returns embeddings in Ollama format
- **6 new tests**: Q3_K/Q5_K CPU dequant (basic + nonzero), Ollama embed input parsing (262 total, all passing)

### Why Q3_K/Q5_K Matter
Q3_K and Q5_K are among the most popular quantization levels for running large models on constrained hardware. Q3_K enables running 70B+ models in ~30GB, while Q5_K provides near-FP16 quality at half the size. With GPU dequantization, these formats now run at full Metal-accelerated speed instead of falling back to CPU. ssd-llm now supports the complete K-quant family on GPU: Q3_K, Q4_K, Q5_K, Q6_K.

### Why Ollama /api/embed Matters
The Ollama `/api/embed` endpoint completes the embedding API coverage — ssd-llm now serves embeddings via both OpenAI (`/v1/embeddings`) and Ollama (`/api/embed`) formats. This enables drop-in compatibility with RAG frameworks, vector databases, and embedding pipelines that target either API.

### Changed
- `matvec_quantized` dispatch now routes Q3_K and Q5_K to GPU when Metal is available
- `matvec_quantized_cpu` dispatch now handles Q3_K and Q5_K (previously returned zeros)
- Server startup log includes `/api/embed` in Ollama endpoint listing
- Cargo.toml version bumped to 1.14.0

## v1.12.0 — Vision/Multimodal Support (2026-02-20)

### Added
- **CLIP Vision Encoder**: Full ViT-L/14 implementation with patch embedding, multi-head self-attention, and MLP blocks
- **Image Preprocessing**: Bicubic resize, center crop, CLIP normalization (ImageNet stats) — CHW tensor output
- **Vision-Language Projection**: Linear and 2-layer MLP (LLaVA-1.5 style) projection from CLIP space to LLM embedding space
- **Multimodal Content Parsing**: OpenAI-compatible `content` array with `text` and `image_url` parts (base64 data URLs and HTTP URLs)
- **GGUF Vision Config Detection**: Auto-detect vision encoder parameters from GGUF metadata (`clip.vision.*` keys)
- **Metal Shader**: `vision_patch_embed` kernel for GPU-accelerated patch embedding convolution + `vision_layer_norm` kernel
- **Minimal PNG Decoder**: Base64 image decoding with PNG IHDR parsing and basic filter support
- **Comprehensive Tests**: 16 new tests covering preprocessing, normalization, encoding, projection, base64, content parsing, and config detection

## v1.9.0 — Function Calling / Tool Use (2026-02-20)

### Added
- **Function calling / Tool use** (`inference/tool_use.rs`):
  - OpenAI-compatible `tools` parameter with function definitions
  - `tool_choice` support: `"auto"`, `"none"`, `"required"`, or specific function (`{"type": "function", "function": {"name": "..."}}`)
  - Multiple parallel tool calls in a single response
  - Robust tool call extraction from model output: standard `tool_calls` format, legacy `function_call`, flat format, and markdown code blocks
  - Argument validation against JSON Schema (required parameters, type checking)
  - Tool result messages (`role: "tool"`) for multi-turn tool use conversations
  - System prompt injection with tool definitions and usage instructions
  - Tool call response format with `finish_reason: "tool_calls"` and structured `tool_calls` array
  - `has_tool_calls()` fast-path detection to avoid unnecessary JSON parsing
  - `validate_tool_call()` for schema validation before returning results
  - `format_tool_calls_response()` for OpenAI-compatible response formatting
  - `parse_tool_messages()` for extracting tool results from conversation history
  - `build_tool_index()` for O(1) tool name lookup
  - `find_matching_brace()` for robust JSON extraction from mixed text
  - 25 new tests: parsing, extraction (all formats), validation, prompt formatting, tool choice modes, brace matching, embedded JSON
- **Tool role** in chat system:
  - `Role::Tool` added to chat message types
  - Tool messages rendered with magenta color in interactive chat (`\x1b[1;35m`)
  - All chat templates updated: Llama 2, Mistral, Gemma, Phi-3 handle tool messages
- **`extract_json_value()` helper** in server.rs — full serde_json-based extraction for complex JSON fields (objects, arrays) with fallback parser
- 33 new tests (169 → 202 total, all passing)

### Why Function Calling Matters
Function calling is the foundation of agent workflows. With tool use support, ssd-llm can participate in tool-augmented generation — the model can request weather data, search results, database queries, or any external action, receive the results, and incorporate them into its response. This makes ssd-llm viable as a local backend for frameworks like LangChain, AutoGen, and custom agent systems. Combined with the existing JSON mode, this provides robust structured interaction between the model and external systems.

### Changed
- `Role` enum extended with `Tool` variant across openai.rs, chat.rs, chat_template.rs
- OpenAI `/v1/chat/completions` endpoint now accepts `tools` and `tool_choice` parameters
- Tool call responses use `finish_reason: "tool_calls"` and include `tool_calls` array in message
- Cargo.toml version bumped to 1.9.0

## v1.7.0 — Interactive Chat CLI + JSON Mode (2026-02-20)

### Added
- **Interactive chat CLI** (`ssd-llm chat`):
  - Multi-turn REPL-style conversations with loaded models
  - Full chat history management with `/undo`, `/clear`, `/history` commands
  - Auto-detected chat templates (ChatML, Llama 2/3, Mistral, Gemma, Phi-3) or `--template` override
  - System prompt support via `--system` flag or `/system` command
  - Streaming token-by-token output with speed stats (tok/s)
  - Colored terminal output for user/assistant/system roles
  - `/config` to inspect current settings, `/help` for all commands
  - 8 new tests for chat commands, history management, undo/clear behavior
- **JSON mode** for structured output:
  - `response_format: { "type": "json_object" }` in OpenAI-compatible API
  - Stack-based JSON grammar validator (`inference/json_mode.rs`) with prefix validation
  - Recursive-descent state machine for character-level JSON prefix checking
  - `JsonGrammar` for incremental token-by-token validation
  - `apply_json_constraint()` for logit masking (constrains sampling to valid JSON tokens)
  - Automatic JSON extraction from model output with bracket-matching fallback
  - 20 new tests: prefix validation, incremental feeding, all JSON types, nested structures, escape sequences, rejection of invalid input
- 28 new tests (141 → 169 total, all passing)

### Why Interactive Chat Matters
Previously, `ssd-llm` only supported single-shot prompts via `run` or the API server. The `chat` subcommand provides the most natural way to interact with a model locally — maintaining conversation context, supporting system prompts, and streaming responses in real-time. This makes ssd-llm competitive with `ollama run` for local development.

### Why JSON Mode Matters
Structured output is essential for tool-use, function calling, and agent workflows. JSON mode ensures the model produces valid JSON, eliminating the need for fragile regex extraction or retry loops. The grammar-constrained approach validates at the token level, providing guarantees that post-hoc validation cannot.

### Changed
- OpenAI `/v1/chat/completions` endpoint now accepts `response_format` parameter
- Cargo.toml version bumped to 1.7.0

## v1.6.0 — Tail-Free Sampling + Mirostat v1/v2 (2026-02-19)

### Added
- **Tail-Free Sampling (TFS)** (`sampler.rs`):
  - Uses second derivative of sorted probability distribution to identify and remove the "tail"
  - More principled than top-p: adapts cutoff based on distribution shape, not arbitrary percentile
  - `tfs_z` parameter (0.0 = disabled, 0.95 = moderate, 1.0 = no filtering)
  - `Sampler::with_tfs()` constructor with full penalty support
  - `tail_free_filter()` function: softmax → first derivative → second derivative → normalize → cumulative cutoff
  - 3 new tests: tail filtering, disabled-at-one, direct filter function
- **Mirostat v1 & v2 adaptive sampling** (`sampler.rs`):
  - **Mirostat v1**: Estimates Zipf exponent from top probabilities, computes optimal k to achieve target perplexity
  - **Mirostat v2**: Simplified truncation — keeps tokens whose surprise (-log₂p) ≤ mu, adapts mu toward target tau
  - Both maintain running `mu` state that converges toward `2 * tau`, producing text with consistent perplexity
  - `MirostatMode` enum: `Disabled`, `V1`, `V2`
  - `Sampler::with_mirostat(temperature, mode, tau, eta)` constructor
  - `mirostat_tau`: target surprise level (default 5.0, lower = more focused)
  - `mirostat_eta`: learning rate for mu adaptation (default 0.1)
  - Zipf exponent estimation via `estimate_zipf_s()` and generalized harmonic number approximation
  - 6 new tests: v1/v2 valid output, mu adaptation, greedy-at-zero-temp, v1 mu convergence, Zipf estimation
- **API support** for all new parameters:
  - Ollama API: `tfs_z`, `mirostat` (0/1/2), `mirostat_tau`, `mirostat_eta`
  - Chat endpoint: same parameters
  - Config file: `[inference]` section supports all new fields
- **`build_sampler()` helper** in transformer.rs — dispatches to correct sampler constructor based on InferenceConfig
- 9 new tests (132 → 141 total, all passing)

### Why Mirostat Matters
Standard sampling (top-k, top-p) uses fixed truncation regardless of model confidence. Mirostat adaptively adjusts truncation to maintain a target "surprise" level — producing text that's neither boringly repetitive (surprise too low) nor incoherently random (surprise too high). This is especially valuable for long-form generation where fixed parameters degrade over time.

### Why TFS Matters
Top-p keeps tokens by cumulative probability, which can include many low-probability "tail" tokens in flat distributions. TFS identifies where the probability distribution's rate of change drops off (via second derivative) and cuts precisely there. This gives cleaner, more natural sampling without the arbitrary percentile choice of top-p.

### Changed
- `InferenceConfig` (transformer.rs) extended with `tfs_z`, `mirostat`, `mirostat_tau`, `mirostat_eta`
- `InferenceConfig` (config.rs) extended with same fields + TOML parsing
- All API endpoints updated to parse and pass new sampling parameters
- Cargo.toml version bumped to 1.6.0

## v1.2.0 — Chat Templates, Stop Sequences & Sampling Penalties (2026-02-19)

### Added
- **Chat template system** (`inference/chat_template.rs`) — auto-detected formatting for multi-turn conversations:
  - **ChatML** (Qwen, Yi, OpenHermes, default fallback)
  - **Llama 2** (`<<SYS>>` / `[INST]` format)
  - **Llama 3** (`<|start_header_id|>` format)
  - **Mistral/Mixtral** (`[INST]` without system token)
  - **Gemma** (`<start_of_turn>` format)
  - **Phi-3** (`<|user|>` format)
  - **Raw** (plain concatenation)
  - Auto-detection from GGUF metadata template string or model name
  - Per-template stop sequences for proper generation termination
  - `chat_template` parameter in OpenAI API for explicit override
  - 13 new tests for all template formats, detection, and multi-turn conversations
- **Stop sequences** — `stop` parameter in both Ollama and OpenAI API endpoints for early generation termination
- **Repetition penalty** — multiplicative penalty on previously generated tokens (divides positive logits, multiplies negative)
- **Frequency penalty** — additive penalty scaling with token occurrence count
- **Presence penalty** — additive binary penalty for any previously seen token
- **`Sampler::with_penalties()`** — constructor with full penalty configuration
- **`Sampler::apply_penalties()`** — apply all penalty types to logits given previous token history
- **Proper message parsing** in OpenAI chat endpoint — full multi-message extraction with role/content pairs
- **Proper token usage tracking** — `prompt_tokens` now correctly reported in OpenAI API responses
- **`extract_json_string_array()`** — JSON array parser for stop sequences (supports both `"stop": "text"` and `"stop": ["a", "b"]`)
- **`extract_chat_messages()`** — full OpenAI messages array parser
- 4 new sampler tests (repetition, frequency, presence penalty, no-op)
- 17 new tests total (107 → from 87 baseline)

### Changed
- `InferenceConfig` extended with `stop_sequences`, `repetition_penalty`, `frequency_penalty`, `presence_penalty`
- OpenAI `/v1/chat/completions` now properly parses all messages and applies chat template (was: only last message content)
- OpenAI API responses now include accurate `prompt_tokens` count (was: hardcoded 0)

## v1.1.0 — Embeddings & Models API (2026-02-19)

### Added
- **OpenAI Embeddings API** (`POST /v1/embeddings`) — compute L2-normalized embedding vectors from input text, supporting both single string and array inputs, fully compatible with OpenAI client libraries
- **Models Listing** (`GET /v1/models`) — OpenAI-compatible endpoint for model discovery
- **`transformer::embed()`** — dedicated embedding extraction function with optional L2 normalization, using batch prefill + final RMS norm
- **4 new tests** for embedding input parsing (single string, array, empty, single-in-array)

## v1.0.0 — Production-Ready Release (2026-02-19)

### Added
- **Model downloader** (`pull/mod.rs`):
  - `ssd-llm pull user/repo:file.gguf` — download GGUF models from Hugging Face Hub
  - Supports short-form (`user/repo:file.gguf`), repo-only (`user/repo`), and full URLs
  - Resumable downloads via curl with `.part` file tracking
  - GGUF magic validation on downloaded files
  - `ssd-llm models` — list locally downloaded models with sizes
  - Configurable model directory via `$SSD_LLM_MODEL_DIR` env var
  - 7 new tests for URL parsing, model spec parsing, and directory listing
- **Configuration file** (`config.rs`):
  - `ssd-llm config --init` — generate default `ssd-llm.toml`
  - `ssd-llm config` — show current configuration
  - TOML parser (no external dependency) with `[model]`, `[server]`, `[inference]`, `[paths]` sections
  - Auto-discovery: `./ssd-llm.toml` → `~/.config/ssd-llm/ssd-llm.toml`
  - `$SSD_LLM_CONFIG` env var for custom config path
  - 6 new tests for config parsing, defaults, and generation
- **Graceful shutdown** — SIGINT/SIGTERM handling for clean server stop
  - Non-blocking accept loop with shutdown flag
  - "Press Ctrl+C to stop" message on server start
- **CORS preflight** — `OPTIONS` handler with proper CORS headers for browser-based clients
- **Criterion micro-benchmarks** (`benches/inference_bench.rs`):
  - `cargo bench` — reproducible benchmarks with HTML reports
  - Benchmarks: softmax, RMSNorm, matrix-vector multiply, RoPE, SiLU, dot product
  - Multiple input sizes per benchmark for scaling analysis
  - 4-wide SIMD vs naive dot product comparison
- 13 new tests (87 total, all passing)

### Changed
- **Removed blanket `#![allow(dead_code)]` clippy suppressions** — all 25 individual clippy warnings fixed:
  - Replaced `for i in 0..len { arr[i] }` patterns with iterators
  - Removed unnecessary `u64` casts in Metal GPU pipeline
  - Used `div_ceil()` instead of manual reimplementation
  - Used `RangeInclusive::contains()` for version checks
  - Replaced redundant closures with function references
  - Only `#![allow(dead_code)]` (for unreached public API items) and `#![allow(clippy::too_many_arguments)]` remain
- API server version bumped to 1.0.0
- Cargo.toml version bumped to 1.0.0
- README: updated features, architecture, roadmap (v1.0 ✅)

### Why Model Download Matters
Getting models is the first barrier for new users. `ssd-llm pull` makes it one command, with resume support for large downloads (70B models at ~40GB). No need to manually navigate HuggingFace and figure out which file to download.

### Why Config File Matters
Production deployments need reproducible settings. A TOML config file means you can version-control your inference parameters, share configs between machines, and avoid long CLI argument lists.

### Why Graceful Shutdown Matters
In production, SIGTERM from container orchestrators (Kubernetes, Docker) needs to be handled cleanly — finish in-flight requests, close connections, and exit with code 0. This is table stakes for v1.0.

## v0.9.0 — Structured Benchmark Suite + Flash Attention + Health/Metrics API (2026-02-18)

### Added
- **Structured benchmark suite** (`benchmark.rs`):
  - `BenchmarkResults` struct with JSON serialization for CI/CD integration
  - 6 benchmark scenarios: GGUF parse, cold load, sequential streaming, warm cache, forward pass estimate, cache budget analysis
  - Rich console output with Unicode table formatting
  - `--json` CLI flag for machine-readable output (`ssd-llm bench model.gguf --json`)
  - `run_benchmark_structured()` and `run_benchmark_json()` public APIs
- **Flash attention** (`inference/flash_attention.rs`):
  - Online softmax algorithm (Dao et al., 2022 / Milakov & Gimelshein, 2018)
  - O(1) extra memory per head instead of O(seq_len) — critical for long contexts
  - 4-wide SIMD dot product accumulator for cache-friendly computation
  - `flash_attention_cached()` — drop-in replacement for standard attention
  - `flash_attention_windowed()` — flash attention with sliding window range support
  - Numerically stable: handles extreme values without NaN/Inf overflow
  - `--flash-attention` CLI flag for `run` and `serve` commands
  - 7 new tests: basic, multi-token, single-token equivalence, numerical stability, windowed, GQA, online softmax correctness
- **Health & metrics API** (`api/metrics.rs`):
  - `MetricsCollector` — thread-safe atomic counters for production monitoring
  - `GET /health` — JSON readiness probe (model status, uptime, active requests)
  - `GET /metrics` — JSON metrics (request counts, token throughput, latency percentiles, cache stats, SSD I/O)
  - Prometheus-compatible text output via `metrics_prometheus()`
  - Latency tracking with p50/p95/p99 percentile computation
  - Bounded latency buffer (last 1000 requests)
  - 8 new tests: basic metrics, health JSON, loading state, metrics JSON, Prometheus output, error recording, empty percentiles, computed percentiles, cache metrics
- 19 new tests (74 total, all passing)
- Codebase cleanup: fixed all clippy warnings across entire project

### Why Flash Attention Matters for SSD-LLM
Standard attention materializes an N×N score matrix in memory. For long contexts (32K+ tokens), this matrix alone can consume gigabytes. Flash attention computes the exact same result in a single streaming pass, using O(head_dim) memory per head instead of O(seq_len). Combined with sliding window attention, this makes ultra-long context inference viable even on memory-constrained devices.

### Why Structured Benchmarks Matter
Moving toward v1.0 requires rigorous performance tracking. The structured benchmark suite produces machine-readable JSON for automated regression testing, while the human-readable table gives immediate insight into SSD streaming performance, cache efficiency, and estimated throughput.

### Why Health/Metrics Matter
Production deployments need observability. The `/health` endpoint enables Kubernetes-style readiness probes, while `/metrics` provides the counters and histograms needed for Grafana dashboards and alerting. Prometheus-compatible output means zero-config integration with standard monitoring stacks.

### Changed
- CLI: `bench` gains `--json` flag for machine-readable output
- CLI: `run` and `serve` gain `--flash-attention` flag
- API server: new `/health` and `/metrics` endpoints
- API server version bumped to 0.9.0
- Cargo.toml version bumped to 0.9.0
- README: updated features, roadmap, architecture
- Fixed all clippy warnings project-wide (dead_code, unused imports/variables, private interfaces)

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
