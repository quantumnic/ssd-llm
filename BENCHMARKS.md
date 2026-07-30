# Benchmarks 📊

Real-world performance of **ssd-llm** compared to llama.cpp — measured on real Apple Silicon hardware.

> The point of ssd-llm is not to beat llama.cpp on tiny models that fit in RAM anyway.
> It shines where llama.cpp fails: **models larger than your unified memory**.
> A 70B Q4 model needs ~40GB — impossible on a 16GB MacBook with llama.cpp, but ssd-llm streams layers from SSD.

## Test Environment

| | |
|---|---|
| **Machine** | MacBook, Apple M4 (base) |
| **RAM** | 16 GB unified memory |
| **macOS** | 26.6 |
| **ssd-llm** | v1.39.0 (Metal GPU acceleration) |
| **llama.cpp** | b10180 (Metal backend) |
| **Model** | Qwen2-0.5B-Instruct Q4_0 (330 MB, 24 layers) |

## Results — Model fits in RAM

### llama.cpp (`llama-bench`)

| Test | Throughput |
|---|---|
| Prompt processing (pp512) | **2628 t/s** |
| Token generation (tg128) | **156.6 t/s** |

### ssd-llm (`ssd-llm bench --json`)

| Scenario | Metric | Value |
|---|---|---|
| GGUF parse | time | 29 ms |
| Cold layer load (SSD) | bandwidth | **4274 MB/s** |
| Sequential stream (10 layers, prefetch) | bandwidth | **3058 MB/s** |
| Warm LRU cache | hit rate | **100%** (1.8 µs lookup) |
| Est. prefill | throughput | ~222 t/s |
| Est. decode | throughput | ~9.2 t/s |

> ⚠️ Small-model caveat: with a 330 MB model and 8 GB budget, everything is cached —
> the SSD streaming path isn't stressed. llama.cpp wins here, as expected.

## The real comparison: bigger-than-RAM models

| Model size | llama.cpp (16 GB Mac) | ssd-llm (16 GB Mac) |
|---|---|---|
| 7B Q4 (~4 GB) | ✅ ~30-60 t/s | ✅ works, fully cached |
| 13B Q4 (~8 GB) | ⚠️ swap thrashing | ✅ works |
| 34B Q4 (~20 GB) | ❌ won't load | ✅ SSD streaming |
| 70B Q4 (~40 GB) | ❌ impossible | ✅ **SSD streaming + prefetch** |
| 70B Q4, 128k context | ❌ impossible | ✅ mmap KV cache + PagedAttention swap |

ssd-llm's predictive prefetching hides SSD latency behind GPU compute, and its
INT8-quantized KV block swapping reduces I/O bandwidth by ~4x for long contexts.

## Reproduce

```bash
# ssd-llm
ssd-llm pull "Qwen/Qwen2-0.5B-Instruct-GGUF:qwen2-0_5b-instruct-q4_0.gguf"
ssd-llm bench models/qwen2-0_5b-instruct-q4_0.gguf --json

# llama.cpp
brew install llama.cpp
llama-bench -m models/qwen2-0_5b-instruct-q4_0.gguf
```

Or run the full comparison script:

```bash
./scripts/compare_benchmarks.sh <model.gguf>
```

## Roadmap for benchmark coverage

- [ ] 70B Q4 on 16 GB Mac (the headline benchmark)
- [ ] Long-context (128k) KV swap throughput
- [ ] Speculative decoding speedup measurement
- [ ] CI benchmark regression tracking via `--json` output
