//! Comprehensive Benchmark Suite for SSD-LLM
//!
//! v0.9: Structured benchmarking with JSON output, multiple scenarios,
//! and comparative metrics for production performance analysis.
//!
//! Scenarios:
//! - Cold start: first layer load from SSD (no OS page cache)
//! - Warm cache: repeated layer loads (LRU cache hits)
//! - Sequential streaming: layer-by-layer with prefetch
//! - Prefill throughput: batch prefill simulation
//! - Decode throughput: per-token forward pass estimation
//!
//! Output: human-readable table + optional JSON for CI/CD integration

use crate::model::cache::LayerCache;
use crate::model::gguf::GgufFile;
use crate::ssd::streamer::SsdStreamer;
use anyhow::Result;
use std::path::Path;
use std::time::Instant;

/// Structured benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub model_info: ModelInfo,
    pub scenarios: Vec<ScenarioResult>,
    pub summary: BenchmarkSummary,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub path: String,
    pub tensors: usize,
    pub layers: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub vocab_size: u32,
    pub file_size_bytes: u64,
    pub architecture: String,
}

#[derive(Debug, Clone)]
pub struct ScenarioResult {
    pub name: String,
    pub description: String,
    pub metrics: Vec<Metric>,
}

#[derive(Debug, Clone)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    pub unit: String,
}

#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub total_duration_ms: f64,
    pub est_prefill_tok_per_sec: f64,
    pub est_decode_tok_per_sec: f64,
    pub ssd_bandwidth_mb_per_sec: f64,
    pub memory_budget_bytes: usize,
    pub cache_efficiency_pct: f64,
}

impl BenchmarkResults {
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\n");

        // Model info
        json.push_str("  \"model\": {\n");
        json.push_str(&format!(
            "    \"path\": \"{}\",\n",
            escape_json(&self.model_info.path)
        ));
        json.push_str(&format!(
            "    \"architecture\": \"{}\",\n",
            self.model_info.architecture
        ));
        json.push_str(&format!("    \"tensors\": {},\n", self.model_info.tensors));
        json.push_str(&format!("    \"layers\": {},\n", self.model_info.layers));
        json.push_str(&format!("    \"n_embd\": {},\n", self.model_info.n_embd));
        json.push_str(&format!("    \"n_head\": {},\n", self.model_info.n_head));
        json.push_str(&format!(
            "    \"n_head_kv\": {},\n",
            self.model_info.n_head_kv
        ));
        json.push_str(&format!(
            "    \"vocab_size\": {},\n",
            self.model_info.vocab_size
        ));
        json.push_str(&format!(
            "    \"file_size_bytes\": {}\n",
            self.model_info.file_size_bytes
        ));
        json.push_str("  },\n");

        // Scenarios
        json.push_str("  \"scenarios\": [\n");
        for (i, s) in self.scenarios.iter().enumerate() {
            json.push_str("    {\n");
            json.push_str(&format!("      \"name\": \"{}\",\n", escape_json(&s.name)));
            json.push_str(&format!(
                "      \"description\": \"{}\",\n",
                escape_json(&s.description)
            ));
            json.push_str("      \"metrics\": {\n");
            for (j, m) in s.metrics.iter().enumerate() {
                let comma = if j + 1 < s.metrics.len() { "," } else { "" };
                json.push_str(&format!(
                    "        \"{}\": {{ \"value\": {:.4}, \"unit\": \"{}\" }}{}\n",
                    escape_json(&m.name),
                    m.value,
                    escape_json(&m.unit),
                    comma
                ));
            }
            json.push_str("      }\n");
            let comma = if i + 1 < self.scenarios.len() {
                ","
            } else {
                ""
            };
            json.push_str(&format!("    }}{}\n", comma));
        }
        json.push_str("  ],\n");

        // Summary
        json.push_str("  \"summary\": {\n");
        json.push_str(&format!(
            "    \"total_duration_ms\": {:.2},\n",
            self.summary.total_duration_ms
        ));
        json.push_str(&format!(
            "    \"est_prefill_tok_per_sec\": {:.1},\n",
            self.summary.est_prefill_tok_per_sec
        ));
        json.push_str(&format!(
            "    \"est_decode_tok_per_sec\": {:.1},\n",
            self.summary.est_decode_tok_per_sec
        ));
        json.push_str(&format!(
            "    \"ssd_bandwidth_mb_per_sec\": {:.0},\n",
            self.summary.ssd_bandwidth_mb_per_sec
        ));
        json.push_str(&format!(
            "    \"memory_budget_bytes\": {},\n",
            self.summary.memory_budget_bytes
        ));
        json.push_str(&format!(
            "    \"cache_efficiency_pct\": {:.1}\n",
            self.summary.cache_efficiency_pct
        ));
        json.push_str("  }\n");

        json.push_str("}\n");
        json
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Run the full benchmark suite
pub fn run_benchmark(model_path: &Path, budget: usize) -> Result<()> {
    let results = run_benchmark_structured(model_path, budget)?;
    print_results(&results);
    Ok(())
}

/// Run benchmark with structured output (for JSON export)
pub fn run_benchmark_structured(model_path: &Path, budget: usize) -> Result<BenchmarkResults> {
    let total_start = Instant::now();

    // Parse GGUF
    let parse_start = Instant::now();
    let gguf = GgufFile::open(model_path)?;
    let parse_time = parse_start.elapsed();

    let model_info = ModelInfo {
        path: model_path.display().to_string(),
        tensors: gguf.tensors.len(),
        layers: gguf.n_layers(),
        n_embd: gguf.n_embd(),
        n_head: gguf.n_head(),
        n_head_kv: gguf.n_head_kv(),
        vocab_size: gguf.vocab_size(),
        file_size_bytes: gguf.file_size,
        architecture: gguf.architecture().to_string(),
    };

    let streamer = SsdStreamer::new(model_path, budget)?;
    let n_layers = gguf.n_layers();
    let mut scenarios = Vec::new();

    // Scenario 1: GGUF Parse
    scenarios.push(ScenarioResult {
        name: "gguf_parse".to_string(),
        description: "GGUF file parsing and metadata extraction".to_string(),
        metrics: vec![
            Metric {
                name: "parse_time_ms".into(),
                value: parse_time.as_secs_f64() * 1000.0,
                unit: "ms".into(),
            },
            Metric {
                name: "tensor_count".into(),
                value: gguf.tensors.len() as f64,
                unit: "count".into(),
            },
            Metric {
                name: "metadata_count".into(),
                value: gguf.metadata.len() as f64,
                unit: "count".into(),
            },
        ],
    });

    let mut avg_bandwidth = 0.0f64;
    let mut cache_efficiency = 0.0f64;

    if n_layers > 0 {
        // Scenario 2: Cold layer load
        let start = Instant::now();
        let layer = streamer.load_layer(&gguf, 0)?;
        let cold_time = start.elapsed();
        let layer_size_mb = layer.size_bytes as f64 / (1024.0 * 1024.0);
        let cold_bandwidth = layer_size_mb / cold_time.as_secs_f64();

        scenarios.push(ScenarioResult {
            name: "cold_load".to_string(),
            description: "First layer load from SSD (cold, no page cache benefit)".to_string(),
            metrics: vec![
                Metric {
                    name: "layer_size_mb".into(),
                    value: layer_size_mb,
                    unit: "MB".into(),
                },
                Metric {
                    name: "load_time_ms".into(),
                    value: cold_time.as_secs_f64() * 1000.0,
                    unit: "ms".into(),
                },
                Metric {
                    name: "bandwidth_mb_s".into(),
                    value: cold_bandwidth,
                    unit: "MB/s".into(),
                },
            ],
        });

        // Scenario 3: Sequential streaming with prefetch
        let mut cache = LayerCache::new(budget);
        let layers_to_bench = (n_layers as usize).min(10);
        let start = Instant::now();
        let mut total_bytes = 0u64;

        for i in 0..layers_to_bench {
            if i + 1 < n_layers as usize {
                streamer.prefetch_layer(&gguf, (i + 1) as u32);
            }
            let layer = streamer.load_layer(&gguf, i as u32)?;
            total_bytes += layer.size_bytes as u64;
            cache.insert(i as u32, layer);
        }

        let seq_time = start.elapsed();
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        avg_bandwidth = total_mb / seq_time.as_secs_f64();
        let avg_per_layer = seq_time.as_secs_f64() * 1000.0 / layers_to_bench as f64;

        scenarios.push(ScenarioResult {
            name: "sequential_stream".to_string(),
            description: format!(
                "Sequential load of {} layers with prefetch",
                layers_to_bench
            ),
            metrics: vec![
                Metric {
                    name: "layers_loaded".into(),
                    value: layers_to_bench as f64,
                    unit: "count".into(),
                },
                Metric {
                    name: "total_mb".into(),
                    value: total_mb,
                    unit: "MB".into(),
                },
                Metric {
                    name: "total_time_ms".into(),
                    value: seq_time.as_secs_f64() * 1000.0,
                    unit: "ms".into(),
                },
                Metric {
                    name: "avg_per_layer_ms".into(),
                    value: avg_per_layer,
                    unit: "ms".into(),
                },
                Metric {
                    name: "bandwidth_mb_s".into(),
                    value: avg_bandwidth,
                    unit: "MB/s".into(),
                },
            ],
        });

        // Scenario 4: Warm cache (re-read same layers)
        let start = Instant::now();
        let mut cache_hits = 0usize;
        for i in 0..layers_to_bench {
            if cache.get(i as u32).is_some() {
                cache_hits += 1;
            }
        }
        let warm_time = start.elapsed();
        cache_efficiency = cache_hits as f64 / layers_to_bench as f64 * 100.0;

        scenarios.push(ScenarioResult {
            name: "warm_cache".to_string(),
            description: "Layer access from LRU cache (no SSD I/O)".to_string(),
            metrics: vec![
                Metric {
                    name: "lookups".into(),
                    value: layers_to_bench as f64,
                    unit: "count".into(),
                },
                Metric {
                    name: "hits".into(),
                    value: cache_hits as f64,
                    unit: "count".into(),
                },
                Metric {
                    name: "hit_rate_pct".into(),
                    value: cache_efficiency,
                    unit: "%".into(),
                },
                Metric {
                    name: "lookup_time_us".into(),
                    value: warm_time.as_secs_f64() * 1_000_000.0,
                    unit: "µs".into(),
                },
            ],
        });

        // Scenario 5: Full model forward pass estimate
        let total_model_mb =
            gguf.tensors.iter().map(|t| t.size_bytes).sum::<u64>() as f64 / (1024.0 * 1024.0);
        let est_full_time = total_model_mb / avg_bandwidth;

        scenarios.push(ScenarioResult {
            name: "forward_pass_estimate".to_string(),
            description: "Estimated full forward pass timing (I/O bound)".to_string(),
            metrics: vec![
                Metric {
                    name: "total_model_mb".into(),
                    value: total_model_mb,
                    unit: "MB".into(),
                },
                Metric {
                    name: "total_model_gb".into(),
                    value: total_model_mb / 1024.0,
                    unit: "GB".into(),
                },
                Metric {
                    name: "est_forward_pass_s".into(),
                    value: est_full_time,
                    unit: "s".into(),
                },
                Metric {
                    name: "est_decode_tok_s".into(),
                    value: 1.0 / est_full_time,
                    unit: "tok/s".into(),
                },
                Metric {
                    name: "est_prefill_tok_s".into(),
                    value: n_layers as f64 / est_full_time,
                    unit: "tok/s".into(),
                },
            ],
        });

        // Scenario 6: Cache budget analysis
        let layers_in_budget =
            (budget as f64 / (total_model_mb * 1024.0 * 1024.0 / n_layers as f64)) as usize;
        let pct_cached = (layers_in_budget as f64 / n_layers as f64 * 100.0).min(100.0);

        scenarios.push(ScenarioResult {
            name: "cache_analysis".to_string(),
            description: "Memory budget vs model size analysis".to_string(),
            metrics: vec![
                Metric {
                    name: "budget_gb".into(),
                    value: budget as f64 / (1024.0 * 1024.0 * 1024.0),
                    unit: "GB".into(),
                },
                Metric {
                    name: "model_gb".into(),
                    value: total_model_mb / 1024.0,
                    unit: "GB".into(),
                },
                Metric {
                    name: "layers_in_budget".into(),
                    value: layers_in_budget as f64,
                    unit: "layers".into(),
                },
                Metric {
                    name: "pct_cached".into(),
                    value: pct_cached,
                    unit: "%".into(),
                },
                Metric {
                    name: "cache_used_mb".into(),
                    value: cache.used_bytes() as f64 / (1024.0 * 1024.0),
                    unit: "MB".into(),
                },
                Metric {
                    name: "cache_hits".into(),
                    value: cache.stats().hits as f64,
                    unit: "count".into(),
                },
                Metric {
                    name: "cache_misses".into(),
                    value: cache.stats().misses as f64,
                    unit: "count".into(),
                },
                Metric {
                    name: "cache_evictions".into(),
                    value: cache.stats().evictions as f64,
                    unit: "count".into(),
                },
            ],
        });
    }

    let total_duration = total_start.elapsed();

    // Compute summary estimates
    let total_model_mb =
        gguf.tensors.iter().map(|t| t.size_bytes).sum::<u64>() as f64 / (1024.0 * 1024.0);
    let est_forward = if avg_bandwidth > 0.0 {
        total_model_mb / avg_bandwidth
    } else {
        0.0
    };

    let summary = BenchmarkSummary {
        total_duration_ms: total_duration.as_secs_f64() * 1000.0,
        est_prefill_tok_per_sec: if est_forward > 0.0 {
            n_layers as f64 / est_forward
        } else {
            0.0
        },
        est_decode_tok_per_sec: if est_forward > 0.0 {
            1.0 / est_forward
        } else {
            0.0
        },
        ssd_bandwidth_mb_per_sec: avg_bandwidth,
        memory_budget_bytes: budget,
        cache_efficiency_pct: cache_efficiency,
    };

    Ok(BenchmarkResults {
        model_info,
        scenarios,
        summary,
    })
}

fn print_results(results: &BenchmarkResults) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              ssd-llm Benchmark Suite v0.9.0                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    let mi = &results.model_info;
    println!("║ Model: {:<52} ║", truncate(&mi.path, 52));
    println!(
        "║ Arch: {:<12}  Layers: {:<5}  Embd: {:<6}  Heads: {}/{:<3}║",
        mi.architecture, mi.layers, mi.n_embd, mi.n_head, mi.n_head_kv
    );
    println!(
        "║ Tensors: {:<8}  Size: {:.2} GB  Vocab: {:<17}║",
        mi.tensors,
        mi.file_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        mi.vocab_size
    );
    println!("╠══════════════════════════════════════════════════════════════╣");

    for scenario in &results.scenarios {
        println!("║ {:^60} ║", scenario.name.to_uppercase());
        println!("║ {:<60} ║", truncate(&scenario.description, 60));
        for m in &scenario.metrics {
            let val_str = if m.value > 1000.0 {
                format!("{:.0}", m.value)
            } else if m.value > 1.0 {
                format!("{:.2}", m.value)
            } else {
                format!("{:.4}", m.value)
            };
            println!("║   {:<30} {:>15} {:<10} ║", m.name, val_str, m.unit);
        }
        println!("╠══════════════════════════════════════════════════════════════╣");
    }

    let s = &results.summary;
    println!("║                        SUMMARY                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  SSD Bandwidth:      {:>10.0} MB/s                       ║",
        s.ssd_bandwidth_mb_per_sec
    );
    println!(
        "║  Est. Decode:        {:>10.1} tok/s                      ║",
        s.est_decode_tok_per_sec
    );
    println!(
        "║  Est. Prefill:       {:>10.1} tok/s                      ║",
        s.est_prefill_tok_per_sec
    );
    println!(
        "║  Cache Efficiency:   {:>10.1}%                           ║",
        s.cache_efficiency_pct
    );
    println!(
        "║  Memory Budget:      {:>10.2} GB                         ║",
        s.memory_budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "║  Total Bench Time:   {:>10.2} ms                         ║",
        s.total_duration_ms
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max + 3..])
    }
}

/// Run benchmark and return JSON string
pub fn run_benchmark_json(model_path: &Path, budget: usize) -> Result<String> {
    let results = run_benchmark_structured(model_path, budget)?;
    Ok(results.to_json())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_results_to_json() {
        let results = BenchmarkResults {
            model_info: ModelInfo {
                path: "test.gguf".into(),
                tensors: 100,
                layers: 32,
                n_embd: 4096,
                n_head: 32,
                n_head_kv: 8,
                vocab_size: 32000,
                file_size_bytes: 4_000_000_000,
                architecture: "llama".into(),
            },
            scenarios: vec![ScenarioResult {
                name: "cold_load".into(),
                description: "Cold layer load".into(),
                metrics: vec![Metric {
                    name: "load_time_ms".into(),
                    value: 5.5,
                    unit: "ms".into(),
                }],
            }],
            summary: BenchmarkSummary {
                total_duration_ms: 100.0,
                est_prefill_tok_per_sec: 50.0,
                est_decode_tok_per_sec: 10.0,
                ssd_bandwidth_mb_per_sec: 3000.0,
                memory_budget_bytes: 8_589_934_592,
                cache_efficiency_pct: 95.0,
            },
        };

        let json = results.to_json();
        assert!(json.contains("\"architecture\": \"llama\""));
        assert!(json.contains("\"cold_load\""));
        assert!(json.contains("\"est_decode_tok_per_sec\""));
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 20), "short");
        assert_eq!(
            truncate("a very long string that should be truncated", 20),
            "...ould be truncated"
        );
    }

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("hello\"world"), "hello\\\"world");
        assert_eq!(escape_json("line\nnewline"), "line\\nnewline");
    }
}
