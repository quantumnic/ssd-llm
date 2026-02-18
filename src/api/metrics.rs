//! Health check and metrics endpoints for production monitoring
//!
//! Provides:
//! - GET /health — readiness probe (JSON with model status)
//! - GET /metrics — Prometheus-style text metrics + JSON metrics
//!
//! Metrics tracked:
//! - Request counts (total, active, errors)
//! - Token generation throughput
//! - Latency histograms (p50, p95, p99)
//! - SSD I/O and cache statistics
//! - Memory usage estimates

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Thread-safe metrics collector
pub struct MetricsCollector {
    pub start_time: Instant,
    pub total_requests: AtomicU64,
    pub active_requests: AtomicU64,
    pub error_count: AtomicU64,
    pub total_prompt_tokens: AtomicU64,
    pub total_generated_tokens: AtomicU64,
    pub total_inference_time_us: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub cache_evictions: AtomicU64,
    pub ssd_bytes_read: AtomicU64,
    latencies_ms: Mutex<Vec<f64>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            total_requests: AtomicU64::new(0),
            active_requests: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_generated_tokens: AtomicU64::new(0),
            total_inference_time_us: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            cache_evictions: AtomicU64::new(0),
            ssd_bytes_read: AtomicU64::new(0),
            latencies_ms: Mutex::new(Vec::new()),
        }
    }

    pub fn record_request_start(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_request_end(&self, latency_ms: f64, prompt_tokens: u64, gen_tokens: u64) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        self.total_prompt_tokens
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.total_generated_tokens
            .fetch_add(gen_tokens, Ordering::Relaxed);
        self.total_inference_time_us
            .fetch_add((latency_ms * 1000.0) as u64, Ordering::Relaxed);

        if let Ok(mut latencies) = self.latencies_ms.lock() {
            latencies.push(latency_ms);
            // Keep bounded (last 1000)
            if latencies.len() > 1000 {
                let excess = latencies.len() - 1000;
                latencies.drain(0..excess);
            }
        }
    }

    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_eviction(&self) {
        self.cache_evictions.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_ssd_read(&self, bytes: u64) {
        self.ssd_bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Generate health check JSON
    pub fn health_json(&self, model_name: &str, model_loaded: bool) -> String {
        let uptime_secs = self.start_time.elapsed().as_secs();
        let status = if model_loaded { "healthy" } else { "loading" };

        format!(
            r#"{{"status":"{}","model":"{}","uptime_seconds":{},"version":"0.9.0","active_requests":{}}}"#,
            status,
            model_name,
            uptime_secs,
            self.active_requests.load(Ordering::Relaxed),
        )
    }

    /// Generate JSON metrics
    pub fn metrics_json(&self, model_name: &str, memory_budget: usize) -> String {
        let uptime = self.start_time.elapsed();
        let total_req = self.total_requests.load(Ordering::Relaxed);
        let total_gen = self.total_generated_tokens.load(Ordering::Relaxed);
        let total_prompt = self.total_prompt_tokens.load(Ordering::Relaxed);
        let total_inf_us = self.total_inference_time_us.load(Ordering::Relaxed);
        let errors = self.error_count.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let evictions = self.cache_evictions.load(Ordering::Relaxed);
        let ssd_bytes = self.ssd_bytes_read.load(Ordering::Relaxed);

        let avg_latency_ms = if total_req > 0 {
            total_inf_us as f64 / 1000.0 / total_req as f64
        } else {
            0.0
        };

        let throughput_tok_s = if uptime.as_secs_f64() > 0.0 {
            total_gen as f64 / uptime.as_secs_f64()
        } else {
            0.0
        };

        let cache_hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64 * 100.0
        } else {
            0.0
        };

        let (p50, p95, p99) = self.compute_percentiles();

        format!(
            concat!(
                "{{",
                "\"model\":\"{}\",",
                "\"uptime_seconds\":{},",
                "\"memory_budget_bytes\":{},",
                "\"requests\":{{\"total\":{},\"active\":{},\"errors\":{}}},",
                "\"tokens\":{{\"prompt_total\":{},\"generated_total\":{},\"throughput_tok_s\":{:.2}}},",
                "\"latency_ms\":{{\"avg\":{:.2},\"p50\":{:.2},\"p95\":{:.2},\"p99\":{:.2}}},",
                "\"cache\":{{\"hits\":{},\"misses\":{},\"evictions\":{},\"hit_rate_pct\":{:.1}}},",
                "\"ssd\":{{\"bytes_read\":{},\"read_mb\":{:.2}}}",
                "}}"
            ),
            model_name,
            uptime.as_secs(),
            memory_budget,
            total_req, self.active_requests.load(Ordering::Relaxed), errors,
            total_prompt, total_gen, throughput_tok_s,
            avg_latency_ms, p50, p95, p99,
            hits, misses, evictions, cache_hit_rate,
            ssd_bytes, ssd_bytes as f64 / (1024.0 * 1024.0),
        )
    }

    /// Generate Prometheus-style text metrics
    pub fn metrics_prometheus(&self, model_name: &str) -> String {
        let total_req = self.total_requests.load(Ordering::Relaxed);
        let active = self.active_requests.load(Ordering::Relaxed);
        let errors = self.error_count.load(Ordering::Relaxed);
        let gen_tokens = self.total_generated_tokens.load(Ordering::Relaxed);
        let prompt_tokens = self.total_prompt_tokens.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let evictions = self.cache_evictions.load(Ordering::Relaxed);
        let ssd_bytes = self.ssd_bytes_read.load(Ordering::Relaxed);
        let uptime = self.start_time.elapsed().as_secs();

        format!(
            concat!(
                "# HELP ssd_llm_uptime_seconds Server uptime in seconds\n",
                "# TYPE ssd_llm_uptime_seconds gauge\n",
                "ssd_llm_uptime_seconds{{model=\"{}\"}} {}\n",
                "# HELP ssd_llm_requests_total Total requests processed\n",
                "# TYPE ssd_llm_requests_total counter\n",
                "ssd_llm_requests_total{{model=\"{}\"}} {}\n",
                "# HELP ssd_llm_requests_active Currently active requests\n",
                "# TYPE ssd_llm_requests_active gauge\n",
                "ssd_llm_requests_active{{model=\"{}\"}} {}\n",
                "# HELP ssd_llm_errors_total Total request errors\n",
                "# TYPE ssd_llm_errors_total counter\n",
                "ssd_llm_errors_total{{model=\"{}\"}} {}\n",
                "# HELP ssd_llm_tokens_generated_total Total tokens generated\n",
                "# TYPE ssd_llm_tokens_generated_total counter\n",
                "ssd_llm_tokens_generated_total{{model=\"{}\"}} {}\n",
                "# HELP ssd_llm_tokens_prompt_total Total prompt tokens processed\n",
                "# TYPE ssd_llm_tokens_prompt_total counter\n",
                "ssd_llm_tokens_prompt_total{{model=\"{}\"}} {}\n",
                "# HELP ssd_llm_cache_hits_total Layer cache hits\n",
                "# TYPE ssd_llm_cache_hits_total counter\n",
                "ssd_llm_cache_hits_total{{model=\"{}\"}} {}\n",
                "# HELP ssd_llm_cache_misses_total Layer cache misses\n",
                "# TYPE ssd_llm_cache_misses_total counter\n",
                "ssd_llm_cache_misses_total{{model=\"{}\"}} {}\n",
                "# HELP ssd_llm_cache_evictions_total Layer cache evictions\n",
                "# TYPE ssd_llm_cache_evictions_total counter\n",
                "ssd_llm_cache_evictions_total{{model=\"{}\"}} {}\n",
                "# HELP ssd_llm_ssd_bytes_read_total Total bytes read from SSD\n",
                "# TYPE ssd_llm_ssd_bytes_read_total counter\n",
                "ssd_llm_ssd_bytes_read_total{{model=\"{}\"}} {}\n",
            ),
            model_name,
            uptime,
            model_name,
            total_req,
            model_name,
            active,
            model_name,
            errors,
            model_name,
            gen_tokens,
            model_name,
            prompt_tokens,
            model_name,
            hits,
            model_name,
            misses,
            model_name,
            evictions,
            model_name,
            ssd_bytes,
        )
    }

    fn compute_percentiles(&self) -> (f64, f64, f64) {
        if let Ok(mut latencies) = self.latencies_ms.lock() {
            if latencies.is_empty() {
                return (0.0, 0.0, 0.0);
            }
            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let len = latencies.len();
            let p50 = latencies[(len as f64 * 0.50) as usize];
            let p95 = latencies[((len as f64 * 0.95) as usize).min(len - 1)];
            let p99 = latencies[((len as f64 * 0.99) as usize).min(len - 1)];
            (p50, p95, p99)
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_basic() {
        let m = MetricsCollector::new();
        m.record_request_start();
        assert_eq!(m.active_requests.load(Ordering::Relaxed), 1);
        assert_eq!(m.total_requests.load(Ordering::Relaxed), 1);

        m.record_request_end(50.0, 10, 20);
        assert_eq!(m.active_requests.load(Ordering::Relaxed), 0);
        assert_eq!(m.total_generated_tokens.load(Ordering::Relaxed), 20);
        assert_eq!(m.total_prompt_tokens.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_health_json() {
        let m = MetricsCollector::new();
        let json = m.health_json("test-model", true);
        assert!(json.contains("\"status\":\"healthy\""));
        assert!(json.contains("\"model\":\"test-model\""));
        assert!(json.contains("\"version\":\"0.9.0\""));
    }

    #[test]
    fn test_health_json_loading() {
        let m = MetricsCollector::new();
        let json = m.health_json("test-model", false);
        assert!(json.contains("\"status\":\"loading\""));
    }

    #[test]
    fn test_metrics_json() {
        let m = MetricsCollector::new();
        m.record_request_start();
        m.record_request_end(100.0, 50, 30);
        m.record_cache_hit();
        m.record_cache_hit();
        m.record_cache_miss();
        m.record_ssd_read(1_000_000);

        let json = m.metrics_json("llama-70b", 8_589_934_592);
        assert!(json.contains("\"total\":1"));
        assert!(json.contains("\"generated_total\":30"));
        assert!(json.contains("\"hits\":2"));
        assert!(json.contains("\"misses\":1"));
    }

    #[test]
    fn test_metrics_prometheus() {
        let m = MetricsCollector::new();
        m.record_request_start();
        m.record_request_end(50.0, 10, 20);

        let prom = m.metrics_prometheus("test");
        assert!(prom.contains("ssd_llm_requests_total{model=\"test\"} 1"));
        assert!(prom.contains("ssd_llm_tokens_generated_total{model=\"test\"} 20"));
    }

    #[test]
    fn test_error_recording() {
        let m = MetricsCollector::new();
        m.record_request_start();
        m.record_error();
        assert_eq!(m.active_requests.load(Ordering::Relaxed), 0);
        assert_eq!(m.error_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_percentiles_empty() {
        let m = MetricsCollector::new();
        let (p50, p95, p99) = m.compute_percentiles();
        assert_eq!(p50, 0.0);
        assert_eq!(p95, 0.0);
        assert_eq!(p99, 0.0);
    }

    #[test]
    fn test_percentiles_computed() {
        let m = MetricsCollector::new();
        // Record 100 latencies: 1.0, 2.0, ..., 100.0
        for i in 1..=100 {
            m.record_request_start();
            m.record_request_end(i as f64, 1, 1);
        }
        let (p50, p95, p99) = m.compute_percentiles();
        assert!((p50 - 50.0).abs() < 2.0, "p50 should be ~50, got {}", p50);
        assert!((p95 - 95.0).abs() < 2.0, "p95 should be ~95, got {}", p95);
        assert!((p99 - 99.0).abs() < 2.0, "p99 should be ~99, got {}", p99);
    }

    #[test]
    fn test_cache_metrics() {
        let m = MetricsCollector::new();
        for _ in 0..10 {
            m.record_cache_hit();
        }
        for _ in 0..5 {
            m.record_cache_miss();
        }
        for _ in 0..3 {
            m.record_cache_eviction();
        }

        assert_eq!(m.cache_hits.load(Ordering::Relaxed), 10);
        assert_eq!(m.cache_misses.load(Ordering::Relaxed), 5);
        assert_eq!(m.cache_evictions.load(Ordering::Relaxed), 3);
    }
}
