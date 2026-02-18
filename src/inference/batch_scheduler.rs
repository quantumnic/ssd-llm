//! Continuous batching scheduler for concurrent request handling
//!
//! Instead of processing one request at a time, the batch scheduler
//! groups multiple requests together so they share layer loads from SSD.
//! Each layer is loaded once and applied to all active sequences.

use crate::inference::kv_cache::KvCache;
use crate::inference::sampler::Sampler;
use crate::inference::tokenizer::SimpleTokenizer;
use crate::inference::transformer::{self, InferenceConfig};
use crate::model::cache::LayerCache;
use crate::model::gguf::GgufFile;
use crate::ssd::prefetch::{PrefetchStrategy, Prefetcher};
use crate::ssd::streamer::SsdStreamer;
use anyhow::Result;
use std::collections::VecDeque;

use std::time::Instant;
use tracing::{debug, info};

/// Unique ID for a batch request
pub type RequestId = u64;

/// State of a request in the batch
#[derive(Debug, Clone, PartialEq)]
pub enum RequestState {
    /// Waiting to start prefill
    Queued,
    /// Currently in prefill phase
    Prefilling,
    /// In decode phase (generating tokens)
    Decoding,
    /// Completed (success or error)
    Completed,
}

/// A single request in the batch
pub struct BatchRequest {
    pub id: RequestId,
    pub prompt_tokens: Vec<u32>,
    pub config: InferenceConfig,
    pub state: RequestState,
    /// KV cache for this request's sequence
    pub kv_cache: KvCache,
    /// Current hidden state
    pub hidden_state: Vec<f32>,
    /// Current position in the sequence
    pub position: usize,
    /// Generated token ids
    pub generated: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Prefill progress: how many prompt tokens have been processed
    pub prefill_idx: usize,
    /// Submission time
    pub submitted_at: Instant,
    /// Whether this request is done
    pub done: bool,
}

/// Result for a completed batch request
#[derive(Clone)]
pub struct BatchResult {
    pub id: RequestId,
    pub text: String,
    pub token_count: usize,
    pub tokens_per_sec: f64,
    pub prompt_tokens: usize,
}

/// Continuous batching scheduler
pub struct BatchScheduler {
    /// Queue of waiting requests
    queue: VecDeque<BatchRequest>,
    /// Active requests being processed
    active: Vec<BatchRequest>,
    /// Completed results
    completed: Vec<BatchResult>,
    /// Maximum concurrent sequences
    max_batch_size: usize,
    /// Next request ID
    next_id: RequestId,
    /// Stats
    pub total_requests: u64,
    pub total_tokens_generated: u64,
}

impl BatchScheduler {
    pub fn new(max_batch_size: usize) -> Self {
        info!("Batch scheduler: max_batch_size={}", max_batch_size);
        Self {
            queue: VecDeque::new(),
            active: Vec::new(),
            completed: Vec::new(),
            max_batch_size,
            next_id: 1,
            total_requests: 0,
            total_tokens_generated: 0,
        }
    }

    /// Submit a new request, returns its ID
    pub fn submit(
        &mut self,
        prompt_tokens: Vec<u32>,
        config: InferenceConfig,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_ctx: usize,
        n_embd: usize,
    ) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;

        let kv_cache = KvCache::new(n_layers, n_kv_heads, head_dim, n_ctx);
        let max_tokens = config.max_tokens;

        let req = BatchRequest {
            id,
            prompt_tokens,
            config,
            state: RequestState::Queued,
            kv_cache,
            hidden_state: vec![0.0f32; n_embd],
            position: 0,
            generated: Vec::new(),
            max_tokens,
            prefill_idx: 0,
            submitted_at: Instant::now(),
            done: false,
        };

        self.queue.push_back(req);
        self.total_requests += 1;
        debug!(
            "Batch scheduler: submitted request {} (queue: {})",
            id,
            self.queue.len()
        );
        id
    }

    /// Run one iteration of the batch scheduler.
    /// Processes one layer across all active sequences (decode phase),
    /// or advances prefill for sequences in prefill phase.
    ///
    /// Returns completed results from this iteration.
    pub fn step(
        &mut self,
        gguf: &GgufFile,
        streamer: &SsdStreamer,
        layer_cache: &mut LayerCache,
        tokenizer: &SimpleTokenizer,
    ) -> Result<Vec<BatchResult>> {
        // Promote queued requests to active if there's space
        while self.active.len() < self.max_batch_size && !self.queue.is_empty() {
            let mut req = self.queue.pop_front().unwrap();
            req.state = RequestState::Prefilling;
            debug!(
                "Batch scheduler: activating request {} (prefill {} tokens)",
                req.id,
                req.prompt_tokens.len()
            );
            self.active.push(req);
        }

        if self.active.is_empty() {
            return Ok(vec![]);
        }

        let n_layers = gguf.n_layers();
        let n_embd = gguf.n_embd() as usize;
        let n_head = gguf.n_head() as usize;
        let n_head_kv = gguf.n_head_kv() as usize;
        let head_dim = n_embd / n_head;
        let vocab_size = gguf.vocab_size() as usize;
        let prefetcher = Prefetcher::new(PrefetchStrategy::LookAhead(2));

        // --- Prefill phase: batch-process prefilling requests ---
        // For each prefilling request, run batch_prefill
        for req in self.active.iter_mut() {
            if req.state != RequestState::Prefilling {
                continue;
            }

            // Load embeddings and do batch prefill
            let embeddings = streamer
                .load_named_tensor_f32(gguf, "token_embd.weight")
                .ok();
            let mut token_embeddings: Vec<Vec<f32>> = Vec::with_capacity(req.prompt_tokens.len());
            for &token_id in &req.prompt_tokens {
                let mut emb = vec![0.0f32; n_embd];
                if let Some(ref emb_data) = embeddings {
                    let s = token_id as usize * n_embd;
                    let e = s + n_embd;
                    if e <= emb_data.len() {
                        emb = emb_data[s..e].to_vec();
                    }
                }
                token_embeddings.push(emb);
            }

            req.hidden_state = transformer::batch_prefill(
                gguf,
                streamer,
                layer_cache,
                &mut req.kv_cache,
                &token_embeddings,
                0,
                &prefetcher,
            )?;
            req.position = req.prompt_tokens.len();
            req.state = RequestState::Decoding;
            debug!(
                "Batch scheduler: request {} prefill complete, switching to decode",
                req.id
            );
        }

        // --- Decode phase: one token per decoding request ---
        // Layer-major: load each layer once, apply to all decoding sequences
        let decoding_ids: Vec<usize> = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, r)| r.state == RequestState::Decoding)
            .map(|(i, _)| i)
            .collect();

        if !decoding_ids.is_empty() {
            for layer_idx in 0..n_layers {
                prefetcher.on_layer_start(layer_idx, n_layers, streamer, gguf, layer_cache);

                if layer_cache.get(layer_idx).is_none() {
                    let layer = streamer.load_layer(gguf, layer_idx)?;
                    layer_cache.insert(layer_idx, layer);
                }

                let cached = layer_cache.get(layer_idx).unwrap();

                for &idx in &decoding_ids {
                    let req = &mut self.active[idx];
                    let position = req.position;

                    // RMS Norm (pre-attention)
                    let mut attn_input = req.hidden_state.clone();
                    let full_name = format!("blk.{}.attn_norm.weight", layer_idx);
                    if let Some(norm_w) = cached.tensors.get(&full_name) {
                        crate::metal::compute::rmsnorm_f32_fast(&mut attn_input, norm_w, 1e-5);
                    }

                    // Self-Attention
                    let wq_name = format!("blk.{}.attn_q.weight", layer_idx);
                    let wk_name = format!("blk.{}.attn_k.weight", layer_idx);
                    let wv_name = format!("blk.{}.attn_v.weight", layer_idx);
                    let wo_name = format!("blk.{}.attn_output.weight", layer_idx);
                    if let (Some(wq), Some(wk), Some(wv), Some(wo)) = (
                        cached.tensors.get(&wq_name),
                        cached.tensors.get(&wk_name),
                        cached.tensors.get(&wv_name),
                        cached.tensors.get(&wo_name),
                    ) {
                        let kv_layer = req.kv_cache.layer_mut(layer_idx as usize);
                        let attn_output = crate::inference::attention::multi_head_attention_cached(
                            &attn_input,
                            wq,
                            wk,
                            wv,
                            wo,
                            n_head,
                            n_head_kv,
                            head_dim,
                            position,
                            kv_layer,
                        );
                        for i in 0..req.hidden_state.len() {
                            req.hidden_state[i] += attn_output.get(i).copied().unwrap_or(0.0);
                        }
                    }

                    // RMS Norm (pre-FFN)
                    let mut ffn_input = req.hidden_state.clone();
                    let ffn_norm_name = format!("blk.{}.ffn_norm.weight", layer_idx);
                    if let Some(norm_w) = cached.tensors.get(&ffn_norm_name) {
                        crate::metal::compute::rmsnorm_f32_fast(&mut ffn_input, norm_w, 1e-5);
                    }

                    // FFN
                    let gate_name = format!("blk.{}.ffn_gate.weight", layer_idx);
                    let up_name = format!("blk.{}.ffn_up.weight", layer_idx);
                    let down_name = format!("blk.{}.ffn_down.weight", layer_idx);
                    if let (Some(w_gate), Some(w_up), Some(w_down)) = (
                        cached.tensors.get(&gate_name),
                        cached.tensors.get(&up_name),
                        cached.tensors.get(&down_name),
                    ) {
                        let ffn_output = crate::inference::feed_forward::feed_forward(
                            &ffn_input, w_gate, w_up, w_down, n_embd,
                        );
                        for (h, f) in req.hidden_state.iter_mut().zip(ffn_output.iter()) {
                            *h += f;
                        }
                    }
                }

                prefetcher.on_layer_done(layer_idx, streamer, gguf);
            }

            // Sample next token for each decoding request
            for &idx in &decoding_ids {
                let req = &mut self.active[idx];
                let sampler =
                    Sampler::new(req.config.temperature, req.config.top_k, req.config.top_p);

                // Final norm + project to vocab
                let mut final_hidden = req.hidden_state.clone();
                if let Ok(norm_w) = streamer.load_named_tensor_f32(gguf, "output_norm.weight") {
                    crate::metal::compute::rmsnorm_f32_fast(&mut final_hidden, &norm_w, 1e-5);
                }

                let logits = if let Ok(output_weight) =
                    streamer.load_named_tensor_f32(gguf, "output.weight")
                {
                    crate::metal::compute::matvec_f32_simd(
                        &output_weight,
                        &final_hidden,
                        vocab_size,
                        n_embd,
                    )
                } else if let Ok(embd_weight) =
                    streamer.load_named_tensor_f32(gguf, "token_embd.weight")
                {
                    crate::metal::compute::matvec_f32_simd(
                        &embd_weight,
                        &final_hidden,
                        vocab_size,
                        n_embd,
                    )
                } else {
                    vec![0.0f32; vocab_size]
                };

                let token_id = sampler.sample(&logits);

                if token_id == tokenizer.eos_id
                    || token_id == 0
                    || req.generated.len() >= req.max_tokens
                {
                    req.done = true;
                } else {
                    req.generated.push(token_id);
                    req.position += 1;

                    // Embed new token for next step
                    if let Ok(embeddings) =
                        streamer.load_named_tensor_f32(gguf, "token_embd.weight")
                    {
                        let s = token_id as usize * n_embd;
                        let e = s + n_embd;
                        if e <= embeddings.len() {
                            req.hidden_state = embeddings[s..e].to_vec();
                        }
                    }
                }
            }
        }

        // Collect completed requests
        let mut results = Vec::new();
        self.active.retain(|req| {
            if req.done {
                let elapsed = req.submitted_at.elapsed();
                let tps = if elapsed.as_secs_f64() > 0.0 {
                    req.generated.len() as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                results.push(BatchResult {
                    id: req.id,
                    text: tokenizer.decode(&req.generated),
                    token_count: req.generated.len(),
                    tokens_per_sec: tps,
                    prompt_tokens: req.prompt_tokens.len(),
                });
                false // remove from active
            } else {
                true // keep
            }
        });

        for r in &results {
            self.total_tokens_generated += r.token_count as u64;
            debug!(
                "Batch scheduler: request {} completed ({} tokens, {:.1} tok/s)",
                r.id, r.token_count, r.tokens_per_sec
            );
        }

        self.completed.extend(results.clone());
        Ok(results)
    }

    /// Number of requests currently being processed
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Number of requests waiting in queue
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Check if there's any work to do
    pub fn has_work(&self) -> bool {
        !self.queue.is_empty() || !self.active.is_empty()
    }

    /// Take a completed result by request ID
    pub fn take_result(&mut self, id: RequestId) -> Option<BatchResult> {
        if let Some(pos) = self.completed.iter().position(|r| r.id == id) {
            Some(self.completed.remove(pos))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_scheduler_submit() {
        let mut sched = BatchScheduler::new(4);
        let id = sched.submit(
            vec![1, 2, 3],
            InferenceConfig {
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
                max_tokens: 10,
            },
            2,
            4,
            64,
            512,
            256,
        );
        assert_eq!(id, 1);
        assert_eq!(sched.queue_len(), 1);
        assert!(sched.has_work());
    }

    #[test]
    fn test_batch_scheduler_max_batch() {
        let mut sched = BatchScheduler::new(2);
        for _ in 0..5 {
            sched.submit(
                vec![1, 2],
                InferenceConfig {
                    temperature: 0.7,
                    top_k: 40,
                    top_p: 0.9,
                    max_tokens: 5,
                },
                1,
                2,
                32,
                256,
                64,
            );
        }
        assert_eq!(sched.queue_len(), 5);
        assert_eq!(sched.active_count(), 0);
    }
}
