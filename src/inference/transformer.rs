//! Transformer forward pass — layer-by-layer streaming inference with KV cache

use crate::inference::attention::multi_head_attention_cached;
use crate::inference::feed_forward::feed_forward;
use crate::inference::kv_cache::KvCache;
use crate::inference::sampler::Sampler;
use crate::inference::tokenizer::SimpleTokenizer;
use crate::model::cache::{CachedLayer, LayerCache};
use crate::model::gguf::GgufFile;
use crate::ssd::prefetch::{PrefetchStrategy, Prefetcher};
use crate::ssd::streamer::SsdStreamer;
use anyhow::{bail, Result};
use std::time::Instant;
use tracing::{debug, info};

pub struct InferenceConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
}

pub struct GenerationResult {
    pub text: String,
    pub token_count: usize,
    pub tokens_per_sec: f64,
    pub prompt_tokens: usize,
    pub kv_cache_bytes: usize,
}

/// Run transformer forward pass (public entry point for speculative decoding)
pub fn forward_pass_pub(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    kv_cache: &mut KvCache,
    hidden_state: &mut Vec<f32>,
    position: usize,
    prefetcher: &Prefetcher,
) -> Result<()> {
    forward_pass(gguf, streamer, layer_cache, kv_cache, hidden_state, position, prefetcher)
}

/// Run transformer forward pass for a single token through all layers (with KV cache)
fn forward_pass(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    kv_cache: &mut KvCache,
    hidden_state: &mut Vec<f32>,
    position: usize,
    prefetcher: &Prefetcher,
) -> Result<()> {
    let n_layers = gguf.n_layers();
    let n_embd = gguf.n_embd() as usize;
    let n_head = gguf.n_head() as usize;
    let n_head_kv = gguf.n_head_kv() as usize;
    let head_dim = n_embd / n_head;

    for layer_idx in 0..n_layers {
        // Issue prefetch for next layer(s)
        prefetcher.on_layer_start(layer_idx, n_layers, streamer, gguf, layer_cache);

        // Load layer into cache if not present
        if layer_cache.get(layer_idx).is_none() {
            let layer = streamer.load_layer(gguf, layer_idx)?;
            layer_cache.insert(layer_idx, layer);
        }

        let cached = layer_cache.get(layer_idx).unwrap();

        // 1. RMS Norm (pre-attention)
        let mut attn_input = hidden_state.clone();
        if let Some(norm_w) = find_tensor_in_layer(cached, "attn_norm.weight", layer_idx) {
            rms_norm(&mut attn_input, norm_w);
        }

        // 2. Self-Attention with KV cache
        if let (Some(wq), Some(wk), Some(wv), Some(wo)) = (
            find_tensor_in_layer(cached, "attn_q.weight", layer_idx),
            find_tensor_in_layer(cached, "attn_k.weight", layer_idx),
            find_tensor_in_layer(cached, "attn_v.weight", layer_idx),
            find_tensor_in_layer(cached, "attn_output.weight", layer_idx),
        ) {
            let kv_layer = kv_cache.layer_mut(layer_idx as usize);
            let attn_output = multi_head_attention_cached(
                &attn_input, wq, wk, wv, wo,
                n_head, n_head_kv, head_dim, position,
                kv_layer,
            );
            // Residual connection
            for i in 0..hidden_state.len() {
                hidden_state[i] += attn_output.get(i).copied().unwrap_or(0.0);
            }
        }

        // 3. RMS Norm (pre-FFN)
        let mut ffn_input = hidden_state.clone();
        if let Some(norm_w) = find_tensor_in_layer(cached, "ffn_norm.weight", layer_idx) {
            rms_norm(&mut ffn_input, norm_w);
        }

        // 4. Feed-Forward Network
        if let (Some(w_gate), Some(w_up), Some(w_down)) = (
            find_tensor_in_layer(cached, "ffn_gate.weight", layer_idx),
            find_tensor_in_layer(cached, "ffn_up.weight", layer_idx),
            find_tensor_in_layer(cached, "ffn_down.weight", layer_idx),
        ) {
            let ffn_output = feed_forward(&ffn_input, w_gate, w_up, w_down, n_embd);
            // Residual connection
            for (h, f) in hidden_state.iter_mut().zip(ffn_output.iter()) {
                *h += f;
            }
        }

        // Signal layer done for eviction
        prefetcher.on_layer_done(layer_idx, streamer, gguf);
    }

    Ok(())
}

/// Helper: find a tensor in a cached layer by suffix
fn find_tensor_in_layer<'a>(cached: &'a CachedLayer, suffix: &str, layer_idx: u32) -> Option<&'a Vec<f32>> {
    let full_name = format!("blk.{}.{}", layer_idx, suffix);
    cached.tensors.get(&full_name)
}

/// RMS Normalization in-place
fn rms_norm(x: &mut Vec<f32>, weight: &[f32]) {
    crate::metal::compute::rmsnorm_f32_fast(x, weight, 1e-5);
}

/// Generate text token by token with KV cache
pub fn generate(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    prompt: &str,
    config: &InferenceConfig,
) -> Result<GenerationResult> {
    let n_embd = gguf.n_embd() as usize;
    let n_layers = gguf.n_layers();
    let n_head = gguf.n_head() as usize;
    let n_head_kv = gguf.n_head_kv() as usize;
    let head_dim = n_embd / n_head;
    let vocab_size = gguf.vocab_size() as usize;
    let n_ctx = gguf.n_ctx() as usize;

    if n_embd == 0 || n_layers == 0 {
        bail!("Invalid model: n_embd={}, n_layers={}", n_embd, n_layers);
    }

    info!(
        "Model: arch={}, layers={}, embd={}, heads={}/{}, vocab={}, ctx={}",
        gguf.architecture(), n_layers, n_embd, n_head, n_head_kv, vocab_size, n_ctx
    );

    let prefetcher = Prefetcher::new(PrefetchStrategy::LookAhead(2));
    let sampler = Sampler::new(config.temperature, config.top_k, config.top_p);

    // Initialize KV cache
    let mut kv_cache = KvCache::new(n_layers as usize, n_head_kv, head_dim, n_ctx);

    let tokenizer = SimpleTokenizer::from_gguf(gguf);
    let tokens = tokenizer.encode(prompt);
    let prompt_len = tokens.len();
    info!("Prompt tokens: {}", prompt_len);

    let mut generated_tokens = Vec::new();
    let mut hidden_state = vec![0.0f32; n_embd];

    let start = Instant::now();

    // Process prompt tokens (prefill phase)
    for (pos, &token_id) in tokens.iter().enumerate() {
        if let Ok(embeddings) = streamer.load_named_tensor_f32(gguf, "token_embd.weight") {
            let embd_start = token_id as usize * n_embd;
            let embd_end = embd_start + n_embd;
            if embd_end <= embeddings.len() {
                hidden_state = embeddings[embd_start..embd_end].to_vec();
            }
        }

        forward_pass(gguf, streamer, layer_cache, &mut kv_cache, &mut hidden_state, pos, &prefetcher)?;
    }

    let prefill_time = start.elapsed();
    info!("Prefill: {} tokens in {:.2}ms ({:.1} tok/s)",
        prompt_len, prefill_time.as_millis(),
        prompt_len as f64 / prefill_time.as_secs_f64()
    );

    // Generation phase (decode)
    let decode_start = Instant::now();
    let mut pos = tokens.len();
    for _ in 0..config.max_tokens {
        if kv_cache.is_full() {
            info!("KV cache full at seq_len={}, stopping", pos);
            break;
        }

        // Final RMS norm + project to vocab
        let mut final_hidden = hidden_state.clone();
        if let Ok(norm_w) = streamer.load_named_tensor_f32(gguf, "output_norm.weight") {
            rms_norm(&mut final_hidden, &norm_w);
        }

        let logits = if let Ok(output_weight) = streamer.load_named_tensor_f32(gguf, "output.weight") {
            matmul_1d(&final_hidden, &output_weight, vocab_size)
        } else if let Ok(embd_weight) = streamer.load_named_tensor_f32(gguf, "token_embd.weight") {
            matmul_1d(&final_hidden, &embd_weight, vocab_size)
        } else {
            vec![0.0f32; vocab_size]
        };

        let token_id = sampler.sample(&logits);

        if token_id == tokenizer.eos_id || token_id == 0 {
            break;
        }

        generated_tokens.push(token_id);

        // Embed new token and forward
        if let Ok(embeddings) = streamer.load_named_tensor_f32(gguf, "token_embd.weight") {
            let embd_start = token_id as usize * n_embd;
            let embd_end = embd_start + n_embd;
            if embd_end <= embeddings.len() {
                hidden_state = embeddings[embd_start..embd_end].to_vec();
            }
        }

        forward_pass(gguf, streamer, layer_cache, &mut kv_cache, &mut hidden_state, pos, &prefetcher)?;
        pos += 1;
    }

    let decode_time = decode_start.elapsed();
    let token_count = generated_tokens.len();
    let tokens_per_sec = if decode_time.as_secs_f64() > 0.0 {
        token_count as f64 / decode_time.as_secs_f64()
    } else {
        0.0
    };

    info!(
        "Decode: {} tokens in {:.2}ms ({:.1} tok/s), KV cache: {:.2} MB",
        token_count, decode_time.as_millis(), tokens_per_sec,
        kv_cache.size_bytes() as f64 / (1024.0 * 1024.0)
    );

    let text = tokenizer.decode(&generated_tokens);

    Ok(GenerationResult {
        text,
        token_count,
        tokens_per_sec,
        prompt_tokens: prompt_len,
        kv_cache_bytes: kv_cache.size_bytes(),
    })
}

/// Matrix-vector multiplication: hidden_state (n_embd) × weight (vocab_size × n_embd) → logits (vocab_size)
fn matmul_1d(x: &[f32], weight: &[f32], output_dim: usize) -> Vec<f32> {
    crate::metal::compute::matvec_f32_simd(weight, x, output_dim, x.len())
}

/// Streaming generator — yields one token at a time
pub struct StreamingGenerator<'a> {
    gguf: &'a GgufFile,
    streamer: &'a SsdStreamer,
    layer_cache: &'a mut LayerCache,
    kv_cache: KvCache,
    hidden_state: Vec<f32>,
    tokenizer: SimpleTokenizer,
    sampler: Sampler,
    prefetcher: Prefetcher,
    position: usize,
    remaining: usize,
    done: bool,
}

impl<'a> StreamingGenerator<'a> {
    /// Get the next generated token as a string, or None if done
    pub fn next_token(&mut self) -> Result<Option<String>> {
        if self.done || self.remaining == 0 {
            return Ok(None);
        }

        if self.kv_cache.is_full() {
            self.done = true;
            return Ok(None);
        }

        let n_embd = self.gguf.n_embd() as usize;
        let vocab_size = self.gguf.vocab_size() as usize;

        // Final RMS norm + project to vocab
        let mut final_hidden = self.hidden_state.clone();
        if let Ok(norm_w) = self.streamer.load_named_tensor_f32(self.gguf, "output_norm.weight") {
            rms_norm(&mut final_hidden, &norm_w);
        }

        let logits = if let Ok(output_weight) = self.streamer.load_named_tensor_f32(self.gguf, "output.weight") {
            matmul_1d(&final_hidden, &output_weight, vocab_size)
        } else if let Ok(embd_weight) = self.streamer.load_named_tensor_f32(self.gguf, "token_embd.weight") {
            matmul_1d(&final_hidden, &embd_weight, vocab_size)
        } else {
            vec![0.0f32; vocab_size]
        };

        let token_id = self.sampler.sample(&logits);

        if token_id == self.tokenizer.eos_id || token_id == 0 {
            self.done = true;
            return Ok(None);
        }

        let token_text = self.tokenizer.decode_token(token_id);

        // Embed new token and forward
        if let Ok(embeddings) = self.streamer.load_named_tensor_f32(self.gguf, "token_embd.weight") {
            let embd_start = token_id as usize * n_embd;
            let embd_end = embd_start + n_embd;
            if embd_end <= embeddings.len() {
                self.hidden_state = embeddings[embd_start..embd_end].to_vec();
            }
        }

        forward_pass(
            self.gguf, self.streamer, self.layer_cache,
            &mut self.kv_cache, &mut self.hidden_state,
            self.position, &self.prefetcher,
        )?;

        self.position += 1;
        self.remaining -= 1;

        Ok(Some(token_text))
    }
}

/// Create a streaming generator that yields tokens one at a time
pub fn generate_streaming<'a>(
    gguf: &'a GgufFile,
    streamer: &'a SsdStreamer,
    layer_cache: &'a mut LayerCache,
    prompt: &str,
    config: &InferenceConfig,
) -> Result<StreamingGenerator<'a>> {
    let n_embd = gguf.n_embd() as usize;
    let n_layers = gguf.n_layers();
    let n_head = gguf.n_head() as usize;
    let n_head_kv = gguf.n_head_kv() as usize;
    let head_dim = n_embd / n_head;
    let n_ctx = gguf.n_ctx() as usize;

    if n_embd == 0 || n_layers == 0 {
        anyhow::bail!("Invalid model: n_embd={}, n_layers={}", n_embd, n_layers);
    }

    let prefetcher = Prefetcher::new(PrefetchStrategy::LookAhead(2));
    let sampler = Sampler::new(config.temperature, config.top_k, config.top_p);
    let tokenizer = SimpleTokenizer::from_gguf(gguf);
    let tokens = tokenizer.encode(prompt);

    let mut kv_cache = KvCache::new(n_layers as usize, n_head_kv, head_dim, n_ctx);
    let mut hidden_state = vec![0.0f32; n_embd];

    // Prefill
    for (pos, &token_id) in tokens.iter().enumerate() {
        if let Ok(embeddings) = streamer.load_named_tensor_f32(gguf, "token_embd.weight") {
            let embd_start = token_id as usize * n_embd;
            let embd_end = embd_start + n_embd;
            if embd_end <= embeddings.len() {
                hidden_state = embeddings[embd_start..embd_end].to_vec();
            }
        }
        forward_pass(gguf, streamer, layer_cache, &mut kv_cache, &mut hidden_state, pos, &prefetcher)?;
    }

    Ok(StreamingGenerator {
        gguf,
        streamer,
        layer_cache,
        kv_cache,
        hidden_state,
        tokenizer,
        sampler,
        prefetcher,
        position: tokens.len(),
        remaining: config.max_tokens,
        done: false,
    })
}
