//! Transformer forward pass — layer-by-layer streaming inference

use crate::inference::attention::multi_head_attention;
use crate::inference::feed_forward::feed_forward;
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
}

/// Run transformer forward pass for a single token through all layers
fn forward_pass(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    cache: &mut LayerCache,
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
        prefetcher.on_layer_start(layer_idx, n_layers, streamer, gguf, cache);

        // Load layer into cache if not present
        if cache.get(layer_idx).is_none() {
            let layer = streamer.load_layer(gguf, layer_idx)?;
            cache.insert(layer_idx, layer);
        }

        let cached = cache.get(layer_idx).unwrap();

        // 1. RMS Norm (pre-attention)
        let attn_norm_weight = find_tensor_in_layer(cached, "attn_norm.weight", layer_idx);
        if let Some(norm_w) = attn_norm_weight {
            rms_norm(hidden_state, norm_w);
        }

        // 2. Self-Attention
        let residual = hidden_state.clone();
        // Try to get attention weight tensors
        if let (Some(wq), Some(wk), Some(wv), Some(wo)) = (
            find_tensor_in_layer(cached, "attn_q.weight", layer_idx),
            find_tensor_in_layer(cached, "attn_k.weight", layer_idx),
            find_tensor_in_layer(cached, "attn_v.weight", layer_idx),
            find_tensor_in_layer(cached, "attn_output.weight", layer_idx),
        ) {
            let attn_output = multi_head_attention(
                hidden_state, wq, wk, wv, wo,
                n_head, n_head_kv, head_dim, position,
            );
            // Residual connection
            let n = hidden_state.len();
            for i in 0..n {
                hidden_state[i] = attn_output.get(i).copied().unwrap_or(hidden_state[i]) + residual[i];
            }
        }

        // 3. RMS Norm (pre-FFN)
        let ffn_norm_weight = find_tensor_in_layer(cached, "ffn_norm.weight", layer_idx);
        if let Some(norm_w) = ffn_norm_weight {
            rms_norm(hidden_state, norm_w);
        }

        // 4. Feed-Forward Network
        let residual = hidden_state.clone();
        if let (Some(w_gate), Some(w_up), Some(w_down)) = (
            find_tensor_in_layer(cached, "ffn_gate.weight", layer_idx),
            find_tensor_in_layer(cached, "ffn_up.weight", layer_idx),
            find_tensor_in_layer(cached, "ffn_down.weight", layer_idx),
        ) {
            let ffn_output = feed_forward(hidden_state, w_gate, w_up, w_down, n_embd);
            for (h, (f, r)) in hidden_state.iter_mut().zip(ffn_output.iter().zip(residual.iter())) {
                *h = f + r;
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

/// RMS Normalization in-place (uses optimized Metal compute path)
fn rms_norm(x: &mut Vec<f32>, weight: &[f32]) {
    crate::metal::compute::rmsnorm_f32_fast(x, weight, 1e-5);
}

/// Generate text token by token
pub fn generate(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    cache: &mut LayerCache,
    prompt: &str,
    config: &InferenceConfig,
) -> Result<GenerationResult> {
    let n_embd = gguf.n_embd() as usize;
    let n_layers = gguf.n_layers();
    let vocab_size = gguf.vocab_size() as usize;

    if n_embd == 0 || n_layers == 0 {
        bail!("Invalid model: n_embd={}, n_layers={}", n_embd, n_layers);
    }

    info!(
        "Model: arch={}, layers={}, embd={}, heads={}, kv_heads={}, vocab={}, ctx={}",
        gguf.architecture(), n_layers, n_embd, gguf.n_head(), gguf.n_head_kv(), vocab_size, gguf.n_ctx()
    );

    let prefetcher = Prefetcher::new(PrefetchStrategy::LookAhead(2));
    let sampler = Sampler::new(config.temperature, config.top_k, config.top_p);

    // Simple tokenization (in v0.1, we use a basic tokenizer)
    let tokenizer = SimpleTokenizer::from_gguf(gguf);
    let tokens = tokenizer.encode(prompt);
    info!("Prompt tokens: {:?}", tokens.len());

    let mut generated_tokens = Vec::new();
    let mut hidden_state = vec![0.0f32; n_embd];

    let start = Instant::now();

    // Process prompt tokens
    for (pos, &token_id) in tokens.iter().enumerate() {
        // Load token embedding
        if let Ok(embeddings) = streamer.load_named_tensor_f32(gguf, "token_embd.weight") {
            let embd_start = token_id as usize * n_embd;
            let embd_end = embd_start + n_embd;
            if embd_end <= embeddings.len() {
                hidden_state = embeddings[embd_start..embd_end].to_vec();
            }
        }

        forward_pass(gguf, streamer, cache, &mut hidden_state, pos, &prefetcher)?;
    }

    // Generate new tokens
    let mut pos = tokens.len();
    for _ in 0..config.max_tokens {
        // Project to vocab (output.weight or token_embd.weight tied)
        let logits = if let Ok(output_weight) = streamer.load_named_tensor_f32(gguf, "output.weight") {
            matmul_1d(&hidden_state, &output_weight, vocab_size)
        } else if let Ok(embd_weight) = streamer.load_named_tensor_f32(gguf, "token_embd.weight") {
            // Tied embeddings
            matmul_1d(&hidden_state, &embd_weight, vocab_size)
        } else {
            vec![0.0f32; vocab_size]
        };

        // Sample
        let token_id = sampler.sample(&logits);

        // Check for EOS
        if token_id == 2 || token_id == 0 {
            break;
        }

        generated_tokens.push(token_id);

        // Embed the new token and run forward pass
        if let Ok(embeddings) = streamer.load_named_tensor_f32(gguf, "token_embd.weight") {
            let embd_start = token_id as usize * n_embd;
            let embd_end = embd_start + n_embd;
            if embd_end <= embeddings.len() {
                hidden_state = embeddings[embd_start..embd_end].to_vec();
            }
        }

        forward_pass(gguf, streamer, cache, &mut hidden_state, pos, &prefetcher)?;
        pos += 1;
    }

    let elapsed = start.elapsed();
    let token_count = generated_tokens.len();
    let tokens_per_sec = token_count as f64 / elapsed.as_secs_f64();

    // Decode tokens
    let text = tokenizer.decode(&generated_tokens);

    Ok(GenerationResult {
        text,
        token_count,
        tokens_per_sec,
    })
}

/// Matrix-vector multiplication: hidden_state (n_embd) × weight (vocab_size × n_embd) → logits (vocab_size)
fn matmul_1d(x: &[f32], weight: &[f32], output_dim: usize) -> Vec<f32> {
    let input_dim = x.len();
    let mut output = vec![0.0f32; output_dim];

    for i in 0..output_dim.min(weight.len() / input_dim) {
        let row_start = i * input_dim;
        let mut sum = 0.0f32;
        for j in 0..input_dim {
            sum += weight[row_start + j] * x[j];
        }
        output[i] = sum;
    }

    output
}
