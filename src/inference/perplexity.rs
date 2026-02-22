//! Perplexity evaluation — measure model quality on text datasets
//!
//! Perplexity (PPL) is the exponential of the average negative log-likelihood per token.
//! Lower perplexity = better model quality. This is the standard metric for comparing
//! quantization methods and evaluating model quality.
//!
//! Supports:
//! - Sliding window evaluation for long texts (llama.cpp-compatible)
//! - Stride-based chunking for texts exceeding context length
//! - Per-chunk and aggregate statistics
//! - JSON output for CI/CD integration

use crate::inference::kv_cache::KvCache;
use crate::inference::tokenizer::SimpleTokenizer;
use crate::inference::transformer::rms_norm;
use crate::model::cache::LayerCache;
use crate::model::gguf::GgufFile;
use crate::ssd::prefetch::{PrefetchStrategy, Prefetcher};
use crate::ssd::streamer::SsdStreamer;
use anyhow::{bail, Result};
use std::time::Instant;
use tracing::info;

/// Configuration for perplexity evaluation
pub struct PerplexityConfig {
    /// Context window size (0 = use model's n_ctx)
    pub context_size: usize,
    /// Stride for sliding window (0 = context_size / 2)
    pub stride: usize,
    /// Show per-chunk results
    pub verbose: bool,
}

/// Result of perplexity evaluation
pub struct PerplexityResult {
    /// Overall perplexity
    pub perplexity: f64,
    /// Average negative log-likelihood per token
    pub nll: f64,
    /// Total tokens evaluated
    pub tokens_evaluated: usize,
    /// Total tokens in input
    pub total_tokens: usize,
    /// Number of chunks processed
    pub chunks: usize,
    /// Per-chunk perplexity values
    pub chunk_perplexities: Vec<f64>,
    /// Time taken in seconds
    pub elapsed_secs: f64,
    /// Tokens per second (evaluation throughput)
    pub tokens_per_sec: f64,
}

impl PerplexityResult {
    /// Format as JSON string
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"perplexity\": {:.4},\n",
                "  \"nll\": {:.6},\n",
                "  \"tokens_evaluated\": {},\n",
                "  \"total_tokens\": {},\n",
                "  \"chunks\": {},\n",
                "  \"elapsed_secs\": {:.2},\n",
                "  \"tokens_per_sec\": {:.1},\n",
                "  \"chunk_perplexities\": [{}]\n",
                "}}"
            ),
            self.perplexity,
            self.nll,
            self.tokens_evaluated,
            self.total_tokens,
            self.chunks,
            self.elapsed_secs,
            self.tokens_per_sec,
            self.chunk_perplexities
                .iter()
                .map(|p| format!("{:.4}", p))
                .collect::<Vec<_>>()
                .join(", "),
        )
    }
}

/// Compute log-softmax of logits, returning log-probabilities
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&x| ((x - max) as f64).exp()).sum();
    let log_sum_exp = max as f64 + sum_exp.ln();
    logits
        .iter()
        .map(|&x| (x as f64 - log_sum_exp) as f32)
        .collect()
}

/// Evaluate perplexity on a text string.
///
/// For each position i in the tokenized text, we compute the model's prediction
/// for token[i] given tokens[0..i], and measure the negative log-likelihood.
/// Perplexity = exp(average NLL).
pub fn evaluate_perplexity(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    text: &str,
    config: &PerplexityConfig,
) -> Result<PerplexityResult> {
    let n_embd = gguf.n_embd() as usize;
    let n_layers = gguf.n_layers();
    let n_head = gguf.n_head() as usize;
    let n_head_kv = gguf.n_head_kv() as usize;
    let head_dim = n_embd / n_head;
    let model_ctx = gguf.n_ctx() as usize;
    let vocab_size = gguf.vocab_size() as usize;

    if n_embd == 0 || n_layers == 0 {
        bail!("Invalid model: n_embd={}, n_layers={}", n_embd, n_layers);
    }

    let ctx_size = if config.context_size > 0 {
        config.context_size.min(model_ctx)
    } else {
        model_ctx
    };

    let stride = if config.stride > 0 {
        config.stride.min(ctx_size)
    } else {
        ctx_size / 2
    };

    let tokenizer = SimpleTokenizer::from_gguf(gguf);
    let tokens = tokenizer.encode(text);
    let total_tokens = tokens.len();

    if total_tokens < 2 {
        bail!(
            "Text too short for perplexity evaluation ({} tokens, need at least 2)",
            total_tokens
        );
    }

    info!(
        "Perplexity eval: {} tokens, ctx={}, stride={}, vocab={}",
        total_tokens, ctx_size, stride, vocab_size
    );

    let prefetcher = Prefetcher::new(PrefetchStrategy::LookAhead(2));
    let start = Instant::now();

    let mut total_nll: f64 = 0.0;
    let mut tokens_evaluated: usize = 0;
    let mut chunk_perplexities: Vec<f64> = Vec::new();

    // Process text in chunks with sliding window
    let mut chunk_start: usize = 0;
    let mut chunk_idx: usize = 0;

    while chunk_start < total_tokens.saturating_sub(1) {
        let chunk_end = (chunk_start + ctx_size).min(total_tokens);
        let chunk_tokens = &tokens[chunk_start..chunk_end];
        let chunk_len = chunk_tokens.len();

        if chunk_len < 2 {
            break;
        }

        // Determine which tokens in this chunk to score.
        // For the first chunk, we score positions 1..chunk_len (can't predict the first token).
        // For subsequent chunks with overlap, we only score the non-overlapping (new) positions
        // to avoid double-counting.
        let score_start = if chunk_start == 0 {
            1 // Skip first token (no context to predict it)
        } else {
            // With stride < ctx_size, we have overlap. Only score the new tokens.
            ctx_size - stride
        };

        // Fresh KV cache for each chunk
        let mut kv_cache = KvCache::new(n_layers as usize, n_head_kv, head_dim, ctx_size);

        // Load embeddings for the chunk
        let embeddings_data = streamer
            .load_named_tensor_f32(gguf, "token_embd.weight")
            .ok();

        let mut token_embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunk_len);
        for &token_id in chunk_tokens {
            let mut emb = vec![0.0f32; n_embd];
            if let Some(ref emb_data) = embeddings_data {
                let s = token_id as usize * n_embd;
                let e = s + n_embd;
                if e <= emb_data.len() {
                    emb = emb_data[s..e].to_vec();
                }
            }
            token_embeddings.push(emb);
        }

        // Batch prefill: get hidden states for ALL positions in the chunk.
        // We need to modify batch_prefill to return all hidden states, not just the last one.
        // Instead, we process token-by-token through layers to collect all logits.
        let chunk_nll = evaluate_chunk_nll(
            gguf,
            streamer,
            layer_cache,
            &mut kv_cache,
            &token_embeddings,
            chunk_tokens,
            score_start,
            vocab_size,
            n_embd,
            &prefetcher,
        )?;

        let scored_count = chunk_len - score_start;
        let chunk_avg_nll = chunk_nll / scored_count as f64;
        let chunk_ppl = chunk_avg_nll.exp();

        total_nll += chunk_nll;
        tokens_evaluated += scored_count;
        chunk_perplexities.push(chunk_ppl);

        chunk_idx += 1;
        if config.verbose {
            println!(
                "  chunk {}: tokens [{}-{}), scored {}, PPL = {:.4}",
                chunk_idx, chunk_start, chunk_end, scored_count, chunk_ppl,
            );
        }

        // Advance by stride
        if chunk_end >= total_tokens {
            break;
        }
        chunk_start += stride;
    }

    let elapsed = start.elapsed();
    let avg_nll = total_nll / tokens_evaluated as f64;
    let perplexity = avg_nll.exp();

    Ok(PerplexityResult {
        perplexity,
        nll: avg_nll,
        tokens_evaluated,
        total_tokens,
        chunks: chunk_perplexities.len(),
        chunk_perplexities,
        elapsed_secs: elapsed.as_secs_f64(),
        tokens_per_sec: tokens_evaluated as f64 / elapsed.as_secs_f64(),
    })
}

/// Evaluate negative log-likelihood for a chunk of tokens.
///
/// Uses batch prefill to process all tokens through the transformer, then computes
/// logits at each position and measures the NLL of the actual next token.
///
/// Returns the sum of NLL for positions [score_start..chunk_len).
fn evaluate_chunk_nll(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    kv_cache: &mut KvCache,
    token_embeddings: &[Vec<f32>],
    chunk_tokens: &[u32],
    score_start: usize,
    vocab_size: usize,
    n_embd: usize,
    prefetcher: &Prefetcher,
) -> Result<f64> {
    use crate::inference::attention::multi_head_attention_cached;
    use crate::inference::transformer::{detect_moe_config, find_tensor_in_layer, run_ffn_or_moe};

    let n_layers = gguf.n_layers();
    let n_head = gguf.n_head() as usize;
    let n_head_kv = gguf.n_head_kv() as usize;
    let head_dim = n_embd / n_head;
    let chunk_len = token_embeddings.len();
    let moe_config = detect_moe_config(gguf);

    // Process all tokens through all layers, keeping all hidden states
    let mut hidden_states: Vec<Vec<f32>> = token_embeddings.to_vec();

    for layer_idx in 0..n_layers {
        prefetcher.on_layer_start(layer_idx, n_layers, streamer, gguf, layer_cache);

        if layer_cache.get(layer_idx).is_none() {
            let layer = streamer.load_layer(gguf, layer_idx)?;
            layer_cache.insert(layer_idx, layer);
        }

        let cached = layer_cache.get(layer_idx).unwrap();

        for (t, hidden_state) in hidden_states.iter_mut().enumerate().take(chunk_len) {
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
                    &attn_input,
                    wq,
                    wk,
                    wv,
                    wo,
                    n_head,
                    n_head_kv,
                    head_dim,
                    t,
                    kv_layer,
                );
                for (i, hs) in hidden_state.iter_mut().enumerate() {
                    *hs += attn_output.get(i).copied().unwrap_or(0.0);
                }
            }

            // 3. RMS Norm (pre-FFN)
            let mut ffn_input = hidden_state.clone();
            if let Some(norm_w) = find_tensor_in_layer(cached, "ffn_norm.weight", layer_idx) {
                rms_norm(&mut ffn_input, norm_w);
            }

            // 4. Feed-Forward Network
            if let Some(ffn_output) =
                run_ffn_or_moe(&ffn_input, cached, layer_idx, moe_config.as_ref(), n_embd)
            {
                for (h, f) in hidden_state.iter_mut().zip(ffn_output.iter()) {
                    *h += f;
                }
            }
        }

        prefetcher.on_layer_done(layer_idx, streamer, gguf);
    }

    // Now compute logits for each position and measure NLL
    let output_norm = streamer
        .load_named_tensor_f32(gguf, "output_norm.weight")
        .ok();
    let output_weight = streamer.load_named_tensor_f32(gguf, "output.weight").ok();
    let embd_weight = if output_weight.is_none() {
        streamer
            .load_named_tensor_f32(gguf, "token_embd.weight")
            .ok()
    } else {
        None
    };

    let mut total_nll: f64 = 0.0;

    // For each position i in [score_start-1 .. chunk_len-2], compute logits and check
    // the probability of token[i+1].
    // Position i predicts token at position i+1.
    for (i, hs) in hidden_states
        .iter()
        .enumerate()
        .take(chunk_len.saturating_sub(1))
        .skip(score_start.saturating_sub(1))
    {
        let target_pos = i + 1;
        if target_pos < score_start {
            continue;
        }

        let mut final_hidden = hs.clone();
        if let Some(ref norm_w) = output_norm {
            rms_norm(&mut final_hidden, norm_w);
        }

        let logits = if let Some(ref w) = output_weight {
            crate::metal::compute::matvec_f32_simd(w, &final_hidden, vocab_size, n_embd)
        } else if let Some(ref w) = embd_weight {
            crate::metal::compute::matvec_f32_simd(w, &final_hidden, vocab_size, n_embd)
        } else {
            vec![0.0f32; vocab_size]
        };

        let log_probs = log_softmax(&logits);
        let target_token = chunk_tokens[target_pos] as usize;
        let token_nll = if target_token < log_probs.len() {
            -(log_probs[target_token] as f64)
        } else {
            // Unknown token, assign max penalty
            30.0
        };

        total_nll += token_nll;
    }

    Ok(total_nll)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_softmax_uniform() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let log_probs = log_softmax(&logits);
        let expected = -(4.0f32).ln();
        for lp in &log_probs {
            assert!(
                (lp - expected).abs() < 1e-5,
                "got {}, expected {}",
                lp,
                expected
            );
        }
    }

    #[test]
    fn test_log_softmax_peaked() {
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let log_probs = log_softmax(&logits);
        // First element should have log-prob close to 0 (high probability)
        assert!(log_probs[0] > -0.01);
        // Others should have very negative log-probs
        assert!(log_probs[1] < -9.0);
    }

    #[test]
    fn test_log_softmax_sums_to_one() {
        let logits = vec![2.0, 1.0, 0.5, -1.0, 3.0];
        let log_probs = log_softmax(&logits);
        let sum: f64 = log_probs.iter().map(|&lp| (lp as f64).exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "probabilities sum to {}", sum);
    }

    #[test]
    fn test_log_softmax_numerical_stability() {
        // Very large logits should not cause overflow
        let logits = vec![1000.0, 999.0, 998.0];
        let log_probs = log_softmax(&logits);
        assert!(log_probs[0].is_finite());
        assert!(log_probs[1].is_finite());
        assert!(log_probs[2].is_finite());
        // First should be highest
        assert!(log_probs[0] > log_probs[1]);
        assert!(log_probs[1] > log_probs[2]);
    }

    #[test]
    fn test_perplexity_result_json() {
        let result = PerplexityResult {
            perplexity: 12.345,
            nll: 2.5135,
            tokens_evaluated: 1000,
            total_tokens: 1024,
            chunks: 2,
            chunk_perplexities: vec![11.5, 13.2],
            elapsed_secs: 45.67,
            tokens_per_sec: 21.9,
        };
        let json = result.to_json();
        assert!(json.contains("\"perplexity\": 12.3450"));
        assert!(json.contains("\"tokens_evaluated\": 1000"));
        assert!(json.contains("\"chunks\": 2"));
        assert!(json.contains("11.5000"));
        assert!(json.contains("13.2000"));
    }
}
