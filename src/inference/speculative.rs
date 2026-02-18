//! Speculative Decoding — accelerate inference using a small draft model
//!
//! Algorithm (Leviathan et al., 2023; Chen et al., 2023):
//! 1. Draft model generates K candidate tokens autoregressively (cheap)
//! 2. Target model verifies all K tokens in a single batched forward pass
//! 3. Accept tokens left-to-right until first rejection
//! 4. On rejection, resample from adjusted distribution: max(0, p_target - p_draft)
//! 5. Accepted tokens = "free" — they cost only the draft model's compute
//!
//! For ssd-llm this is especially powerful: the draft model (e.g. 1B) fits in RAM
//! while the target model (e.g. 70B) streams from SSD. Speculative decoding reduces
//! the number of expensive SSD-streaming forward passes.

use crate::inference::kv_cache::KvCache;
use crate::inference::sampler::Sampler;
use crate::inference::tokenizer::SimpleTokenizer;
use crate::model::cache::LayerCache;
use crate::model::gguf::GgufFile;
use crate::ssd::prefetch::{PrefetchStrategy, Prefetcher};
use crate::ssd::streamer::SsdStreamer;
use anyhow::{bail, Result};
use std::time::Instant;
use tracing::{debug, info};

/// Configuration for speculative decoding
pub struct SpeculativeConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
    /// Number of tokens to draft per speculation step (K)
    pub draft_ahead: usize,
}

/// Result of speculative decoding
pub struct SpeculativeResult {
    pub text: String,
    pub token_count: usize,
    pub tokens_per_sec: f64,
    pub prompt_tokens: usize,
    pub kv_cache_bytes: usize,
    /// Total draft tokens proposed
    pub draft_tokens_total: usize,
    /// Total draft tokens accepted
    pub draft_tokens_accepted: usize,
    /// Acceptance rate as fraction
    pub acceptance_rate: f64,
    /// Number of target model forward passes (= speculation rounds + resamples)
    pub target_forward_passes: usize,
}

/// Single forward pass through a model, returning logits
fn forward_single(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    kv_cache: &mut KvCache,
    token_id: u32,
    position: usize,
    prefetcher: &Prefetcher,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let n_embd = gguf.n_embd() as usize;
    let vocab_size = gguf.vocab_size() as usize;

    // Embed token
    let mut hidden_state = vec![0.0f32; n_embd];
    if let Ok(embeddings) = streamer.load_named_tensor_f32(gguf, "token_embd.weight") {
        let start = token_id as usize * n_embd;
        let end = start + n_embd;
        if end <= embeddings.len() {
            hidden_state = embeddings[start..end].to_vec();
        }
    }

    // Run through all layers
    crate::inference::transformer::forward_pass_pub(
        gguf, streamer, layer_cache, kv_cache,
        &mut hidden_state, position, prefetcher,
    )?;

    // Project to logits
    let mut final_hidden = hidden_state.clone();
    if let Ok(norm_w) = streamer.load_named_tensor_f32(gguf, "output_norm.weight") {
        crate::metal::compute::rmsnorm_f32_fast(&mut final_hidden, &norm_w, 1e-5);
    }

    let logits = if let Ok(output_weight) = streamer.load_named_tensor_f32(gguf, "output.weight") {
        crate::metal::compute::matvec_f32_simd(&output_weight, &final_hidden, vocab_size, n_embd)
    } else if let Ok(embd_weight) = streamer.load_named_tensor_f32(gguf, "token_embd.weight") {
        crate::metal::compute::matvec_f32_simd(&embd_weight, &final_hidden, vocab_size, n_embd)
    } else {
        vec![0.0f32; vocab_size]
    };

    Ok((logits, hidden_state))
}

/// Compute probability distribution from logits with temperature
fn logits_to_probs(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let scaled: Vec<f32> = if temperature < 1e-6 {
        logits.to_vec()
    } else {
        logits.iter().map(|&l| l / temperature).collect()
    };
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
    probs
}

/// Sample from a probability distribution
fn sample_from_probs(probs: &[f32], rng: &mut XorShift64) -> u32 {
    let r = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r <= cumulative {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Simple RNG for speculative decoding acceptance checks
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self { state: seed ^ 0x517cc1b727220a95 }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

/// Run speculative decoding with a draft model
///
/// The draft model runs autoregressively to propose K tokens.
/// The target model then verifies them, accepting a prefix and
/// resampling the first rejected position.
pub fn generate_speculative(
    // Target (large) model
    target_gguf: &GgufFile,
    target_streamer: &SsdStreamer,
    target_cache: &mut LayerCache,
    // Draft (small) model
    draft_gguf: &GgufFile,
    draft_streamer: &SsdStreamer,
    draft_cache: &mut LayerCache,
    // Shared
    prompt: &str,
    config: &SpeculativeConfig,
) -> Result<SpeculativeResult> {
    let t_n_embd = target_gguf.n_embd() as usize;
    let t_n_layers = target_gguf.n_layers();
    let t_n_head = target_gguf.n_head() as usize;
    let t_n_head_kv = target_gguf.n_head_kv() as usize;
    let t_head_dim = t_n_embd / t_n_head;
    let t_vocab_size = target_gguf.vocab_size() as usize;
    let t_n_ctx = target_gguf.n_ctx() as usize;

    let d_n_embd = draft_gguf.n_embd() as usize;
    let d_n_layers = draft_gguf.n_layers();
    let d_n_head = draft_gguf.n_head() as usize;
    let d_n_head_kv = draft_gguf.n_head_kv() as usize;
    let d_head_dim = d_n_embd / d_n_head;
    let d_vocab_size = draft_gguf.vocab_size() as usize;
    let d_n_ctx = draft_gguf.n_ctx() as usize;

    if t_n_embd == 0 || d_n_embd == 0 {
        bail!("Invalid model dimensions");
    }

    // Vocab sizes should match (same tokenizer family)
    if t_vocab_size != d_vocab_size {
        info!(
            "Warning: vocab size mismatch (target={}, draft={}), using min",
            t_vocab_size, d_vocab_size
        );
    }
    let vocab_size = t_vocab_size.min(d_vocab_size);

    info!(
        "Speculative decoding: target={}L/{}d, draft={}L/{}d, K={}",
        t_n_layers, t_n_embd, d_n_layers, d_n_embd, config.draft_ahead
    );

    let target_prefetcher = Prefetcher::new(PrefetchStrategy::LookAhead(2));
    let draft_prefetcher = Prefetcher::new(PrefetchStrategy::LookAhead(2));

    // Initialize KV caches
    let mut target_kv = KvCache::new(t_n_layers as usize, t_n_head_kv, t_head_dim, t_n_ctx);
    let mut draft_kv = KvCache::new(d_n_layers as usize, d_n_head_kv, d_head_dim, d_n_ctx);

    // Use target model's tokenizer
    let tokenizer = SimpleTokenizer::from_gguf(target_gguf);
    let tokens = tokenizer.encode(prompt);
    let prompt_len = tokens.len();
    info!("Prompt tokens: {}", prompt_len);

    let start = Instant::now();

    // Prefill both models with prompt
    for (pos, &token_id) in tokens.iter().enumerate() {
        forward_single(
            target_gguf, target_streamer, target_cache, &mut target_kv,
            token_id, pos, &target_prefetcher,
        )?;
        forward_single(
            draft_gguf, draft_streamer, draft_cache, &mut draft_kv,
            token_id, pos, &draft_prefetcher,
        )?;
    }

    let prefill_time = start.elapsed();
    info!(
        "Prefill: {} tokens in {:.2}ms",
        prompt_len, prefill_time.as_millis()
    );

    let decode_start = Instant::now();
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut rng = XorShift64::new();
    let mut pos = tokens.len();
    let mut draft_total = 0usize;
    let mut draft_accepted = 0usize;
    let mut target_passes = 0usize;

    // We need to get the initial target logits for the first position after prompt
    // Run one target forward pass to get logits at end of prompt
    let last_prompt_token = *tokens.last().unwrap_or(&0);

    while generated_tokens.len() < config.max_tokens {
        if target_kv.is_full() || draft_kv.is_full() {
            info!("KV cache full, stopping");
            break;
        }

        // === DRAFT PHASE ===
        // Draft model generates K candidate tokens
        let draft_start_pos = pos;
        let draft_kv_start = draft_kv.seq_len();
        let target_kv_start = target_kv.seq_len();
        let mut draft_tokens: Vec<u32> = Vec::with_capacity(config.draft_ahead);
        let mut draft_logits_list: Vec<Vec<f32>> = Vec::with_capacity(config.draft_ahead);

        // Get the last accepted token to start drafting from
        let mut current_token = if generated_tokens.is_empty() {
            last_prompt_token
        } else {
            *generated_tokens.last().unwrap()
        };

        for k in 0..config.draft_ahead {
            if draft_kv.is_full() { break; }

            let (logits, _) = forward_single(
                draft_gguf, draft_streamer, draft_cache, &mut draft_kv,
                current_token, pos + k, &draft_prefetcher,
            )?;

            let probs = logits_to_probs(&logits[..vocab_size], config.temperature);
            let token_id = sample_from_probs(&probs, &mut rng);

            if token_id == tokenizer.eos_id || token_id == 0 {
                break;
            }

            draft_tokens.push(token_id);
            draft_logits_list.push(logits);
            current_token = token_id;
        }

        draft_total += draft_tokens.len();

        if draft_tokens.is_empty() {
            // Draft produced nothing (EOS), do one target pass to confirm
            let (target_logits, _) = forward_single(
                target_gguf, target_streamer, target_cache, &mut target_kv,
                current_token, pos, &target_prefetcher,
            )?;
            target_passes += 1;

            let target_probs = logits_to_probs(&target_logits[..vocab_size], config.temperature);
            let final_token = sample_from_probs(&target_probs, &mut rng);

            // Also advance draft KV to stay in sync
            let _ = forward_single(
                draft_gguf, draft_streamer, draft_cache, &mut draft_kv,
                current_token, pos, &draft_prefetcher,
            );

            if final_token == tokenizer.eos_id || final_token == 0 {
                break;
            }
            generated_tokens.push(final_token);
            pos += 1;
            continue;
        }

        // === VERIFY PHASE ===
        // Target model processes the original token + all K draft tokens
        // We run target forward pass for each draft token position to get logits
        let mut target_logits_list: Vec<Vec<f32>> = Vec::with_capacity(draft_tokens.len() + 1);

        // Get target logits at starting position (before first draft token)
        let verify_start_token = if generated_tokens.is_empty() {
            last_prompt_token
        } else {
            *generated_tokens.last().unwrap()
        };

        let (logits_0, _) = forward_single(
            target_gguf, target_streamer, target_cache, &mut target_kv,
            verify_start_token, pos, &target_prefetcher,
        )?;
        target_logits_list.push(logits_0);
        target_passes += 1;

        // Run target forward pass for each draft token
        for (k, &draft_tok) in draft_tokens.iter().enumerate() {
            let (logits, _) = forward_single(
                target_gguf, target_streamer, target_cache, &mut target_kv,
                draft_tok, pos + 1 + k, &target_prefetcher,
            )?;
            target_logits_list.push(logits);
            target_passes += 1;
        }

        // === ACCEPTANCE PHASE ===
        // For each draft token k, compare p_target(x_k) vs p_draft(x_k)
        // Accept with probability min(1, p_target / p_draft)
        let mut n_accepted = 0usize;

        for k in 0..draft_tokens.len() {
            let draft_probs = logits_to_probs(&draft_logits_list[k][..vocab_size], config.temperature);
            let target_probs = logits_to_probs(&target_logits_list[k][..vocab_size], config.temperature);

            let token = draft_tokens[k] as usize;
            let p_draft = if token < draft_probs.len() { draft_probs[token] } else { 0.0 };
            let p_target = if token < target_probs.len() { target_probs[token] } else { 0.0 };

            // Accept with probability min(1, p_target / p_draft)
            let accept_prob = if p_draft > 0.0 { (p_target / p_draft).min(1.0) } else { 0.0 };
            let r = rng.next_f32();

            if r < accept_prob {
                // Accept this draft token
                generated_tokens.push(draft_tokens[k]);
                n_accepted += 1;
                debug!("Accept draft token {} (p_t={:.4}, p_d={:.4}, ratio={:.4})",
                    draft_tokens[k], p_target, p_draft, accept_prob);
            } else {
                // Reject — resample from adjusted distribution: max(0, p_target - p_draft)
                debug!("Reject draft token {} at position {} (p_t={:.4}, p_d={:.4})",
                    draft_tokens[k], k, p_target, p_draft);

                let mut adjusted: Vec<f32> = target_probs.iter().zip(draft_probs.iter())
                    .map(|(&pt, &pd)| (pt - pd).max(0.0))
                    .collect();
                let adj_sum: f32 = adjusted.iter().sum();
                if adj_sum > 0.0 {
                    for p in adjusted.iter_mut() { *p /= adj_sum; }
                } else {
                    // Fallback to target distribution
                    adjusted = target_probs.clone();
                }
                let resampled = sample_from_probs(&adjusted, &mut rng);
                if resampled != tokenizer.eos_id && resampled != 0 {
                    generated_tokens.push(resampled);
                }
                break;
            }
        }

        // If all K tokens were accepted, sample one bonus token from the last target logits
        if n_accepted == draft_tokens.len() {
            let last_target_probs = logits_to_probs(
                &target_logits_list[draft_tokens.len()][..vocab_size],
                config.temperature,
            );
            let bonus_token = sample_from_probs(&last_target_probs, &mut rng);
            if bonus_token != tokenizer.eos_id && bonus_token != 0 {
                generated_tokens.push(bonus_token);
                n_accepted += 1; // count the bonus
            }
        }

        draft_accepted += n_accepted.min(draft_tokens.len());

        // Rollback KV caches to only include accepted positions
        let accepted_positions = pos + generated_tokens.len() - (generated_tokens.len().saturating_sub(
            generated_tokens.len() - (generated_tokens.len() - n_accepted.min(generated_tokens.len()))
        ));
        // Simpler: we know how many tokens we just added
        let new_total_pos = prompt_len + generated_tokens.len();
        // Rollback both KV caches to match actual accepted length
        target_kv.rollback(new_total_pos);
        draft_kv.rollback(draft_kv_start); // draft needs full re-sync

        // Re-sync draft KV cache by replaying accepted tokens
        let draft_replay_start = draft_kv.seq_len();
        for i in draft_replay_start..new_total_pos {
            let tok = if i < prompt_len {
                tokens[i]
            } else {
                generated_tokens[i - prompt_len]
            };
            let _ = forward_single(
                draft_gguf, draft_streamer, draft_cache, &mut draft_kv,
                tok, i, &draft_prefetcher,
            );
        }

        pos = new_total_pos;

        // Check for EOS in generated tokens
        if let Some(&last) = generated_tokens.last() {
            if last == tokenizer.eos_id || last == 0 {
                generated_tokens.pop(); // remove EOS from output
                break;
            }
        }
    }

    let decode_time = decode_start.elapsed();
    let token_count = generated_tokens.len();
    let tokens_per_sec = if decode_time.as_secs_f64() > 0.0 {
        token_count as f64 / decode_time.as_secs_f64()
    } else {
        0.0
    };

    let acceptance_rate = if draft_total > 0 {
        draft_accepted as f64 / draft_total as f64
    } else {
        0.0
    };

    info!(
        "Speculative decode: {} tokens in {:.2}ms ({:.1} tok/s)",
        token_count, decode_time.as_millis(), tokens_per_sec
    );
    info!(
        "Draft: {}/{} accepted ({:.1}%), target passes: {} (vs {} without speculation)",
        draft_accepted, draft_total, acceptance_rate * 100.0,
        target_passes, token_count
    );

    let text = tokenizer.decode(&generated_tokens);

    Ok(SpeculativeResult {
        text,
        token_count,
        tokens_per_sec,
        prompt_tokens: prompt_len,
        kv_cache_bytes: target_kv.size_bytes() + draft_kv.size_bytes(),
        draft_tokens_total: draft_total,
        draft_tokens_accepted: draft_accepted,
        acceptance_rate,
        target_forward_passes: target_passes,
    })
}

/// Streaming speculative generator
pub struct SpeculativeStreamingGenerator<'a> {
    // Target model
    target_gguf: &'a GgufFile,
    target_streamer: &'a SsdStreamer,
    target_cache: &'a mut LayerCache,
    target_kv: KvCache,
    target_prefetcher: Prefetcher,
    // Draft model
    draft_gguf: &'a GgufFile,
    draft_streamer: &'a SsdStreamer,
    draft_cache: &'a mut LayerCache,
    draft_kv: KvCache,
    draft_prefetcher: Prefetcher,
    // State
    tokenizer: SimpleTokenizer,
    rng: XorShift64,
    temperature: f32,
    vocab_size: usize,
    draft_ahead: usize,
    position: usize,
    prompt_len: usize,
    remaining: usize,
    done: bool,
    // Buffer for tokens accepted in a single speculation round
    pending_tokens: Vec<u32>,
    pending_idx: usize,
    last_token: u32,
    all_generated: Vec<u32>,
    // Stats
    pub draft_total: usize,
    pub draft_accepted: usize,
    pub target_passes: usize,
}

impl<'a> SpeculativeStreamingGenerator<'a> {
    /// Get the next token string, or None if done
    pub fn next_token(&mut self) -> Result<Option<String>> {
        if self.done {
            return Ok(None);
        }

        // Return buffered tokens first
        if self.pending_idx < self.pending_tokens.len() {
            let token_id = self.pending_tokens[self.pending_idx];
            self.pending_idx += 1;
            return Ok(Some(self.tokenizer.decode_token(token_id)));
        }

        if self.remaining == 0 {
            self.done = true;
            return Ok(None);
        }

        // Run one speculation round
        self.pending_tokens.clear();
        self.pending_idx = 0;

        self.speculate_round()?;

        if self.pending_tokens.is_empty() {
            self.done = true;
            return Ok(None);
        }

        let token_id = self.pending_tokens[0];
        self.pending_idx = 1;
        self.remaining = self.remaining.saturating_sub(self.pending_tokens.len());
        Ok(Some(self.tokenizer.decode_token(token_id)))
    }

    fn speculate_round(&mut self) -> Result<()> {
        if self.target_kv.is_full() || self.draft_kv.is_full() {
            return Ok(());
        }

        let draft_kv_start = self.draft_kv.seq_len();

        // Draft K tokens
        let mut draft_tokens = Vec::with_capacity(self.draft_ahead);
        let mut draft_logits_list = Vec::with_capacity(self.draft_ahead);
        let mut current_token = self.last_token;

        for k in 0..self.draft_ahead {
            if self.draft_kv.is_full() { break; }
            let (logits, _) = forward_single(
                self.draft_gguf, self.draft_streamer, self.draft_cache, &mut self.draft_kv,
                current_token, self.position + k, &self.draft_prefetcher,
            )?;
            let probs = logits_to_probs(&logits[..self.vocab_size], self.temperature);
            let token_id = sample_from_probs(&probs, &mut self.rng);
            if token_id == self.tokenizer.eos_id || token_id == 0 { break; }
            draft_tokens.push(token_id);
            draft_logits_list.push(logits);
            current_token = token_id;
        }

        self.draft_total += draft_tokens.len();

        if draft_tokens.is_empty() {
            // No draft tokens — do single target pass
            let (target_logits, _) = forward_single(
                self.target_gguf, self.target_streamer, self.target_cache, &mut self.target_kv,
                self.last_token, self.position, &self.target_prefetcher,
            )?;
            self.target_passes += 1;
            let _ = forward_single(
                self.draft_gguf, self.draft_streamer, self.draft_cache, &mut self.draft_kv,
                self.last_token, self.position, &self.draft_prefetcher,
            );
            let probs = logits_to_probs(&target_logits[..self.vocab_size], self.temperature);
            let token_id = sample_from_probs(&probs, &mut self.rng);
            if token_id != self.tokenizer.eos_id && token_id != 0 {
                self.pending_tokens.push(token_id);
                self.last_token = token_id;
                self.all_generated.push(token_id);
                self.position += 1;
            }
            return Ok(());
        }

        // Verify with target model
        let mut target_logits_list = Vec::with_capacity(draft_tokens.len() + 1);
        let (logits_0, _) = forward_single(
            self.target_gguf, self.target_streamer, self.target_cache, &mut self.target_kv,
            self.last_token, self.position, &self.target_prefetcher,
        )?;
        target_logits_list.push(logits_0);
        self.target_passes += 1;

        for (k, &draft_tok) in draft_tokens.iter().enumerate() {
            let (logits, _) = forward_single(
                self.target_gguf, self.target_streamer, self.target_cache, &mut self.target_kv,
                draft_tok, self.position + 1 + k, &self.target_prefetcher,
            )?;
            target_logits_list.push(logits);
            self.target_passes += 1;
        }

        // Accept/reject
        let mut n_accepted = 0usize;
        for k in 0..draft_tokens.len() {
            let draft_probs = logits_to_probs(&draft_logits_list[k][..self.vocab_size], self.temperature);
            let target_probs = logits_to_probs(&target_logits_list[k][..self.vocab_size], self.temperature);
            let token = draft_tokens[k] as usize;
            let p_d = if token < draft_probs.len() { draft_probs[token] } else { 0.0 };
            let p_t = if token < target_probs.len() { target_probs[token] } else { 0.0 };
            let accept_prob = if p_d > 0.0 { (p_t / p_d).min(1.0) } else { 0.0 };

            if self.rng.next_f32() < accept_prob {
                self.pending_tokens.push(draft_tokens[k]);
                n_accepted += 1;
            } else {
                // Reject and resample
                let mut adjusted: Vec<f32> = target_probs.iter().zip(draft_probs.iter())
                    .map(|(&pt, &pd)| (pt - pd).max(0.0))
                    .collect();
                let s: f32 = adjusted.iter().sum();
                if s > 0.0 { for p in adjusted.iter_mut() { *p /= s; } }
                else { adjusted = target_probs; }
                let resampled = sample_from_probs(&adjusted, &mut self.rng);
                if resampled != self.tokenizer.eos_id && resampled != 0 {
                    self.pending_tokens.push(resampled);
                }
                break;
            }
        }

        // Bonus token if all accepted
        if n_accepted == draft_tokens.len() {
            let last_probs = logits_to_probs(
                &target_logits_list[draft_tokens.len()][..self.vocab_size],
                self.temperature,
            );
            let bonus = sample_from_probs(&last_probs, &mut self.rng);
            if bonus != self.tokenizer.eos_id && bonus != 0 {
                self.pending_tokens.push(bonus);
            }
        }

        self.draft_accepted += n_accepted.min(draft_tokens.len());

        // Update state
        for &tok in &self.pending_tokens {
            self.all_generated.push(tok);
        }
        if let Some(&last) = self.pending_tokens.last() {
            self.last_token = last;
        }

        // Rollback and re-sync KV caches
        let new_total = self.prompt_len + self.all_generated.len();
        self.target_kv.rollback(new_total);
        self.draft_kv.rollback(draft_kv_start);

        // Re-sync draft
        let replay_from = self.draft_kv.seq_len();
        for i in replay_from..new_total {
            let tok = if i < self.prompt_len {
                // This shouldn't happen since draft_kv_start >= prompt_len
                0
            } else {
                self.all_generated[i - self.prompt_len]
            };
            let _ = forward_single(
                self.draft_gguf, self.draft_streamer, self.draft_cache, &mut self.draft_kv,
                tok, i, &self.draft_prefetcher,
            );
        }

        self.position = new_total;
        Ok(())
    }
}

/// Create a speculative streaming generator
pub fn generate_speculative_streaming<'a>(
    target_gguf: &'a GgufFile,
    target_streamer: &'a SsdStreamer,
    target_cache: &'a mut LayerCache,
    draft_gguf: &'a GgufFile,
    draft_streamer: &'a SsdStreamer,
    draft_cache: &'a mut LayerCache,
    prompt: &str,
    config: &SpeculativeConfig,
) -> Result<SpeculativeStreamingGenerator<'a>> {
    let t_n_embd = target_gguf.n_embd() as usize;
    let t_n_layers = target_gguf.n_layers();
    let t_n_head = target_gguf.n_head() as usize;
    let t_n_head_kv = target_gguf.n_head_kv() as usize;
    let t_head_dim = t_n_embd / t_n_head;
    let t_n_ctx = target_gguf.n_ctx() as usize;

    let d_n_embd = draft_gguf.n_embd() as usize;
    let d_n_layers = draft_gguf.n_layers();
    let d_n_head = draft_gguf.n_head() as usize;
    let d_n_head_kv = draft_gguf.n_head_kv() as usize;
    let d_head_dim = d_n_embd / d_n_head;
    let d_n_ctx = draft_gguf.n_ctx() as usize;

    let vocab_size = (target_gguf.vocab_size() as usize).min(draft_gguf.vocab_size() as usize);

    let target_prefetcher = Prefetcher::new(PrefetchStrategy::LookAhead(2));
    let draft_prefetcher = Prefetcher::new(PrefetchStrategy::LookAhead(2));

    let mut target_kv = KvCache::new(t_n_layers as usize, t_n_head_kv, t_head_dim, t_n_ctx);
    let mut draft_kv = KvCache::new(d_n_layers as usize, d_n_head_kv, d_head_dim, d_n_ctx);

    let tokenizer = SimpleTokenizer::from_gguf(target_gguf);
    let tokens = tokenizer.encode(prompt);
    let prompt_len = tokens.len();

    // Prefill both
    for (pos, &token_id) in tokens.iter().enumerate() {
        forward_single(
            target_gguf, target_streamer, target_cache, &mut target_kv,
            token_id, pos, &target_prefetcher,
        )?;
        forward_single(
            draft_gguf, draft_streamer, draft_cache, &mut draft_kv,
            token_id, pos, &draft_prefetcher,
        )?;
    }

    let last_token = *tokens.last().unwrap_or(&0);

    Ok(SpeculativeStreamingGenerator {
        target_gguf,
        target_streamer,
        target_cache,
        target_kv,
        target_prefetcher,
        draft_gguf,
        draft_streamer,
        draft_cache,
        draft_kv,
        draft_prefetcher,
        tokenizer,
        rng: XorShift64::new(),
        temperature: config.temperature,
        vocab_size,
        draft_ahead: config.draft_ahead,
        position: prompt_len,
        prompt_len,
        remaining: config.max_tokens,
        done: false,
        pending_tokens: Vec::new(),
        pending_idx: 0,
        last_token,
        all_generated: Vec::new(),
        draft_total: 0,
        draft_accepted: 0,
        target_passes: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logits_to_probs_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = logits_to_probs(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Probs should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_logits_to_probs_temperature_sharpening() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs_hot = logits_to_probs(&logits, 2.0);
        let probs_cold = logits_to_probs(&logits, 0.1);
        // Cold temperature should make distribution sharper (max prob higher)
        let max_hot = probs_hot.iter().cloned().fold(0.0f32, f32::max);
        let max_cold = probs_cold.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_cold > max_hot, "Cold temp should be sharper");
    }

    #[test]
    fn test_xorshift_range() {
        let mut rng = XorShift64::new();
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0, "RNG value {} out of range", v);
        }
    }

    #[test]
    fn test_sample_from_probs_deterministic() {
        let probs = vec![0.0, 0.0, 1.0, 0.0];
        let mut rng = XorShift64::new();
        // With prob=1.0 at index 2, should always return 2
        for _ in 0..100 {
            assert_eq!(sample_from_probs(&probs, &mut rng), 2);
        }
    }

    #[test]
    fn test_kv_cache_rollback() {
        let mut cache = KvCache::new(2, 4, 64, 2048);
        // Add 5 positions
        for _ in 0..5 {
            let k = vec![1.0f32; 4 * 64];
            let v = vec![2.0f32; 4 * 64];
            cache.layer_mut(0).append(k.clone(), v.clone());
            cache.layer_mut(1).append(k, v);
        }
        assert_eq!(cache.seq_len(), 5);

        // Rollback to 3
        cache.rollback(3);
        assert_eq!(cache.seq_len(), 3);
        assert_eq!(cache.layer(0).seq_len(), 3);
        assert_eq!(cache.layer(1).seq_len(), 3);
    }
}
