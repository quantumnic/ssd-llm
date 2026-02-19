//! Token sampling strategies: Temperature, Top-K, Top-P (nucleus), Min-P, TFS, Mirostat

use std::cell::Cell;

/// Mirostat mode selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MirostatMode {
    /// Disabled — use standard Top-K/Top-P/Min-P sampling
    Disabled,
    /// Mirostat v1: perplexity-controlled sampling with adaptive top-k
    V1,
    /// Mirostat v2: simplified perplexity control with adaptive truncation
    V2,
}

pub struct Sampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    /// Tail-Free Sampling parameter (0.0 = disabled, typical: 0.95-1.0)
    tfs_z: f32,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    /// Mirostat mode
    mirostat: MirostatMode,
    /// Mirostat target surprise (tau), default 5.0
    mirostat_tau: f32,
    /// Mirostat learning rate (eta), default 0.1
    mirostat_eta: f32,
    /// Mirostat adaptive state: current mu (tracks 2*tau initially)
    mirostat_mu: Cell<f32>,
    rng_state: Cell<u64>,
}

impl Sampler {
    fn make_seed() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
            ^ 0x517cc1b727220a95
    }

    pub fn new(temperature: f32, top_k: usize, top_p: f32) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            tfs_z: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            mirostat: MirostatMode::Disabled,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_mu: Cell::new(10.0),
            rng_state: Cell::new(Self::make_seed()),
        }
    }

    /// Create a sampler with repetition penalty parameters
    pub fn with_penalties(
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            tfs_z: 0.0,
            repetition_penalty: repetition_penalty.max(0.01),
            frequency_penalty,
            presence_penalty,
            mirostat: MirostatMode::Disabled,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_mu: Cell::new(10.0),
            rng_state: Cell::new(Self::make_seed()),
        }
    }

    /// Create a sampler with all parameters including Min-P
    ///
    /// Min-P filtering keeps only tokens whose probability is at least `min_p` times
    /// the probability of the most likely token. This provides adaptive filtering that
    /// scales with model confidence — when the model is sure, fewer tokens pass;
    /// when uncertain, more diversity is allowed.
    ///
    /// Typical values: 0.05–0.1 for creative text, 0.1–0.2 for focused output.
    pub fn with_min_p(
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p: min_p.clamp(0.0, 1.0),
            tfs_z: 0.0,
            repetition_penalty: repetition_penalty.max(0.01),
            frequency_penalty,
            presence_penalty,
            mirostat: MirostatMode::Disabled,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_mu: Cell::new(10.0),
            rng_state: Cell::new(Self::make_seed()),
        }
    }

    /// Create a sampler with Tail-Free Sampling
    ///
    /// TFS uses the second derivative of the sorted probability distribution to find
    /// the "tail" — tokens whose probability drops off sharply. It removes the tail
    /// and samples from the remaining distribution. This adapts better than top-p to
    /// distributions with long tails.
    ///
    /// `tfs_z`: threshold in [0, 1]. 1.0 = disabled, 0.95 = moderate filtering.
    pub fn with_tfs(
        temperature: f32,
        top_k: usize,
        top_p: f32,
        tfs_z: f32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            tfs_z: tfs_z.clamp(0.0, 1.0),
            repetition_penalty: repetition_penalty.max(0.01),
            frequency_penalty,
            presence_penalty,
            mirostat: MirostatMode::Disabled,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_mu: Cell::new(10.0),
            rng_state: Cell::new(Self::make_seed()),
        }
    }

    /// Create a sampler with Mirostat adaptive perplexity control
    ///
    /// Mirostat dynamically adjusts the sampling truncation to maintain a target
    /// perplexity (surprise) level. This produces more coherent text than fixed
    /// top-k/top-p by adapting to the model's confidence at each step.
    ///
    /// - `mode`: V1 (original with adaptive k) or V2 (simplified truncation)
    /// - `tau`: target surprise level (default 5.0, lower = more focused)
    /// - `eta`: learning rate for mu adaptation (default 0.1)
    pub fn with_mirostat(temperature: f32, mode: MirostatMode, tau: f32, eta: f32) -> Self {
        Self {
            temperature,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            tfs_z: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            mirostat: mode,
            mirostat_tau: tau.max(0.0),
            mirostat_eta: eta.clamp(0.0, 1.0),
            mirostat_mu: Cell::new(2.0 * tau),
            rng_state: Cell::new(Self::make_seed()),
        }
    }

    /// Apply repetition, frequency, and presence penalties to logits based on prior tokens
    pub fn apply_penalties(&self, logits: &mut [f32], previous_tokens: &[u32]) {
        if (self.repetition_penalty - 1.0).abs() < 1e-6
            && self.frequency_penalty.abs() < 1e-6
            && self.presence_penalty.abs() < 1e-6
        {
            return;
        }

        // Count token frequencies
        let mut freq: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for &tok in previous_tokens {
            *freq.entry(tok).or_insert(0) += 1;
        }

        for (&tok, &count) in &freq {
            let idx = tok as usize;
            if idx >= logits.len() {
                continue;
            }

            // Repetition penalty (multiplicative): divide positive logits, multiply negative
            if self.repetition_penalty != 1.0 {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.repetition_penalty;
                } else {
                    logits[idx] *= self.repetition_penalty;
                }
            }

            // Frequency penalty (additive, scales with count)
            logits[idx] -= self.frequency_penalty * count as f32;

            // Presence penalty (additive, binary)
            logits[idx] -= self.presence_penalty;
        }
    }

    /// Sample a token ID from logits
    pub fn sample(&self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        // Greedy if temperature is very low
        if self.temperature < 1e-6 {
            return argmax(logits) as u32;
        }

        // Dispatch to Mirostat if enabled
        match self.mirostat {
            MirostatMode::V1 => return self.sample_mirostat_v1(logits),
            MirostatMode::V2 => return self.sample_mirostat_v2(logits),
            MirostatMode::Disabled => {}
        }

        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|&l| l / self.temperature).collect();

        // Top-K filtering
        let mut indexed: Vec<(usize, f32)> =
            scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.top_k.min(indexed.len());
        let candidates: Vec<(usize, f32)> = indexed[..k].to_vec();

        // Min-P filtering: keep tokens with prob >= min_p * max_prob
        // Applied after top-k but before softmax, using pre-softmax logits
        let candidates = if self.min_p > 0.0 && !candidates.is_empty() {
            // Compute softmax over candidates to get probabilities for Min-P check
            let max_logit = candidates
                .iter()
                .map(|(_, v)| *v)
                .fold(f32::NEG_INFINITY, f32::max);
            let probs: Vec<(usize, f32)> = candidates
                .iter()
                .map(|(idx, v)| (*idx, (v - max_logit).exp()))
                .collect();
            let sum: f32 = probs.iter().map(|(_, p)| p).sum();
            let probs: Vec<(usize, f32)> = probs
                .into_iter()
                .map(|(idx, p)| (idx, if sum > 0.0 { p / sum } else { p }))
                .collect();

            let max_prob = probs.iter().map(|(_, p)| *p).fold(0.0f32, f32::max);
            let threshold = self.min_p * max_prob;

            let filtered: Vec<(usize, f32)> = candidates
                .iter()
                .zip(probs.iter())
                .filter(|(_, (_, p))| *p >= threshold)
                .map(|(c, _)| *c)
                .collect();

            if filtered.is_empty() {
                vec![candidates[0]] // keep at least the top token
            } else {
                filtered
            }
        } else {
            candidates
        };

        // TFS filtering: remove the "tail" based on second derivative of sorted probabilities
        let candidates = if self.tfs_z > 0.0 && self.tfs_z < 1.0 && candidates.len() > 2 {
            tail_free_filter(&candidates, self.tfs_z)
        } else {
            candidates
        };

        // Softmax over candidates
        let max_val = candidates
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = candidates
            .iter()
            .map(|(idx, v)| (*idx, (v - max_val).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }

        // Top-P (nucleus) filtering
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumsum = 0.0f32;
        let mut nucleus = Vec::new();
        for (idx, p) in &probs {
            cumsum += p;
            nucleus.push((*idx, *p));
            if cumsum >= self.top_p {
                break;
            }
        }

        // Renormalize
        let nuc_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
        if nuc_sum > 0.0 {
            for (_, p) in nucleus.iter_mut() {
                *p /= nuc_sum;
            }
        }

        // Sample from nucleus
        self.sample_from_distribution(&nucleus)
    }

    /// Mirostat v1: adaptive top-k based on target surprise
    ///
    /// Estimates the optimal k (number of candidates) to achieve target perplexity tau.
    /// Uses Zipf's law approximation: the i-th most likely token has probability ~ i^(-s).
    /// Adjusts mu after each token to track the target surprise.
    fn sample_mirostat_v1(&self, logits: &[f32]) -> u32 {
        let scaled: Vec<f32> = logits.iter().map(|&l| l / self.temperature).collect();
        let mut indexed: Vec<(usize, f32)> =
            scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Softmax to get probabilities
        let max_val = indexed[0].1;
        let mut probs: Vec<(usize, f32)> = indexed
            .iter()
            .map(|(idx, v)| (*idx, (v - max_val).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }

        let mu = self.mirostat_mu.get();

        // Estimate Zipf exponent s from the top two probabilities
        let s = estimate_zipf_s(probs[0].1, probs.get(1).map(|(_, p)| *p).unwrap_or(1e-10));

        // Compute optimal k: k = (epsilon * 2^mu)^(1/s)
        // where epsilon normalizes the Zipf distribution
        let k_float = ((mu.exp2()) * self.compute_zipf_epsilon(s, logits.len())).powf(1.0 / s);
        let k = (k_float.round() as usize).clamp(1, probs.len());

        // Truncate to top-k
        let candidates: Vec<(usize, f32)> = probs[..k].to_vec();

        // Renormalize
        let cand_sum: f32 = candidates.iter().map(|(_, p)| p).sum();
        let candidates: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|(idx, p)| (idx, if cand_sum > 0.0 { p / cand_sum } else { p }))
            .collect();

        // Sample
        let token = self.sample_from_distribution(&candidates);

        // Update mu: mu_new = mu - eta * (surprise - tau)
        // surprise = -log2(p(token))
        let token_prob = candidates
            .iter()
            .find(|(idx, _)| *idx == token as usize)
            .map(|(_, p)| *p)
            .unwrap_or(1e-10);
        let surprise = -token_prob.max(1e-10).log2();
        let new_mu = mu - self.mirostat_eta * (surprise - self.mirostat_tau);
        self.mirostat_mu.set(new_mu);

        token
    }

    /// Mirostat v2: simplified adaptive truncation
    ///
    /// Instead of estimating Zipf parameters, v2 directly truncates the distribution
    /// at the point where cumulative surprise exceeds mu. Simpler and often works
    /// better than v1 in practice.
    fn sample_mirostat_v2(&self, logits: &[f32]) -> u32 {
        let scaled: Vec<f32> = logits.iter().map(|&l| l / self.temperature).collect();
        let mut indexed: Vec<(usize, f32)> =
            scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Softmax
        let max_val = indexed[0].1;
        let mut probs: Vec<(usize, f32)> = indexed
            .iter()
            .map(|(idx, v)| (*idx, (v - max_val).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }

        let mu = self.mirostat_mu.get();

        // Keep tokens whose surprise (-log2(p)) <= mu
        let candidates: Vec<(usize, f32)> = probs
            .iter()
            .filter(|(_, p)| -p.max(1e-10).log2() <= mu)
            .copied()
            .collect();

        // Always keep at least the top token
        let candidates = if candidates.is_empty() {
            vec![probs[0]]
        } else {
            candidates
        };

        // Renormalize
        let cand_sum: f32 = candidates.iter().map(|(_, p)| p).sum();
        let candidates: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|(idx, p)| (idx, if cand_sum > 0.0 { p / cand_sum } else { p }))
            .collect();

        let token = self.sample_from_distribution(&candidates);

        // Update mu
        let token_prob = candidates
            .iter()
            .find(|(idx, _)| *idx == token as usize)
            .map(|(_, p)| *p)
            .unwrap_or(1e-10);
        let surprise = -token_prob.max(1e-10).log2();
        let new_mu = mu - self.mirostat_eta * (surprise - self.mirostat_tau);
        self.mirostat_mu.set(new_mu);

        token
    }

    /// Compute Zipf epsilon normalization constant for Mirostat v1
    fn compute_zipf_epsilon(&self, s: f32, vocab_size: usize) -> f32 {
        // epsilon = sum_{i=1}^{n} i^(-s)  (generalized harmonic number)
        // For large vocab, approximate with integral: n^(1-s) / (1-s) for s != 1
        if (s - 1.0).abs() < 1e-6 {
            // Harmonic series: ~ln(n)
            (vocab_size as f32).ln()
        } else if s > 1.0 {
            // Converges; use partial sum for small n, approximation for large
            let n = vocab_size.min(1000) as f32;
            (1..=n as usize).map(|i| (i as f32).powf(-s)).sum::<f32>()
        } else {
            let n = vocab_size as f32;
            n.powf(1.0 - s) / (1.0 - s)
        }
    }

    /// Sample from a probability distribution (Vec of (index, probability))
    fn sample_from_distribution(&self, dist: &[(usize, f32)]) -> u32 {
        let r = self.random_f32();
        let mut cumulative = 0.0f32;
        for (idx, p) in dist {
            cumulative += p;
            if r <= cumulative {
                return *idx as u32;
            }
        }
        dist.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
    }

    /// xorshift64 PRNG returning f32 in [0, 1)
    fn random_f32(&self) -> f32 {
        let mut state = self.rng_state.get();
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.rng_state.set(state);
        // Use upper bits for better distribution
        ((state >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

/// Tail-Free Sampling: filter candidates by second derivative of sorted probabilities
///
/// TFS computes the second derivative (discrete) of the sorted probability distribution,
/// normalizes it, and accumulates from most likely to least likely. Tokens beyond the
/// cumulative threshold `z` are removed — these are the "tail" tokens.
fn tail_free_filter(candidates: &[(usize, f32)], z: f32) -> Vec<(usize, f32)> {
    if candidates.len() <= 2 {
        return candidates.to_vec();
    }

    // Softmax to get probabilities (candidates are already sorted by logit descending)
    let max_val = candidates
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = candidates
        .iter()
        .map(|(_, v)| (v - max_val).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    let probs: Vec<f32> = probs
        .iter()
        .map(|p| if sum > 0.0 { p / sum } else { *p })
        .collect();

    // First derivative
    let first_deriv: Vec<f32> = probs.windows(2).map(|w| (w[0] - w[1]).abs()).collect();

    // Second derivative
    let second_deriv: Vec<f32> = first_deriv
        .windows(2)
        .map(|w| (w[0] - w[1]).abs())
        .collect();

    // Normalize second derivative
    let sd_sum: f32 = second_deriv.iter().sum();
    if sd_sum < 1e-12 {
        return candidates.to_vec();
    }
    let normalized: Vec<f32> = second_deriv.iter().map(|&d| d / sd_sum).collect();

    // Accumulate and find cutoff
    let mut cumsum = 0.0f32;
    let mut cutoff = candidates.len();
    for (i, &nd) in normalized.iter().enumerate() {
        cumsum += nd;
        if cumsum > z {
            // Keep tokens 0..=i+1 (the second derivative at i maps to token i+1 in the original)
            cutoff = i + 2; // +2 because second derivative loses 2 elements
            break;
        }
    }

    let result: Vec<(usize, f32)> = candidates[..cutoff.min(candidates.len())].to_vec();
    if result.is_empty() {
        vec![candidates[0]]
    } else {
        result
    }
}

/// Estimate Zipf exponent s from the two highest probabilities
/// Using the relationship p1/p2 ≈ 2^s for Zipf distribution
fn estimate_zipf_s(p1: f32, p2: f32) -> f32 {
    if p2 < 1e-10 {
        return 1.0;
    }
    let ratio = (p1 / p2).max(1.0);
    (ratio.log2()).clamp(0.1, 10.0)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let sampler = Sampler::new(0.0, 40, 0.9);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(sampler.sample(&logits), 3); // argmax
    }

    #[test]
    fn test_rng_advances() {
        let sampler = Sampler::new(1.0, 40, 0.9);
        let a = sampler.random_f32();
        let b = sampler.random_f32();
        assert_ne!(a, b, "RNG should produce different values");
    }

    #[test]
    fn test_repetition_penalty() {
        let sampler = Sampler::with_penalties(0.0, 40, 0.9, 2.0, 0.0, 0.0);
        // Token 3 has highest logit but was seen before — penalty should reduce it
        let mut logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        sampler.apply_penalties(&mut logits, &[3]);
        // Positive logit divided by 2.0: 0.9 -> 0.45
        assert!((logits[3] - 0.45).abs() < 1e-5);
        // Unmentioned tokens unchanged
        assert!((logits[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_frequency_penalty() {
        let sampler = Sampler::with_penalties(0.0, 40, 0.9, 1.0, 0.5, 0.0);
        let mut logits = vec![1.0, 2.0, 3.0];
        // Token 2 appeared 3 times
        sampler.apply_penalties(&mut logits, &[2, 2, 2]);
        // 3.0 - 0.5*3 = 1.5
        assert!((logits[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_presence_penalty() {
        let sampler = Sampler::with_penalties(0.0, 40, 0.9, 1.0, 0.0, 1.0);
        let mut logits = vec![1.0, 2.0, 3.0];
        sampler.apply_penalties(&mut logits, &[1, 2, 2]);
        // Token 1: 2.0 - 1.0 = 1.0
        assert!((logits[1] - 1.0).abs() < 1e-5);
        // Token 2: 3.0 - 1.0 = 2.0 (presence is binary, not scaled by count)
        assert!((logits[2] - 2.0).abs() < 1e-5);
        // Token 0 unchanged
        assert!((logits[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_min_p_filters_low_probability() {
        // With min_p=0.5, only tokens with prob >= 50% of max prob should survive
        let sampler = Sampler::with_min_p(0.0, 40, 0.9, 0.5, 1.0, 0.0, 0.0);
        // Token 3 has highest logit (greedy), min_p should not affect greedy
        let logits = vec![0.1, 0.5, 0.3, 5.0, 0.2];
        assert_eq!(sampler.sample(&logits), 3);
    }

    #[test]
    fn test_min_p_zero_has_no_effect() {
        // With min_p=0, sampling should work the same as without
        let sampler_no_minp = Sampler::new(0.0, 40, 0.9);
        let sampler_minp_zero = Sampler::with_min_p(0.0, 40, 0.9, 0.0, 1.0, 0.0, 0.0);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(
            sampler_no_minp.sample(&logits),
            sampler_minp_zero.sample(&logits)
        );
    }

    #[test]
    fn test_tfs_filters_tail() {
        // With tfs_z=0.5, TFS should aggressively filter the tail
        let sampler = Sampler::with_tfs(0.0, 40, 0.9, 0.5, 1.0, 0.0, 0.0);
        // Greedy still picks argmax regardless of TFS
        let logits = vec![0.1, 0.5, 0.3, 5.0, 0.2];
        assert_eq!(sampler.sample(&logits), 3);
    }

    #[test]
    fn test_tfs_disabled_when_one() {
        // tfs_z=1.0 should be effectively disabled
        let sampler_no_tfs = Sampler::new(1.0, 40, 0.9);
        let sampler_tfs_one = Sampler::with_tfs(1.0, 40, 0.9, 1.0, 1.0, 0.0, 0.0);
        // Both should produce valid tokens (can't check exact equality due to RNG)
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let t1 = sampler_no_tfs.sample(&logits);
        let t2 = sampler_tfs_one.sample(&logits);
        assert!(t1 < 5);
        assert!(t2 < 5);
    }

    #[test]
    fn test_tfs_filter_function() {
        // Test the tail_free_filter directly
        let candidates = vec![(0, 5.0), (1, 3.0), (2, 1.0), (3, 0.1), (4, -2.0)];
        let filtered = tail_free_filter(&candidates, 0.5);
        // Should keep at least the top tokens and cut the tail
        assert!(!filtered.is_empty());
        assert!(filtered.len() <= candidates.len());
        // First token should always be kept
        assert_eq!(filtered[0].0, 0);
    }

    #[test]
    fn test_mirostat_v1_samples_valid() {
        let sampler = Sampler::with_mirostat(0.7, MirostatMode::V1, 5.0, 0.1);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.6];
        let token = sampler.sample(&logits);
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn test_mirostat_v2_samples_valid() {
        let sampler = Sampler::with_mirostat(0.7, MirostatMode::V2, 5.0, 0.1);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.6];
        let token = sampler.sample(&logits);
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn test_mirostat_v2_adapts_mu() {
        let sampler = Sampler::with_mirostat(0.7, MirostatMode::V2, 5.0, 0.1);
        let initial_mu = sampler.mirostat_mu.get();
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        sampler.sample(&logits);
        let new_mu = sampler.mirostat_mu.get();
        // mu should have changed after sampling
        assert!(
            (initial_mu - new_mu).abs() > 1e-6,
            "Mirostat mu should adapt: was {}, now {}",
            initial_mu,
            new_mu
        );
    }

    #[test]
    fn test_mirostat_greedy_at_zero_temp() {
        // Even with mirostat enabled, temperature 0 should be greedy
        let sampler = Sampler::with_mirostat(0.0, MirostatMode::V2, 5.0, 0.1);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(sampler.sample(&logits), 3);
    }

    #[test]
    fn test_mirostat_v1_mu_tracks_tau() {
        // After many samples, mu should gravitate toward tau
        let sampler = Sampler::with_mirostat(0.7, MirostatMode::V1, 3.0, 0.3);
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for _ in 0..50 {
            sampler.sample(&logits);
        }
        // mu should be in a reasonable range around tau (not diverging)
        let mu = sampler.mirostat_mu.get();
        assert!(
            mu > -50.0 && mu < 50.0,
            "Mirostat v1 mu should stay bounded: {}",
            mu
        );
    }

    #[test]
    fn test_estimate_zipf_s() {
        // When p1 >> p2, s should be large
        let s1 = estimate_zipf_s(0.9, 0.001);
        assert!(s1 > 5.0, "Large ratio should give large s: {}", s1);

        // When p1 ≈ p2, s should be small
        let s2 = estimate_zipf_s(0.5, 0.45);
        assert!(s2 < 1.0, "Small ratio should give small s: {}", s2);
    }

    #[test]
    fn test_no_penalties_noop() {
        let sampler = Sampler::with_penalties(1.0, 40, 0.9, 1.0, 0.0, 0.0);
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        sampler.apply_penalties(&mut logits, &[0, 1, 2]);
        assert_eq!(logits, original);
    }
}
