//! Token sampling strategies: Temperature, Top-K, Top-P (nucleus)

use std::cell::Cell;

pub struct Sampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    rng_state: Cell<u64>,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            temperature,
            top_k,
            top_p,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            rng_state: Cell::new(seed ^ 0x517cc1b727220a95),
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
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            temperature,
            top_k,
            top_p,
            repetition_penalty: repetition_penalty.max(0.01),
            frequency_penalty,
            presence_penalty,
            rng_state: Cell::new(seed ^ 0x517cc1b727220a95),
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

        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|&l| l / self.temperature).collect();

        // Top-K filtering
        let mut indexed: Vec<(usize, f32)> =
            scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.top_k.min(indexed.len());
        let candidates: Vec<(usize, f32)> = indexed[..k].to_vec();

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
        let r = self.random_f32();
        let mut cumulative = 0.0f32;
        for (idx, p) in &nucleus {
            cumulative += p;
            if r <= cumulative {
                return *idx as u32;
            }
        }

        nucleus.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
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
    fn test_no_penalties_noop() {
        let sampler = Sampler::with_penalties(1.0, 40, 0.9, 1.0, 0.0, 0.0);
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        sampler.apply_penalties(&mut logits, &[0, 1, 2]);
        assert_eq!(logits, original);
    }
}
