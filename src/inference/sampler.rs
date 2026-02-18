//! Token sampling strategies: Temperature, Top-K, Top-P (nucleus)

use std::cell::Cell;

pub struct Sampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
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
            rng_state: Cell::new(seed ^ 0x517cc1b727220a95),
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
}
