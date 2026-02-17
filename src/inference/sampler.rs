//! Token sampling strategies: Temperature, Top-K, Top-P (nucleus)

pub struct Sampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng_state: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32) -> Self {
        // Simple seed from system time
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            temperature,
            top_k,
            top_p,
            rng_state: seed,
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
        let mut indexed: Vec<(usize, f32)> = scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.top_k.min(indexed.len());
        let mut candidates: Vec<(usize, f32)> = indexed[..k].to_vec();

        // Softmax over candidates
        let max_val = candidates.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = candidates
            .iter()
            .map(|(idx, v)| (*idx, (v - max_val).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= sum;
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
        for (_, p) in nucleus.iter_mut() {
            *p /= nuc_sum;
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

        // Fallback
        nucleus.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
    }

    /// Simple xorshift64 PRNG returning f32 in [0, 1)
    fn random_f32(&self) -> f32 {
        // Use a simple hash of the state for sampling
        let mut state = self.rng_state;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f32) / (u64::MAX as f32)
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
