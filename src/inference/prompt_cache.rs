//! Prompt prefix caching for KV state reuse
//!
//! When multiple requests share a common prompt prefix (e.g., system prompt),
//! the KV cache from prefill can be reused, skipping expensive SSD reads.
//! Uses a hash-trie structure keyed on token sequences.

use crate::inference::kv_cache::KvCache;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

/// A cached KV state for a specific token prefix
#[derive(Clone)]
struct CachedKvState {
    /// The token sequence this cache covers
    tokens: Vec<u32>,
    /// Per-layer key caches: [layer][position][kv_dim]
    keys: Vec<Vec<Vec<f32>>>,
    /// Per-layer value caches: [layer][position][kv_dim]
    values: Vec<Vec<Vec<f32>>>,
    /// The hidden state after processing these tokens
    hidden_state: Vec<f32>,
    /// Last access time for eviction
    last_access: Instant,
    /// Size in bytes
    size_bytes: usize,
}

/// Prompt cache with LRU eviction
pub struct PromptCache {
    /// Cache entries keyed by hash of token prefix
    entries: HashMap<u64, CachedKvState>,
    /// Maximum cache size in bytes
    max_bytes: usize,
    /// Current cache size
    used_bytes: usize,
    /// Stats
    pub hits: u64,
    pub misses: u64,
    pub partial_hits: u64,
}

impl PromptCache {
    pub fn new(max_bytes: usize) -> Self {
        info!(
            "Prompt cache initialized: {:.1} MB budget",
            max_bytes as f64 / (1024.0 * 1024.0)
        );
        Self {
            entries: HashMap::new(),
            max_bytes,
            used_bytes: 0,
            hits: 0,
            misses: 0,
            partial_hits: 0,
        }
    }

    /// Try to find a cached KV state matching a token prefix.
    /// Returns (matched_length, hidden_state) and restores the KV cache.
    pub fn lookup(&mut self, tokens: &[u32], kv_cache: &mut KvCache) -> Option<(usize, Vec<f32>)> {
        // Try exact match first
        let hash = hash_tokens(tokens);
        if let Some(entry) = self.entries.get_mut(&hash) {
            if entry.tokens == tokens {
                entry.last_access = Instant::now();
                restore_kv_cache(kv_cache, &entry.keys, &entry.values);
                self.hits += 1;
                info!("Prompt cache: exact hit for {} tokens", tokens.len());
                return Some((tokens.len(), entry.hidden_state.clone()));
            }
        }

        // Try progressively shorter prefixes
        let mut best_match: Option<(usize, &CachedKvState)> = None;
        for len in (1..tokens.len()).rev() {
            let prefix = &tokens[..len];
            let prefix_hash = hash_tokens(prefix);
            if let Some(entry) = self.entries.get(&prefix_hash) {
                if entry.tokens == prefix {
                    best_match = Some((len, entry));
                    break;
                }
            }
        }

        if let Some((len, _entry)) = best_match {
            // Update last_access via hash lookup
            let prefix_hash = hash_tokens(&tokens[..len]);
            if let Some(entry) = self.entries.get_mut(&prefix_hash) {
                entry.last_access = Instant::now();
                restore_kv_cache(kv_cache, &entry.keys, &entry.values);
                self.partial_hits += 1;
                info!(
                    "Prompt cache: partial hit — {} of {} tokens cached",
                    len,
                    tokens.len()
                );
                return Some((len, entry.hidden_state.clone()));
            }
        }

        self.misses += 1;
        debug!("Prompt cache miss for {} tokens", tokens.len());
        None
    }

    /// Store a KV state for a token prefix
    pub fn store(&mut self, tokens: &[u32], kv_cache: &KvCache, hidden_state: &[f32]) {
        let hash = hash_tokens(tokens);

        // Calculate size
        let n_layers = kv_cache.n_layers();
        let mut keys = Vec::with_capacity(n_layers);
        let mut values = Vec::with_capacity(n_layers);
        let mut size_bytes = tokens.len() * 4 + hidden_state.len() * 4; // tokens + hidden

        for i in 0..n_layers {
            let layer = kv_cache.layer(i);
            keys.push(layer.keys.clone());
            values.push(layer.values.clone());
            size_bytes += layer.size_bytes();
        }

        // Evict old entries if needed
        while self.used_bytes + size_bytes > self.max_bytes && !self.entries.is_empty() {
            self.evict_oldest();
        }

        if size_bytes > self.max_bytes {
            debug!(
                "Prompt cache: entry too large ({:.1} MB), skipping",
                size_bytes as f64 / (1024.0 * 1024.0)
            );
            return;
        }

        self.entries.insert(
            hash,
            CachedKvState {
                tokens: tokens.to_vec(),
                keys,
                values,
                hidden_state: hidden_state.to_vec(),
                last_access: Instant::now(),
                size_bytes,
            },
        );
        self.used_bytes += size_bytes;

        debug!(
            "Prompt cache: stored {} tokens ({:.1} MB), total: {:.1}/{:.1} MB",
            tokens.len(),
            size_bytes as f64 / (1024.0 * 1024.0),
            self.used_bytes as f64 / (1024.0 * 1024.0),
            self.max_bytes as f64 / (1024.0 * 1024.0),
        );
    }

    fn evict_oldest(&mut self) {
        let oldest = self
            .entries
            .iter()
            .min_by_key(|(_, v)| v.last_access)
            .map(|(k, v)| (*k, v.size_bytes));

        if let Some((key, size)) = oldest {
            self.entries.remove(&key);
            self.used_bytes -= size;
            debug!(
                "Prompt cache: evicted entry ({:.1} MB)",
                size as f64 / (1024.0 * 1024.0)
            );
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }
}

/// Restore KV cache from stored state
fn restore_kv_cache(kv_cache: &mut KvCache, keys: &[Vec<Vec<f32>>], values: &[Vec<Vec<f32>>]) {
    kv_cache.clear();
    for (layer_idx, (k, v)) in keys.iter().zip(values.iter()).enumerate() {
        let layer = kv_cache.layer_mut(layer_idx);
        for (ki, vi) in k.iter().zip(v.iter()) {
            layer.append(ki.clone(), vi.clone());
        }
    }
}

/// FNV-1a hash for token sequences
fn hash_tokens(tokens: &[u32]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &t in tokens {
        let bytes = t.to_le_bytes();
        for &b in &bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::KvCache;

    #[test]
    fn test_hash_consistency() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        assert_eq!(hash_tokens(&tokens), hash_tokens(&tokens));
        assert_ne!(hash_tokens(&tokens), hash_tokens(&[1, 2, 3, 4, 6]));
    }

    #[test]
    fn test_prompt_cache_store_and_lookup() {
        let mut cache = PromptCache::new(100 * 1024 * 1024); // 100MB
        let tokens = vec![1u32, 2, 3];
        let hidden = vec![0.5f32; 64];

        let mut kv = KvCache::new(2, 4, 16, 512);
        // Add some data to KV cache
        kv.layer_mut(0).append(vec![1.0; 64], vec![2.0; 64]);
        kv.layer_mut(0).append(vec![1.0; 64], vec![2.0; 64]);
        kv.layer_mut(1).append(vec![3.0; 64], vec![4.0; 64]);
        kv.layer_mut(1).append(vec![3.0; 64], vec![4.0; 64]);

        cache.store(&tokens, &kv, &hidden);

        // Lookup exact match
        let mut kv2 = KvCache::new(2, 4, 16, 512);
        let result = cache.lookup(&tokens, &mut kv2);
        assert!(result.is_some());
        let (len, h) = result.unwrap();
        assert_eq!(len, 3);
        assert_eq!(h, hidden);
        assert_eq!(kv2.layer(0).seq_len(), 2);
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn test_prompt_cache_partial_hit() {
        let mut cache = PromptCache::new(100 * 1024 * 1024);
        let prefix = vec![1u32, 2, 3];
        let full = vec![1u32, 2, 3, 4, 5];
        let hidden = vec![0.5f32; 64];

        let kv = KvCache::new(1, 2, 8, 512);
        cache.store(&prefix, &kv, &hidden);

        let mut kv2 = KvCache::new(1, 2, 8, 512);
        let result = cache.lookup(&full, &mut kv2);
        assert!(result.is_some());
        let (len, _) = result.unwrap();
        assert_eq!(len, 3); // matched prefix only
        assert_eq!(cache.partial_hits, 1);
    }

    #[test]
    fn test_prompt_cache_eviction() {
        let mut cache = PromptCache::new(1024); // tiny budget
        let tokens1 = vec![1u32, 2];
        let tokens2 = vec![3u32, 4];
        let hidden = vec![0.0f32; 64];

        let kv = KvCache::new(1, 1, 4, 16);
        cache.store(&tokens1, &kv, &hidden);
        cache.store(&tokens2, &kv, &hidden);
        // At least one should have been evicted or stored
        assert!(cache.len() >= 1);
    }
}
