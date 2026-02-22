//! LRU layer cache with configurable memory budget and adaptive layer pinning

use std::collections::{HashMap, VecDeque};
use tracing::debug;

#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub prefetch_hits: u64,
}

/// Cached tensor data for a layer
#[derive(Clone)]
pub struct CachedLayer {
    pub layer_idx: u32,
    pub tensors: HashMap<String, Vec<f32>>,
    pub size_bytes: usize,
}

/// LRU cache for transformer layers with a memory budget
pub struct LayerCache {
    budget: usize,
    used: usize,
    /// LRU order: front = least recently used, back = most recently used
    lru_order: VecDeque<u32>,
    layers: HashMap<u32, CachedLayer>,
    /// Layers that have been prefetched
    prefetched: HashMap<u32, CachedLayer>,
    stats: CacheStats,
    /// Layers that should be pinned (never evicted)
    pinned: Vec<u32>,
}

impl LayerCache {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            used: 0,
            lru_order: VecDeque::new(),
            layers: HashMap::new(),
            prefetched: HashMap::new(),
            stats: CacheStats::default(),
            pinned: Vec::new(),
        }
    }

    /// Get a layer from cache, returns None on miss
    pub fn get(&mut self, layer_idx: u32) -> Option<&CachedLayer> {
        // Check prefetch buffer first
        if let Some(layer) = self.prefetched.remove(&layer_idx) {
            self.stats.prefetch_hits += 1;
            self.insert_internal(layer_idx, layer);
        }

        if self.layers.contains_key(&layer_idx) {
            self.stats.hits += 1;
            // Move to back of LRU
            self.lru_order.retain(|&x| x != layer_idx);
            self.lru_order.push_back(layer_idx);
            self.layers.get(&layer_idx)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a layer into cache, evicting LRU entries if needed
    pub fn insert(&mut self, layer_idx: u32, layer: CachedLayer) {
        self.insert_internal(layer_idx, layer);
    }

    fn insert_internal(&mut self, layer_idx: u32, layer: CachedLayer) {
        let size = layer.size_bytes;

        // Remove if already present
        if let Some(existing) = self.layers.remove(&layer_idx) {
            self.used -= existing.size_bytes;
            self.lru_order.retain(|&x| x != layer_idx);
        }

        // Evict until we have space
        while self.used + size > self.budget && !self.lru_order.is_empty() {
            if let Some(evict_idx) = self.find_evictable() {
                self.evict(evict_idx);
            } else {
                break; // all pinned, can't evict
            }
        }

        self.layers.insert(layer_idx, layer);
        self.lru_order.push_back(layer_idx);
        self.used += size;
        debug!(
            "Cache insert layer {} ({} bytes), used: {}/{}",
            layer_idx, size, self.used, self.budget
        );
    }

    /// Insert into prefetch buffer (doesn't count toward main budget until accessed)
    pub fn insert_prefetch(&mut self, layer_idx: u32, layer: CachedLayer) {
        self.prefetched.insert(layer_idx, layer);
    }

    /// Pin a layer so it's never evicted
    pub fn pin(&mut self, layer_idx: u32) {
        if !self.pinned.contains(&layer_idx) {
            self.pinned.push(layer_idx);
        }
    }

    fn find_evictable(&self) -> Option<u32> {
        self.lru_order
            .iter()
            .find(|idx| !self.pinned.contains(idx))
            .copied()
    }

    fn evict(&mut self, layer_idx: u32) {
        if let Some(layer) = self.layers.remove(&layer_idx) {
            self.used -= layer.size_bytes;
            self.lru_order.retain(|&x| x != layer_idx);
            self.stats.evictions += 1;
            debug!("Cache evict layer {}", layer_idx);
        }
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn used_bytes(&self) -> usize {
        self.used
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    pub fn contains(&self, layer_idx: u32) -> bool {
        self.layers.contains_key(&layer_idx) || self.prefetched.contains_key(&layer_idx)
    }

    /// Set the memory budget (used by adaptive memory pressure)
    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
    }

    /// Get list of pinned layers
    pub fn pinned(&self) -> &[u32] {
        &self.pinned
    }

    /// Unpin a layer, allowing it to be evicted
    pub fn unpin(&mut self, layer_idx: u32) {
        self.pinned.retain(|&x| x != layer_idx);
    }
}

/// Adaptive layer pinning: tracks per-layer access frequency and automatically
/// pins the hottest layers in RAM. Designed for transformer inference where
/// embeddings and early/late attention layers are accessed more frequently.
pub struct AdaptiveLayerPinner {
    /// Per-layer access count
    access_counts: HashMap<u32, u64>,
    /// Total accesses across all layers
    total_accesses: u64,
    /// How many layers to auto-pin (0 = disabled)
    max_pinned: usize,
    /// Minimum access count before a layer is eligible for pinning
    min_accesses: u64,
    /// Decay factor applied periodically to prevent stale pinning (0.0..1.0)
    decay_factor: f64,
    /// Accesses since last decay
    accesses_since_decay: u64,
    /// Decay every N accesses
    decay_interval: u64,
}

impl AdaptiveLayerPinner {
    /// Create a new adaptive pinner
    ///
    /// - `max_pinned`: maximum number of layers to auto-pin (e.g., 4-8)
    /// - `min_accesses`: minimum hits before eligible (e.g., 10)
    pub fn new(max_pinned: usize, min_accesses: u64) -> Self {
        Self {
            access_counts: HashMap::new(),
            total_accesses: 0,
            max_pinned,
            min_accesses,
            decay_factor: 0.9,
            accesses_since_decay: 0,
            decay_interval: 500,
        }
    }

    /// Record an access to a layer
    pub fn record_access(&mut self, layer_idx: u32) {
        *self.access_counts.entry(layer_idx).or_insert(0) += 1;
        self.total_accesses += 1;
        self.accesses_since_decay += 1;

        // Periodic decay to adapt to changing access patterns
        if self.accesses_since_decay >= self.decay_interval {
            self.decay();
            self.accesses_since_decay = 0;
        }
    }

    /// Apply exponential decay to all access counts
    fn decay(&mut self) {
        for count in self.access_counts.values_mut() {
            *count = (*count as f64 * self.decay_factor) as u64;
        }
        // Remove layers that decayed to zero
        self.access_counts.retain(|_, v| *v > 0);
    }

    /// Get the top-N hottest layers that should be pinned
    pub fn recommended_pins(&self) -> Vec<u32> {
        if self.max_pinned == 0 {
            return Vec::new();
        }

        let mut candidates: Vec<(u32, u64)> = self
            .access_counts
            .iter()
            .filter(|(_, &count)| count >= self.min_accesses)
            .map(|(&idx, &count)| (idx, count))
            .collect();

        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        candidates
            .into_iter()
            .take(self.max_pinned)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Apply recommended pins to a cache, unpinning layers that are no longer hot
    pub fn apply_to_cache(&self, cache: &mut LayerCache) {
        let recommended = self.recommended_pins();

        // Unpin layers no longer recommended
        let current_pinned: Vec<u32> = cache.pinned().to_vec();
        for &idx in &current_pinned {
            if !recommended.contains(&idx) {
                cache.unpin(idx);
                debug!("Adaptive unpin layer {}", idx);
            }
        }

        // Pin newly recommended layers
        for &idx in &recommended {
            if !current_pinned.contains(&idx) {
                cache.pin(idx);
                debug!("Adaptive pin layer {} (hot)", idx);
            }
        }
    }

    /// Get access count for a specific layer
    pub fn access_count(&self, layer_idx: u32) -> u64 {
        self.access_counts.get(&layer_idx).copied().unwrap_or(0)
    }

    /// Total recorded accesses
    pub fn total_accesses(&self) -> u64 {
        self.total_accesses
    }

    /// Number of tracked layers
    pub fn tracked_layers(&self) -> usize {
        self.access_counts.len()
    }

    /// Get a snapshot of all access counts (for metrics/debugging)
    pub fn access_snapshot(&self) -> Vec<(u32, u64)> {
        let mut snapshot: Vec<(u32, u64)> =
            self.access_counts.iter().map(|(&k, &v)| (k, v)).collect();
        snapshot.sort_by_key(|(idx, _)| *idx);
        snapshot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_pinner_basic() {
        let mut pinner = AdaptiveLayerPinner::new(2, 5);

        // Access layer 0 many times (embeddings)
        for _ in 0..20 {
            pinner.record_access(0);
        }
        // Access layer 31 many times (final layer)
        for _ in 0..15 {
            pinner.record_access(31);
        }
        // Access middle layers less
        for _ in 0..3 {
            pinner.record_access(16);
        }

        let pins = pinner.recommended_pins();
        assert_eq!(pins.len(), 2);
        assert_eq!(pins[0], 0); // highest access
        assert_eq!(pins[1], 31); // second highest
    }

    #[test]
    fn test_adaptive_pinner_min_accesses() {
        let mut pinner = AdaptiveLayerPinner::new(4, 10);

        // All layers below threshold
        for i in 0..5 {
            for _ in 0..5 {
                pinner.record_access(i);
            }
        }

        let pins = pinner.recommended_pins();
        assert!(
            pins.is_empty(),
            "No layers should meet min_accesses threshold"
        );
    }

    #[test]
    fn test_adaptive_pinner_apply_to_cache() {
        let mut cache = LayerCache::new(1_000_000);
        let mut pinner = AdaptiveLayerPinner::new(2, 5);

        // Simulate access pattern
        for _ in 0..20 {
            pinner.record_access(0);
        }
        for _ in 0..15 {
            pinner.record_access(5);
        }

        pinner.apply_to_cache(&mut cache);
        assert!(cache.pinned().contains(&0));
        assert!(cache.pinned().contains(&5));

        // Now layer 5 cools down and layer 10 heats up
        // (We won't trigger decay in this small test, just override)
        for _ in 0..30 {
            pinner.record_access(10);
        }

        pinner.apply_to_cache(&mut cache);
        assert!(cache.pinned().contains(&0));
        assert!(cache.pinned().contains(&10));
    }

    #[test]
    fn test_adaptive_pinner_decay() {
        let mut pinner = AdaptiveLayerPinner::new(2, 1);
        pinner.decay_interval = 10; // decay every 10 accesses for testing

        // Layer 0 gets 5 accesses
        for _ in 0..5 {
            pinner.record_access(0);
        }
        assert_eq!(pinner.access_count(0), 5);

        // Trigger decay by doing 10 more accesses on layer 1
        for _ in 0..10 {
            pinner.record_access(1);
        }

        // Layer 0 should have decayed: 5 * 0.9 = 4
        assert_eq!(pinner.access_count(0), 4);
    }

    #[test]
    fn test_adaptive_pinner_disabled() {
        let mut pinner = AdaptiveLayerPinner::new(0, 1);
        for _ in 0..100 {
            pinner.record_access(0);
        }
        assert!(pinner.recommended_pins().is_empty());
    }

    #[test]
    fn test_cache_unpin() {
        let mut cache = LayerCache::new(1_000_000);
        cache.pin(5);
        cache.pin(10);
        assert_eq!(cache.pinned().len(), 2);

        cache.unpin(5);
        assert_eq!(cache.pinned().len(), 1);
        assert!(!cache.pinned().contains(&5));
        assert!(cache.pinned().contains(&10));
    }

    #[test]
    fn test_access_snapshot() {
        let mut pinner = AdaptiveLayerPinner::new(4, 1);
        pinner.record_access(3);
        pinner.record_access(3);
        pinner.record_access(1);

        let snapshot = pinner.access_snapshot();
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot[0], (1, 1));
        assert_eq!(snapshot[1], (3, 2));
    }
}
