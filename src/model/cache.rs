//! LRU layer cache with configurable memory budget

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
}
