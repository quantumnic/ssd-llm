//! Predictive prefetcher for layer-by-layer streaming

use crate::model::gguf::GgufFile;
use crate::ssd::streamer::SsdStreamer;
use crate::model::cache::LayerCache;
use tracing::debug;

/// Prefetch strategy
#[derive(Debug, Clone)]
pub enum PrefetchStrategy {
    /// Prefetch the next N layers
    LookAhead(usize),
    /// No prefetching
    None,
}

impl Default for PrefetchStrategy {
    fn default() -> Self {
        Self::LookAhead(1)
    }
}

/// Predictive prefetcher that issues madvise hints ahead of computation
pub struct Prefetcher {
    strategy: PrefetchStrategy,
}

impl Prefetcher {
    pub fn new(strategy: PrefetchStrategy) -> Self {
        Self { strategy }
    }

    /// Called before processing layer `current_layer`.
    /// Issues prefetch for upcoming layers.
    pub fn on_layer_start(
        &self,
        current_layer: u32,
        total_layers: u32,
        streamer: &SsdStreamer,
        gguf: &GgufFile,
        cache: &LayerCache,
    ) {
        match &self.strategy {
            PrefetchStrategy::LookAhead(n) => {
                for i in 1..=(*n as u32) {
                    let next = current_layer + i;
                    if next < total_layers && !cache.contains(next) {
                        streamer.prefetch_layer(gguf, next);
                        debug!("Prefetcher: issued WILLNEED for layer {}", next);
                    }
                }
            }
            PrefetchStrategy::None => {}
        }
    }

    /// Called after processing a layer — can evict old layers
    pub fn on_layer_done(
        &self,
        completed_layer: u32,
        streamer: &SsdStreamer,
        gguf: &GgufFile,
    ) {
        // Evict layer that's 2 behind (keep 1 buffer for potential re-use)
        if completed_layer >= 2 {
            streamer.evict_layer(gguf, completed_layer - 2);
            debug!("Prefetcher: evicted layer {} from OS cache", completed_layer - 2);
        }
    }
}
