//! mmap pool manager with madvise hints
//!
//! This module manages multiple memory-mapped regions and provides
//! fine-grained control over OS page cache behavior using madvise.

use std::collections::HashMap;

/// Page cache hints for mmap regions
#[derive(Debug, Clone, Copy)]
pub enum PageHint {
    /// MADV_WILLNEED — prefetch into page cache
    WillNeed,
    /// MADV_DONTNEED — evict from page cache
    DontNeed,
    /// MADV_SEQUENTIAL — expect sequential access
    Sequential,
    /// MADV_RANDOM — expect random access
    Random,
}

/// Tracks mmap regions and their access patterns
pub struct MmapPoolManager {
    /// Track which regions are currently in-use
    active_regions: HashMap<String, (u64, u64)>, // name -> (offset, size)
    /// Total bytes currently marked as active
    active_bytes: u64,
    /// Budget limit
    budget: u64,
}

impl MmapPoolManager {
    pub fn new(budget: u64) -> Self {
        Self {
            active_regions: HashMap::new(),
            active_bytes: 0,
            budget,
        }
    }

    /// Register a region as active
    pub fn activate(&mut self, name: String, offset: u64, size: u64) -> bool {
        if self.active_bytes + size > self.budget {
            return false;
        }
        self.active_regions.insert(name, (offset, size));
        self.active_bytes += size;
        true
    }

    /// Deactivate a region
    pub fn deactivate(&mut self, name: &str) -> Option<(u64, u64)> {
        if let Some((offset, size)) = self.active_regions.remove(name) {
            self.active_bytes -= size;
            Some((offset, size))
        } else {
            None
        }
    }

    pub fn active_bytes(&self) -> u64 {
        self.active_bytes
    }

    pub fn budget(&self) -> u64 {
        self.budget
    }
}
