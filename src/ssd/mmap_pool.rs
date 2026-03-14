//! mmap pool manager with madvise hints
//!
//! This module manages multiple memory-mapped regions and provides
//! fine-grained control over OS page cache behavior using madvise.

use std::collections::HashMap;

/// Page cache hints for mmap regions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    /// Remaining bytes available before hitting the budget
    pub fn available_bytes(&self) -> u64 {
        self.budget.saturating_sub(self.active_bytes)
    }

    /// Utilization ratio (0.0–1.0). Returns 0.0 when budget is zero.
    pub fn utilization(&self) -> f64 {
        if self.budget == 0 {
            return 0.0;
        }
        self.active_bytes as f64 / self.budget as f64
    }

    /// Number of currently active regions
    pub fn region_count(&self) -> usize {
        self.active_regions.len()
    }

    /// Check whether a named region is currently active
    pub fn is_region_active(&self, name: &str) -> bool {
        self.active_regions.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activate_within_budget() {
        let mut pool = MmapPoolManager::new(1024);
        assert!(pool.activate("layer0".into(), 0, 512));
        assert_eq!(pool.active_bytes(), 512);
        assert_eq!(pool.region_count(), 1);
        assert!(pool.is_region_active("layer0"));
    }

    #[test]
    fn test_activate_exceeds_budget() {
        let mut pool = MmapPoolManager::new(1024);
        assert!(pool.activate("layer0".into(), 0, 800));
        assert!(!pool.activate("layer1".into(), 800, 300));
        assert_eq!(pool.region_count(), 1);
        assert!(!pool.is_region_active("layer1"));
    }

    #[test]
    fn test_deactivate_frees_space() {
        let mut pool = MmapPoolManager::new(1024);
        pool.activate("layer0".into(), 0, 600);
        assert_eq!(pool.available_bytes(), 424);
        pool.deactivate("layer0");
        assert_eq!(pool.available_bytes(), 1024);
        assert_eq!(pool.region_count(), 0);
    }

    #[test]
    fn test_deactivate_missing_returns_none() {
        let mut pool = MmapPoolManager::new(1024);
        assert!(pool.deactivate("nonexistent").is_none());
    }

    #[test]
    fn test_utilization() {
        let mut pool = MmapPoolManager::new(1000);
        assert!((pool.utilization() - 0.0).abs() < f64::EPSILON);
        pool.activate("a".into(), 0, 250);
        assert!((pool.utilization() - 0.25).abs() < f64::EPSILON);
        pool.activate("b".into(), 250, 750);
        assert!((pool.utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_utilization_zero_budget() {
        let pool = MmapPoolManager::new(0);
        assert!((pool.utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_available_bytes() {
        let mut pool = MmapPoolManager::new(2048);
        assert_eq!(pool.available_bytes(), 2048);
        pool.activate("r".into(), 0, 1000);
        assert_eq!(pool.available_bytes(), 1048);
    }

    #[test]
    fn test_is_region_active() {
        let mut pool = MmapPoolManager::new(4096);
        assert!(!pool.is_region_active("x"));
        pool.activate("x".into(), 0, 100);
        assert!(pool.is_region_active("x"));
        pool.deactivate("x");
        assert!(!pool.is_region_active("x"));
    }
}
