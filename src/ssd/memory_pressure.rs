//! Adaptive Memory Pressure Monitor for macOS
//!
//! Uses the Mach `host_statistics64` API to query real-time VM statistics
//! and detect memory pressure levels. The monitor runs on a background thread
//! and provides [`MemoryAdvice`] to the layer cache and block swap systems,
//! allowing them to dynamically adjust their budgets:
//!
//! - **Normal** — full budget, aggressive prefetch
//! - **Warning** — reduce budget to 75%, moderate prefetch
//! - **Critical** — reduce budget to 50%, disable prefetch, force evictions
//! - **Urgent** — reduce budget to 25%, emergency eviction, pause new loads
//!
//! On non-macOS platforms the monitor is a no-op that always reports Normal.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// System memory pressure level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PressureLevel {
    /// Plenty of free memory
    Normal = 0,
    /// System starting to compress/swap — shed non-essential caches
    Warning = 1,
    /// Significant pressure — aggressive eviction, reduce budgets
    Critical = 2,
    /// Extreme pressure — emergency mode, minimal footprint
    Urgent = 3,
}

impl PressureLevel {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Normal,
            1 => Self::Warning,
            2 => Self::Critical,
            3 => Self::Urgent,
            _ => Self::Normal,
        }
    }

    /// Budget multiplier: fraction of the configured budget to actually use
    pub fn budget_fraction(self) -> f64 {
        match self {
            Self::Normal => 1.0,
            Self::Warning => 0.75,
            Self::Critical => 0.50,
            Self::Urgent => 0.25,
        }
    }

    /// Whether prefetching should be active at this pressure level
    pub fn allow_prefetch(self) -> bool {
        matches!(self, Self::Normal | Self::Warning)
    }

    /// Whether new layer loads should be allowed (false = only serve from cache)
    pub fn allow_new_loads(self) -> bool {
        !matches!(self, Self::Urgent)
    }
}

impl std::fmt::Display for PressureLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::Warning => write!(f, "warning"),
            Self::Critical => write!(f, "critical"),
            Self::Urgent => write!(f, "urgent"),
        }
    }
}

/// Snapshot of system memory state
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Total physical RAM in bytes
    pub total_bytes: u64,
    /// Free (immediately reusable) pages in bytes
    pub free_bytes: u64,
    /// Active (recently referenced) pages in bytes
    pub active_bytes: u64,
    /// Inactive (not recently referenced, reclaimable) pages in bytes
    pub inactive_bytes: u64,
    /// Wired (kernel, not evictable) pages in bytes
    pub wired_bytes: u64,
    /// Compressed pages in bytes
    pub compressed_bytes: u64,
    /// Available = free + inactive (reclaimable without swapping)
    pub available_bytes: u64,
    /// Computed pressure level
    pub pressure: PressureLevel,
    /// Timestamp of this measurement
    pub timestamp: Instant,
}

/// Shared pressure state readable from any thread without locking
#[derive(Clone)]
pub struct PressureState {
    level: Arc<AtomicU8>,
}

impl PressureState {
    fn new() -> Self {
        Self {
            level: Arc::new(AtomicU8::new(PressureLevel::Normal as u8)),
        }
    }

    /// Get the current pressure level (lock-free)
    pub fn current(&self) -> PressureLevel {
        PressureLevel::from_u8(self.level.load(Ordering::Relaxed))
    }

    fn set(&self, level: PressureLevel) {
        self.level.store(level as u8, Ordering::Relaxed);
    }
}

/// Configuration for the memory pressure monitor
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// How often to poll VM statistics (default: 500ms)
    pub poll_interval: Duration,
    /// Available-memory fraction below which we enter Warning (default: 0.20 = 20%)
    pub warning_threshold: f64,
    /// Available-memory fraction below which we enter Critical (default: 0.10 = 10%)
    pub critical_threshold: f64,
    /// Available-memory fraction below which we enter Urgent (default: 0.05 = 5%)
    pub urgent_threshold: f64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(500),
            warning_threshold: 0.20,
            critical_threshold: 0.10,
            urgent_threshold: 0.05,
        }
    }
}

/// Memory pressure monitor handle
pub struct MemoryPressureMonitor {
    state: PressureState,
    _config: MonitorConfig,
}

impl MemoryPressureMonitor {
    /// Create a new monitor and start its background polling thread.
    /// Returns the monitor handle with a shared [`PressureState`].
    pub fn start(config: MonitorConfig) -> Self {
        let state = PressureState::new();
        let bg_state = state.clone();
        let bg_config = config.clone();

        std::thread::Builder::new()
            .name("mem-pressure".into())
            .spawn(move || {
                Self::poll_loop(bg_state, bg_config);
            })
            .expect("failed to spawn memory pressure monitor thread");

        info!(
            "Memory pressure monitor started (poll every {:?})",
            config.poll_interval
        );

        Self {
            state,
            _config: config,
        }
    }

    /// Start with default config
    pub fn start_default() -> Self {
        Self::start(MonitorConfig::default())
    }

    /// Get shared pressure state (cheap clone, lock-free reads)
    pub fn pressure_state(&self) -> PressureState {
        self.state.clone()
    }

    /// Take a single memory snapshot right now
    pub fn snapshot() -> Option<MemorySnapshot> {
        query_vm_stats()
    }

    fn poll_loop(state: PressureState, config: MonitorConfig) {
        let mut prev_level = PressureLevel::Normal;

        loop {
            if let Some(snap) = query_vm_stats() {
                let level = snap.pressure;
                if level != prev_level {
                    match level {
                        PressureLevel::Normal => {
                            info!(
                                "Memory pressure → normal (available: {} MB)",
                                snap.available_bytes / (1024 * 1024)
                            );
                        }
                        PressureLevel::Warning => {
                            warn!(
                                "Memory pressure → WARNING (available: {} MB)",
                                snap.available_bytes / (1024 * 1024)
                            );
                        }
                        PressureLevel::Critical => {
                            warn!(
                                "Memory pressure → CRITICAL (available: {} MB)",
                                snap.available_bytes / (1024 * 1024)
                            );
                        }
                        PressureLevel::Urgent => {
                            warn!(
                                "Memory pressure → URGENT (available: {} MB) — emergency mode",
                                snap.available_bytes / (1024 * 1024)
                            );
                        }
                    }
                    prev_level = level;
                }
                state.set(level);
            }

            std::thread::sleep(config.poll_interval);
        }
    }
}

/// Query macOS VM statistics via `host_statistics64`
#[cfg(target_os = "macos")]
fn query_vm_stats() -> Option<MemorySnapshot> {
    use std::mem;

    // Mach types
    type MachPort = u32;
    type KernReturn = i32;

    #[repr(C)]
    #[derive(Default)]
    struct VmStatistics64 {
        free_count: u64,
        active_count: u64,
        inactive_count: u64,
        wire_count: u64,
        zero_fill_count: u64,
        reactivations: u64,
        pageins: u64,
        pageouts: u64,
        faults: u64,
        cow_faults: u64,
        lookups: u64,
        hits: u64,
        purges: u64,
        purgeable_count: u64,
        speculative_count: u64,
        decompressions: u64,
        compressions: u64,
        swapins: u64,
        swapouts: u64,
        compressor_page_count: u64,
        throttled_count: u64,
        external_page_count: u64,
        internal_page_count: u64,
        total_uncompressed_pages_in_compressor: u64,
    }

    const HOST_VM_INFO64: i32 = 4;
    const KERN_SUCCESS: KernReturn = 0;

    extern "C" {
        fn mach_host_self() -> MachPort;
        fn host_statistics64(
            host: MachPort,
            flavor: i32,
            info: *mut VmStatistics64,
            count: *mut u32,
        ) -> KernReturn;
        fn host_page_size(host: MachPort, page_size: *mut u64) -> KernReturn;
    }

    unsafe {
        let host = mach_host_self();

        let mut page_size: u64 = 0;
        if host_page_size(host, &mut page_size) != KERN_SUCCESS {
            debug!("host_page_size failed");
            return None;
        }

        let mut vm_stat = VmStatistics64::default();
        let mut count = (mem::size_of::<VmStatistics64>() / mem::size_of::<u32>()) as u32;

        if host_statistics64(host, HOST_VM_INFO64, &mut vm_stat, &mut count) != KERN_SUCCESS {
            debug!("host_statistics64 failed");
            return None;
        }

        let free = vm_stat.free_count.saturating_mul(page_size);
        let active = vm_stat.active_count.saturating_mul(page_size);
        let inactive = vm_stat.inactive_count.saturating_mul(page_size);
        let wired = vm_stat.wire_count.saturating_mul(page_size);
        let compressed = vm_stat.compressor_page_count.saturating_mul(page_size);

        let total = free
            .saturating_add(active)
            .saturating_add(inactive)
            .saturating_add(wired)
            .saturating_add(compressed);
        let available = free.saturating_add(inactive);

        let ratio = if total > 0 {
            available as f64 / total as f64
        } else {
            1.0
        };

        // Use default thresholds for classification
        let pressure = if ratio < 0.05 {
            PressureLevel::Urgent
        } else if ratio < 0.10 {
            PressureLevel::Critical
        } else if ratio < 0.20 {
            PressureLevel::Warning
        } else {
            PressureLevel::Normal
        };

        Some(MemorySnapshot {
            total_bytes: total,
            free_bytes: free,
            active_bytes: active,
            inactive_bytes: inactive,
            wired_bytes: wired,
            compressed_bytes: compressed,
            available_bytes: available,
            pressure,
            timestamp: Instant::now(),
        })
    }
}

/// Fallback for non-macOS: always reports Normal
#[cfg(not(target_os = "macos"))]
fn query_vm_stats() -> Option<MemorySnapshot> {
    Some(MemorySnapshot {
        total_bytes: 0,
        free_bytes: 0,
        active_bytes: 0,
        inactive_bytes: 0,
        wired_bytes: 0,
        compressed_bytes: 0,
        available_bytes: 0,
        pressure: PressureLevel::Normal,
        timestamp: Instant::now(),
    })
}

/// Adaptive layer cache that wraps [`super::super::model::cache::LayerCache`]
/// and adjusts its effective budget based on memory pressure.
pub struct AdaptiveCache {
    /// The configured (maximum) budget
    base_budget: usize,
    /// Current effective budget
    effective_budget: usize,
    /// Shared pressure state
    pressure: PressureState,
    /// Timestamp of last budget adjustment
    last_adjust: Instant,
}

impl AdaptiveCache {
    /// Create a new adaptive cache wrapper
    pub fn new(base_budget: usize, pressure: PressureState) -> Self {
        Self {
            base_budget,
            effective_budget: base_budget,
            pressure,
            last_adjust: Instant::now(),
        }
    }

    /// Recompute effective budget based on current pressure.
    /// Returns the new budget in bytes.
    pub fn update_budget(&mut self) -> usize {
        let level = self.pressure.current();
        let new_budget = (self.base_budget as f64 * level.budget_fraction()) as usize;

        if new_budget != self.effective_budget {
            debug!(
                "Adaptive cache budget: {} → {} MB (pressure: {})",
                self.effective_budget / (1024 * 1024),
                new_budget / (1024 * 1024),
                level,
            );
            self.effective_budget = new_budget;
            self.last_adjust = Instant::now();
        }

        self.effective_budget
    }

    /// Get current effective budget without recalculating
    pub fn effective_budget(&self) -> usize {
        self.effective_budget
    }

    /// Get base (configured) budget
    pub fn base_budget(&self) -> usize {
        self.base_budget
    }

    /// Whether prefetching is advisable right now
    pub fn should_prefetch(&self) -> bool {
        self.pressure.current().allow_prefetch()
    }

    /// Whether new layer loads are allowed
    pub fn allow_new_loads(&self) -> bool {
        self.pressure.current().allow_new_loads()
    }

    /// Current pressure level
    pub fn pressure_level(&self) -> PressureLevel {
        self.pressure.current()
    }

    /// Time since last budget adjustment
    pub fn time_since_adjust(&self) -> Duration {
        self.last_adjust.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_level_from_u8() {
        assert_eq!(PressureLevel::from_u8(0), PressureLevel::Normal);
        assert_eq!(PressureLevel::from_u8(1), PressureLevel::Warning);
        assert_eq!(PressureLevel::from_u8(2), PressureLevel::Critical);
        assert_eq!(PressureLevel::from_u8(3), PressureLevel::Urgent);
        assert_eq!(PressureLevel::from_u8(255), PressureLevel::Normal);
    }

    #[test]
    fn test_budget_fractions() {
        assert_eq!(PressureLevel::Normal.budget_fraction(), 1.0);
        assert_eq!(PressureLevel::Warning.budget_fraction(), 0.75);
        assert_eq!(PressureLevel::Critical.budget_fraction(), 0.50);
        assert_eq!(PressureLevel::Urgent.budget_fraction(), 0.25);
    }

    #[test]
    fn test_prefetch_policy() {
        assert!(PressureLevel::Normal.allow_prefetch());
        assert!(PressureLevel::Warning.allow_prefetch());
        assert!(!PressureLevel::Critical.allow_prefetch());
        assert!(!PressureLevel::Urgent.allow_prefetch());
    }

    #[test]
    fn test_new_loads_policy() {
        assert!(PressureLevel::Normal.allow_new_loads());
        assert!(PressureLevel::Warning.allow_new_loads());
        assert!(PressureLevel::Critical.allow_new_loads());
        assert!(!PressureLevel::Urgent.allow_new_loads());
    }

    #[test]
    fn test_pressure_state_atomic() {
        let state = PressureState::new();
        assert_eq!(state.current(), PressureLevel::Normal);

        state.set(PressureLevel::Critical);
        assert_eq!(state.current(), PressureLevel::Critical);

        state.set(PressureLevel::Normal);
        assert_eq!(state.current(), PressureLevel::Normal);
    }

    #[test]
    fn test_pressure_display() {
        assert_eq!(format!("{}", PressureLevel::Normal), "normal");
        assert_eq!(format!("{}", PressureLevel::Warning), "warning");
        assert_eq!(format!("{}", PressureLevel::Critical), "critical");
        assert_eq!(format!("{}", PressureLevel::Urgent), "urgent");
    }

    #[test]
    fn test_adaptive_cache_budget_adjustment() {
        let state = PressureState::new();
        let mut cache = AdaptiveCache::new(8 * 1024 * 1024 * 1024, state.clone());

        // Normal: full budget
        assert_eq!(cache.update_budget(), 8 * 1024 * 1024 * 1024);

        // Warning: 75%
        state.set(PressureLevel::Warning);
        assert_eq!(cache.update_budget(), 6 * 1024 * 1024 * 1024);

        // Critical: 50%
        state.set(PressureLevel::Critical);
        assert_eq!(cache.update_budget(), 4 * 1024 * 1024 * 1024);

        // Urgent: 25%
        state.set(PressureLevel::Urgent);
        assert_eq!(cache.update_budget(), 2 * 1024 * 1024 * 1024);

        // Back to normal
        state.set(PressureLevel::Normal);
        assert_eq!(cache.update_budget(), 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_adaptive_cache_policies() {
        let state = PressureState::new();
        let cache = AdaptiveCache::new(1024, state.clone());

        assert!(cache.should_prefetch());
        assert!(cache.allow_new_loads());

        state.set(PressureLevel::Critical);
        assert!(!cache.should_prefetch());
        assert!(cache.allow_new_loads());

        state.set(PressureLevel::Urgent);
        assert!(!cache.should_prefetch());
        assert!(!cache.allow_new_loads());
    }

    #[test]
    fn test_snapshot_available() {
        // On macOS this will return real stats, on other platforms a zero stub
        let snap = MemoryPressureMonitor::snapshot();
        assert!(snap.is_some());

        let snap = snap.unwrap();
        #[cfg(target_os = "macos")]
        {
            // On macOS, total should be > 0
            assert!(snap.total_bytes > 0, "total_bytes should be > 0 on macOS");
            assert!(
                snap.total_bytes > 1_000_000_000,
                "total should be at least 1GB, got {} bytes",
                snap.total_bytes
            );
        }
        #[cfg(not(target_os = "macos"))]
        {
            assert_eq!(snap.pressure, PressureLevel::Normal);
        }
    }

    #[test]
    fn test_monitor_config_defaults() {
        let config = MonitorConfig::default();
        assert_eq!(config.poll_interval, Duration::from_millis(500));
        assert_eq!(config.warning_threshold, 0.20);
        assert_eq!(config.critical_threshold, 0.10);
        assert_eq!(config.urgent_threshold, 0.05);
    }
}
