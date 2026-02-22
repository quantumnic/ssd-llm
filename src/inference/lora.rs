//! LoRA (Low-Rank Adaptation) adapter support
//!
//! Loads LoRA adapters from GGUF files and applies them to base model weights
//! at inference time. Supports multiple simultaneous adapters with configurable
//! scaling factors.
//!
//! LoRA decomposition: W' = W + (alpha/r) * B @ A
//! where A is (r × d_in) and B is (d_out × r)

use crate::model::gguf::GgufFile;
use crate::ssd::streamer::SsdStreamer;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// A single LoRA adapter loaded from a GGUF file
pub struct LoraAdapter {
    /// Path to the adapter GGUF file
    pub path: PathBuf,
    /// LoRA rank (r)
    pub rank: usize,
    /// LoRA alpha scaling factor
    pub alpha: f32,
    /// User-specified scaling multiplier (default 1.0)
    pub scale: f32,
    /// Loaded A and B matrices per tensor name
    /// Key: base tensor name (e.g., "blk.0.attn_q.weight")
    /// Value: (A matrix [r × d_in], B matrix [d_out × r])
    pub adapters: HashMap<String, LoraWeight>,
}

/// A pair of LoRA low-rank matrices for a single weight
pub struct LoraWeight {
    /// A matrix: shape [r, d_in], stored row-major
    pub lora_a: Vec<f32>,
    /// B matrix: shape [d_out, r], stored row-major
    pub lora_b: Vec<f32>,
    /// Input dimension
    pub d_in: usize,
    /// Output dimension
    pub d_out: usize,
    /// Rank
    pub rank: usize,
}

impl LoraWeight {
    /// Compute the LoRA delta: (alpha/r) * scale * B @ A
    /// Returns a vector of size d_out × d_in (row-major)
    pub fn compute_delta(&self, alpha: f32, scale: f32) -> Vec<f32> {
        let factor = (alpha / self.rank as f32) * scale;
        let mut delta = vec![0.0f32; self.d_out * self.d_in];

        // B @ A: [d_out, r] @ [r, d_in] = [d_out, d_in]
        for row in 0..self.d_out {
            for col in 0..self.d_in {
                let mut sum = 0.0f32;
                for k in 0..self.rank {
                    sum += self.lora_b[row * self.rank + k] * self.lora_a[k * self.d_in + col];
                }
                delta[row * self.d_in + col] = sum * factor;
            }
        }

        delta
    }

    /// Apply the LoRA delta directly to a weight vector in-place
    /// weight: [d_out × d_in] row-major, modified in-place
    pub fn apply_to_weight(&self, weight: &mut [f32], alpha: f32, scale: f32) {
        let factor = (alpha / self.rank as f32) * scale;

        // B @ A added to weight: W' = W + factor * B @ A
        for row in 0..self.d_out {
            for col in 0..self.d_in {
                let mut sum = 0.0f32;
                for k in 0..self.rank {
                    sum += self.lora_b[row * self.rank + k] * self.lora_a[k * self.d_in + col];
                }
                let idx = row * self.d_in + col;
                if idx < weight.len() {
                    weight[idx] += sum * factor;
                }
            }
        }
    }
}

impl LoraAdapter {
    /// Load a LoRA adapter from a GGUF file
    pub fn load(path: &Path, scale: f32) -> Result<Self> {
        let gguf = GgufFile::open(path)
            .with_context(|| format!("Failed to open LoRA adapter: {}", path.display()))?;

        let streamer = SsdStreamer::new(path, 512 * 1024 * 1024)?;

        // Try to read LoRA parameters from GGUF metadata
        let rank = gguf
            .get_u32("adapter.lora.rank")
            .or_else(|| gguf.get_u32("training.lora.rank"))
            .unwrap_or(16) as usize;

        let alpha = gguf
            .get_f32("adapter.lora.alpha")
            .or_else(|| gguf.get_f32("training.lora.alpha"))
            .unwrap_or(rank as f32);

        info!(
            "Loading LoRA adapter: {} (rank={}, alpha={}, scale={})",
            path.display(),
            rank,
            alpha,
            scale
        );

        // Scan for LoRA tensor pairs (*.lora_a / *.lora_b)
        let tensor_names: Vec<String> = gguf.tensor_names();
        let mut adapters = HashMap::new();

        // Collect all lora_a tensors and find matching lora_b
        for name in &tensor_names {
            if !name.ends_with(".lora_a") {
                continue;
            }

            let base_name = name.strip_suffix(".lora_a").unwrap();
            let b_name = format!("{}.lora_b", base_name);

            if !tensor_names.contains(&b_name) {
                warn!("LoRA tensor {} has no matching B matrix, skipping", name);
                continue;
            }

            let a_info = gguf.find_tensor(name).unwrap();
            let b_info = gguf.find_tensor(&b_name).unwrap();

            let lora_a = streamer.load_tensor_f32(a_info)?;
            let lora_b = streamer.load_tensor_f32(b_info)?;

            // A: [r, d_in], B: [d_out, r]
            let a_dims = &a_info.dimensions;
            let b_dims = &b_info.dimensions;

            let (a_rank, d_in) = if a_dims.len() >= 2 {
                (a_dims[0] as usize, a_dims[1] as usize)
            } else {
                let total = a_dims.iter().product::<u64>() as usize;
                (rank, total / rank)
            };

            let (d_out, b_rank) = if b_dims.len() >= 2 {
                (b_dims[0] as usize, b_dims[1] as usize)
            } else {
                let total = b_dims.iter().product::<u64>() as usize;
                (total / rank, rank)
            };

            if a_rank != b_rank {
                warn!(
                    "LoRA rank mismatch for {}: A rank={}, B rank={}",
                    base_name, a_rank, b_rank
                );
                continue;
            }

            debug!(
                "LoRA adapter: {} -> rank={}, d_in={}, d_out={}",
                base_name, a_rank, d_in, d_out
            );

            adapters.insert(
                base_name.to_string(),
                LoraWeight {
                    lora_a,
                    lora_b,
                    d_in,
                    d_out,
                    rank: a_rank,
                },
            );
        }

        if adapters.is_empty() {
            bail!(
                "No LoRA adapter pairs found in {}. Expected tensors named *.lora_a / *.lora_b",
                path.display()
            );
        }

        info!(
            "Loaded LoRA adapter with {} weight pairs from {}",
            adapters.len(),
            path.display()
        );

        Ok(Self {
            path: path.to_path_buf(),
            rank,
            alpha,
            scale,
            adapters,
        })
    }

    /// Check if this adapter has weights for a given tensor name
    pub fn has_weight(&self, tensor_name: &str) -> bool {
        self.adapters.contains_key(tensor_name)
    }

    /// Get the effective scale factor: alpha / rank * user_scale
    pub fn effective_scale(&self) -> f32 {
        (self.alpha / self.rank as f32) * self.scale
    }
}

/// Manager for multiple LoRA adapters with hot-swapping support
pub struct LoraManager {
    /// Named adapters for hot-swap support: (name, adapter)
    adapters: Vec<(String, LoraAdapter)>,
    /// Set of currently active adapter names (subset of loaded adapters)
    active: std::collections::HashSet<String>,
    /// Cache of merged deltas: tensor_name -> pre-computed delta vector
    delta_cache: HashMap<String, Vec<f32>>,
    /// Generation counter — incremented on any adapter change, invalidates cache
    generation: u64,
    /// Generation at which delta_cache was last computed
    cache_generation: u64,
}

impl LoraManager {
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
            active: std::collections::HashSet::new(),
            delta_cache: HashMap::new(),
            generation: 0,
            cache_generation: 0,
        }
    }

    /// Load and add a LoRA adapter with auto-generated name
    pub fn add_adapter(&mut self, path: &Path, scale: f32) -> Result<()> {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| format!("adapter_{}", self.adapters.len()));
        self.add_named_adapter(&name, path, scale)
    }

    /// Load and add a LoRA adapter with explicit name
    pub fn add_named_adapter(&mut self, name: &str, path: &Path, scale: f32) -> Result<()> {
        // Remove existing adapter with same name if present
        self.adapters.retain(|(n, _)| n != name);
        self.active.remove(name);

        let adapter = LoraAdapter::load(path, scale)?;
        let name_str = name.to_string();
        self.active.insert(name_str.clone());
        self.adapters.push((name_str, adapter));
        self.generation += 1;
        self.delta_cache.clear();
        info!(
            "Added and activated LoRA adapter '{}' from {}",
            name,
            path.display()
        );
        Ok(())
    }

    /// Hot-swap: activate a loaded adapter by name (no-op if already active)
    pub fn activate(&mut self, name: &str) -> Result<()> {
        if !self.adapters.iter().any(|(n, _)| n == name) {
            bail!("LoRA adapter '{}' not loaded", name);
        }
        if self.active.insert(name.to_string()) {
            self.generation += 1;
            self.delta_cache.clear();
            info!("Activated LoRA adapter '{}'", name);
        }
        Ok(())
    }

    /// Hot-swap: deactivate an adapter by name (keeps it loaded for fast re-activation)
    pub fn deactivate(&mut self, name: &str) -> Result<()> {
        if !self.adapters.iter().any(|(n, _)| n == name) {
            bail!("LoRA adapter '{}' not loaded", name);
        }
        if self.active.remove(name) {
            self.generation += 1;
            self.delta_cache.clear();
            info!("Deactivated LoRA adapter '{}'", name);
        }
        Ok(())
    }

    /// Unload an adapter entirely (frees memory)
    pub fn remove_adapter(&mut self, name: &str) -> Result<()> {
        let before = self.adapters.len();
        self.adapters.retain(|(n, _)| n != name);
        self.active.remove(name);
        if self.adapters.len() == before {
            bail!("LoRA adapter '{}' not found", name);
        }
        self.generation += 1;
        self.delta_cache.clear();
        info!("Removed LoRA adapter '{}'", name);
        Ok(())
    }

    /// Update the scale of a loaded adapter
    pub fn set_scale(&mut self, name: &str, scale: f32) -> Result<()> {
        for (n, adapter) in &mut self.adapters {
            if n == name {
                adapter.scale = scale;
                self.generation += 1;
                self.delta_cache.clear();
                info!("Updated LoRA adapter '{}' scale to {}", name, scale);
                return Ok(());
            }
        }
        bail!("LoRA adapter '{}' not found", name);
    }

    /// Apply all *active* LoRA adapters to a weight tensor in-place
    pub fn apply_to_weight(&mut self, tensor_name: &str, weight: &mut [f32]) {
        for (name, adapter) in &self.adapters {
            if !self.active.contains(name) {
                continue;
            }
            if let Some(lora_weight) = adapter.adapters.get(tensor_name) {
                lora_weight.apply_to_weight(weight, adapter.alpha, adapter.scale);
            }
        }
    }

    /// Check if any active adapter has weights for this tensor
    pub fn has_weight(&self, tensor_name: &str) -> bool {
        self.adapters
            .iter()
            .filter(|(n, _)| self.active.contains(n))
            .any(|(_, a)| a.adapters.contains_key(tensor_name))
    }

    /// Get the number of loaded adapters (active + inactive)
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }

    /// Get the number of currently active adapters
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// List all loaded adapters with their status
    pub fn list(&self) -> Vec<(String, bool)> {
        self.adapters
            .iter()
            .map(|(n, _)| (n.clone(), self.active.contains(n)))
            .collect()
    }

    /// Get summary info for all loaded adapters
    pub fn summary(&self) -> Vec<LoraAdapterInfo> {
        self.adapters
            .iter()
            .map(|(name, a)| LoraAdapterInfo {
                path: a.path.display().to_string(),
                name: name.clone(),
                active: self.active.contains(name),
                rank: a.rank,
                alpha: a.alpha,
                scale: a.scale,
                num_weights: a.adapters.len(),
            })
            .collect()
    }

    /// Return true if no adapters are loaded
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
}

impl Default for LoraManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary info about a loaded adapter
#[derive(Debug, Clone)]
pub struct LoraAdapterInfo {
    pub path: String,
    pub name: String,
    pub active: bool,
    pub rank: usize,
    pub alpha: f32,
    pub scale: f32,
    pub num_weights: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_weight_compute_delta() {
        // rank=2, d_in=3, d_out=2
        // A = [[1,0,0],[0,1,0]] (2x3)
        // B = [[1,0],[0,1]] (2x2, identity)
        // delta = B @ A = [[1,0,0],[0,1,0]] scaled by alpha/r
        let lw = LoraWeight {
            lora_a: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            lora_b: vec![1.0, 0.0, 0.0, 1.0],
            d_in: 3,
            d_out: 2,
            rank: 2,
        };

        let delta = lw.compute_delta(2.0, 1.0);
        // factor = 2.0/2 * 1.0 = 1.0
        assert_eq!(delta.len(), 6);
        assert!((delta[0] - 1.0).abs() < 1e-6); // [0,0]
        assert!((delta[1] - 0.0).abs() < 1e-6); // [0,1]
        assert!((delta[2] - 0.0).abs() < 1e-6); // [0,2]
        assert!((delta[3] - 0.0).abs() < 1e-6); // [1,0]
        assert!((delta[4] - 1.0).abs() < 1e-6); // [1,1]
        assert!((delta[5] - 0.0).abs() < 1e-6); // [1,2]
    }

    #[test]
    fn test_lora_weight_apply_to_weight() {
        let lw = LoraWeight {
            lora_a: vec![1.0, 0.0, 0.0, 1.0],
            lora_b: vec![2.0, 0.0, 0.0, 3.0],
            d_in: 2,
            d_out: 2,
            rank: 2,
        };

        // B @ A = [[2,0],[0,3]], factor = 4.0/2 * 0.5 = 1.0
        let mut weight = vec![10.0, 20.0, 30.0, 40.0];
        lw.apply_to_weight(&mut weight, 4.0, 0.5);
        assert!((weight[0] - 12.0).abs() < 1e-6); // 10 + 2
        assert!((weight[1] - 20.0).abs() < 1e-6); // 20 + 0
        assert!((weight[2] - 30.0).abs() < 1e-6); // 30 + 0
        assert!((weight[3] - 43.0).abs() < 1e-6); // 40 + 3
    }

    #[test]
    fn test_lora_weight_scaling() {
        let lw = LoraWeight {
            lora_a: vec![1.0, 0.0, 0.0, 1.0],
            lora_b: vec![1.0, 0.0, 0.0, 1.0],
            d_in: 2,
            d_out: 2,
            rank: 2,
        };

        // alpha=8, r=2, scale=0.25 -> factor = (8/2)*0.25 = 1.0
        let delta = lw.compute_delta(8.0, 0.25);
        assert!((delta[0] - 1.0).abs() < 1e-6);
        assert!((delta[3] - 1.0).abs() < 1e-6);

        // alpha=2, r=2, scale=2.0 -> factor = (2/2)*2.0 = 2.0
        let delta2 = lw.compute_delta(2.0, 2.0);
        assert!((delta2[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_lora_manager_empty() {
        let mgr = LoraManager::new();
        assert!(mgr.is_empty());
        assert_eq!(mgr.adapter_count(), 0);
        assert_eq!(mgr.active_count(), 0);
        assert!(!mgr.has_weight("blk.0.attn_q.weight"));
    }

    #[test]
    fn test_lora_adapter_info() {
        let info = LoraAdapterInfo {
            path: "/tmp/adapter.gguf".to_string(),
            name: "test_adapter".to_string(),
            active: true,
            rank: 16,
            alpha: 32.0,
            scale: 1.0,
            num_weights: 42,
        };
        assert_eq!(info.rank, 16);
        assert!((info.alpha - 32.0).abs() < 1e-6);
        assert!(info.active);
        assert_eq!(info.name, "test_adapter");
    }

    #[test]
    fn test_lora_manager_hot_swap_api() {
        // Test the hot-swap API surface without loading real GGUF files
        let mgr = LoraManager::new();
        assert!(mgr.is_empty());
        assert_eq!(mgr.list().len(), 0);

        // activate/deactivate should fail on non-existent adapters
        let mut mgr = LoraManager::new();
        assert!(mgr.activate("nonexistent").is_err());
        assert!(mgr.deactivate("nonexistent").is_err());
        assert!(mgr.remove_adapter("nonexistent").is_err());
        assert!(mgr.set_scale("nonexistent", 0.5).is_err());
    }
}
