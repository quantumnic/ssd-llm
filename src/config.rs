//! Configuration file support for ssd-llm
//!
//! Loads settings from `ssd-llm.toml` (or `$SSD_LLM_CONFIG`).
//! CLI arguments override config file values.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Top-level configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub model: ModelConfig,
    pub server: ServerConfig,
    pub inference: InferenceConfig,
    pub paths: PathsConfig,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub path: Option<PathBuf>,
    pub memory_budget: String,
    pub draft_model: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_batch: usize,
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
    pub draft_ahead: usize,
    pub adaptive_draft: bool,
    pub prompt_cache: bool,
    pub tensor_parallel: usize,
    pub sliding_window: usize,
    pub sink_tokens: usize,
    pub mmap_kv: bool,
    pub flash_attention: bool,
    pub kv_quantize: bool,
    pub swap_quantize: bool,
    pub tfs_z: f32,
    pub mirostat: u8,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub grammar: String,
    pub adaptive_memory: bool,
    pub adaptive_pin: usize,
}

#[derive(Debug, Clone)]
pub struct PathsConfig {
    pub model_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                path: None,
                memory_budget: "8G".to_string(),
                draft_model: None,
            },
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 11434,
                max_batch: 4,
            },
            inference: InferenceConfig {
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
                max_tokens: 128,
                draft_ahead: 5,
                adaptive_draft: false,
                prompt_cache: false,
                tensor_parallel: 0,
                sliding_window: 0,
                sink_tokens: 4,
                mmap_kv: false,
                flash_attention: false,
                kv_quantize: false,
                swap_quantize: false,
                tfs_z: 0.0,
                mirostat: 0,
                mirostat_tau: 5.0,
                mirostat_eta: 0.1,
                grammar: String::new(),
                adaptive_memory: false,
                adaptive_pin: 0,
            },
            paths: PathsConfig {
                model_dir: PathBuf::from("models"),
            },
        }
    }
}

impl Config {
    /// Load config from default locations, falling back to defaults
    pub fn load() -> Result<Self> {
        let config_path = if let Ok(path) = std::env::var("SSD_LLM_CONFIG") {
            Some(PathBuf::from(path))
        } else {
            Self::find_config_file()
        };

        match config_path {
            Some(path) if path.exists() => Self::load_from_file(&path),
            _ => Ok(Self::default()),
        }
    }

    /// Find config file in standard locations
    fn find_config_file() -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = vec![PathBuf::from("ssd-llm.toml")];
        if let Some(dir) = dirs_config() {
            candidates.push(dir.join("ssd-llm.toml"));
        }
        candidates.into_iter().find(|c| c.exists())
    }

    /// Load and parse a TOML config file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        Self::parse_toml(&content)
    }

    /// Parse TOML content into Config (simple parser, no external TOML crate)
    fn parse_toml(content: &str) -> Result<Self> {
        let mut config = Self::default();
        let kv_map = parse_toml_simple(content);

        for (key, value) in &kv_map {
            let full_key = key.as_str();
            match full_key {
                "model.path" => config.model.path = Some(PathBuf::from(value)),
                "model.memory_budget" => config.model.memory_budget = value.clone(),
                "model.draft_model" => config.model.draft_model = Some(PathBuf::from(value)),
                "server.host" => config.server.host = value.clone(),
                "server.port" => {
                    config.server.port = value.parse().unwrap_or(config.server.port);
                }
                "server.max_batch" => {
                    config.server.max_batch = value.parse().unwrap_or(config.server.max_batch);
                }
                "inference.temperature" => {
                    config.inference.temperature =
                        value.parse().unwrap_or(config.inference.temperature);
                }
                "inference.top_k" => {
                    config.inference.top_k = value.parse().unwrap_or(config.inference.top_k);
                }
                "inference.top_p" => {
                    config.inference.top_p = value.parse().unwrap_or(config.inference.top_p);
                }
                "inference.max_tokens" => {
                    config.inference.max_tokens =
                        value.parse().unwrap_or(config.inference.max_tokens);
                }
                "inference.flash_attention" => {
                    config.inference.flash_attention = value == "true";
                }
                "inference.sliding_window" => {
                    config.inference.sliding_window =
                        value.parse().unwrap_or(config.inference.sliding_window);
                }
                "inference.mmap_kv" => {
                    config.inference.mmap_kv = value == "true";
                }
                "inference.prompt_cache" => {
                    config.inference.prompt_cache = value == "true";
                }
                "inference.kv_quantize" => {
                    config.inference.kv_quantize = value == "true";
                }
                "inference.swap_quantize" => {
                    config.inference.swap_quantize = value == "true";
                }
                "inference.tfs_z" => {
                    config.inference.tfs_z = value.parse().unwrap_or(0.0);
                }
                "inference.mirostat" => {
                    config.inference.mirostat = value.parse().unwrap_or(0);
                }
                "inference.mirostat_tau" => {
                    config.inference.mirostat_tau = value.parse().unwrap_or(5.0);
                }
                "inference.mirostat_eta" => {
                    config.inference.mirostat_eta = value.parse().unwrap_or(0.1);
                }
                "inference.adaptive_memory" => {
                    config.inference.adaptive_memory = value == "true";
                }
                "inference.adaptive_pin" => {
                    config.inference.adaptive_pin = value.parse().unwrap_or(0);
                }
                "paths.model_dir" => config.paths.model_dir = PathBuf::from(value),
                _ => {} // ignore unknown keys
            }
        }

        Ok(config)
    }

    /// Generate a default config file content
    pub fn default_toml() -> String {
        r#"# ssd-llm configuration file

[model]
# path = "models/llama-2-7b.Q4_0.gguf"
memory_budget = "8G"
# draft_model = "models/tinyllama-1b.Q4_0.gguf"

[server]
host = "127.0.0.1"
port = 11434
max_batch = 4

[inference]
temperature = 0.7
top_k = 40
top_p = 0.9
max_tokens = 128
flash_attention = false
sliding_window = 0
mmap_kv = false
prompt_cache = false
kv_quantize = false
swap_quantize = false
tfs_z = 0.0
mirostat = 0
mirostat_tau = 5.0
mirostat_eta = 0.1

[paths]
model_dir = "models"
"#
        .to_string()
    }
}

/// Simple TOML parser — handles `[section]` headers and `key = value` pairs
fn parse_toml_simple(content: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut section = String::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].trim().to_string();
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim();
            // Strip quotes
            let value = value
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .unwrap_or(value);
            let full_key = if section.is_empty() {
                key.to_string()
            } else {
                format!("{}.{}", section, key)
            };
            map.insert(full_key, value.to_string());
        }
    }
    map
}

/// Get platform config directory
fn dirs_config() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join(".config").join("ssd-llm"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::env::var("XDG_CONFIG_HOME")
            .ok()
            .map(|d| PathBuf::from(d).join("ssd-llm"))
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| PathBuf::from(h).join(".config").join("ssd-llm"))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.server.port, 11434);
        assert_eq!(config.inference.temperature, 0.7);
        assert_eq!(config.model.memory_budget, "8G");
    }

    #[test]
    fn test_parse_toml() {
        let toml = r#"
[model]
path = "test.gguf"
memory_budget = "16G"

[server]
port = 8080

[inference]
flash_attention = true
temperature = 0.5
"#;
        let config = Config::parse_toml(toml).unwrap();
        assert_eq!(config.model.path, Some(PathBuf::from("test.gguf")));
        assert_eq!(config.model.memory_budget, "16G");
        assert_eq!(config.server.port, 8080);
        assert!(config.inference.flash_attention);
        assert_eq!(config.inference.temperature, 0.5);
    }

    #[test]
    fn test_parse_toml_empty() {
        let config = Config::parse_toml("").unwrap();
        assert_eq!(config.server.port, 11434);
    }

    #[test]
    fn test_parse_toml_comments() {
        let toml = r#"
# This is a comment
[server]
# port = 9999
port = 8080
"#;
        let config = Config::parse_toml(toml).unwrap();
        assert_eq!(config.server.port, 8080);
    }

    #[test]
    fn test_default_toml_parseable() {
        let toml = Config::default_toml();
        let config = Config::parse_toml(&toml).unwrap();
        assert_eq!(config.server.port, 11434);
    }

    #[test]
    fn test_parse_toml_simple() {
        let content = "[section]\nkey = \"value\"\nnum = 42";
        let map = parse_toml_simple(content);
        assert_eq!(map.get("section.key").unwrap(), "value");
        assert_eq!(map.get("section.num").unwrap(), "42");
    }
}

impl Config {
    /// Validate configuration values, returning all errors found
    pub fn validate(&self) -> Result<()> {
        let mut errors = Vec::new();

        // Server validation
        if self.server.port == 0 {
            errors.push("server.port must be > 0".to_string());
        }
        if self.server.max_batch == 0 {
            errors.push("server.max_batch must be > 0".to_string());
        }
        if self.server.host.is_empty() {
            errors.push("server.host must not be empty".to_string());
        }

        // Inference validation
        if self.inference.temperature < 0.0 {
            errors.push(format!(
                "inference.temperature must be >= 0.0, got {}",
                self.inference.temperature
            ));
        }
        if self.inference.top_p <= 0.0 || self.inference.top_p > 1.0 {
            errors.push(format!(
                "inference.top_p must be in (0.0, 1.0], got {}",
                self.inference.top_p
            ));
        }
        if self.inference.top_k == 0 {
            errors.push("inference.top_k must be > 0".to_string());
        }
        if self.inference.max_tokens == 0 {
            errors.push("inference.max_tokens must be > 0".to_string());
        }
        if self.inference.tfs_z < 0.0 || self.inference.tfs_z > 1.0 {
            errors.push(format!(
                "inference.tfs_z must be in [0.0, 1.0], got {}",
                self.inference.tfs_z
            ));
        }
        if self.inference.mirostat > 2 {
            errors.push(format!(
                "inference.mirostat must be 0, 1, or 2, got {}",
                self.inference.mirostat
            ));
        }
        if self.inference.mirostat_tau <= 0.0 {
            errors.push(format!(
                "inference.mirostat_tau must be > 0.0, got {}",
                self.inference.mirostat_tau
            ));
        }
        if self.inference.mirostat_eta <= 0.0 {
            errors.push(format!(
                "inference.mirostat_eta must be > 0.0, got {}",
                self.inference.mirostat_eta
            ));
        }

        // Memory budget validation (suffix + numeric prefix)
        if self.model.memory_budget_bytes().is_none() {
            errors.push(format!(
                "model.memory_budget must be a positive number followed by G, M, or K, got '{}'",
                self.model.memory_budget
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            anyhow::bail!(
                "Configuration validation failed:\n  - {}",
                errors.join("\n  - ")
            )
        }
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn test_validate_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_bad_temperature() {
        let mut config = Config::default();
        config.inference.temperature = -0.5;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("temperature"));
    }

    #[test]
    fn test_validate_bad_top_p() {
        let mut config = Config::default();
        config.inference.top_p = 1.5;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("top_p"));
    }

    #[test]
    fn test_validate_zero_top_p() {
        let mut config = Config::default();
        config.inference.top_p = 0.0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("top_p"));
    }

    #[test]
    fn test_validate_bad_memory_budget() {
        let mut config = Config::default();
        config.model.memory_budget = "8".to_string();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("memory_budget"));
    }

    #[test]
    fn test_validate_bad_mirostat() {
        let mut config = Config::default();
        config.inference.mirostat = 5;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("mirostat"));
    }

    #[test]
    fn test_validate_multiple_errors() {
        let mut config = Config::default();
        config.inference.temperature = -1.0;
        config.inference.top_p = 2.0;
        config.server.port = 0;
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("temperature"));
        assert!(msg.contains("top_p"));
        assert!(msg.contains("port"));
    }

    #[test]
    fn test_validate_empty_host() {
        let mut config = Config::default();
        config.server.host = String::new();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("host"));
    }

    #[test]
    fn test_validate_tfs_z_out_of_range() {
        let mut config = Config::default();
        config.inference.tfs_z = 1.5;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("tfs_z"));
    }

    #[test]
    fn test_validate_memory_budget_variants() {
        for suffix in &["G", "M", "K"] {
            let mut config = Config::default();
            config.model.memory_budget = format!("16{}", suffix);
            assert!(config.validate().is_ok(), "Should accept {}", suffix);
        }
    }
}

impl ModelConfig {
    /// Parse memory_budget string (e.g. "8G", "512M", "1024K") into bytes.
    /// Returns `None` if the format is invalid.
    pub fn memory_budget_bytes(&self) -> Option<usize> {
        let s = self.memory_budget.trim();
        if s.is_empty() {
            return None;
        }
        let (num_part, multiplier) = match s.as_bytes().last()? {
            b'G' | b'g' => (&s[..s.len() - 1], 1024 * 1024 * 1024),
            b'M' | b'm' => (&s[..s.len() - 1], 1024 * 1024),
            b'K' | b'k' => (&s[..s.len() - 1], 1024),
            _ => return None,
        };
        let num: usize = num_part.parse().ok()?;
        num.checked_mul(multiplier)
    }
}

#[cfg(test)]
mod memory_budget_tests {
    use super::*;

    #[test]
    fn test_memory_budget_bytes_gigabytes() {
        let m = ModelConfig {
            path: None,
            memory_budget: "8G".to_string(),
            draft_model: None,
        };
        assert_eq!(m.memory_budget_bytes(), Some(8 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_memory_budget_bytes_megabytes() {
        let m = ModelConfig {
            path: None,
            memory_budget: "512M".to_string(),
            draft_model: None,
        };
        assert_eq!(m.memory_budget_bytes(), Some(512 * 1024 * 1024));
    }

    #[test]
    fn test_memory_budget_bytes_kilobytes() {
        let m = ModelConfig {
            path: None,
            memory_budget: "1024K".to_string(),
            draft_model: None,
        };
        assert_eq!(m.memory_budget_bytes(), Some(1024 * 1024));
    }

    #[test]
    fn test_memory_budget_bytes_lowercase() {
        let m = ModelConfig {
            path: None,
            memory_budget: "4g".to_string(),
            draft_model: None,
        };
        assert_eq!(m.memory_budget_bytes(), Some(4 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_memory_budget_bytes_invalid_no_suffix() {
        let m = ModelConfig {
            path: None,
            memory_budget: "8".to_string(),
            draft_model: None,
        };
        assert_eq!(m.memory_budget_bytes(), None);
    }

    #[test]
    fn test_memory_budget_bytes_invalid_no_number() {
        let m = ModelConfig {
            path: None,
            memory_budget: "G".to_string(),
            draft_model: None,
        };
        assert_eq!(m.memory_budget_bytes(), None);
    }

    #[test]
    fn test_memory_budget_bytes_empty() {
        let m = ModelConfig {
            path: None,
            memory_budget: "".to_string(),
            draft_model: None,
        };
        assert_eq!(m.memory_budget_bytes(), None);
    }

    #[test]
    fn test_memory_budget_bytes_invalid_text() {
        let m = ModelConfig {
            path: None,
            memory_budget: "lotsG".to_string(),
            draft_model: None,
        };
        assert_eq!(m.memory_budget_bytes(), None);
    }
}
