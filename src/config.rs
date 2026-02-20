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
    pub tfs_z: f32,
    pub mirostat: u8,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub grammar: String,
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
                tfs_z: 0.0,
                mirostat: 0,
                mirostat_tau: 5.0,
                mirostat_eta: 0.1,
                grammar: String::new(),
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
