//! Ollama model management API endpoints
//!
//! Implements the remaining Ollama REST API endpoints for full drop-in compatibility:
//! - POST /api/show — model information and metadata
//! - POST /api/pull — download models from HuggingFace Hub
//! - POST /api/copy — copy/alias a local model
//! - DELETE /api/delete — remove a local model
//! - GET /api/ps — list running models / process status

use crate::model::gguf::GgufFile;
use crate::pull;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{error, info};

/// Model information for /api/show
#[derive(Debug)]
pub struct ModelInfo {
    pub name: String,
    pub architecture: String,
    pub parameter_count: String,
    pub quantization: String,
    pub context_length: u32,
    pub embedding_length: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub layer_count: u32,
    pub vocab_size: u32,
    pub file_size: u64,
    pub is_moe: bool,
    pub n_experts: u32,
    pub n_experts_used: u32,
    pub template: String,
    pub system_prompt: String,
}

/// Extract model info from a loaded GGUF file
pub fn model_info_from_gguf(gguf: &GgufFile, name: &str, _path: &Path) -> ModelInfo {
    let arch = gguf.architecture().to_string();

    // Detect quantization from most common tensor type
    let quantization = detect_quantization(gguf);

    // Read chat template if present
    let template = gguf
        .metadata
        .get("tokenizer.chat_template")
        .and_then(|v| v.as_string())
        .unwrap_or("")
        .to_string();

    // Read system prompt if present
    let system_prompt = gguf
        .metadata
        .get("general.description")
        .and_then(|v| v.as_string())
        .unwrap_or("")
        .to_string();

    // Estimate parameter count from tensor sizes
    let total_elements: u64 = gguf
        .tensors
        .iter()
        .map(|t| t.dimensions.iter().product::<u64>())
        .sum();
    let parameter_count = format_param_count(total_elements);

    ModelInfo {
        name: name.to_string(),
        architecture: arch,
        parameter_count,
        quantization,
        context_length: gguf.n_ctx(),
        embedding_length: gguf.n_embd(),
        head_count: gguf.n_head(),
        head_count_kv: gguf.n_head_kv(),
        layer_count: gguf.n_layers(),
        vocab_size: gguf.vocab_size(),
        file_size: gguf.file_size,
        is_moe: gguf.is_moe(),
        n_experts: gguf.n_experts(),
        n_experts_used: gguf.n_experts_used(),
        template,
        system_prompt,
    }
}

/// Public accessor for quantization detection (used by server.rs)
pub fn detect_quantization_pub(gguf: &GgufFile) -> String {
    detect_quantization(gguf)
}

/// Detect quantization level from tensor types
fn detect_quantization(gguf: &GgufFile) -> String {
    use std::collections::HashMap;
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for tensor in &gguf.tensors {
        *type_counts
            .entry(format!("{:?}", tensor.dtype))
            .or_insert(0) += 1;
    }
    // Find most common non-F32 type (skip output/embedding which are often F16/F32)
    let weight_tensors: Vec<_> = gguf
        .tensors
        .iter()
        .filter(|t| {
            t.name.contains("weight") && !t.name.contains("output") && !t.name.contains("embed")
        })
        .collect();

    if weight_tensors.is_empty() {
        return "unknown".to_string();
    }

    let mut wt_counts: HashMap<String, usize> = HashMap::new();
    for t in &weight_tensors {
        *wt_counts.entry(format!("{:?}", t.dtype)).or_insert(0) += 1;
    }

    wt_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(name, _)| name)
        .unwrap_or_else(|| "unknown".to_string())
}

/// Format parameter count as human-readable string
fn format_param_count(elements: u64) -> String {
    if elements >= 1_000_000_000 {
        format!("{:.1}B", elements as f64 / 1_000_000_000.0)
    } else if elements >= 1_000_000 {
        format!("{:.0}M", elements as f64 / 1_000_000.0)
    } else {
        format!("{}", elements)
    }
}

/// Format ModelInfo as JSON for /api/show response
pub fn model_info_to_json(info: &ModelInfo) -> String {
    let moe_section = if info.is_moe {
        format!(
            r#","expert_count":{},"expert_used_count":{}"#,
            info.n_experts, info.n_experts_used
        )
    } else {
        String::new()
    };

    let modelfile = format!("# ssd-llm model\\nFROM {}\\n", escape_json_str(&info.name));
    let parameters = "temperature 0.7\\ntop_p 0.9\\ntop_k 40";

    format!(
        concat!(
            "{{",
            "\"modelfile\":\"{}\",",
            "\"parameters\":\"{}\",",
            "\"template\":\"{}\",",
            "\"system\":\"{}\",",
            "\"details\":{{",
            "\"parent_model\":\"\",",
            "\"format\":\"gguf\",",
            "\"family\":\"{}\",",
            "\"families\":[\"{}\"],",
            "\"parameter_size\":\"{}\",",
            "\"quantization_level\":\"{}\"",
            "}},",
            "\"model_info\":{{",
            "\"general.architecture\":\"{}\",",
            "\"general.parameter_count\":\"{}\"",
            "{},",
            "\"context_length\":{},",
            "\"embedding_length\":{},",
            "\"block_count\":{},",
            "\"attention.head_count\":{},",
            "\"attention.head_count_kv\":{},",
            "\"vocab_size\":{}",
            "}},",
            "\"model_size\":{}",
            "}}"
        ),
        modelfile,
        parameters,
        escape_json_str(&info.template),
        escape_json_str(&info.system_prompt),
        escape_json_str(&info.architecture),
        escape_json_str(&info.architecture),
        info.parameter_count,
        info.quantization,
        escape_json_str(&info.architecture),
        info.parameter_count,
        moe_section,
        info.context_length,
        info.embedding_length,
        info.layer_count,
        info.head_count,
        info.head_count_kv,
        info.vocab_size,
        info.file_size,
    )
}

/// Handle POST /api/pull — download a model
///
/// Request: {"name": "user/repo:file.gguf", "stream": false}
/// Response: {"status": "success"} or streaming progress
pub fn handle_pull(body: &str, model_dir: &Path) -> PullResult {
    let name = extract_field(body, "name").unwrap_or_default();
    let _streaming = extract_bool_field(body, "stream").unwrap_or(true);

    if name.is_empty() {
        return PullResult::Error("'name' is required".to_string());
    }

    info!("Pull request: {}", name);

    // Parse spec
    let (repo, filename) = match pull::parse_model_spec(&name) {
        Ok(v) => v,
        Err(e) => return PullResult::Error(format!("Invalid model spec: {}", e)),
    };

    // If no filename specified, try to list and pick first
    let filename = match filename {
        Some(f) => f,
        None => {
            info!("No filename specified, listing GGUF files in {}", repo);
            match pull::list_gguf_files(&repo) {
                Ok(files) if !files.is_empty() => {
                    info!("Found {} GGUF files, using: {}", files.len(), files[0]);
                    files[0].clone()
                }
                Ok(_) => return PullResult::Error(format!("No GGUF files found in {}", repo)),
                Err(e) => return PullResult::Error(format!("Failed to list files: {}", e)),
            }
        }
    };

    // Download
    match pull::download_model(&repo, &filename, model_dir, false) {
        Ok(path) => {
            info!("Pull complete: {}", path.display());
            PullResult::Success {
                status: "success".to_string(),
                digest: format!("sha256:{}", hex_digest_stub(&filename)),
                total: fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
            }
        }
        Err(e) => {
            error!("Pull failed: {}", e);
            PullResult::Error(e.to_string())
        }
    }
}

/// Result of a pull operation
pub enum PullResult {
    Success {
        status: String,
        digest: String,
        total: u64,
    },
    Error(String),
}

impl PullResult {
    pub fn to_json(&self) -> (u16, String) {
        match self {
            PullResult::Success {
                status,
                digest,
                total,
            } => (
                200,
                format!(
                    r#"{{"status":"{}","digest":"{}","total":{}}}"#,
                    status, digest, total
                ),
            ),
            PullResult::Error(msg) => (400, format!(r#"{{"error":"{}"}}"#, escape_json_str(msg))),
        }
    }
}

/// Handle POST /api/copy — copy a model file under a new name
///
/// Request: {"source": "model-name", "destination": "new-name"}
pub fn handle_copy(body: &str, model_dir: &Path) -> (u16, String) {
    let source = extract_field(body, "source").unwrap_or_default();
    let destination = extract_field(body, "destination").unwrap_or_default();

    if source.is_empty() || destination.is_empty() {
        return (
            400,
            r#"{"error":"'source' and 'destination' are required"}"#.to_string(),
        );
    }

    let src_path = resolve_model_path(&source, model_dir);
    let dst_path = model_dir.join(sanitize_filename(&destination));

    match &src_path {
        Some(p) => {
            info!("Copying {} -> {}", p.display(), dst_path.display());
            if let Err(e) = fs::create_dir_all(model_dir) {
                return (500, format!(r#"{{"error":"Failed to create dir: {}"}}"#, e));
            }
            match fs::copy(p, &dst_path) {
                Ok(_) => (200, r#"{"status":"success"}"#.to_string()),
                Err(e) => (
                    500,
                    format!(
                        r#"{{"error":"Copy failed: {}"}}"#,
                        escape_json_str(&e.to_string())
                    ),
                ),
            }
        }
        None => (
            404,
            format!(
                r#"{{"error":"model '{}' not found"}}"#,
                escape_json_str(&source)
            ),
        ),
    }
}

/// Handle DELETE /api/delete — delete a local model
///
/// Request: {"name": "model-name"}
pub fn handle_delete(body: &str, model_dir: &Path) -> (u16, String) {
    let name = extract_field(body, "name").unwrap_or_default();

    if name.is_empty() {
        return (400, r#"{"error":"'name' is required"}"#.to_string());
    }

    match resolve_model_path(&name, model_dir) {
        Some(path) => {
            info!("Deleting model: {}", path.display());
            match fs::remove_file(&path) {
                Ok(()) => (200, r#"{"status":"success"}"#.to_string()),
                Err(e) => (
                    500,
                    format!(
                        r#"{{"error":"Delete failed: {}"}}"#,
                        escape_json_str(&e.to_string())
                    ),
                ),
            }
        }
        None => (
            404,
            format!(
                r#"{{"error":"model '{}' not found"}}"#,
                escape_json_str(&name)
            ),
        ),
    }
}

/// Build JSON for GET /api/ps — running model processes
pub fn running_model_json(model_name: &str, model_size: u64, quantization: &str) -> String {
    format!(
        r#"{{"models":[{{"name":"{}","model":"{}","size":{},"digest":"local","details":{{"family":"gguf","quantization_level":"{}"}},"expires_at":"2099-01-01T00:00:00Z","size_vram":0}}]}}"#,
        escape_json_str(model_name),
        escape_json_str(model_name),
        model_size,
        escape_json_str(quantization),
    )
}

/// Resolve a model name to a file path in the model directory
fn resolve_model_path(name: &str, model_dir: &Path) -> Option<PathBuf> {
    // Try exact name first
    let exact = model_dir.join(name);
    if exact.exists() {
        return Some(exact);
    }

    // Try with .gguf extension
    let with_ext = model_dir.join(format!("{}.gguf", name));
    if with_ext.exists() {
        return Some(with_ext);
    }

    // Try case-insensitive search
    if let Ok(entries) = fs::read_dir(model_dir) {
        let lower = name.to_lowercase();
        for entry in entries.flatten() {
            let entry_name = entry.file_name().to_string_lossy().to_lowercase();
            if entry_name == lower || entry_name == format!("{}.gguf", lower) {
                return Some(entry.path());
            }
        }
    }

    None
}

/// Sanitize a filename (no path traversal)
fn sanitize_filename(name: &str) -> String {
    let base = name.replace(['/', '\\'], "_").replace("..", "_");
    if base.ends_with(".gguf") {
        base
    } else {
        format!("{}.gguf", base)
    }
}

/// Generate a stub hex digest from filename (deterministic but not cryptographic)
fn hex_digest_stub(filename: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    for byte in filename.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3); // FNV prime
    }
    format!(
        "{:016x}{:016x}{:016x}{:016x}",
        hash,
        hash.rotate_left(16),
        hash.rotate_left(32),
        hash.rotate_left(48)
    )
}

// --- Simple JSON helpers ---

fn extract_field(json: &str, key: &str) -> Option<String> {
    let pattern = format!(r#""{}""#, key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start().strip_prefix(':')?;
    let after = after.trim_start().strip_prefix('"')?;
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

fn extract_bool_field(json: &str, key: &str) -> Option<bool> {
    let pattern = format!(r#""{}""#, key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start().strip_prefix(':')?;
    let after = after.trim_start();
    if after.starts_with("true") {
        Some(true)
    } else if after.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn escape_json_str(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_format_param_count() {
        assert_eq!(format_param_count(7_000_000_000), "7.0B");
        assert_eq!(format_param_count(70_500_000_000), "70.5B");
        assert_eq!(format_param_count(350_000_000), "350M");
        assert_eq!(format_param_count(999), "999");
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("my-model"), "my-model.gguf");
        assert_eq!(sanitize_filename("my-model.gguf"), "my-model.gguf");
        assert_eq!(sanitize_filename("../evil"), "__evil.gguf");
        assert_eq!(sanitize_filename("path/to/model"), "path_to_model.gguf");
    }

    #[test]
    fn test_extract_field() {
        let json = r#"{"name": "llama2", "stream": true}"#;
        assert_eq!(extract_field(json, "name"), Some("llama2".to_string()));
    }

    #[test]
    fn test_extract_bool_field() {
        let json = r#"{"stream": true, "verbose": false}"#;
        assert_eq!(extract_bool_field(json, "stream"), Some(true));
        assert_eq!(extract_bool_field(json, "verbose"), Some(false));
        assert_eq!(extract_bool_field(json, "missing"), None);
    }

    #[test]
    fn test_hex_digest_stub() {
        let d1 = hex_digest_stub("model-a.gguf");
        let d2 = hex_digest_stub("model-b.gguf");
        assert_ne!(d1, d2);
        assert_eq!(d1.len(), 64);
        // Deterministic
        assert_eq!(d1, hex_digest_stub("model-a.gguf"));
    }

    #[test]
    fn test_resolve_model_path_not_found() {
        let dir = env::temp_dir().join("ssd-llm-test-resolve");
        let _ = fs::create_dir_all(&dir);
        assert_eq!(resolve_model_path("nonexistent", &dir), None);
    }

    #[test]
    fn test_resolve_model_path_exact() {
        let dir = env::temp_dir().join("ssd-llm-test-resolve-exact");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("test.gguf");
        fs::write(&file, b"test").unwrap();
        assert_eq!(resolve_model_path("test.gguf", &dir), Some(file.clone()));
        let _ = fs::remove_file(&file);
    }

    #[test]
    fn test_resolve_model_path_with_ext() {
        let dir = env::temp_dir().join("ssd-llm-test-resolve-ext");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("mymodel.gguf");
        fs::write(&file, b"test").unwrap();
        assert_eq!(resolve_model_path("mymodel", &dir), Some(file.clone()));
        let _ = fs::remove_file(&file);
    }

    #[test]
    fn test_pull_result_error_json() {
        let result = PullResult::Error("bad request".to_string());
        let (status, json) = result.to_json();
        assert_eq!(status, 400);
        assert!(json.contains("bad request"));
    }

    #[test]
    fn test_pull_result_success_json() {
        let result = PullResult::Success {
            status: "success".to_string(),
            digest: "sha256:abc123".to_string(),
            total: 1024,
        };
        let (status, json) = result.to_json();
        assert_eq!(status, 200);
        assert!(json.contains("success"));
        assert!(json.contains("1024"));
    }

    #[test]
    fn test_handle_copy_missing_fields() {
        let dir = env::temp_dir().join("ssd-llm-test-copy");
        let _ = fs::create_dir_all(&dir);
        let (status, _) = handle_copy(r#"{"source": "a"}"#, &dir);
        assert_eq!(status, 400);
    }

    #[test]
    fn test_handle_delete_missing_name() {
        let dir = env::temp_dir().join("ssd-llm-test-delete");
        let _ = fs::create_dir_all(&dir);
        let (status, _) = handle_delete(r#"{}"#, &dir);
        assert_eq!(status, 400);
    }

    #[test]
    fn test_handle_delete_not_found() {
        let dir = env::temp_dir().join("ssd-llm-test-delete-nf");
        let _ = fs::create_dir_all(&dir);
        let (status, json) = handle_delete(r#"{"name": "nonexistent"}"#, &dir);
        assert_eq!(status, 404);
        assert!(json.contains("not found"));
    }

    #[test]
    fn test_running_model_json() {
        let json = running_model_json("llama2", 4_000_000_000, "Q4_0");
        assert!(json.contains("llama2"));
        assert!(json.contains("4000000000"));
        assert!(json.contains("Q4_0"));
    }

    #[test]
    fn test_escape_json_str() {
        assert_eq!(escape_json_str("hello"), "hello");
        assert_eq!(escape_json_str("he\"llo"), "he\\\"llo");
        assert_eq!(escape_json_str("line\nnew"), "line\\nnew");
    }
}
