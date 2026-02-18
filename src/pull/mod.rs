//! Model downloader — fetch GGUF files from Hugging Face Hub
//!
//! Supports:
//! - Direct HuggingFace URLs
//! - Short-form identifiers: `user/repo:filename.gguf`
//! - Progress bar with resumable downloads

use anyhow::{bail, Context, Result};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Default directory for downloaded models
const DEFAULT_MODEL_DIR: &str = "models";

/// Download state for progress tracking
pub struct DownloadProgress {
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub filename: String,
}

/// Parse a model specifier into (repo, filename).
///
/// Formats:
/// - `user/repo:filename.gguf` → ("user/repo", "filename.gguf")
/// - `user/repo` → ("user/repo", auto-detect first GGUF)
/// - Full URL → download directly
pub fn parse_model_spec(spec: &str) -> Result<(String, Option<String>)> {
    if spec.starts_with("http://") || spec.starts_with("https://") {
        // Extract repo and filename from HF URL
        // https://huggingface.co/user/repo/resolve/main/file.gguf
        if let Some(rest) = spec.strip_prefix("https://huggingface.co/") {
            let parts: Vec<&str> = rest.splitn(5, '/').collect();
            if parts.len() >= 5 && parts[2] == "resolve" {
                let repo = format!("{}/{}", parts[0], parts[1]);
                let filename = parts[4].to_string();
                return Ok((repo, Some(filename)));
            }
        }
        bail!(
            "Unsupported URL format. Use: https://huggingface.co/user/repo/resolve/main/file.gguf"
        );
    }

    if let Some((repo, file)) = spec.split_once(':') {
        Ok((repo.to_string(), Some(file.to_string())))
    } else {
        Ok((spec.to_string(), None))
    }
}

/// Build the HuggingFace download URL
fn hf_download_url(repo: &str, filename: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", repo, filename)
}

/// List GGUF files in a HuggingFace repo via the API
pub fn list_gguf_files(repo: &str) -> Result<Vec<String>> {
    let url = format!("https://huggingface.co/api/models/{}", repo);

    let response = ureq_get(&url)?;
    // Parse the siblings array for .gguf files
    let mut gguf_files = Vec::new();
    // Simple JSON parsing for siblings[].rfilename
    for line in response.split('"') {
        if line.ends_with(".gguf") {
            gguf_files.push(line.to_string());
        }
    }
    gguf_files.sort();
    gguf_files.dedup();
    Ok(gguf_files)
}

/// Simple HTTP GET using std::net (no external HTTP crate dependency)
fn ureq_get(url: &str) -> Result<String> {
    use std::io::BufRead;
    use std::net::TcpStream;

    let (host, path, use_tls) = parse_url(url)?;
    let port = if use_tls { 443 } else { 80 };

    if use_tls {
        // For HTTPS, we shell out to curl since we don't want to add a TLS dependency
        let output = std::process::Command::new("curl")
            .args(["-sL", url])
            .output()
            .context("Failed to execute curl")?;
        if !output.status.success() {
            bail!("curl failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        return Ok(String::from_utf8_lossy(&output.stdout).to_string());
    }

    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(&addr)?;
    let request = format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: ssd-llm/1.0\r\nConnection: close\r\n\r\n",
        path, host
    );
    stream.write_all(request.as_bytes())?;

    let mut reader = std::io::BufReader::new(stream);
    // Skip headers
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.trim().is_empty() {
            break;
        }
    }
    let mut body = String::new();
    reader.read_to_string(&mut body)?;
    Ok(body)
}

fn parse_url(url: &str) -> Result<(String, String, bool)> {
    let use_tls = url.starts_with("https://");
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .context("Invalid URL")?;
    let (host, path) = without_scheme
        .split_once('/')
        .unwrap_or((without_scheme, ""));
    Ok((host.to_string(), format!("/{}", path), use_tls))
}

/// Download a model file with progress reporting
pub fn download_model(
    repo: &str,
    filename: &str,
    output_dir: &Path,
    force: bool,
) -> Result<PathBuf> {
    let output_path = output_dir.join(filename);

    // Check if already downloaded
    if output_path.exists() && !force {
        println!("✓ Model already exists: {}", output_path.display());
        println!("  Use --force to re-download");
        return Ok(output_path);
    }

    // Create output directory
    fs::create_dir_all(output_dir)?;

    let url = hf_download_url(repo, filename);
    println!("📥 Downloading: {}", url);
    println!("   → {}", output_path.display());

    // Use curl for reliable HTTPS download with progress
    let temp_path = output_path.with_extension("gguf.part");
    let mut args = vec![
        "-L".to_string(),
        "--progress-bar".to_string(),
        "-o".to_string(),
        temp_path.to_string_lossy().to_string(),
    ];

    // Resume partial download if exists
    if temp_path.exists() {
        args.push("-C".to_string());
        args.push("-".to_string());
        println!("   Resuming partial download...");
    }

    args.push(url);

    let status = std::process::Command::new("curl")
        .args(&args)
        .status()
        .context("Failed to execute curl. Is curl installed?")?;

    if !status.success() {
        bail!("Download failed (curl exit code: {:?})", status.code());
    }

    // Rename temp file to final
    fs::rename(&temp_path, &output_path)?;

    // Verify it's a valid GGUF
    let file = fs::File::open(&output_path)?;
    let mut magic = [0u8; 4];
    std::io::BufReader::new(file).read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        fs::remove_file(&output_path)?;
        bail!("Downloaded file is not a valid GGUF file");
    }

    let size = fs::metadata(&output_path)?.len();
    println!(
        "✓ Downloaded: {} ({:.2} GB)",
        filename,
        size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    Ok(output_path)
}

/// Get default model directory (next to binary or in workspace)
pub fn default_model_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SSD_LLM_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(DEFAULT_MODEL_DIR)
}

/// List locally downloaded models
pub fn list_local_models(dir: &Path) -> Result<Vec<(String, u64)>> {
    let mut models = Vec::new();
    if !dir.exists() {
        return Ok(models);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let size = fs::metadata(&path)?.len();
            models.push((name, size));
        }
    }
    models.sort();
    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_spec_with_file() {
        let (repo, file) =
            parse_model_spec("TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q4_0.gguf").unwrap();
        assert_eq!(repo, "TheBloke/Llama-2-7B-GGUF");
        assert_eq!(file, Some("llama-2-7b.Q4_0.gguf".to_string()));
    }

    #[test]
    fn test_parse_model_spec_repo_only() {
        let (repo, file) = parse_model_spec("TheBloke/Llama-2-7B-GGUF").unwrap();
        assert_eq!(repo, "TheBloke/Llama-2-7B-GGUF");
        assert_eq!(file, None);
    }

    #[test]
    fn test_parse_model_spec_url() {
        let (repo, file) = parse_model_spec(
            "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf",
        )
        .unwrap();
        assert_eq!(repo, "TheBloke/Llama-2-7B-GGUF");
        assert_eq!(file, Some("llama-2-7b.Q4_0.gguf".to_string()));
    }

    #[test]
    fn test_parse_url() {
        let (host, path, tls) = parse_url("https://example.com/foo/bar").unwrap();
        assert_eq!(host, "example.com");
        assert_eq!(path, "/foo/bar");
        assert!(tls);
    }

    #[test]
    fn test_default_model_dir() {
        let dir = default_model_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn test_hf_download_url() {
        let url = hf_download_url("TheBloke/Llama-2-7B-GGUF", "llama-2-7b.Q4_0.gguf");
        assert_eq!(
            url,
            "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf"
        );
    }

    #[test]
    fn test_list_local_models_empty() {
        let dir = std::env::temp_dir().join("ssd-llm-test-empty-dir");
        let models = list_local_models(&dir).unwrap();
        assert!(models.is_empty());
    }
}
