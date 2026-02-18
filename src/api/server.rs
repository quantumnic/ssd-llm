//! Ollama-compatible HTTP API server
//!
//! Implements the Ollama REST API for drop-in compatibility:
//! - POST /api/generate — text generation
//! - POST /api/chat — chat completions
//! - GET /api/tags — list loaded models
//! - GET /api/version — server version
//! - POST /v1/chat/completions — OpenAI-compatible endpoint

use crate::inference::transformer::{self, InferenceConfig, GenerationResult};
use crate::model::cache::LayerCache;
use crate::model::gguf::GgufFile;
use crate::ssd::streamer::SsdStreamer;
use anyhow::Result;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{error, info, warn};

/// Server configuration
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_path: PathBuf,
    pub memory_budget: usize,
}

/// Model context shared across requests
struct ModelContext {
    gguf: GgufFile,
    streamer: SsdStreamer,
    cache: LayerCache,
    model_name: String,
    model_path: PathBuf,
}

/// The API server
pub struct ApiServer {
    config: ServerConfig,
}

impl ApiServer {
    pub fn new(config: ServerConfig) -> Self {
        Self { config }
    }

    /// Start listening and handling requests
    pub fn run(&self) -> Result<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = TcpListener::bind(&addr)?;
        info!("ssd-llm API server listening on http://{}", addr);
        info!("Ollama-compatible API: POST /api/generate, POST /api/chat, GET /api/tags");
        info!("OpenAI-compatible API: POST /v1/chat/completions");

        // Load model
        info!("Loading model: {}", self.config.model_path.display());
        let gguf = GgufFile::open(&self.config.model_path)?;
        let streamer = SsdStreamer::new(&self.config.model_path, self.config.memory_budget)?;
        let cache = LayerCache::new(self.config.memory_budget);

        let model_name = self.config.model_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        info!("Model loaded: {} ({} layers, {} embd, {} vocab)",
            model_name, gguf.n_layers(), gguf.n_embd(), gguf.vocab_size());

        let ctx = Arc::new(Mutex::new(ModelContext {
            gguf,
            streamer,
            cache,
            model_name,
            model_path: self.config.model_path.clone(),
        }));

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let ctx = Arc::clone(&ctx);
                    // Simple single-threaded for now (v0.2)
                    if let Err(e) = handle_connection(stream, &ctx) {
                        error!("Request error: {}", e);
                    }
                }
                Err(e) => error!("Accept error: {}", e),
            }
        }

        Ok(())
    }
}

fn handle_connection(mut stream: TcpStream, ctx: &Arc<Mutex<ModelContext>>) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;

    let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
    if parts.len() < 2 {
        return send_response(&mut stream, 400, "Bad Request");
    }
    let method = parts[0];
    let path = parts[1];

    // Read headers
    let mut content_length = 0usize;
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            break;
        }
        if let Some(val) = trimmed.strip_prefix("Content-Length:") {
            content_length = val.trim().parse().unwrap_or(0);
        }
        headers.push(trimmed);
    }

    // Read body
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    let body_str = String::from_utf8_lossy(&body).to_string();

    info!("{} {} (body: {} bytes)", method, path, content_length);

    match (method, path) {
        ("GET", "/api/tags") => handle_tags(&mut stream, ctx),
        ("GET", "/api/version") => handle_version(&mut stream),
        ("POST", "/api/generate") => handle_generate(&mut stream, ctx, &body_str),
        ("POST", "/api/chat") => handle_chat(&mut stream, ctx, &body_str),
        ("POST", "/v1/chat/completions") => handle_openai_chat(&mut stream, ctx, &body_str),
        ("GET", "/") => send_json_response(&mut stream, 200,
            r#"{"status":"ssd-llm is running","version":"0.2.0"}"#),
        _ => send_response(&mut stream, 404, "Not Found"),
    }
}

fn handle_tags(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>) -> Result<()> {
    let ctx = ctx.lock().unwrap();
    let resp = format!(
        r#"{{"models":[{{"name":"{}","size":{},"digest":"local","details":{{"family":"{}","parameter_size":"{}","quantization_level":"local"}}}}]}}"#,
        ctx.model_name,
        ctx.gguf.file_size,
        ctx.gguf.architecture(),
        format!("{}B", ctx.gguf.n_layers()),
    );
    send_json_response(stream, 200, &resp)
}

fn handle_version(stream: &mut TcpStream) -> Result<()> {
    send_json_response(stream, 200, r#"{"version":"0.2.0-ssd-llm"}"#)
}

fn handle_generate(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>, body: &str) -> Result<()> {
    // Parse minimal JSON fields
    let prompt = extract_json_string(body, "prompt").unwrap_or_default();
    let max_tokens = extract_json_number(body, "num_predict").unwrap_or(128) as usize;
    let temperature = extract_json_float(body, "temperature").unwrap_or(0.7);
    let top_k = extract_json_number(body, "top_k").unwrap_or(40) as usize;
    let top_p = extract_json_float(body, "top_p").unwrap_or(0.9);

    let config = InferenceConfig {
        temperature,
        top_k,
        top_p,
        max_tokens,
    };

    let mut ctx = ctx.lock().unwrap();
    let ModelContext { ref gguf, ref streamer, ref mut cache, ref model_name, .. } = *ctx;
    let start = Instant::now();

    match transformer::generate(gguf, streamer, cache, &prompt, &config) {
        Ok(result) => {
            let elapsed = start.elapsed();
            let resp = format!(
                r#"{{"model":"{}","created_at":"{}","response":"{}","done":true,"total_duration":{},"eval_count":{},"eval_duration":{}}}"#,
                model_name,
                chrono_now(),
                escape_json(&result.text),
                elapsed.as_nanos(),
                result.token_count,
                elapsed.as_nanos(),
            );
            send_json_response(stream, 200, &resp)
        }
        Err(e) => {
            let resp = format!(r#"{{"error":"{}"}}"#, escape_json(&e.to_string()));
            send_json_response(stream, 500, &resp)
        }
    }
}

fn handle_chat(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>, body: &str) -> Result<()> {
    // Extract last user message from messages array (simplified parsing)
    let prompt = extract_last_message_content(body).unwrap_or_default();

    let config = InferenceConfig {
        temperature: extract_json_float(body, "temperature").unwrap_or(0.7),
        top_k: extract_json_number(body, "top_k").unwrap_or(40) as usize,
        top_p: extract_json_float(body, "top_p").unwrap_or(0.9),
        max_tokens: extract_json_number(body, "num_predict").unwrap_or(128) as usize,
    };

    let mut ctx = ctx.lock().unwrap();
    let ModelContext { ref gguf, ref streamer, ref mut cache, ref model_name, .. } = *ctx;
    let start = Instant::now();

    match transformer::generate(gguf, streamer, cache, &prompt, &config) {
        Ok(result) => {
            let elapsed = start.elapsed();
            let resp = format!(
                r#"{{"model":"{}","created_at":"{}","message":{{"role":"assistant","content":"{}"}},"done":true,"total_duration":{},"eval_count":{}}}"#,
                model_name,
                chrono_now(),
                escape_json(&result.text),
                elapsed.as_nanos(),
                result.token_count,
            );
            send_json_response(stream, 200, &resp)
        }
        Err(e) => {
            let resp = format!(r#"{{"error":"{}"}}"#, escape_json(&e.to_string()));
            send_json_response(stream, 500, &resp)
        }
    }
}

fn handle_openai_chat(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>, body: &str) -> Result<()> {
    let prompt = extract_last_message_content(body).unwrap_or_default();
    let max_tokens = extract_json_number(body, "max_tokens").unwrap_or(128) as usize;
    let temperature = extract_json_float(body, "temperature").unwrap_or(0.7);

    let config = InferenceConfig {
        temperature,
        top_k: 40,
        top_p: extract_json_float(body, "top_p").unwrap_or(0.9),
        max_tokens,
    };

    let mut ctx = ctx.lock().unwrap();
    let ModelContext { ref gguf, ref streamer, ref mut cache, ref model_name, .. } = *ctx;
    let _start = Instant::now();

    match transformer::generate(gguf, streamer, cache, &prompt, &config) {
        Ok(result) => {
            let resp = format!(
                r#"{{"id":"chatcmpl-ssd","object":"chat.completion","created":{},"model":"{}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":0,"completion_tokens":{},"total_tokens":{}}}}}"#,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                model_name,
                escape_json(&result.text),
                result.token_count,
                result.token_count,
            );
            send_json_response(stream, 200, &resp)
        }
        Err(e) => {
            let resp = format!(r#"{{"error":{{"message":"{}","type":"server_error"}}}}"#, escape_json(&e.to_string()));
            send_json_response(stream, 500, &resp)
        }
    }
}

// --- Helpers ---

fn send_response(stream: &mut TcpStream, status: u16, body: &str) -> Result<()> {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    };
    let resp = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status, status_text, body.len(), body
    );
    stream.write_all(resp.as_bytes())?;
    Ok(())
}

fn send_json_response(stream: &mut TcpStream, status: u16, json: &str) -> Result<()> {
    let resp = format!(
        "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        status, json.len(), json
    );
    stream.write_all(resp.as_bytes())?;
    Ok(())
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn chrono_now() -> String {
    // Simple ISO 8601 timestamp
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}Z", duration.as_secs())
}

/// Extract a string field from JSON (minimal parser, no serde dependency)
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!(r#""{}""#, key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start().strip_prefix(':')?;
    let after = after.trim_start().strip_prefix('"')?;
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

fn extract_json_number(json: &str, key: &str) -> Option<i64> {
    let pattern = format!(r#""{}""#, key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start().strip_prefix(':')?;
    let after = after.trim_start();
    let end = after.find(|c: char| !c.is_ascii_digit() && c != '-')?;
    after[..end].parse().ok()
}

fn extract_json_float(json: &str, key: &str) -> Option<f32> {
    let pattern = format!(r#""{}""#, key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start().strip_prefix(':')?;
    let after = after.trim_start();
    let end = after.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').unwrap_or(after.len());
    after[..end].parse().ok()
}

/// Extract the content of the last message in a messages array
fn extract_last_message_content(json: &str) -> Option<String> {
    // Find all "content" fields and take the last one
    let mut last_content = None;
    let mut search_from = 0;
    while let Some(pos) = json[search_from..].find(r#""content""#) {
        let abs_pos = search_from + pos;
        let after = &json[abs_pos + 9..]; // len of "content"
        if let Some(content) = after.trim_start().strip_prefix(':') {
            let content = content.trim_start();
            if let Some(content) = content.strip_prefix('"') {
                if let Some(end) = content.find('"') {
                    last_content = Some(content[..end].to_string());
                }
            }
        }
        search_from = abs_pos + 9;
    }
    last_content
}
