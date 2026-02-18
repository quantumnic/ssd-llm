//! Ollama-compatible HTTP API server
//!
//! v0.4: Added streaming responses (chunked transfer encoding).
//! Implements the Ollama REST API for drop-in compatibility:
//! - POST /api/generate — text generation (streaming + non-streaming)
//! - POST /api/chat — chat completions (streaming + non-streaming)
//! - GET /api/tags — list loaded models
//! - GET /api/version — server version
//! - POST /v1/chat/completions — OpenAI-compatible endpoint (streaming SSE)

use crate::inference::transformer::{self, InferenceConfig, GenerationResult};
use crate::inference::tokenizer::SimpleTokenizer;
use crate::model::cache::LayerCache;
use crate::model::gguf::GgufFile;
use crate::ssd::streamer::SsdStreamer;
use anyhow::Result;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{error, info, warn};

/// Server configuration
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_path: PathBuf,
    pub memory_budget: usize,
    pub draft_model_path: Option<PathBuf>,
    pub draft_ahead: usize,
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
        info!("Streaming: supported (default for generate/chat)");

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
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() { break; }
        if let Some(val) = trimmed.strip_prefix("Content-Length:") {
            content_length = val.trim().parse().unwrap_or(0);
        }
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
            r#"{"status":"ssd-llm is running","version":"0.5.0"}"#),
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
    send_json_response(stream, 200, r#"{"version":"0.5.0-ssd-llm"}"#)
}

fn handle_generate(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>, body: &str) -> Result<()> {
    let prompt = extract_json_string(body, "prompt").unwrap_or_default();
    let max_tokens = extract_json_number(body, "num_predict").unwrap_or(128) as usize;
    let temperature = extract_json_float(body, "temperature").unwrap_or(0.7);
    let top_k = extract_json_number(body, "top_k").unwrap_or(40) as usize;
    let top_p = extract_json_float(body, "top_p").unwrap_or(0.9);
    let streaming = extract_json_bool(body, "stream").unwrap_or(true);

    let config = InferenceConfig {
        temperature,
        top_k,
        top_p,
        max_tokens,
    };

    if streaming {
        return handle_generate_streaming(stream, ctx, &prompt, &config);
    }

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

/// Handle streaming generate — sends one JSON object per token via chunked transfer
fn handle_generate_streaming(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>, prompt: &str, config: &InferenceConfig) -> Result<()> {
    // Send chunked response header
    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n")?;

    let mut ctx = ctx.lock().unwrap();
    let ModelContext { ref gguf, ref streamer, ref mut cache, ref model_name, .. } = *ctx;
    let start = Instant::now();

    // Use the streaming generator
    match transformer::generate_streaming(gguf, streamer, cache, prompt, config) {
        Ok(mut gen) => {
            let mut total_tokens = 0usize;
            while let Some(token_text) = gen.next_token()? {
                total_tokens += 1;
                let chunk = format!(
                    r#"{{"model":"{}","created_at":"{}","response":"{}","done":false}}"#,
                    model_name, chrono_now(), escape_json(&token_text),
                );
                write_chunk(stream, &chunk)?;
                let _ = stream.flush();
            }
            // Final message
            let elapsed = start.elapsed();
            let final_chunk = format!(
                r#"{{"model":"{}","created_at":"{}","response":"","done":true,"total_duration":{},"eval_count":{},"eval_duration":{}}}"#,
                model_name, chrono_now(), elapsed.as_nanos(), total_tokens, elapsed.as_nanos(),
            );
            write_chunk(stream, &final_chunk)?;
            write_chunk(stream, "")?; // empty chunk = end
            Ok(())
        }
        Err(e) => {
            let chunk = format!(r#"{{"error":"{}","done":true}}"#, escape_json(&e.to_string()));
            write_chunk(stream, &chunk)?;
            write_chunk(stream, "")?;
            Ok(())
        }
    }
}

fn handle_chat(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>, body: &str) -> Result<()> {
    let prompt = extract_last_message_content(body).unwrap_or_default();
    let streaming = extract_json_bool(body, "stream").unwrap_or(true);

    let config = InferenceConfig {
        temperature: extract_json_float(body, "temperature").unwrap_or(0.7),
        top_k: extract_json_number(body, "top_k").unwrap_or(40) as usize,
        top_p: extract_json_float(body, "top_p").unwrap_or(0.9),
        max_tokens: extract_json_number(body, "num_predict").unwrap_or(128) as usize,
    };

    if streaming {
        return handle_chat_streaming(stream, ctx, &prompt, &config);
    }

    let mut ctx = ctx.lock().unwrap();
    let ModelContext { ref gguf, ref streamer, ref mut cache, ref model_name, .. } = *ctx;
    let start = Instant::now();

    match transformer::generate(gguf, streamer, cache, &prompt, &config) {
        Ok(result) => {
            let elapsed = start.elapsed();
            let resp = format!(
                r#"{{"model":"{}","created_at":"{}","message":{{"role":"assistant","content":"{}"}},"done":true,"total_duration":{},"eval_count":{}}}"#,
                model_name, chrono_now(), escape_json(&result.text), elapsed.as_nanos(), result.token_count,
            );
            send_json_response(stream, 200, &resp)
        }
        Err(e) => {
            let resp = format!(r#"{{"error":"{}"}}"#, escape_json(&e.to_string()));
            send_json_response(stream, 500, &resp)
        }
    }
}

/// Handle streaming chat — Ollama format
fn handle_chat_streaming(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>, prompt: &str, config: &InferenceConfig) -> Result<()> {
    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n")?;

    let mut ctx = ctx.lock().unwrap();
    let ModelContext { ref gguf, ref streamer, ref mut cache, ref model_name, .. } = *ctx;
    let start = Instant::now();

    match transformer::generate_streaming(gguf, streamer, cache, prompt, config) {
        Ok(mut gen) => {
            let mut total_tokens = 0usize;
            while let Some(token_text) = gen.next_token()? {
                total_tokens += 1;
                let chunk = format!(
                    r#"{{"model":"{}","created_at":"{}","message":{{"role":"assistant","content":"{}"}},"done":false}}"#,
                    model_name, chrono_now(), escape_json(&token_text),
                );
                write_chunk(stream, &chunk)?;
                let _ = stream.flush();
            }
            let elapsed = start.elapsed();
            let final_chunk = format!(
                r#"{{"model":"{}","created_at":"{}","message":{{"role":"assistant","content":""}},"done":true,"total_duration":{},"eval_count":{}}}"#,
                model_name, chrono_now(), elapsed.as_nanos(), total_tokens,
            );
            write_chunk(stream, &final_chunk)?;
            write_chunk(stream, "")?;
            Ok(())
        }
        Err(e) => {
            let chunk = format!(r#"{{"error":"{}","done":true}}"#, escape_json(&e.to_string()));
            write_chunk(stream, &chunk)?;
            write_chunk(stream, "")?;
            Ok(())
        }
    }
}

fn handle_openai_chat(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>, body: &str) -> Result<()> {
    let prompt = extract_last_message_content(body).unwrap_or_default();
    let max_tokens = extract_json_number(body, "max_tokens").unwrap_or(128) as usize;
    let temperature = extract_json_float(body, "temperature").unwrap_or(0.7);
    let streaming = extract_json_bool(body, "stream").unwrap_or(false);

    let config = InferenceConfig {
        temperature,
        top_k: 40,
        top_p: extract_json_float(body, "top_p").unwrap_or(0.9),
        max_tokens,
    };

    if streaming {
        return handle_openai_streaming(stream, ctx, &prompt, &config);
    }

    let mut ctx = ctx.lock().unwrap();
    let ModelContext { ref gguf, ref streamer, ref mut cache, ref model_name, .. } = *ctx;

    match transformer::generate(gguf, streamer, cache, &prompt, &config) {
        Ok(result) => {
            let resp = format!(
                r#"{{"id":"chatcmpl-ssd","object":"chat.completion","created":{},"model":"{}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":0,"completion_tokens":{},"total_tokens":{}}}}}"#,
                unix_timestamp(), model_name, escape_json(&result.text),
                result.token_count, result.token_count,
            );
            send_json_response(stream, 200, &resp)
        }
        Err(e) => {
            let resp = format!(r#"{{"error":{{"message":"{}","type":"server_error"}}}}"#, escape_json(&e.to_string()));
            send_json_response(stream, 500, &resp)
        }
    }
}

/// Handle streaming OpenAI chat — SSE format
fn handle_openai_streaming(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>, prompt: &str, config: &InferenceConfig) -> Result<()> {
    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n")?;

    let mut ctx = ctx.lock().unwrap();
    let ModelContext { ref gguf, ref streamer, ref mut cache, ref model_name, .. } = *ctx;

    match transformer::generate_streaming(gguf, streamer, cache, prompt, config) {
        Ok(mut gen) => {
            while let Some(token_text) = gen.next_token()? {
                let chunk = format!(
                    r#"{{"id":"chatcmpl-ssd","object":"chat.completion.chunk","created":{},"model":"{}","choices":[{{"index":0,"delta":{{"content":"{}"}},"finish_reason":null}}]}}"#,
                    unix_timestamp(), model_name, escape_json(&token_text),
                );
                write!(stream, "data: {}\n\n", chunk)?;
                let _ = stream.flush();
            }
            // Send [DONE]
            write!(stream, "data: [DONE]\n\n")?;
            let _ = stream.flush();
            Ok(())
        }
        Err(e) => {
            let err = format!(r#"{{"error":{{"message":"{}"}}}}"#, escape_json(&e.to_string()));
            write!(stream, "data: {}\n\n", err)?;
            Ok(())
        }
    }
}

// --- Helpers ---

fn write_chunk(stream: &mut TcpStream, data: &str) -> Result<()> {
    if data.is_empty() {
        stream.write_all(b"0\r\n\r\n")?;
    } else {
        let line = format!("{}\n", data); // newline-delimited JSON
        write!(stream, "{:x}\r\n{}\r\n", line.len(), line)?;
    }
    Ok(())
}

fn send_response(stream: &mut TcpStream, status: u16, body: &str) -> Result<()> {
    let status_text = match status {
        200 => "OK", 400 => "Bad Request", 404 => "Not Found",
        500 => "Internal Server Error", _ => "Unknown",
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
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}Z", duration.as_secs())
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

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

fn extract_json_bool(json: &str, key: &str) -> Option<bool> {
    let pattern = format!(r#""{}""#, key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start().strip_prefix(':')?;
    let after = after.trim_start();
    if after.starts_with("true") { Some(true) }
    else if after.starts_with("false") { Some(false) }
    else { None }
}

fn extract_last_message_content(json: &str) -> Option<String> {
    let mut last_content = None;
    let mut search_from = 0;
    while let Some(pos) = json[search_from..].find(r#""content""#) {
        let abs_pos = search_from + pos;
        let after = &json[abs_pos + 9..];
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
