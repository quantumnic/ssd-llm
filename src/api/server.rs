//! Ollama-compatible HTTP API server
//!
//! v0.4: Added streaming responses (chunked transfer encoding).
//! Implements the Ollama REST API for drop-in compatibility:
//! - POST /api/generate — text generation (streaming + non-streaming)
//! - POST /api/chat — chat completions (streaming + non-streaming)
//! - GET /api/tags — list loaded models
//! - GET /api/version — server version
//! - POST /v1/chat/completions — OpenAI-compatible endpoint (streaming SSE)

use crate::inference::transformer::{self, InferenceConfig};
use crate::model::cache::LayerCache;
use crate::model::gguf::GgufFile;
use crate::ssd::streamer::SsdStreamer;
use anyhow::Result;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
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
    pub adaptive_draft: bool,
    pub prompt_cache: bool,
    pub max_batch: usize,
    pub tensor_parallel: usize,
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
        info!(
            "OpenAI-compatible API: POST /v1/chat/completions, POST /v1/embeddings, GET /v1/models"
        );
        info!("Streaming: supported (default for generate/chat)");

        // Load model
        info!("Loading model: {}", self.config.model_path.display());
        let gguf = GgufFile::open(&self.config.model_path)?;
        let streamer = SsdStreamer::new(&self.config.model_path, self.config.memory_budget)?;
        let cache = LayerCache::new(self.config.memory_budget);

        let model_name = self
            .config
            .model_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        info!(
            "Model loaded: {} ({} layers, {} embd, {} vocab)",
            model_name,
            gguf.n_layers(),
            gguf.n_embd(),
            gguf.vocab_size()
        );

        let ctx = Arc::new(Mutex::new(ModelContext {
            gguf,
            streamer,
            cache,
            model_name,
            model_path: self.config.model_path.clone(),
        }));

        // Graceful shutdown via SIGINT/SIGTERM
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);

        // Set up signal handler
        ctrlc_register(move || {
            warn!("Shutdown signal received, stopping server...");
            shutdown_clone.store(true, Ordering::SeqCst);
        });

        // Set non-blocking so we can check shutdown flag
        listener.set_nonblocking(true)?;

        info!("Server ready. Press Ctrl+C to stop.");

        while !shutdown.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((stream, _addr)) => {
                    let ctx = Arc::clone(&ctx);
                    if let Err(e) = handle_connection(stream, &ctx) {
                        error!("Request error: {}", e);
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => error!("Accept error: {}", e),
            }
        }

        info!("Server stopped gracefully.");
        Ok(())
    }
}

/// Register a ctrl-c handler (simplified, no external crate)
fn ctrlc_register<F: FnOnce() + Send + 'static>(handler: F) {
    let handler = std::sync::Mutex::new(Some(handler));
    unsafe {
        libc::signal(libc::SIGINT, (signal_handler as *const ()) as usize);
        libc::signal(libc::SIGTERM, (signal_handler as *const ()) as usize);
    }
    // Store handler in a static
    *SHUTDOWN_HANDLER.lock().unwrap() = Some(Box::new(move || {
        if let Some(f) = handler.lock().unwrap().take() {
            f();
        }
    }));
}

static SHUTDOWN_HANDLER: std::sync::Mutex<Option<Box<dyn FnOnce() + Send>>> =
    std::sync::Mutex::new(None);

extern "C" fn signal_handler(_sig: libc::c_int) {
    if let Some(handler) = SHUTDOWN_HANDLER.lock().unwrap().take() {
        handler();
    }
}

fn handle_connection(mut stream: TcpStream, ctx: &Arc<Mutex<ModelContext>>) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;

    let parts: Vec<&str> = request_line.split_whitespace().collect();
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
        if trimmed.is_empty() {
            break;
        }
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
        ("POST", "/v1/embeddings") => handle_embeddings(&mut stream, ctx, &body_str),
        ("GET", "/v1/models") => handle_openai_models(&mut stream, ctx),
        ("GET", "/health") => handle_health(&mut stream, ctx),
        ("GET", "/metrics") => handle_metrics(&mut stream, ctx),
        ("OPTIONS", _) => handle_cors_preflight(&mut stream),
        ("GET", "/") => send_json_response(
            &mut stream,
            200,
            r#"{"status":"ssd-llm is running","version":"1.1.0"}"#,
        ),
        _ => send_response(&mut stream, 404, "Not Found"),
    }
}

fn handle_tags(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>) -> Result<()> {
    let ctx = ctx.lock().unwrap();
    let resp = format!(
        r#"{{"models":[{{"name":"{}","size":{},"digest":"local","details":{{"family":"{}","parameter_size":"{}B","quantization_level":"local"}}}}]}}"#,
        ctx.model_name,
        ctx.gguf.file_size,
        ctx.gguf.architecture(),
        ctx.gguf.n_layers(),
    );
    send_json_response(stream, 200, &resp)
}

fn handle_cors_preflight(stream: &mut TcpStream) -> Result<()> {
    let resp = "HTTP/1.1 204 No Content\r\n\
        Access-Control-Allow-Origin: *\r\n\
        Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
        Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
        Access-Control-Max-Age: 86400\r\n\
        Content-Length: 0\r\n\
        Connection: close\r\n\r\n";
    stream.write_all(resp.as_bytes())?;
    Ok(())
}

fn handle_version(stream: &mut TcpStream) -> Result<()> {
    send_json_response(stream, 200, r#"{"version":"1.1.0-ssd-llm"}"#)
}

fn handle_health(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>) -> Result<()> {
    let ctx = ctx.lock().unwrap();
    let metrics = crate::api::metrics::MetricsCollector::new();
    let json = metrics.health_json(&ctx.model_name, true);
    send_json_response(stream, 200, &json)
}

fn handle_metrics(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>) -> Result<()> {
    let ctx = ctx.lock().unwrap();
    let metrics = crate::api::metrics::MetricsCollector::new();
    // Check Accept header for prometheus format (simplified — always return JSON for now)
    let json = metrics.metrics_json(&ctx.model_name, 0);
    send_json_response(stream, 200, &json)
}

fn handle_generate(
    stream: &mut TcpStream,
    ctx: &Arc<Mutex<ModelContext>>,
    body: &str,
) -> Result<()> {
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
        stop_sequences: Vec::new(),
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
    };

    if streaming {
        return handle_generate_streaming(stream, ctx, &prompt, &config);
    }

    let mut ctx = ctx.lock().unwrap();
    let ModelContext {
        ref gguf,
        ref streamer,
        ref mut cache,
        ref model_name,
        ..
    } = *ctx;
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
fn handle_generate_streaming(
    stream: &mut TcpStream,
    ctx: &Arc<Mutex<ModelContext>>,
    prompt: &str,
    config: &InferenceConfig,
) -> Result<()> {
    // Send chunked response header
    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n")?;

    let mut ctx = ctx.lock().unwrap();
    let ModelContext {
        ref gguf,
        ref streamer,
        ref mut cache,
        ref model_name,
        ..
    } = *ctx;
    let start = Instant::now();

    // Use the streaming generator
    match transformer::generate_streaming(gguf, streamer, cache, prompt, config) {
        Ok(mut gen) => {
            let mut total_tokens = 0usize;
            while let Some(token_text) = gen.next_token()? {
                total_tokens += 1;
                let chunk = format!(
                    r#"{{"model":"{}","created_at":"{}","response":"{}","done":false}}"#,
                    model_name,
                    chrono_now(),
                    escape_json(&token_text),
                );
                write_chunk(stream, &chunk)?;
                let _ = stream.flush();
            }
            // Final message
            let elapsed = start.elapsed();
            let final_chunk = format!(
                r#"{{"model":"{}","created_at":"{}","response":"","done":true,"total_duration":{},"eval_count":{},"eval_duration":{}}}"#,
                model_name,
                chrono_now(),
                elapsed.as_nanos(),
                total_tokens,
                elapsed.as_nanos(),
            );
            write_chunk(stream, &final_chunk)?;
            write_chunk(stream, "")?; // empty chunk = end
            Ok(())
        }
        Err(e) => {
            let chunk = format!(
                r#"{{"error":"{}","done":true}}"#,
                escape_json(&e.to_string())
            );
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
        stop_sequences: extract_json_string_array(body, "stop").unwrap_or_default(),
        repetition_penalty: extract_json_float(body, "repeat_penalty").unwrap_or(1.0),
        frequency_penalty: extract_json_float(body, "frequency_penalty").unwrap_or(0.0),
        presence_penalty: extract_json_float(body, "presence_penalty").unwrap_or(0.0),
    };

    if streaming {
        return handle_chat_streaming(stream, ctx, &prompt, &config);
    }

    let mut ctx = ctx.lock().unwrap();
    let ModelContext {
        ref gguf,
        ref streamer,
        ref mut cache,
        ref model_name,
        ..
    } = *ctx;
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

/// Handle streaming chat — Ollama format
fn handle_chat_streaming(
    stream: &mut TcpStream,
    ctx: &Arc<Mutex<ModelContext>>,
    prompt: &str,
    config: &InferenceConfig,
) -> Result<()> {
    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n")?;

    let mut ctx = ctx.lock().unwrap();
    let ModelContext {
        ref gguf,
        ref streamer,
        ref mut cache,
        ref model_name,
        ..
    } = *ctx;
    let start = Instant::now();

    match transformer::generate_streaming(gguf, streamer, cache, prompt, config) {
        Ok(mut gen) => {
            let mut total_tokens = 0usize;
            while let Some(token_text) = gen.next_token()? {
                total_tokens += 1;
                let chunk = format!(
                    r#"{{"model":"{}","created_at":"{}","message":{{"role":"assistant","content":"{}"}},"done":false}}"#,
                    model_name,
                    chrono_now(),
                    escape_json(&token_text),
                );
                write_chunk(stream, &chunk)?;
                let _ = stream.flush();
            }
            let elapsed = start.elapsed();
            let final_chunk = format!(
                r#"{{"model":"{}","created_at":"{}","message":{{"role":"assistant","content":""}},"done":true,"total_duration":{},"eval_count":{}}}"#,
                model_name,
                chrono_now(),
                elapsed.as_nanos(),
                total_tokens,
            );
            write_chunk(stream, &final_chunk)?;
            write_chunk(stream, "")?;
            Ok(())
        }
        Err(e) => {
            let chunk = format!(
                r#"{{"error":"{}","done":true}}"#,
                escape_json(&e.to_string())
            );
            write_chunk(stream, &chunk)?;
            write_chunk(stream, "")?;
            Ok(())
        }
    }
}

fn handle_openai_chat(
    stream: &mut TcpStream,
    ctx: &Arc<Mutex<ModelContext>>,
    body: &str,
) -> Result<()> {
    use crate::inference::chat_template::{format_chat, ChatTemplateFormat};

    // Parse all messages and format with appropriate chat template
    let messages = extract_chat_messages(body);
    let prompt = if messages.is_empty() {
        extract_last_message_content(body).unwrap_or_default()
    } else {
        // Detect template from model name
        let model_name = {
            let ctx = ctx.lock().unwrap();
            ctx.model_name.clone()
        };
        let template_name = extract_json_string(body, "chat_template");
        let format = if let Some(ref tmpl) = template_name {
            ChatTemplateFormat::from_gguf_template(tmpl)
        } else {
            ChatTemplateFormat::from_model_name(&model_name)
        };
        format_chat(&messages, format)
    };

    let max_tokens = extract_json_number(body, "max_tokens").unwrap_or(128) as usize;
    let temperature = extract_json_float(body, "temperature").unwrap_or(0.7);
    let streaming = extract_json_bool(body, "stream").unwrap_or(false);

    let config = InferenceConfig {
        temperature,
        top_k: 40,
        top_p: extract_json_float(body, "top_p").unwrap_or(0.9),
        max_tokens,
        stop_sequences: extract_json_string_array(body, "stop").unwrap_or_default(),
        repetition_penalty: 1.0,
        frequency_penalty: extract_json_float(body, "frequency_penalty").unwrap_or(0.0),
        presence_penalty: extract_json_float(body, "presence_penalty").unwrap_or(0.0),
    };

    if streaming {
        return handle_openai_streaming(stream, ctx, &prompt, &config);
    }

    let mut ctx = ctx.lock().unwrap();
    let ModelContext {
        ref gguf,
        ref streamer,
        ref mut cache,
        ref model_name,
        ..
    } = *ctx;

    match transformer::generate(gguf, streamer, cache, &prompt, &config) {
        Ok(result) => {
            let resp = format!(
                r#"{{"id":"chatcmpl-ssd","object":"chat.completion","created":{},"model":"{}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
                unix_timestamp(),
                model_name,
                escape_json(&result.text),
                result.prompt_tokens,
                result.token_count,
                result.prompt_tokens + result.token_count,
            );
            send_json_response(stream, 200, &resp)
        }
        Err(e) => {
            let resp = format!(
                r#"{{"error":{{"message":"{}","type":"server_error"}}}}"#,
                escape_json(&e.to_string())
            );
            send_json_response(stream, 500, &resp)
        }
    }
}

/// Handle streaming OpenAI chat — SSE format
fn handle_openai_streaming(
    stream: &mut TcpStream,
    ctx: &Arc<Mutex<ModelContext>>,
    prompt: &str,
    config: &InferenceConfig,
) -> Result<()> {
    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n")?;

    let mut ctx = ctx.lock().unwrap();
    let ModelContext {
        ref gguf,
        ref streamer,
        ref mut cache,
        ref model_name,
        ..
    } = *ctx;

    match transformer::generate_streaming(gguf, streamer, cache, prompt, config) {
        Ok(mut gen) => {
            while let Some(token_text) = gen.next_token()? {
                let chunk = format!(
                    r#"{{"id":"chatcmpl-ssd","object":"chat.completion.chunk","created":{},"model":"{}","choices":[{{"index":0,"delta":{{"content":"{}"}},"finish_reason":null}}]}}"#,
                    unix_timestamp(),
                    model_name,
                    escape_json(&token_text),
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
            let err = format!(
                r#"{{"error":{{"message":"{}"}}}}"#,
                escape_json(&e.to_string())
            );
            write!(stream, "data: {}\n\n", err)?;
            Ok(())
        }
    }
}

fn handle_embeddings(
    stream: &mut TcpStream,
    ctx: &Arc<Mutex<ModelContext>>,
    body: &str,
) -> Result<()> {
    // Support both "input": "string" and "input": ["string", ...]
    let inputs = extract_embedding_inputs(body);
    if inputs.is_empty() {
        return send_json_response(
            stream,
            400,
            r#"{"error":{"message":"'input' is required","type":"invalid_request_error"}}"#,
        );
    }

    let mut ctx = ctx.lock().unwrap();
    let ModelContext {
        ref gguf,
        ref streamer,
        ref mut cache,
        ref model_name,
        ..
    } = *ctx;

    let mut data_entries = Vec::new();
    let mut total_tokens = 0usize;

    for (idx, input) in inputs.iter().enumerate() {
        match transformer::embed(gguf, streamer, cache, input, true) {
            Ok(result) => {
                total_tokens += result.prompt_tokens;
                // Format embedding array as JSON
                let emb_json: Vec<String> = result
                    .embedding
                    .iter()
                    .map(|v| format!("{:.8}", v))
                    .collect();
                data_entries.push(format!(
                    r#"{{"object":"embedding","embedding":[{}],"index":{}}}"#,
                    emb_json.join(","),
                    idx
                ));
            }
            Err(e) => {
                let resp = format!(
                    r#"{{"error":{{"message":"{}","type":"server_error"}}}}"#,
                    escape_json(&e.to_string())
                );
                return send_json_response(stream, 500, &resp);
            }
        }
    }

    let resp = format!(
        r#"{{"object":"list","data":[{}],"model":"{}","usage":{{"prompt_tokens":{},"total_tokens":{}}}}}"#,
        data_entries.join(","),
        model_name,
        total_tokens,
        total_tokens,
    );
    send_json_response(stream, 200, &resp)
}

fn handle_openai_models(stream: &mut TcpStream, ctx: &Arc<Mutex<ModelContext>>) -> Result<()> {
    let ctx = ctx.lock().unwrap();
    let resp = format!(
        r#"{{"object":"list","data":[{{"id":"{}","object":"model","created":{},"owned_by":"ssd-llm"}}]}}"#,
        ctx.model_name,
        unix_timestamp(),
    );
    send_json_response(stream, 200, &resp)
}

/// Extract embedding inputs — supports "input": "text" and "input": ["text1", "text2"]
fn extract_embedding_inputs(json: &str) -> Vec<String> {
    let mut results = Vec::new();
    if let Some(pos) = json.find(r#""input""#) {
        let after = &json[pos + 7..];
        let after = after.trim_start();
        if let Some(after) = after.strip_prefix(':') {
            let after = after.trim_start();
            if after.starts_with('[') {
                // Array of strings
                if let Some(end) = after.find(']') {
                    let arr = &after[1..end];
                    let mut in_string = false;
                    let mut current = String::new();
                    let mut escaped = false;
                    for ch in arr.chars() {
                        if escaped {
                            current.push(ch);
                            escaped = false;
                            continue;
                        }
                        match ch {
                            '\\' if in_string => {
                                escaped = true;
                                current.push(ch);
                            }
                            '"' => {
                                if in_string {
                                    results.push(current.clone());
                                    current.clear();
                                }
                                in_string = !in_string;
                            }
                            _ if in_string => current.push(ch),
                            _ => {}
                        }
                    }
                }
            } else if let Some(inner) = after.strip_prefix('"') {
                // Single string
                if let Some(end) = inner.find('"') {
                    results.push(inner[..end].to_string());
                }
            }
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_embedding_inputs_single_string() {
        let json = r#"{"input": "hello world", "model": "test"}"#;
        let inputs = extract_embedding_inputs(json);
        assert_eq!(inputs, vec!["hello world"]);
    }

    #[test]
    fn test_extract_embedding_inputs_array() {
        let json = r#"{"input": ["hello", "world"], "model": "test"}"#;
        let inputs = extract_embedding_inputs(json);
        assert_eq!(inputs, vec!["hello", "world"]);
    }

    #[test]
    fn test_extract_embedding_inputs_empty() {
        let json = r#"{"model": "test"}"#;
        let inputs = extract_embedding_inputs(json);
        assert!(inputs.is_empty());
    }

    #[test]
    fn test_extract_embedding_inputs_single_in_array() {
        let json = r#"{"input": ["only one"]}"#;
        let inputs = extract_embedding_inputs(json);
        assert_eq!(inputs, vec!["only one"]);
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
    let end = after
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(after.len());
    after[..end].parse().ok()
}

fn extract_json_bool(json: &str, key: &str) -> Option<bool> {
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

/// Extract an array of strings from JSON, e.g. "stop": ["foo", "bar"]
fn extract_json_string_array(json: &str, key: &str) -> Option<Vec<String>> {
    let pattern = format!(r#""{}""#, key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start().strip_prefix(':')?;
    let after = after.trim_start();

    // Handle single string value
    if after.starts_with('"') {
        let inner = after.strip_prefix('"')?;
        let end = inner.find('"')?;
        return Some(vec![inner[..end].to_string()]);
    }

    // Handle array
    if !after.starts_with('[') {
        return None;
    }
    let bracket_end = after.find(']')?;
    let array_content = &after[1..bracket_end];

    let mut result = Vec::new();
    let mut search = array_content;
    while let Some(start) = search.find('"') {
        let inner = &search[start + 1..];
        if let Some(end) = inner.find('"') {
            result.push(inner[..end].to_string());
            search = &inner[end + 1..];
        } else {
            break;
        }
    }
    Some(result)
}

/// Extract all chat messages (role + content pairs) from an OpenAI messages array
fn extract_chat_messages(json: &str) -> Vec<crate::api::openai::ChatMessage> {
    use crate::api::openai::{ChatMessage, Role};

    let mut messages = Vec::new();

    // Find "messages" array
    let Some(msgs_pos) = json.find(r#""messages""#) else {
        return messages;
    };
    let after = &json[msgs_pos + 10..];
    let Some(arr_start) = after.find('[') else {
        return messages;
    };
    let arr_content = &after[arr_start..];

    // Find each message object by looking for "role" keys
    let mut search_from = 0;
    while let Some(role_pos) = arr_content[search_from..].find(r#""role""#) {
        let abs_pos = search_from + role_pos;
        let role_after = &arr_content[abs_pos + 6..];
        let role_str = role_after
            .trim_start()
            .strip_prefix(':')
            .and_then(|s| s.trim_start().strip_prefix('"'))
            .and_then(|s| s.find('"').map(|end| &s[..end]));

        // Find matching "content" after this "role"
        let content_after = &arr_content[abs_pos..];
        let content_str = content_after.find(r#""content""#).and_then(|cp| {
            let ca = &content_after[cp + 9..];
            ca.trim_start().strip_prefix(':').and_then(|s| {
                let s = s.trim_start();
                s.strip_prefix('"')
                    .and_then(|s| s.find('"').map(|end| &s[..end]))
            })
        });

        if let (Some(role), Some(content)) = (role_str, content_str) {
            messages.push(ChatMessage {
                role: Role::from_str(role),
                content: content.to_string(),
            });
        }

        search_from = abs_pos + 6;
    }

    messages
}
