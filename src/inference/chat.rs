//! Interactive chat CLI for multi-turn conversations
//!
//! Provides a REPL-style interface where users can have multi-turn conversations
//! with a loaded model. Maintains chat history and applies the appropriate chat
//! template format based on model detection.

use crate::api::openai::{ChatMessage, Role};
use crate::inference::chat_template::{format_chat, ChatTemplateFormat};
use crate::inference::transformer::{self, InferenceConfig};
use crate::model::cache::LayerCache;
use crate::model::gguf::GgufFile;
use crate::ssd::streamer::SsdStreamer;
use anyhow::Result;
use std::io::{self, BufRead, Write};

/// Configuration for interactive chat sessions
pub struct ChatConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
    pub system_prompt: Option<String>,
    pub template: ChatTemplateFormat,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            max_tokens: 512,
            system_prompt: None,
            template: ChatTemplateFormat::ChatML,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// Run an interactive chat session
pub fn run_interactive(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    cache: &mut LayerCache,
    config: &ChatConfig,
) -> Result<()> {
    let mut history: Vec<ChatMessage> = Vec::new();

    // Add system prompt if provided
    if let Some(ref system) = config.system_prompt {
        history.push(ChatMessage {
            role: Role::System,
            content: system.clone(),
        });
    }

    let template_stops = config.template.stop_sequences();

    println!("╭──────────────────────────────────────────╮");
    println!("│  ssd-llm interactive chat                │");
    println!("│  Type /help for commands, /quit to exit  │");
    println!(
        "│  Template: {:?}{} │",
        config.template,
        " ".repeat(22 - format!("{:?}", config.template).len())
    );
    println!("╰──────────────────────────────────────────╯");
    println!();

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        // Print prompt
        print!("\x1b[1;32myou>\x1b[0m ");
        io::stdout().flush()?;

        // Read user input
        let mut input = String::new();
        if reader.read_line(&mut input)? == 0 {
            // EOF
            println!("\nGoodbye!");
            break;
        }
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            match handle_command(&input, &mut history, config) {
                CommandResult::Continue => continue,
                CommandResult::Quit => {
                    println!("Goodbye! 👋");
                    break;
                }
                CommandResult::Error(msg) => {
                    println!("\x1b[1;31mError:\x1b[0m {}", msg);
                    continue;
                }
            }
        }

        // Add user message to history
        history.push(ChatMessage {
            role: Role::User,
            content: input,
        });

        // Format prompt with chat template
        let prompt = format_chat(&history, config.template);

        // Build inference config
        let inf_config = InferenceConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            max_tokens: config.max_tokens,
            stop_sequences: template_stops.clone(),
            repetition_penalty: config.repetition_penalty,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            tfs_z: 0.0,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
        };

        // Generate response with streaming output
        print!("\x1b[1;36massistant>\x1b[0m ");
        io::stdout().flush()?;

        match transformer::generate_streaming(gguf, streamer, cache, &prompt, &inf_config) {
            Ok(mut gen) => {
                let mut response = String::new();
                let start = std::time::Instant::now();
                let mut token_count = 0usize;

                while let Some(token_text) = gen.next_token()? {
                    print!("{}", token_text);
                    io::stdout().flush()?;
                    response.push_str(&token_text);
                    token_count += 1;
                }

                let elapsed = start.elapsed();
                let tps = if elapsed.as_secs_f64() > 0.0 {
                    token_count as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };

                println!();
                println!(
                    "\x1b[2m[{} tokens, {:.1} tok/s, {:.2}s]\x1b[0m",
                    token_count,
                    tps,
                    elapsed.as_secs_f64()
                );
                println!();

                // Add assistant response to history
                history.push(ChatMessage {
                    role: Role::Assistant,
                    content: response,
                });
            }
            Err(e) => {
                println!();
                println!("\x1b[1;31mGeneration error:\x1b[0m {}", e);
                println!();
                // Remove the user message since generation failed
                history.pop();
            }
        }
    }

    Ok(())
}

enum CommandResult {
    Continue,
    Quit,
    Error(String),
}

fn handle_command(cmd: &str, history: &mut Vec<ChatMessage>, config: &ChatConfig) -> CommandResult {
    let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
    let command = parts[0].to_lowercase();

    match command.as_str() {
        "/quit" | "/exit" | "/q" => CommandResult::Quit,
        "/help" | "/h" | "/?" => {
            println!("╭─ Commands ─────────────────────────────╮");
            println!("│ /help         Show this help            │");
            println!("│ /quit         Exit chat                 │");
            println!("│ /clear        Clear conversation history│");
            println!("│ /history      Show conversation history │");
            println!("│ /system <msg> Set system prompt          │");
            println!("│ /config       Show current settings      │");
            println!("│ /undo         Remove last exchange       │");
            println!("╰─────────────────────────────────────────╯");
            CommandResult::Continue
        }
        "/clear" | "/reset" => {
            let had_system = history
                .first()
                .map(|m| m.role == Role::System)
                .unwrap_or(false);
            let system_msg = if had_system {
                Some(history[0].clone())
            } else {
                None
            };
            history.clear();
            if let Some(msg) = system_msg {
                history.push(msg);
                println!("History cleared (system prompt retained).");
            } else {
                println!("History cleared.");
            }
            CommandResult::Continue
        }
        "/history" | "/hist" => {
            if history.is_empty() {
                println!("(empty history)");
            } else {
                for (i, msg) in history.iter().enumerate() {
                    let role_color = match msg.role {
                        Role::System => "\x1b[1;33m",
                        Role::User => "\x1b[1;32m",
                        Role::Assistant => "\x1b[1;36m",
                        Role::Tool => "\x1b[1;35m",
                    };
                    let preview = if msg.content.len() > 80 {
                        format!("{}...", &msg.content[..77])
                    } else {
                        msg.content.clone()
                    };
                    println!(
                        "  [{}] {}{}\x1b[0m: {}",
                        i,
                        role_color,
                        msg.role.as_str(),
                        preview
                    );
                }
            }
            CommandResult::Continue
        }
        "/system" => {
            if let Some(msg) = parts.get(1) {
                // Remove existing system prompt if any
                if history
                    .first()
                    .map(|m| m.role == Role::System)
                    .unwrap_or(false)
                {
                    history.remove(0);
                }
                history.insert(
                    0,
                    ChatMessage {
                        role: Role::System,
                        content: msg.to_string(),
                    },
                );
                println!("System prompt set.");
            } else {
                println!("Usage: /system <your system prompt>");
            }
            CommandResult::Continue
        }
        "/config" | "/settings" => {
            println!("╭─ Settings ─────────────────────────────╮");
            println!("│ Temperature:  {:<24} │", config.temperature);
            println!("│ Top-K:        {:<24} │", config.top_k);
            println!("│ Top-P:        {:<24} │", config.top_p);
            println!("│ Max tokens:   {:<24} │", config.max_tokens);
            println!("│ Template:     {:<24?} │", config.template);
            println!("│ Rep. penalty: {:<24} │", config.repetition_penalty);
            println!("│ History msgs: {:<24} │", history.len());
            println!("╰─────────────────────────────────────────╯");
            CommandResult::Continue
        }
        "/undo" => {
            // Remove last user + assistant pair
            let mut removed = 0;
            while let Some(last) = history.last() {
                if last.role == Role::System {
                    break;
                }
                history.pop();
                removed += 1;
                if removed >= 2 {
                    break;
                }
            }
            if removed > 0 {
                println!("Removed last {} message(s).", removed);
            } else {
                println!("Nothing to undo.");
            }
            CommandResult::Continue
        }
        _ => CommandResult::Error(format!("Unknown command: {}. Type /help for help.", cmd)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_config_default() {
        let config = ChatConfig::default();
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tokens, 512);
        assert_eq!(config.template, ChatTemplateFormat::ChatML);
    }

    #[test]
    fn test_handle_quit_commands() {
        let mut history = Vec::new();
        let config = ChatConfig::default();

        assert!(matches!(
            handle_command("/quit", &mut history, &config),
            CommandResult::Quit
        ));
        assert!(matches!(
            handle_command("/exit", &mut history, &config),
            CommandResult::Quit
        ));
        assert!(matches!(
            handle_command("/q", &mut history, &config),
            CommandResult::Quit
        ));
    }

    #[test]
    fn test_handle_unknown_command() {
        let mut history = Vec::new();
        let config = ChatConfig::default();

        assert!(matches!(
            handle_command("/foobar", &mut history, &config),
            CommandResult::Error(_)
        ));
    }

    #[test]
    fn test_undo_removes_messages() {
        let mut history = vec![
            ChatMessage {
                role: Role::User,
                content: "hello".to_string(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "hi there".to_string(),
            },
        ];
        let config = ChatConfig::default();
        handle_command("/undo", &mut history, &config);
        assert!(history.is_empty());
    }

    #[test]
    fn test_undo_preserves_system() {
        let mut history = vec![
            ChatMessage {
                role: Role::System,
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "hello".to_string(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "hi".to_string(),
            },
        ];
        let config = ChatConfig::default();
        handle_command("/undo", &mut history, &config);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::System);
    }

    #[test]
    fn test_clear_retains_system() {
        let mut history = vec![
            ChatMessage {
                role: Role::System,
                content: "system".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "user".to_string(),
            },
        ];
        let config = ChatConfig::default();
        handle_command("/clear", &mut history, &config);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::System);
    }

    #[test]
    fn test_system_command() {
        let mut history = Vec::new();
        let config = ChatConfig::default();
        handle_command("/system Be concise.", &mut history, &config);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::System);
        assert_eq!(history[0].content, "Be concise.");
    }

    #[test]
    fn test_system_replaces_existing() {
        let mut history = vec![ChatMessage {
            role: Role::System,
            content: "old".to_string(),
        }];
        let config = ChatConfig::default();
        handle_command("/system new prompt", &mut history, &config);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "new prompt");
    }
}
