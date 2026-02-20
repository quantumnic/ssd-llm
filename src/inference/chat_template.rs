//! Chat template system for formatting multi-turn conversations
//!
//! Supports auto-detection from GGUF metadata and common model families:
//! - ChatML (Qwen, Yi, OpenHermes, etc.)
//! - Llama 2 (Meta Llama 2 chat)
//! - Llama 3 (Meta Llama 3/3.1 Instruct)
//! - Mistral/Mixtral ([INST] format)
//! - Gemma (Google Gemma Instruct)
//! - Phi-3 (<|user|> format)
//! - Zephyr (same as ChatML with system handled separately)
//! - Raw (no template, just concatenate content)

use crate::api::openai::{ChatMessage, Role};

/// Known chat template formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChatTemplateFormat {
    /// `<|im_start|>role\ncontent<|im_end|>`
    ChatML,
    /// `[INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] assistant`
    Llama2,
    /// `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,
    /// `[INST] user [/INST] assistant</s>`
    Mistral,
    /// `<start_of_turn>role\ncontent<end_of_turn>`
    Gemma,
    /// `<|user|>\ncontent<|end|>\n<|assistant|>\n`
    Phi3,
    /// No template — just concatenate message content
    Raw,
}

impl ChatTemplateFormat {
    /// Try to detect template format from GGUF metadata chat_template string
    pub fn from_gguf_template(template: &str) -> Self {
        let t = template.to_lowercase();
        if t.contains("<|im_start|>") {
            ChatTemplateFormat::ChatML
        } else if t.contains("<|start_header_id|>") {
            ChatTemplateFormat::Llama3
        } else if t.contains("<<sys>>") {
            ChatTemplateFormat::Llama2
        } else if t.contains("[inst]") {
            ChatTemplateFormat::Mistral
        } else if t.contains("<start_of_turn>") {
            ChatTemplateFormat::Gemma
        } else if t.contains("<|user|>") {
            ChatTemplateFormat::Phi3
        } else {
            ChatTemplateFormat::ChatML // default fallback
        }
    }

    /// Try to detect from model name/path
    pub fn from_model_name(name: &str) -> Self {
        let n = name.to_lowercase();
        if n.contains("llama-3") || n.contains("llama3") {
            ChatTemplateFormat::Llama3
        } else if n.contains("llama-2") || n.contains("llama2") {
            ChatTemplateFormat::Llama2
        } else if n.contains("mistral") || n.contains("mixtral") {
            ChatTemplateFormat::Mistral
        } else if n.contains("gemma") {
            ChatTemplateFormat::Gemma
        } else if n.contains("phi-3") || n.contains("phi3") {
            ChatTemplateFormat::Phi3
        } else {
            // ChatML is a safe default (also used by Qwen, Yi, OpenHermes)
            ChatTemplateFormat::ChatML
        }
    }

    /// Get the stop sequences for this template format
    pub fn stop_sequences(&self) -> Vec<String> {
        match self {
            ChatTemplateFormat::ChatML => vec!["<|im_end|>".to_string()],
            ChatTemplateFormat::Llama2 => vec!["</s>".to_string()],
            ChatTemplateFormat::Llama3 => vec!["<|eot_id|>".to_string()],
            ChatTemplateFormat::Mistral => vec!["</s>".to_string()],
            ChatTemplateFormat::Gemma => vec!["<end_of_turn>".to_string()],
            ChatTemplateFormat::Phi3 => vec!["<|end|>".to_string()],
            ChatTemplateFormat::Raw => vec![],
        }
    }
}

/// Format chat messages using the specified template
pub fn format_chat(messages: &[ChatMessage], format: ChatTemplateFormat) -> String {
    match format {
        ChatTemplateFormat::ChatML => format_chatml(messages),
        ChatTemplateFormat::Llama2 => format_llama2(messages),
        ChatTemplateFormat::Llama3 => format_llama3(messages),
        ChatTemplateFormat::Mistral => format_mistral(messages),
        ChatTemplateFormat::Gemma => format_gemma(messages),
        ChatTemplateFormat::Phi3 => format_phi3(messages),
        ChatTemplateFormat::Raw => format_raw(messages),
    }
}

fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role.as_str(),
            msg.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn format_llama2(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    let system_msg = messages
        .iter()
        .find(|m| m.role == Role::System)
        .map(|m| m.content.as_str());

    let non_system: Vec<&ChatMessage> =
        messages.iter().filter(|m| m.role != Role::System).collect();

    for (i, msg) in non_system.iter().enumerate() {
        match msg.role {
            Role::User => {
                prompt.push_str("<s>[INST] ");
                if i == 0 {
                    if let Some(sys) = system_msg {
                        prompt.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", sys));
                    }
                }
                prompt.push_str(&format!("{} [/INST]", msg.content));
            }
            Role::Assistant => {
                prompt.push_str(&format!(" {}</s>", msg.content));
            }
            _ => {}
        }
    }
    prompt
}

fn format_llama3(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for msg in messages {
        prompt.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role.as_str(),
            msg.content
        ));
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

fn format_mistral(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    let mut user_content = String::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                // Mistral has no system token; prepend to first user message
                user_content.push_str(&msg.content);
                user_content.push_str("\n\n");
            }
            Role::User => {
                user_content.push_str(&msg.content);
                prompt.push_str(&format!("[INST] {} [/INST]", user_content));
                user_content.clear();
            }
            Role::Assistant => {
                prompt.push_str(&format!(" {}</s> ", msg.content));
            }
            Role::Tool => {
                prompt.push_str(&format!("[INST] Tool result: {} [/INST]", msg.content));
            }
        }
    }
    prompt
}

fn format_gemma(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role {
            Role::System | Role::User | Role::Tool => "user",
            Role::Assistant => "model",
        };
        prompt.push_str(&format!(
            "<start_of_turn>{}\n{}<end_of_turn>\n",
            role, msg.content
        ));
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

fn format_phi3(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        prompt.push_str(&format!("<|{}|>\n{}<|end|>\n", role, msg.content));
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

fn format_raw(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: Role::System,
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "Hello".to_string(),
            },
        ]
    }

    #[test]
    fn test_chatml_format() {
        let msgs = make_messages();
        let result = format_chat(&msgs, ChatTemplateFormat::ChatML);
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_llama3_format() {
        let msgs = make_messages();
        let result = format_chat(&msgs, ChatTemplateFormat::Llama3);
        assert!(result.starts_with("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(result.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_llama2_format() {
        let msgs = make_messages();
        let result = format_chat(&msgs, ChatTemplateFormat::Llama2);
        assert!(result.contains("<<SYS>>"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("[INST]"));
        assert!(result.contains("[/INST]"));
    }

    #[test]
    fn test_mistral_format() {
        let msgs = make_messages();
        let result = format_chat(&msgs, ChatTemplateFormat::Mistral);
        assert!(result.contains("[INST]"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("Hello"));
    }

    #[test]
    fn test_gemma_format() {
        let msgs = make_messages();
        let result = format_chat(&msgs, ChatTemplateFormat::Gemma);
        assert!(result.contains("<start_of_turn>user"));
        assert!(result.contains("<end_of_turn>"));
        assert!(result.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_phi3_format() {
        let msgs = make_messages();
        let result = format_chat(&msgs, ChatTemplateFormat::Phi3);
        assert!(result.contains("<|system|>"));
        assert!(result.contains("<|user|>"));
        assert!(result.contains("<|end|>"));
        assert!(result.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_detect_from_gguf_template() {
        assert_eq!(
            ChatTemplateFormat::from_gguf_template("{% for m in messages %}<|im_start|>"),
            ChatTemplateFormat::ChatML
        );
        assert_eq!(
            ChatTemplateFormat::from_gguf_template("<|start_header_id|>{{role}}"),
            ChatTemplateFormat::Llama3
        );
        assert_eq!(
            ChatTemplateFormat::from_gguf_template("<<SYS>>system<</SYS>>"),
            ChatTemplateFormat::Llama2
        );
        assert_eq!(
            ChatTemplateFormat::from_gguf_template("[INST] {{user}} [/INST]"),
            ChatTemplateFormat::Mistral
        );
    }

    #[test]
    fn test_detect_from_model_name() {
        assert_eq!(
            ChatTemplateFormat::from_model_name("Meta-Llama-3.1-8B-Instruct"),
            ChatTemplateFormat::Llama3
        );
        assert_eq!(
            ChatTemplateFormat::from_model_name("Mistral-7B-Instruct-v0.3"),
            ChatTemplateFormat::Mistral
        );
        assert_eq!(
            ChatTemplateFormat::from_model_name("gemma-2-9b-it"),
            ChatTemplateFormat::Gemma
        );
    }

    #[test]
    fn test_stop_sequences() {
        assert_eq!(
            ChatTemplateFormat::ChatML.stop_sequences(),
            vec!["<|im_end|>"]
        );
        assert_eq!(
            ChatTemplateFormat::Llama3.stop_sequences(),
            vec!["<|eot_id|>"]
        );
        assert!(ChatTemplateFormat::Raw.stop_sequences().is_empty());
    }

    #[test]
    fn test_raw_format() {
        let msgs = make_messages();
        let result = format_chat(&msgs, ChatTemplateFormat::Raw);
        assert_eq!(result, "You are helpful.\nHello");
    }

    #[test]
    fn test_multi_turn_chatml() {
        let msgs = vec![
            ChatMessage {
                role: Role::System,
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "Hi".to_string(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "Hello!".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "How are you?".to_string(),
            },
        ];
        let result = format_chat(&msgs, ChatTemplateFormat::ChatML);
        assert!(result.contains("<|im_start|>assistant\nHello!<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_multi_turn_llama2() {
        let msgs = vec![
            ChatMessage {
                role: Role::User,
                content: "Hi".to_string(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "Hello!".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "Bye".to_string(),
            },
        ];
        let result = format_chat(&msgs, ChatTemplateFormat::Llama2);
        assert!(result.contains("[INST] Hi [/INST]"));
        assert!(result.contains(" Hello!</s>"));
        assert!(result.contains("[INST] Bye [/INST]"));
    }
}
