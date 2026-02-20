//! OpenAI-compatible API types and helpers
//!
//! Shared types for the OpenAI chat completions API format.
//! The actual endpoint handler is in server.rs.

/// OpenAI chat message role
#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        }
    }
}

/// A chat message
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// Format a list of chat messages into a single prompt string
/// Uses the ChatML format: <|im_start|>role\ncontent<|im_end|>
pub fn format_chat_prompt(messages: &[ChatMessage]) -> String {
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
