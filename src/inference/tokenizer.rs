//! Simple tokenizer for GGUF models
//! 
//! v0.1: Basic lookup tokenizer from GGUF vocabulary.
//! TODO v0.2: Full BPE tokenizer with merges

use crate::model::gguf::{GgufFile, MetadataValue};
use std::collections::HashMap;

pub struct SimpleTokenizer {
    /// token_id → token string
    id_to_token: Vec<String>,
    /// token string → token_id
    token_to_id: HashMap<String, u32>,
    /// BOS token ID
    pub bos_id: u32,
    /// EOS token ID
    pub eos_id: u32,
}

impl SimpleTokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let mut id_to_token = Vec::new();
        let mut token_to_id = HashMap::new();

        // Extract tokens from metadata
        if let Some(MetadataValue::Array(tokens)) = gguf.metadata.get("tokenizer.ggml.tokens") {
            for (i, token) in tokens.iter().enumerate() {
                let s = match token {
                    MetadataValue::String(s) => s.clone(),
                    _ => format!("[token_{}]", i),
                };
                token_to_id.insert(s.clone(), i as u32);
                id_to_token.push(s);
            }
        }

        let bos_id = gguf.metadata.get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(1);
        let eos_id = gguf.metadata.get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(2);

        Self { id_to_token, token_to_id, bos_id, eos_id }
    }

    /// Simple greedy tokenization (character/byte-level fallback)
    /// For v0.1, this does a simple longest-match tokenization
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos_id];
        let bytes = text.as_bytes();
        let mut pos = 0;

        while pos < bytes.len() {
            let mut best_len = 0;
            let mut best_id = None;

            // Try longest match first (up to 16 chars)
            for len in (1..=16.min(bytes.len() - pos)).rev() {
                if let Ok(substr) = std::str::from_utf8(&bytes[pos..pos + len]) {
                    if let Some(&id) = self.token_to_id.get(substr) {
                        best_len = len;
                        best_id = Some(id);
                        break;
                    }
                }
            }

            // Try with space prefix (common in SentencePiece)
            if best_id.is_none() {
                for len in (1..=16.min(bytes.len() - pos)).rev() {
                    if let Ok(substr) = std::str::from_utf8(&bytes[pos..pos + len]) {
                        let with_space = format!("▁{}", substr);
                        if let Some(&id) = self.token_to_id.get(&with_space) {
                            best_len = len;
                            best_id = Some(id);
                            break;
                        }
                    }
                }
            }

            if let Some(id) = best_id {
                tokens.push(id);
                pos += best_len;
            } else {
                // Fallback: try single byte token <0xNN>
                let byte_token = format!("<0x{:02X}>", bytes[pos]);
                if let Some(&id) = self.token_to_id.get(&byte_token) {
                    tokens.push(id);
                } else {
                    // Skip unknown byte
                    tracing::warn!("Unknown byte at position {}: 0x{:02X}", pos, bytes[pos]);
                }
                pos += 1;
            }
        }

        tokens
    }

    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut text = String::new();
        for &id in tokens {
            if let Some(token) = self.id_to_token.get(id as usize) {
                // Replace SentencePiece space marker
                text.push_str(&token.replace('▁', " "));
            } else {
                text.push_str(&format!("[unk_{}]", id));
            }
        }
        text
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}
