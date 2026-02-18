//! BPE Tokenizer for GGUF models
//!
//! v0.4: Full Byte-Pair Encoding (BPE) tokenizer with merge rules from GGUF metadata.
//! Supports SentencePiece-style tokenization as used by LLaMA models.

use crate::model::gguf::{GgufFile, MetadataValue};
use std::collections::HashMap;
use tracing::{debug, warn};

/// BPE merge rule: (left_token, right_token) → merged_token
#[derive(Debug, Clone)]
struct MergeRule {
    left: String,
    right: String,
    merged: String,
    rank: u32,
}

pub struct SimpleTokenizer {
    /// token_id → token string
    id_to_token: Vec<String>,
    /// token string → token_id
    token_to_id: HashMap<String, u32>,
    /// BPE merge rules, ordered by priority (lower rank = higher priority)
    merges: Vec<MergeRule>,
    /// Pair → merge rank for O(1) lookup
    merge_ranks: HashMap<(String, String), u32>,
    /// Token scores from GGUF (used for SentencePiece-style BPE)
    token_scores: Vec<f32>,
    /// BOS token ID
    pub bos_id: u32,
    /// EOS token ID
    pub eos_id: u32,
    /// Whether to add BOS token
    pub add_bos: bool,
    /// Whether this tokenizer has BPE merges
    has_merges: bool,
}

impl SimpleTokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let mut id_to_token = Vec::new();
        let mut token_to_id = HashMap::new();
        let mut token_scores = Vec::new();

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

        // Extract token scores (used for SentencePiece BPE priority)
        if let Some(MetadataValue::Array(scores)) = gguf.metadata.get("tokenizer.ggml.scores") {
            for score in scores {
                match score {
                    MetadataValue::Float32(v) => token_scores.push(*v),
                    _ => token_scores.push(0.0),
                }
            }
        }

        // Extract BPE merges if available
        let mut merges = Vec::new();
        let mut merge_ranks = HashMap::new();
        let mut has_merges = false;

        if let Some(MetadataValue::Array(merge_list)) = gguf.metadata.get("tokenizer.ggml.merges") {
            has_merges = true;
            for (rank, merge) in merge_list.iter().enumerate() {
                if let MetadataValue::String(s) = merge {
                    if let Some((left, right)) = s.split_once(' ') {
                        let merged = format!("{}{}", left, right);
                        let rule = MergeRule {
                            left: left.to_string(),
                            right: right.to_string(),
                            merged: merged.clone(),
                            rank: rank as u32,
                        };
                        merge_ranks.insert((left.to_string(), right.to_string()), rank as u32);
                        merges.push(rule);
                    }
                }
            }
            debug!("Loaded {} BPE merge rules", merges.len());
        } else if !token_scores.is_empty() {
            // SentencePiece-style: use token scores as merge priority
            // Build merge rules from vocabulary scores
            has_merges = true;
            debug!("Using SentencePiece score-based BPE ({} tokens with scores)", token_scores.len());
        }

        let bos_id = gguf.metadata.get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(1);
        let eos_id = gguf.metadata.get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(2);
        let add_bos = gguf.metadata.get("tokenizer.ggml.add_bos_token")
            .and_then(|v| match v {
                MetadataValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(true);

        Self {
            id_to_token,
            token_to_id,
            merges,
            merge_ranks,
            token_scores,
            bos_id,
            eos_id,
            add_bos,
            has_merges,
        }
    }

    /// Encode text to token IDs using BPE
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        if self.add_bos {
            tokens.push(self.bos_id);
        }

        if text.is_empty() {
            return tokens;
        }

        if self.has_merges && !self.merge_ranks.is_empty() {
            // Standard BPE with explicit merge rules
            tokens.extend(self.encode_bpe(text));
        } else if self.has_merges && !self.token_scores.is_empty() {
            // SentencePiece-style BPE using token scores
            tokens.extend(self.encode_sentencepiece(text));
        } else {
            // Fallback: greedy longest-match
            tokens.extend(self.encode_greedy(text));
        }

        tokens
    }

    /// BPE encoding with explicit merge rules
    fn encode_bpe(&self, text: &str) -> Vec<u32> {
        // Start with UTF-8 bytes as initial tokens
        let mut symbols: Vec<String> = Vec::new();

        // Split into initial characters/bytes
        for ch in text.chars() {
            let s = ch.to_string();
            symbols.push(s);
        }

        // Iteratively apply the highest-priority merge
        loop {
            if symbols.len() < 2 {
                break;
            }

            // Find the best merge (lowest rank)
            let mut best_rank = u32::MAX;
            let mut best_idx = usize::MAX;

            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break; // No more merges possible
            }

            // Apply the merge
            let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
            symbols[best_idx] = merged;
            symbols.remove(best_idx + 1);
        }

        // Convert to token IDs
        let mut ids = Vec::new();
        for sym in &symbols {
            if let Some(&id) = self.token_to_id.get(sym) {
                ids.push(id);
            } else {
                // Try with SentencePiece space prefix
                let with_space = format!("▁{}", sym);
                if let Some(&id) = self.token_to_id.get(&with_space) {
                    ids.push(id);
                } else {
                    // Fall back to byte-level encoding
                    ids.extend(self.encode_bytes(sym.as_bytes()));
                }
            }
        }

        ids
    }

    /// SentencePiece-style BPE using token scores for merge priority
    fn encode_sentencepiece(&self, text: &str) -> Vec<u32> {
        // Prepend space (SentencePiece convention)
        let text = format!("▁{}", text.replace(' ', "▁"));

        // Start with individual UTF-8 characters
        let chars: Vec<char> = text.chars().collect();
        let mut symbols: Vec<String> = chars.iter().map(|c| c.to_string()).collect();

        // Iteratively merge the pair that produces the highest-scored token
        loop {
            if symbols.len() < 2 {
                break;
            }

            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;
            let mut best_merged = String::new();

            for i in 0..symbols.len() - 1 {
                let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                if let Some(&id) = self.token_to_id.get(&merged) {
                    let score = self.token_scores.get(id as usize).copied().unwrap_or(f32::NEG_INFINITY);
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                        best_merged = merged;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            symbols[best_idx] = best_merged;
            symbols.remove(best_idx + 1);
        }

        // Convert to IDs
        let mut ids = Vec::new();
        for sym in &symbols {
            if let Some(&id) = self.token_to_id.get(sym) {
                ids.push(id);
            } else {
                ids.extend(self.encode_bytes(sym.as_bytes()));
            }
        }

        ids
    }

    /// Greedy longest-match tokenization (fallback)
    fn encode_greedy(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let bytes = text.as_bytes();
        let mut pos = 0;

        while pos < bytes.len() {
            let mut best_len = 0;
            let mut best_id = None;

            // Try longest match (up to 32 chars for longer tokens)
            for len in (1..=32.min(bytes.len() - pos)).rev() {
                if let Ok(substr) = std::str::from_utf8(&bytes[pos..pos + len]) {
                    if let Some(&id) = self.token_to_id.get(substr) {
                        best_len = len;
                        best_id = Some(id);
                        break;
                    }
                    // Try with SentencePiece space prefix
                    let with_space = format!("▁{}", substr);
                    if let Some(&id) = self.token_to_id.get(&with_space) {
                        best_len = len;
                        best_id = Some(id);
                        break;
                    }
                }
            }

            if let Some(id) = best_id {
                ids.push(id);
                pos += best_len;
            } else {
                // Byte-level fallback
                ids.extend(self.encode_bytes(&bytes[pos..pos + 1]));
                pos += 1;
            }
        }

        ids
    }

    /// Encode raw bytes as byte-level tokens (<0xNN>)
    fn encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        let mut ids = Vec::new();
        for &byte in bytes {
            let byte_token = format!("<0x{:02X}>", byte);
            if let Some(&id) = self.token_to_id.get(&byte_token) {
                ids.push(id);
            } else {
                warn!("Unknown byte: 0x{:02X}", byte);
            }
        }
        ids
    }

    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut text = String::new();
        for &id in tokens {
            if id == self.bos_id || id == self.eos_id {
                continue;
            }
            if let Some(token) = self.id_to_token.get(id as usize) {
                // Replace SentencePiece space marker
                text.push_str(&token.replace('▁', " "));
            } else {
                text.push_str(&format!("[unk_{}]", id));
            }
        }
        text
    }

    /// Decode a single token ID
    pub fn decode_token(&self, id: u32) -> String {
        if id == self.bos_id || id == self.eos_id {
            return String::new();
        }
        if let Some(token) = self.id_to_token.get(id as usize) {
            token.replace('▁', " ")
        } else {
            format!("[unk_{}]", id)
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Check if a token ID is a special token (BOS, EOS, padding, etc.)
    pub fn is_special(&self, id: u32) -> bool {
        id == self.bos_id || id == self.eos_id || id == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tokenizer() -> SimpleTokenizer {
        let mut id_to_token = vec![
            "<unk>".to_string(), "<s>".to_string(), "</s>".to_string(),
            "▁".to_string(), "h".to_string(), "e".to_string(), "l".to_string(),
            "o".to_string(), "▁he".to_string(), "ll".to_string(), "▁hello".to_string(),
        ];
        let mut token_to_id = HashMap::new();
        for (i, t) in id_to_token.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        // Scores: higher = more likely to be merged
        let token_scores = vec![0.0, 0.0, 0.0, -1.0, -2.0, -2.0, -2.0, -2.0, -0.5, -0.5, 0.5];

        SimpleTokenizer {
            id_to_token,
            token_to_id,
            merges: Vec::new(),
            merge_ranks: HashMap::new(),
            token_scores,
            bos_id: 1,
            eos_id: 2,
            add_bos: true,
            has_merges: true,
        }
    }

    #[test]
    fn test_sentencepiece_bpe() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello");
        assert!(tokens[0] == 1, "Should start with BOS");
        // The tokenizer should produce valid token IDs (not empty)
        assert!(tokens.len() > 1, "Should produce tokens beyond BOS");
        // All non-BOS tokens should be valid vocab IDs
        for &id in &tokens[1..] {
            assert!((id as usize) < tok.vocab_size(), "Token ID {} out of range", id);
        }
    }

    #[test]
    fn test_decode_roundtrip() {
        let tok = make_test_tokenizer();
        let decoded = tok.decode(&[10]);
        assert_eq!(decoded, " hello");
    }

    #[test]
    fn test_decode_token() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.decode_token(10), " hello");
        assert_eq!(tok.decode_token(1), ""); // BOS should be empty
    }
}
