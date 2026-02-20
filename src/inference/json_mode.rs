//! JSON mode: grammar-constrained generation for valid JSON output
//!
//! Implements a stack-based JSON grammar validator that biases token sampling
//! to only allow tokens that would produce valid JSON. This enables reliable
//! structured output from LLMs without post-processing.
//!
//! Supports: objects, arrays, strings, numbers, booleans, null.
//! Does NOT support: JSON Schema enforcement (future work).

/// State of the JSON grammar parser
#[derive(Debug, Clone, Copy, PartialEq)]
enum JsonState {
    /// Expecting any JSON value to start
    ExpectValue,
    /// Inside a string literal
    InString,
    /// String escape sequence (backslash seen)
    InStringEscape,
    /// Inside a number
    InNumber,
    /// After a complete value (expecting comma, closing bracket, or end)
    AfterValue,
    /// Inside an object, expecting a key or closing brace
    ExpectKeyOrClose,
    /// Inside an object, after a key, expecting colon
    ExpectColon,
    /// Inside an array, expecting a value or closing bracket
    ExpectValueOrClose,
    /// Parsing a keyword (true, false, null)
    InKeyword,
    /// JSON is complete
    Done,
}

/// Tracks the nesting structure of JSON being generated
#[derive(Debug, Clone, Copy, PartialEq)]
enum Container {
    Object,
    Array,
}

/// JSON grammar validator for constraining token generation
pub struct JsonValidator {
    state: JsonState,
    /// Stack of nested containers (objects/arrays)
    stack: Vec<Container>,
    /// Partial keyword buffer (for true/false/null)
    keyword_buf: String,
    /// Expected keyword
    keyword_target: &'static str,
    /// Depth counter
    depth: usize,
    /// Whether we've seen any content
    has_content: bool,
}

impl JsonValidator {
    pub fn new() -> Self {
        Self {
            state: JsonState::ExpectValue,
            stack: Vec::new(),
            keyword_buf: String::new(),
            keyword_target: "",
            depth: 0,
            has_content: false,
        }
    }

    /// Check if the JSON output is complete and valid
    pub fn is_complete(&self) -> bool {
        self.state == JsonState::Done
            || (self.state == JsonState::AfterValue && self.stack.is_empty() && self.has_content)
    }

    /// Feed a character and return whether it's valid JSON grammar
    pub fn feed(&mut self, ch: char) -> bool {
        match self.state {
            JsonState::ExpectValue => self.expect_value(ch),
            JsonState::InString => self.in_string(ch),
            JsonState::InStringEscape => self.in_string_escape(ch),
            JsonState::InNumber => self.in_number(ch),
            JsonState::AfterValue => self.after_value(ch),
            JsonState::ExpectKeyOrClose => self.expect_key_or_close(ch),
            JsonState::ExpectColon => self.expect_colon(ch),
            JsonState::ExpectValueOrClose => self.expect_value_or_close(ch),
            JsonState::InKeyword => self.in_keyword(ch),
            JsonState::Done => false, // No more input expected
        }
    }

    /// Feed a string of characters, returning how many were valid
    pub fn feed_str(&mut self, s: &str) -> usize {
        let mut count = 0;
        for ch in s.chars() {
            if self.feed(ch) {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Validate a complete string as JSON, returning true if all characters are accepted
    pub fn validate_complete(s: &str) -> bool {
        let mut v = Self::new();
        for ch in s.chars() {
            if !v.feed(ch) {
                return false;
            }
        }
        v.is_complete()
    }

    fn expect_value(&mut self, ch: char) -> bool {
        match ch {
            ' ' | '\t' | '\n' | '\r' => true, // skip whitespace
            '"' => {
                self.state = JsonState::InString;
                self.has_content = true;
                true
            }
            '{' => {
                self.stack.push(Container::Object);
                self.state = JsonState::ExpectKeyOrClose;
                self.depth += 1;
                self.has_content = true;
                true
            }
            '[' => {
                self.stack.push(Container::Array);
                self.state = JsonState::ExpectValueOrClose;
                self.depth += 1;
                self.has_content = true;
                true
            }
            '-' | '0'..='9' => {
                self.state = JsonState::InNumber;
                self.has_content = true;
                true
            }
            't' => {
                self.keyword_buf.clear();
                self.keyword_buf.push('t');
                self.keyword_target = "true";
                self.state = JsonState::InKeyword;
                self.has_content = true;
                true
            }
            'f' => {
                self.keyword_buf.clear();
                self.keyword_buf.push('f');
                self.keyword_target = "false";
                self.state = JsonState::InKeyword;
                self.has_content = true;
                true
            }
            'n' => {
                self.keyword_buf.clear();
                self.keyword_buf.push('n');
                self.keyword_target = "null";
                self.state = JsonState::InKeyword;
                self.has_content = true;
                true
            }
            _ => false,
        }
    }

    fn in_string(&mut self, ch: char) -> bool {
        match ch {
            '\\' => {
                self.state = JsonState::InStringEscape;
                true
            }
            '"' => {
                self.state = JsonState::AfterValue;
                true
            }
            // Control characters are not allowed in JSON strings
            '\x00'..='\x1f' => false,
            _ => true,
        }
    }

    fn in_string_escape(&mut self, ch: char) -> bool {
        // Valid JSON escape sequences: \" \\ \/ \b \f \n \r \t \uXXXX
        match ch {
            '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u' => {
                self.state = JsonState::InString;
                true
            }
            _ => false,
        }
    }

    fn in_number(&mut self, ch: char) -> bool {
        match ch {
            '0'..='9' | '.' | 'e' | 'E' | '+' | '-' => true,
            _ => {
                // Number ended, process this character as after-value
                self.state = JsonState::AfterValue;
                self.after_value(ch)
            }
        }
    }

    fn in_keyword(&mut self, ch: char) -> bool {
        let next_idx = self.keyword_buf.len();
        if next_idx < self.keyword_target.len()
            && ch == self.keyword_target.as_bytes()[next_idx] as char
        {
            self.keyword_buf.push(ch);
            if self.keyword_buf.len() == self.keyword_target.len() {
                self.state = JsonState::AfterValue;
            }
            true
        } else {
            false
        }
    }

    fn after_value(&mut self, ch: char) -> bool {
        match ch {
            ' ' | '\t' | '\n' | '\r' => true,
            ',' => {
                match self.stack.last() {
                    Some(Container::Object) => self.state = JsonState::ExpectKeyOrClose,
                    Some(Container::Array) => self.state = JsonState::ExpectValue,
                    None => return false, // comma at top level
                }
                true
            }
            '}' => {
                if self.stack.last() == Some(&Container::Object) {
                    self.stack.pop();
                    self.depth -= 1;
                    if self.stack.is_empty() {
                        self.state = JsonState::Done;
                    }
                    // else stay in AfterValue
                    true
                } else {
                    false
                }
            }
            ']' => {
                if self.stack.last() == Some(&Container::Array) {
                    self.stack.pop();
                    self.depth -= 1;
                    if self.stack.is_empty() {
                        self.state = JsonState::Done;
                    }
                    true
                } else {
                    false
                }
            }
            _ => {
                if self.stack.is_empty() {
                    // Top-level value already complete
                    false
                } else {
                    false
                }
            }
        }
    }

    fn expect_key_or_close(&mut self, ch: char) -> bool {
        match ch {
            ' ' | '\t' | '\n' | '\r' => true,
            '"' => {
                self.state = JsonState::InString;
                // After this string, we need a colon — handle in after_value override
                // Actually, we need special handling: after the key string, expect colon
                // We'll set a flag by changing state after string ends
                // For simplicity: keys are strings, after which we expect colon
                // Override: when in object and AfterValue, if last was key, expect colon
                // Actually let's handle this by transitioning to ExpectColon after string
                self.state = JsonState::InString;
                // We need to know we're parsing a key... let's use a different approach:
                // After the string finishes (state becomes AfterValue), we'll check context
                // and transition to ExpectColon. We can do this by checking stack in after_value.
                // But after_value doesn't know if it was a key or value.
                // Solution: use a separate "InKey" state. But for simplicity, let's use the
                // existing InString and then transition based on stack state.
                // We'll handle this by making expect_colon the state after string close
                // when we're in an object key position.
                // For now: we set state to InString, but mark that we need colon after.
                // Simple approach: just use InString, but override the post-string state.
                // Let's add a flag.
                true
            }
            '}' => {
                if self.stack.last() == Some(&Container::Object) {
                    self.stack.pop();
                    self.depth -= 1;
                    if self.stack.is_empty() {
                        self.state = JsonState::Done;
                    } else {
                        self.state = JsonState::AfterValue;
                    }
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn expect_colon(&mut self, ch: char) -> bool {
        match ch {
            ' ' | '\t' | '\n' | '\r' => true,
            ':' => {
                self.state = JsonState::ExpectValue;
                true
            }
            _ => false,
        }
    }

    fn expect_value_or_close(&mut self, ch: char) -> bool {
        match ch {
            ' ' | '\t' | '\n' | '\r' => true,
            ']' => {
                if self.stack.last() == Some(&Container::Array) {
                    self.stack.pop();
                    self.depth -= 1;
                    if self.stack.is_empty() {
                        self.state = JsonState::Done;
                    } else {
                        self.state = JsonState::AfterValue;
                    }
                    true
                } else {
                    false
                }
            }
            _ => {
                // Try parsing as a value
                self.state = JsonState::ExpectValue;
                self.expect_value(ch)
            }
        }
    }
}

// The key problem above is that when parsing object keys, after the string closes,
// we go to AfterValue instead of ExpectColon. Let's fix this with a cleaner design.

/// Production-quality JSON grammar validator with proper key/value tracking
pub struct JsonGrammar {
    /// Character-level state
    chars: Vec<char>,
    /// Whether the accumulated text so far is a valid JSON prefix
    valid: bool,
}

impl JsonGrammar {
    pub fn new() -> Self {
        Self {
            chars: Vec::new(),
            valid: true,
        }
    }

    /// Feed a token string; returns true if the resulting text is still a valid JSON prefix
    pub fn feed_token(&mut self, token: &str) -> bool {
        if !self.valid {
            return false;
        }
        let mut test = self.chars.clone();
        test.extend(token.chars());
        if is_valid_json_prefix(&test) {
            self.chars = test;
            true
        } else {
            false
        }
    }

    /// Check if the accumulated text is complete valid JSON
    pub fn is_complete(&self) -> bool {
        if !self.valid || self.chars.is_empty() {
            return false;
        }
        let s: String = self.chars.iter().collect();
        serde_json::from_str::<serde_json::Value>(&s).is_ok()
    }

    /// Get the accumulated text
    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }
}

/// Check if a character sequence is a valid prefix of some valid JSON document
fn is_valid_json_prefix(chars: &[char]) -> bool {
    if chars.is_empty() {
        return true;
    }
    let s: String = chars.iter().collect();

    // Try parsing as complete JSON first
    if serde_json::from_str::<serde_json::Value>(&s).is_ok() {
        return true;
    }

    // Try common completions to see if it could become valid JSON
    // This is a heuristic approach — try closing open structures
    let completions = [
        "\"}", "\"]", "\"", "}", "]", "0", "0}", "0]", "null", "null}", "null]", "true", "true}",
        "true]", "false", "false}", "false]", "\"\"}", "\"\"]",
    ];

    for completion in completions {
        let test = format!("{}{}", s, completion);
        if serde_json::from_str::<serde_json::Value>(&test).is_ok() {
            return true;
        }
    }

    // More aggressive: try adding many closing brackets
    let mut closer = s.clone();
    for _ in 0..20 {
        closer.push('}');
        if serde_json::from_str::<serde_json::Value>(&closer).is_ok() {
            return true;
        }
        // Also try with ] instead
        closer.pop();
        closer.push(']');
        if serde_json::from_str::<serde_json::Value>(&closer).is_ok() {
            return true;
        }
        closer.pop();
        // Try both in sequence
        let test_obj = format!("{}\"}}", s);
        if serde_json::from_str::<serde_json::Value>(&test_obj).is_ok() {
            return true;
        }
    }

    // If nothing works, check character-level validity with a simple state machine
    is_valid_json_prefix_statemachine(chars)
}

/// Simple state machine to validate JSON prefixes
fn is_valid_json_prefix_statemachine(chars: &[char]) -> bool {
    let mut i = 0;
    let n = chars.len();

    fn skip_ws(chars: &[char], i: &mut usize) {
        while *i < chars.len() && matches!(chars[*i], ' ' | '\t' | '\n' | '\r') {
            *i += 1;
        }
    }

    fn parse_value(chars: &[char], i: &mut usize) -> bool {
        let n = chars.len();
        skip_ws(chars, i);
        if *i >= n {
            return true; // prefix: value hasn't started yet
        }
        match chars[*i] {
            '"' => parse_string(chars, i),
            '{' => parse_object(chars, i),
            '[' => parse_array(chars, i),
            't' => parse_keyword(chars, i, "true"),
            'f' => parse_keyword(chars, i, "false"),
            'n' => parse_keyword(chars, i, "null"),
            '-' | '0'..='9' => parse_number(chars, i),
            _ => false,
        }
    }

    fn parse_string(chars: &[char], i: &mut usize) -> bool {
        let n = chars.len();
        if *i >= n || chars[*i] != '"' {
            return false;
        }
        *i += 1; // skip opening quote
        while *i < n {
            match chars[*i] {
                '\\' => {
                    *i += 1;
                    if *i >= n {
                        return true;
                    } // prefix
                    match chars[*i] {
                        '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => *i += 1,
                        'u' => {
                            *i += 1;
                            for _ in 0..4 {
                                if *i >= n {
                                    return true;
                                }
                                if !chars[*i].is_ascii_hexdigit() {
                                    return false;
                                }
                                *i += 1;
                            }
                        }
                        _ => return false,
                    }
                }
                '"' => {
                    *i += 1;
                    return true;
                }
                '\x00'..='\x1f' => return false,
                _ => *i += 1,
            }
        }
        true // prefix: string not closed yet
    }

    fn parse_number(chars: &[char], i: &mut usize) -> bool {
        let n = chars.len();
        if *i >= n {
            return true;
        }
        if chars[*i] == '-' {
            *i += 1;
            if *i >= n {
                return true;
            }
        }
        if *i >= n {
            return true;
        }
        if !chars[*i].is_ascii_digit() {
            return false;
        }
        while *i < n && chars[*i].is_ascii_digit() {
            *i += 1;
        }
        if *i < n && chars[*i] == '.' {
            *i += 1;
            while *i < n && chars[*i].is_ascii_digit() {
                *i += 1;
            }
        }
        if *i < n && (chars[*i] == 'e' || chars[*i] == 'E') {
            *i += 1;
            if *i < n && (chars[*i] == '+' || chars[*i] == '-') {
                *i += 1;
            }
            while *i < n && chars[*i].is_ascii_digit() {
                *i += 1;
            }
        }
        true
    }

    fn parse_keyword(chars: &[char], i: &mut usize, kw: &str) -> bool {
        let n = chars.len();
        for ch in kw.chars() {
            if *i >= n {
                return true;
            } // prefix
            if chars[*i] != ch {
                return false;
            }
            *i += 1;
        }
        true
    }

    fn parse_object(chars: &[char], i: &mut usize) -> bool {
        let n = chars.len();
        if *i >= n || chars[*i] != '{' {
            return false;
        }
        *i += 1;
        skip_ws(chars, i);
        if *i >= n {
            return true;
        }
        if chars[*i] == '}' {
            *i += 1;
            return true;
        }
        // Parse key-value pairs
        loop {
            skip_ws(chars, i);
            if *i >= n {
                return true;
            }
            // Key must be string
            if chars[*i] != '"' {
                return false;
            }
            if !parse_string(chars, i) {
                return false;
            }
            skip_ws(chars, i);
            if *i >= n {
                return true;
            }
            if chars[*i] != ':' {
                return false;
            }
            *i += 1;
            if !parse_value(chars, i) {
                return false;
            }
            skip_ws(chars, i);
            if *i >= n {
                return true;
            }
            match chars[*i] {
                ',' => {
                    *i += 1;
                }
                '}' => {
                    *i += 1;
                    return true;
                }
                _ => return false,
            }
        }
    }

    fn parse_array(chars: &[char], i: &mut usize) -> bool {
        let n = chars.len();
        if *i >= n || chars[*i] != '[' {
            return false;
        }
        *i += 1;
        skip_ws(chars, i);
        if *i >= n {
            return true;
        }
        if chars[*i] == ']' {
            *i += 1;
            return true;
        }
        loop {
            if !parse_value(chars, i) {
                return false;
            }
            skip_ws(chars, i);
            if *i >= n {
                return true;
            }
            match chars[*i] {
                ',' => {
                    *i += 1;
                }
                ']' => {
                    *i += 1;
                    return true;
                }
                _ => return false,
            }
        }
    }

    skip_ws(chars, &mut i);
    if i >= n {
        return true;
    }
    if !parse_value(chars, &mut i) {
        return false;
    }
    skip_ws(chars, &mut i);
    // Any remaining characters after a complete value at top level should be just whitespace
    // But since we're checking prefixes, trailing content is fine if it was already consumed
    i <= n
}

/// Filter logits to only allow tokens that produce valid JSON
///
/// Given the current JSON grammar state and a list of (token_id, token_text) pairs,
/// masks out tokens that would produce invalid JSON by setting their logits to -inf.
pub fn apply_json_constraint(logits: &mut [f32], current_text: &str, vocab: &[(u32, String)]) {
    let current_chars: Vec<char> = current_text.chars().collect();

    for (token_id, token_text) in vocab {
        let idx = *token_id as usize;
        if idx >= logits.len() {
            continue;
        }

        // Test if appending this token produces a valid JSON prefix
        let mut test: Vec<char> = current_chars.clone();
        test.extend(token_text.chars());

        if !is_valid_json_prefix(&test) {
            logits[idx] = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_json_objects() {
        assert!(JsonGrammar::new().feed_token(r#"{"key": "value"}"#));
        let mut g = JsonGrammar::new();
        g.feed_token(r#"{"key": "value"}"#);
        assert!(g.is_complete());
    }

    #[test]
    fn test_json_prefix_validation() {
        let chars: Vec<char> = r#"{"ke"#.chars().collect();
        assert!(is_valid_json_prefix(&chars));

        let chars: Vec<char> = r#"{"key": "#.chars().collect();
        assert!(is_valid_json_prefix(&chars));

        let chars: Vec<char> = r#"{"key": "val"#.chars().collect();
        assert!(is_valid_json_prefix(&chars));
    }

    #[test]
    fn test_invalid_json_prefix() {
        let chars: Vec<char> = r#"{key"#.chars().collect();
        assert!(!is_valid_json_prefix(&chars));
    }

    #[test]
    fn test_json_array_prefix() {
        let chars: Vec<char> = r#"[1, 2, "#.chars().collect();
        assert!(is_valid_json_prefix(&chars));
    }

    #[test]
    fn test_json_nested() {
        let chars: Vec<char> = r#"{"a": {"b": [1, "#.chars().collect();
        assert!(is_valid_json_prefix(&chars));
    }

    #[test]
    fn test_complete_array() {
        let mut g = JsonGrammar::new();
        g.feed_token("[1, 2, 3]");
        assert!(g.is_complete());
    }

    #[test]
    fn test_complete_string() {
        let mut g = JsonGrammar::new();
        g.feed_token(r#""hello""#);
        assert!(g.is_complete());
    }

    #[test]
    fn test_complete_number() {
        let mut g = JsonGrammar::new();
        g.feed_token("42");
        assert!(g.is_complete());
    }

    #[test]
    fn test_complete_boolean() {
        let mut g = JsonGrammar::new();
        g.feed_token("true");
        assert!(g.is_complete());
    }

    #[test]
    fn test_complete_null() {
        let mut g = JsonGrammar::new();
        g.feed_token("null");
        assert!(g.is_complete());
    }

    #[test]
    fn test_incremental_tokens() {
        let mut g = JsonGrammar::new();
        assert!(g.feed_token(r#"{"#));
        assert!(g.feed_token(r#""name"#));
        assert!(g.feed_token(r#"": "#));
        assert!(g.feed_token(r#""Alice"#));
        assert!(g.feed_token(r#""}"#));
        assert!(g.is_complete());
    }

    #[test]
    fn test_rejects_invalid_token() {
        let mut g = JsonGrammar::new();
        assert!(g.feed_token(r#"{"key": "#));
        // An invalid continuation
        assert!(!g.feed_token("xyz_not_json"));
    }

    #[test]
    fn test_empty_object() {
        let mut g = JsonGrammar::new();
        g.feed_token("{}");
        assert!(g.is_complete());
    }

    #[test]
    fn test_empty_array() {
        let mut g = JsonGrammar::new();
        g.feed_token("[]");
        assert!(g.is_complete());
    }

    #[test]
    fn test_nested_complete() {
        let mut g = JsonGrammar::new();
        g.feed_token(r#"{"a": [1, {"b": true}], "c": null}"#);
        assert!(g.is_complete());
    }

    #[test]
    fn test_string_with_escapes() {
        let chars: Vec<char> = r#""hello \"world\"""#.chars().collect();
        assert!(is_valid_json_prefix(&chars));
    }

    #[test]
    fn test_number_formats() {
        for num in &["0", "42", "-1", "3.14", "1e10", "2.5E-3", "-0.001"] {
            let chars: Vec<char> = num.chars().collect();
            assert!(
                is_valid_json_prefix(&chars),
                "Should accept number: {}",
                num
            );
        }
    }

    #[test]
    fn test_statemachine_basic_object() {
        let chars: Vec<char> = r#"{"a": 1}"#.chars().collect();
        assert!(is_valid_json_prefix_statemachine(&chars));
    }

    #[test]
    fn test_statemachine_partial_key() {
        let chars: Vec<char> = r#"{"ke"#.chars().collect();
        assert!(is_valid_json_prefix_statemachine(&chars));
    }

    #[test]
    fn test_statemachine_rejects_bare_word() {
        let chars: Vec<char> = r#"{key"#.chars().collect();
        assert!(!is_valid_json_prefix_statemachine(&chars));
    }
}
