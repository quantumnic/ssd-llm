//! GBNF Grammar-Constrained Generation
//!
//! Implements a grammar engine compatible with llama.cpp's GBNF format.
//! Grammars define production rules that constrain token generation to only
//! produce outputs matching the grammar. This generalizes JSON mode to
//! arbitrary structured outputs (SQL, XML, function signatures, etc.).
//!
//! GBNF format:
//! ```text
//! root   ::= expr
//! expr   ::= term (("+" | "-") term)*
//! term   ::= [0-9]+
//! ```
//!
//! The engine works by:
//! 1. Parsing GBNF into a set of production rules
//! 2. Maintaining a parse stack during generation
//! 3. For each token candidate, checking if appending it would remain
//!    consistent with the grammar
//! 4. Masking out invalid tokens before sampling

use std::collections::HashMap;
use std::fmt;

/// A character range in a character class, e.g. [a-z] or [abc]
#[derive(Debug, Clone, PartialEq)]
pub struct CharRange {
    pub start: char,
    pub end: char,
}

/// An element within an alternative sequence
#[derive(Debug, Clone, PartialEq)]
pub enum GrammarElement {
    /// A literal string that must be matched exactly
    Literal(String),
    /// A character class like [a-zA-Z0-9_]
    CharClass {
        ranges: Vec<CharRange>,
        negated: bool,
    },
    /// Reference to another rule by name
    RuleRef(String),
    /// A group of alternatives (nested)
    Group(Vec<Vec<GrammarElement>>),
    /// Optional: element? (0 or 1)
    Optional(Box<GrammarElement>),
    /// Repeat: element* (0 or more)
    Repeat(Box<GrammarElement>),
    /// Repeat1: element+ (1 or more)
    RepeatOne(Box<GrammarElement>),
}

/// A production rule: name ::= alt1 | alt2 | ...
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    /// Each alternative is a sequence of elements
    pub alternatives: Vec<Vec<GrammarElement>>,
}

/// Parsed GBNF grammar
#[derive(Debug, Clone)]
pub struct Grammar {
    pub rules: HashMap<String, Rule>,
    pub root: String,
}

/// Errors during grammar parsing
#[derive(Debug)]
pub enum GrammarError {
    /// Syntax error in the grammar definition
    Syntax(String),
    /// Reference to undefined rule
    UndefinedRule(String),
    /// No root rule defined
    NoRoot,
    /// Empty grammar
    Empty,
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GrammarError::Syntax(msg) => write!(f, "grammar syntax error: {}", msg),
            GrammarError::UndefinedRule(name) => write!(f, "undefined rule: {}", name),
            GrammarError::NoRoot => write!(f, "no root rule defined"),
            GrammarError::Empty => write!(f, "empty grammar"),
        }
    }
}

impl std::error::Error for GrammarError {}

/// Parse a GBNF grammar string into a Grammar
pub fn parse_grammar(input: &str) -> Result<Grammar, GrammarError> {
    let mut rules = HashMap::new();
    let mut first_rule = None;

    for line in input.lines() {
        let line = line.trim();
        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Find ::= separator
        let Some(sep_pos) = line.find("::=") else {
            continue;
        };

        let name = line[..sep_pos].trim().to_string();
        let body = line[sep_pos + 3..].trim();

        if name.is_empty() {
            return Err(GrammarError::Syntax("empty rule name".into()));
        }

        if first_rule.is_none() {
            first_rule = Some(name.clone());
        }

        let alternatives = parse_alternatives(body)?;

        rules.insert(name.clone(), Rule { name, alternatives });
    }

    let root = first_rule.ok_or(GrammarError::Empty)?;

    // Validate all rule references
    validate_references(&rules)?;

    Ok(Grammar { rules, root })
}

/// Parse the right-hand side of a rule into alternatives
fn parse_alternatives(input: &str) -> Result<Vec<Vec<GrammarElement>>, GrammarError> {
    let mut alternatives = Vec::new();
    let mut current = Vec::new();
    let mut chars = input.chars().peekable();

    while chars.peek().is_some() {
        skip_whitespace(&mut chars);

        match chars.peek() {
            None => break,
            Some('|') => {
                chars.next();
                alternatives.push(std::mem::take(&mut current));
            }
            Some('"') => {
                let lit = parse_quoted_string(&mut chars)?;
                let elem = GrammarElement::Literal(lit);
                current.push(apply_quantifier(&mut chars, elem));
            }
            Some('[') => {
                let (ranges, negated) = parse_char_class(&mut chars)?;
                let elem = GrammarElement::CharClass { ranges, negated };
                current.push(apply_quantifier(&mut chars, elem));
            }
            Some('(') => {
                chars.next();
                let group = parse_group(&mut chars)?;
                let elem = GrammarElement::Group(group);
                current.push(apply_quantifier(&mut chars, elem));
            }
            Some(c) if c.is_alphanumeric() || *c == '_' || *c == '-' => {
                let name = parse_identifier(&mut chars);
                let elem = GrammarElement::RuleRef(name);
                current.push(apply_quantifier(&mut chars, elem));
            }
            Some(c) => {
                return Err(GrammarError::Syntax(format!(
                    "unexpected character: '{}'",
                    c
                )));
            }
        }
    }

    alternatives.push(current);
    Ok(alternatives)
}

fn skip_whitespace(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    while let Some(c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
}

fn parse_quoted_string(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Result<String, GrammarError> {
    let quote = chars.next().unwrap(); // consume opening quote
    let mut result = String::new();

    loop {
        match chars.next() {
            None => return Err(GrammarError::Syntax("unterminated string".into())),
            Some('\\') => match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('\'') => result.push('\''),
                Some(c) => {
                    result.push('\\');
                    result.push(c);
                }
                None => return Err(GrammarError::Syntax("unterminated escape".into())),
            },
            Some(c) if c == quote => break,
            Some(c) => result.push(c),
        }
    }
    Ok(result)
}

fn parse_char_class(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Result<(Vec<CharRange>, bool), GrammarError> {
    chars.next(); // consume '['
    let negated = chars.peek() == Some(&'^');
    if negated {
        chars.next();
    }

    let mut ranges = Vec::new();

    loop {
        match chars.next() {
            None => return Err(GrammarError::Syntax("unterminated char class".into())),
            Some(']') => break,
            Some('\\') => {
                let c = parse_escape_char(chars)?;
                if chars.peek() == Some(&'-') {
                    chars.next();
                    let end = match chars.next() {
                        Some('\\') => parse_escape_char(chars)?,
                        Some(c) => c,
                        None => {
                            return Err(GrammarError::Syntax("unterminated range".into()));
                        }
                    };
                    ranges.push(CharRange { start: c, end });
                } else {
                    ranges.push(CharRange { start: c, end: c });
                }
            }
            Some(c) => {
                if chars.peek() == Some(&'-') {
                    chars.next();
                    let end = match chars.next() {
                        Some('\\') => parse_escape_char(chars)?,
                        Some(']') => {
                            // e.g. [a-] — treat '-' as literal
                            ranges.push(CharRange { start: c, end: c });
                            ranges.push(CharRange {
                                start: '-',
                                end: '-',
                            });
                            break;
                        }
                        Some(e) => e,
                        None => {
                            return Err(GrammarError::Syntax("unterminated range".into()));
                        }
                    };
                    ranges.push(CharRange { start: c, end });
                } else {
                    ranges.push(CharRange { start: c, end: c });
                }
            }
        }
    }
    Ok((ranges, negated))
}

fn parse_escape_char(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Result<char, GrammarError> {
    match chars.next() {
        Some('n') => Ok('\n'),
        Some('t') => Ok('\t'),
        Some('r') => Ok('\r'),
        Some('\\') => Ok('\\'),
        Some(']') => Ok(']'),
        Some('[') => Ok('['),
        Some('-') => Ok('-'),
        Some(c) => Ok(c),
        None => Err(GrammarError::Syntax("unterminated escape".into())),
    }
}

fn parse_group(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Result<Vec<Vec<GrammarElement>>, GrammarError> {
    let mut alternatives = Vec::new();
    let mut current = Vec::new();

    loop {
        skip_whitespace(chars);
        match chars.peek() {
            None => return Err(GrammarError::Syntax("unterminated group".into())),
            Some(')') => {
                chars.next();
                alternatives.push(current);
                return Ok(alternatives);
            }
            Some('|') => {
                chars.next();
                alternatives.push(std::mem::take(&mut current));
            }
            Some('"') => {
                let lit = parse_quoted_string(chars)?;
                let elem = GrammarElement::Literal(lit);
                current.push(apply_quantifier(chars, elem));
            }
            Some('[') => {
                let (ranges, negated) = parse_char_class(chars)?;
                let elem = GrammarElement::CharClass { ranges, negated };
                current.push(apply_quantifier(chars, elem));
            }
            Some('(') => {
                chars.next();
                let group = parse_group(chars)?;
                let elem = GrammarElement::Group(group);
                current.push(apply_quantifier(chars, elem));
            }
            Some(c) if c.is_alphanumeric() || *c == '_' || *c == '-' => {
                let name = parse_identifier(chars);
                let elem = GrammarElement::RuleRef(name);
                current.push(apply_quantifier(chars, elem));
            }
            Some(c) => {
                return Err(GrammarError::Syntax(format!("unexpected '{}' in group", c)));
            }
        }
    }
}

fn parse_identifier(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut name = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_alphanumeric() || c == '_' || c == '-' {
            name.push(c);
            chars.next();
        } else {
            break;
        }
    }
    name
}

fn apply_quantifier(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    elem: GrammarElement,
) -> GrammarElement {
    match chars.peek() {
        Some('?') => {
            chars.next();
            GrammarElement::Optional(Box::new(elem))
        }
        Some('*') => {
            chars.next();
            GrammarElement::Repeat(Box::new(elem))
        }
        Some('+') => {
            chars.next();
            GrammarElement::RepeatOne(Box::new(elem))
        }
        _ => elem,
    }
}

fn validate_references(rules: &HashMap<String, Rule>) -> Result<(), GrammarError> {
    for rule in rules.values() {
        for alt in &rule.alternatives {
            validate_elements(alt, rules)?;
        }
    }
    Ok(())
}

fn validate_elements(
    elements: &[GrammarElement],
    rules: &HashMap<String, Rule>,
) -> Result<(), GrammarError> {
    for elem in elements {
        match elem {
            GrammarElement::RuleRef(name) => {
                if !rules.contains_key(name) {
                    return Err(GrammarError::UndefinedRule(name.clone()));
                }
            }
            GrammarElement::Group(alts) => {
                for alt in alts {
                    validate_elements(alt, rules)?;
                }
            }
            GrammarElement::Optional(inner)
            | GrammarElement::Repeat(inner)
            | GrammarElement::RepeatOne(inner) => {
                validate_elements(&[inner.as_ref().clone()], rules)?;
            }
            _ => {}
        }
    }
    Ok(())
}

// ─── Grammar State Machine ──────────────────────────────────────────

/// A position within a grammar rule being matched
#[derive(Debug, Clone)]
struct StackFrame {
    /// Rule name being matched
    rule: String,
    /// Which alternative is being tried (index into rule.alternatives)
    alt_idx: usize,
    /// Position within the alternative's element sequence
    elem_idx: usize,
    /// For literals: how many chars of the literal have been consumed
    lit_offset: usize,
    /// For repeat/repeat_one: how many repetitions completed
    rep_count: usize,
}

/// Grammar acceptor: tracks parse state and determines valid next characters
#[derive(Debug, Clone)]
pub struct GrammarAcceptor {
    grammar: Grammar,
    /// Stack of active parse frames
    stack: Vec<StackFrame>,
    /// Whether the grammar has been fully satisfied
    complete: bool,
}

impl GrammarAcceptor {
    /// Create a new acceptor starting from the root rule
    pub fn new(grammar: Grammar) -> Self {
        let root = grammar.root.clone();
        let mut acc = GrammarAcceptor {
            grammar,
            stack: Vec::new(),
            complete: false,
        };
        acc.push_rule(&root, 0);
        acc
    }

    /// Get the set of characters that are valid as the next input
    pub fn allowed_chars(&self) -> Vec<CharRange> {
        if self.complete {
            return Vec::new();
        }

        let mut allowed = Vec::new();
        self.collect_allowed(&mut allowed, 0);
        allowed
    }

    /// Check if a specific character is allowed next
    pub fn is_char_allowed(&self, c: char) -> bool {
        if self.complete {
            return false;
        }
        let allowed = self.allowed_chars();
        char_in_ranges(c, &allowed)
    }

    /// Check if the grammar can accept end-of-input now
    pub fn can_finish(&self) -> bool {
        if self.complete {
            return true;
        }
        self.can_finish_from(0)
    }

    /// Feed a character into the acceptor, advancing the state.
    /// Returns true if the character was accepted.
    pub fn accept(&mut self, c: char) -> bool {
        if self.complete {
            return false;
        }

        if self.try_advance(c) {
            // After advancing, simplify the stack
            self.reduce();
            return true;
        }
        false
    }

    /// Feed a string of characters, returning how many were accepted
    pub fn accept_str(&mut self, s: &str) -> usize {
        let mut count = 0;
        for c in s.chars() {
            if !self.accept(c) {
                break;
            }
            count += 1;
        }
        count
    }

    /// Check if a candidate token string is fully compatible with the grammar.
    /// Creates a temporary clone to test without modifying state.
    pub fn would_accept(&self, token: &str) -> AcceptResult {
        let mut test = self.clone();
        let accepted = test.accept_str(token);
        if accepted == token.len() {
            if test.can_finish() {
                AcceptResult::Full
            } else {
                AcceptResult::Partial
            }
        } else {
            AcceptResult::Rejected
        }
    }

    fn push_rule(&mut self, rule_name: &str, alt_idx: usize) {
        self.stack.push(StackFrame {
            rule: rule_name.to_string(),
            alt_idx,
            elem_idx: 0,
            lit_offset: 0,
            rep_count: 0,
        });
    }

    fn collect_allowed(&self, allowed: &mut Vec<CharRange>, stack_depth: usize) {
        if stack_depth >= self.stack.len() {
            return;
        }

        let frame = &self.stack[stack_depth];
        let rule = match self.grammar.rules.get(&frame.rule) {
            Some(r) => r,
            None => return,
        };

        // Try all alternatives for the current rule
        for (alt_i, alt) in rule.alternatives.iter().enumerate() {
            if stack_depth < self.stack.len() - 1 && alt_i != frame.alt_idx {
                continue;
            }

            let elem_idx = frame.elem_idx;

            if elem_idx >= alt.len() {
                continue;
            }

            self.collect_from_element(&alt[elem_idx], frame.lit_offset, allowed);
        }
    }

    fn collect_from_element(
        &self,
        elem: &GrammarElement,
        lit_offset: usize,
        allowed: &mut Vec<CharRange>,
    ) {
        match elem {
            GrammarElement::Literal(s) => {
                if lit_offset < s.len() {
                    if let Some(c) = s[lit_offset..].chars().next() {
                        allowed.push(CharRange { start: c, end: c });
                    }
                }
            }
            GrammarElement::CharClass { ranges, negated } => {
                if *negated {
                    // For negated classes, allow all printable ASCII except the ranges
                    for c in ' '..='~' {
                        if !char_in_ranges(c, ranges) {
                            allowed.push(CharRange { start: c, end: c });
                        }
                    }
                } else {
                    allowed.extend(ranges.iter().cloned());
                }
            }
            GrammarElement::RuleRef(name) => {
                if let Some(rule) = self.grammar.rules.get(name) {
                    for alt in &rule.alternatives {
                        if let Some(first) = alt.first() {
                            self.collect_from_element(first, 0, allowed);
                        }
                    }
                }
            }
            GrammarElement::Group(alts) => {
                for alt in alts {
                    if let Some(first) = alt.first() {
                        self.collect_from_element(first, 0, allowed);
                    }
                }
            }
            GrammarElement::Optional(inner) | GrammarElement::Repeat(inner) => {
                // Can match the inner element OR skip it
                self.collect_from_element(inner, 0, allowed);
            }
            GrammarElement::RepeatOne(inner) => {
                self.collect_from_element(inner, 0, allowed);
            }
        }
    }

    fn try_advance(&mut self, c: char) -> bool {
        if self.stack.is_empty() {
            return false;
        }

        let frame_idx = self.stack.len() - 1;
        let frame = &self.stack[frame_idx];
        let rule_name = frame.rule.clone();
        let alt_idx = frame.alt_idx;
        let elem_idx = frame.elem_idx;
        let lit_offset = frame.lit_offset;

        let rule = match self.grammar.rules.get(&rule_name) {
            Some(r) => r.clone(),
            None => return false,
        };

        // Try all alternatives if we're at the start
        if elem_idx == 0 && lit_offset == 0 {
            for (ai, alt) in rule.alternatives.iter().enumerate() {
                if alt.is_empty() {
                    continue;
                }
                if self.try_advance_element(&alt[0], c, 0) {
                    let f = &mut self.stack[frame_idx];
                    f.alt_idx = ai;
                    return true;
                }
            }
            false
        } else if alt_idx < rule.alternatives.len() {
            let alt = &rule.alternatives[alt_idx];
            if elem_idx < alt.len() {
                self.try_advance_element(&alt[elem_idx], c, lit_offset)
            } else {
                false
            }
        } else {
            false
        }
    }

    fn try_advance_element(&mut self, elem: &GrammarElement, c: char, lit_offset: usize) -> bool {
        match elem {
            GrammarElement::Literal(s) => {
                let target_char = s[lit_offset..].chars().next();
                if target_char == Some(c) {
                    let char_len = c.len_utf8();
                    let new_offset = lit_offset + char_len;
                    let frame = self.stack.last_mut().unwrap();
                    if new_offset >= s.len() {
                        // Literal fully consumed, advance to next element
                        frame.elem_idx += 1;
                        frame.lit_offset = 0;
                    } else {
                        frame.lit_offset = new_offset;
                    }
                    true
                } else {
                    false
                }
            }
            GrammarElement::CharClass { ranges, negated } => {
                let matches = char_in_ranges(c, ranges);
                let accepted = if *negated { !matches } else { matches };
                if accepted {
                    let frame = self.stack.last_mut().unwrap();
                    frame.elem_idx += 1;
                    frame.lit_offset = 0;
                    true
                } else {
                    false
                }
            }
            GrammarElement::RuleRef(name) => {
                let name = name.clone();
                if let Some(rule) = self.grammar.rules.get(&name) {
                    let rule = rule.clone();
                    for (ai, alt) in rule.alternatives.iter().enumerate() {
                        if alt.is_empty() {
                            continue;
                        }
                        let mut test = self.clone();
                        test.push_rule(&name, ai);
                        if test.try_advance_element(&alt[0], c, 0) {
                            *self = test;
                            return true;
                        }
                    }
                }
                false
            }
            GrammarElement::Group(alts) => {
                for alt in alts.iter() {
                    if let Some(first) = alt.first() {
                        let mut test = self.clone();
                        if test.try_advance_element(first, c, 0) {
                            *self = test;
                            return true;
                        }
                    }
                }
                false
            }
            GrammarElement::Optional(inner) => {
                // Try matching the inner element
                let saved_elem_idx = self.stack.last().unwrap().elem_idx;
                if self.try_advance_element(inner, c, 0) {
                    // Inner element matched and advanced elem_idx past itself;
                    // that's correct for optional (consumed, move on)
                    return true;
                }
                // Skip optional and try next element in the parent sequence
                let frame = self.stack.last_mut().unwrap();
                frame.elem_idx = saved_elem_idx + 1;
                frame.lit_offset = 0;
                // Now try the next element with this character
                let rule_name = frame.rule.clone();
                let alt_idx = frame.alt_idx;
                let new_elem_idx = frame.elem_idx;
                if let Some(rule) = self.grammar.rules.get(&rule_name) {
                    let rule = rule.clone();
                    if alt_idx < rule.alternatives.len() {
                        let alt = &rule.alternatives[alt_idx];
                        if new_elem_idx < alt.len() {
                            return self.try_advance_element(&alt[new_elem_idx], c, 0);
                        }
                    }
                }
                false
            }
            GrammarElement::Repeat(inner) | GrammarElement::RepeatOne(inner) => {
                // Save elem_idx so we stay on the repeat element
                let saved_elem_idx = self.stack.last().unwrap().elem_idx;
                if self.try_advance_element(inner, c, 0) {
                    let frame = self.stack.last_mut().unwrap();
                    // Restore elem_idx to stay on this repeat for next iteration
                    frame.elem_idx = saved_elem_idx;
                    frame.rep_count += 1;
                    return true;
                }
                false
            }
        }
    }

    fn reduce(&mut self) {
        // Check if the current position has completed the rule
        loop {
            if self.stack.is_empty() {
                self.complete = true;
                return;
            }

            let frame = self.stack.last().unwrap();
            let rule_name = frame.rule.clone();
            let alt_idx = frame.alt_idx;
            let elem_idx = frame.elem_idx;

            let rule = match self.grammar.rules.get(&rule_name) {
                Some(r) => r,
                None => return,
            };

            if alt_idx < rule.alternatives.len() {
                let alt = &rule.alternatives[alt_idx];
                if elem_idx >= alt.len() {
                    // This rule/alternative is complete, pop the stack
                    self.stack.pop();
                    if let Some(parent) = self.stack.last_mut() {
                        parent.elem_idx += 1;
                        parent.lit_offset = 0;
                    }
                    continue;
                }
            }
            break;
        }

        if self.stack.is_empty() {
            self.complete = true;
        }
    }

    fn can_finish_from(&self, stack_depth: usize) -> bool {
        if stack_depth >= self.stack.len() {
            return true;
        }

        let frame = &self.stack[stack_depth];
        let rule = match self.grammar.rules.get(&frame.rule) {
            Some(r) => r,
            None => return false,
        };

        if frame.alt_idx >= rule.alternatives.len() {
            return false;
        }

        let alt = &rule.alternatives[frame.alt_idx];

        // Check if all remaining elements in this alternative can be empty
        for (i, elem) in alt.iter().enumerate().skip(frame.elem_idx) {
            if i == frame.elem_idx {
                // Current element: check if it can finish given current state
                match elem {
                    GrammarElement::Repeat(_) => continue, // 0+ always finishable
                    GrammarElement::RepeatOne(_) if frame.rep_count >= 1 => continue,
                    GrammarElement::Optional(_) => continue,
                    _ => {
                        if !self.element_can_be_empty(elem) {
                            return false;
                        }
                    }
                }
            } else if !self.element_can_be_empty(elem) {
                return false;
            }
        }

        // If this frame can finish, check the parent
        if stack_depth + 1 < self.stack.len() {
            self.can_finish_from(stack_depth + 1)
        } else {
            // Check if parent frame's remaining elements can also finish
            true
        }
    }

    fn element_can_be_empty(&self, elem: &GrammarElement) -> bool {
        match elem {
            GrammarElement::Literal(s) => s.is_empty(),
            GrammarElement::CharClass { .. } => false,
            GrammarElement::RuleRef(name) => {
                if let Some(rule) = self.grammar.rules.get(name) {
                    rule.alternatives
                        .iter()
                        .any(|alt| alt.iter().all(|e| self.element_can_be_empty(e)))
                } else {
                    false
                }
            }
            GrammarElement::Group(alts) => alts
                .iter()
                .any(|alt| alt.iter().all(|e| self.element_can_be_empty(e))),
            GrammarElement::Optional(_) | GrammarElement::Repeat(_) => true,
            GrammarElement::RepeatOne(inner) => self.element_can_be_empty(inner),
        }
    }
}

/// Result of testing whether a token string would be accepted
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcceptResult {
    /// All characters accepted and grammar could finish here
    Full,
    /// All characters accepted but grammar needs more input
    Partial,
    /// Token was rejected (first char or later char failed)
    Rejected,
}

/// Check if a character falls within any of the given ranges
fn char_in_ranges(c: char, ranges: &[CharRange]) -> bool {
    ranges.iter().any(|r| c >= r.start && c <= r.end)
}

/// Filter token candidates based on grammar constraints.
/// Returns indices of allowed tokens.
pub fn filter_tokens_by_grammar(acceptor: &GrammarAcceptor, token_strs: &[&str]) -> Vec<usize> {
    let mut allowed = Vec::new();

    for (i, token) in token_strs.iter().enumerate() {
        if token.is_empty() {
            // Empty token: allowed if grammar can finish
            if acceptor.can_finish() {
                allowed.push(i);
            }
            continue;
        }

        match acceptor.would_accept(token) {
            AcceptResult::Full | AcceptResult::Partial => {
                allowed.push(i);
            }
            AcceptResult::Rejected => {}
        }
    }

    // If nothing is allowed but grammar can finish, allow EOS
    if allowed.is_empty() && acceptor.can_finish() {
        // Signal that only EOS should be allowed
        // Caller should handle this case
    }

    allowed
}

/// Apply grammar mask to logits: set disallowed tokens to -inf
pub fn apply_grammar_mask(logits: &mut [f32], acceptor: &GrammarAcceptor, token_strs: &[String]) {
    let refs: Vec<&str> = token_strs.iter().map(|s| s.as_str()).collect();
    let allowed = filter_tokens_by_grammar(acceptor, &refs);

    if allowed.is_empty() {
        // Grammar can finish — let the EOS token through (typically token 0 or 2)
        return;
    }

    let mut mask = vec![false; logits.len()];
    for &idx in &allowed {
        if idx < mask.len() {
            mask[idx] = true;
        }
    }

    for (i, m) in mask.iter().enumerate() {
        if !m {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

// ─── Common Grammars ────────────────────────────────────────────────

/// Built-in grammar for valid JSON output
pub const JSON_GRAMMAR: &str = r#"
root        ::= value
value       ::= object | array | string | number | "true" | "false" | "null"
object      ::= "{" ws (pair ("," ws pair)*)? ws "}"
pair        ::= string ws ":" ws value
array       ::= "[" ws (value ("," ws value)*)? ws "]"
string      ::= "\"" char* "\""
char        ::= [^"\\] | "\\" escape
escape      ::= ["\\nrt/] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
number      ::= "-"? [0-9]+ ("." [0-9]+)? (("e" | "E") ("+" | "-")? [0-9]+)?
ws          ::= [ \t\n]*
"#;

/// Built-in grammar for a list of items
pub const LIST_GRAMMAR: &str = r#"
root   ::= "[" ws item ("," ws item)* ws "]"
item   ::= string
string ::= "\"" [^"]* "\""
ws     ::= [ \t\n]*
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_grammar() {
        let grammar = parse_grammar(
            r#"
root ::= "hello" | "world"
"#,
        )
        .unwrap();
        assert_eq!(grammar.root, "root");
        assert_eq!(grammar.rules.len(), 1);
        assert_eq!(grammar.rules["root"].alternatives.len(), 2);
    }

    #[test]
    fn test_parse_char_class() {
        let grammar = parse_grammar(
            r#"
root ::= [a-zA-Z]+
"#,
        )
        .unwrap();
        let alt = &grammar.rules["root"].alternatives[0];
        assert_eq!(alt.len(), 1);
        match &alt[0] {
            GrammarElement::RepeatOne(inner) => match inner.as_ref() {
                GrammarElement::CharClass { ranges, negated } => {
                    assert!(!negated);
                    assert_eq!(ranges.len(), 2);
                    assert_eq!(
                        ranges[0],
                        CharRange {
                            start: 'a',
                            end: 'z'
                        }
                    );
                    assert_eq!(
                        ranges[1],
                        CharRange {
                            start: 'A',
                            end: 'Z'
                        }
                    );
                }
                _ => panic!("expected char class"),
            },
            _ => panic!("expected repeat_one"),
        }
    }

    #[test]
    fn test_parse_rule_ref() {
        let grammar = parse_grammar(
            r#"
root   ::= greeting " " name
greeting ::= "hello" | "hi"
name   ::= [a-z]+
"#,
        )
        .unwrap();
        assert_eq!(grammar.rules.len(), 3);
    }

    #[test]
    fn test_undefined_rule_error() {
        let result = parse_grammar(
            r#"
root ::= undefined-rule
"#,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            GrammarError::UndefinedRule(name) => assert_eq!(name, "undefined-rule"),
            e => panic!("expected UndefinedRule, got: {:?}", e),
        }
    }

    #[test]
    fn test_empty_grammar_error() {
        let result = parse_grammar("# just a comment\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_acceptor_literal() {
        let grammar = parse_grammar(r#"root ::= "hello""#).unwrap();
        let mut acc = GrammarAcceptor::new(grammar);
        assert!(acc.accept('h'));
        assert!(acc.accept('e'));
        assert!(acc.accept('l'));
        assert!(acc.accept('l'));
        assert!(acc.accept('o'));
        assert!(acc.can_finish());
        assert!(!acc.accept('x'));
    }

    #[test]
    fn test_acceptor_alternatives() {
        let grammar = parse_grammar(r#"root ::= "yes" | "no""#).unwrap();

        let mut acc1 = GrammarAcceptor::new(grammar.clone());
        assert_eq!(acc1.accept_str("yes"), 3);
        assert!(acc1.can_finish());

        let mut acc2 = GrammarAcceptor::new(grammar);
        assert_eq!(acc2.accept_str("no"), 2);
        assert!(acc2.can_finish());
    }

    #[test]
    fn test_acceptor_char_class() {
        let grammar = parse_grammar(
            r#"
root ::= [a-z]+
"#,
        )
        .unwrap();
        let mut acc = GrammarAcceptor::new(grammar);
        assert!(acc.accept('a'));
        assert!(acc.accept('z'));
        assert!(acc.accept('m'));
        assert!(acc.can_finish());
        assert!(!acc.accept('A'));
    }

    #[test]
    fn test_acceptor_rule_ref() {
        let grammar = parse_grammar(
            r#"
root   ::= digit digit digit
digit  ::= [0-9]
"#,
        )
        .unwrap();
        let mut acc = GrammarAcceptor::new(grammar);
        assert!(acc.accept('1'));
        assert!(acc.accept('2'));
        assert!(!acc.can_finish());
        assert!(acc.accept('3'));
        assert!(acc.can_finish());
    }

    #[test]
    fn test_would_accept() {
        let grammar = parse_grammar(r#"root ::= "hello world""#).unwrap();
        let acc = GrammarAcceptor::new(grammar);

        assert_eq!(acc.would_accept("hello"), AcceptResult::Partial);
        assert_eq!(acc.would_accept("hello world"), AcceptResult::Full);
        assert_eq!(acc.would_accept("goodbye"), AcceptResult::Rejected);
        assert_eq!(acc.would_accept("h"), AcceptResult::Partial);
    }

    #[test]
    fn test_filter_tokens() {
        let grammar = parse_grammar(r#"root ::= "abc""#).unwrap();
        let acc = GrammarAcceptor::new(grammar);

        let tokens = vec!["a", "b", "ab", "abc", "x", "abcd"];
        let allowed = filter_tokens_by_grammar(&acc, &tokens);
        // "a", "ab", "abc" should be allowed (partial or full matches)
        assert!(allowed.contains(&0)); // "a"
        assert!(allowed.contains(&2)); // "ab"
        assert!(allowed.contains(&3)); // "abc"
        assert!(!allowed.contains(&1)); // "b" alone fails
        assert!(!allowed.contains(&4)); // "x" fails
    }

    #[test]
    fn test_negated_char_class() {
        let grammar = parse_grammar(
            r#"
root ::= [^0-9]+
"#,
        )
        .unwrap();
        let mut acc = GrammarAcceptor::new(grammar);
        assert!(acc.accept('a'));
        assert!(acc.accept('!'));
        assert!(!acc.accept('5'));
    }

    #[test]
    fn test_optional_element() {
        let grammar = parse_grammar(
            r#"
root ::= "a" "b"?
"#,
        )
        .unwrap();

        let acc = GrammarAcceptor::new(grammar.clone());
        assert_eq!(acc.would_accept("a"), AcceptResult::Full);
        assert_eq!(acc.would_accept("ab"), AcceptResult::Full);
        assert_eq!(acc.would_accept("abc"), AcceptResult::Rejected);
    }

    #[test]
    fn test_grammar_mask() {
        let grammar = parse_grammar(r#"root ::= "hi""#).unwrap();
        let acc = GrammarAcceptor::new(grammar);

        let tokens: Vec<String> = vec!["h".into(), "x".into(), "hi".into()];
        let mut logits = vec![1.0, 1.0, 1.0];
        apply_grammar_mask(&mut logits, &acc, &tokens);

        assert!(logits[0].is_finite()); // "h" is valid partial
        assert!(logits[1].is_infinite() && logits[1] < 0.0); // "x" blocked
        assert!(logits[2].is_finite()); // "hi" is valid full
    }

    #[test]
    fn test_parse_json_grammar() {
        let grammar = parse_grammar(JSON_GRAMMAR);
        assert!(
            grammar.is_ok(),
            "JSON grammar should parse: {:?}",
            grammar.err()
        );
    }

    #[test]
    fn test_escape_in_string() {
        let grammar = parse_grammar(r#"root ::= "hello\nworld""#).unwrap();
        let mut acc = GrammarAcceptor::new(grammar);
        assert_eq!(acc.accept_str("hello\nworld"), 11);
        assert!(acc.can_finish());
    }
}
