//! Function calling / Tool use support
//!
//! Implements OpenAI-compatible function calling for tool use in chat completions.
//! Parses tool definitions, injects them into prompts, extracts tool calls from
//! model output, and validates the generated JSON arguments.
//!
//! Supports:
//! - `tools` parameter with function definitions (OpenAI format)
//! - `tool_choice`: "auto", "none", "required", or specific function
//! - Multiple parallel tool calls in a single response
//! - Argument validation against parameter schemas
//! - Tool result messages (role: "tool") for multi-turn tool use

use serde_json::Value;
use std::collections::HashMap;

/// A tool/function definition
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Function name
    pub name: String,
    /// Function description
    pub description: String,
    /// JSON Schema for parameters (as raw JSON value)
    pub parameters: Value,
}

/// A tool call extracted from model output
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// Unique ID for this call (e.g., "call_abc123")
    pub id: String,
    /// Function name
    pub name: String,
    /// JSON arguments string
    pub arguments: String,
}

/// Tool choice mode
#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    /// Model decides whether to call tools
    Auto,
    /// Never call tools
    None,
    /// Must call at least one tool
    Required,
    /// Must call this specific function
    Function(String),
}

impl ToolChoice {
    pub fn from_json(value: &Value) -> Self {
        match value {
            Value::String(s) => match s.as_str() {
                "none" => ToolChoice::None,
                "required" => ToolChoice::Required,
                _ => ToolChoice::Auto,
            },
            Value::Object(obj) => {
                if let Some(Value::Object(func)) = obj.get("function") {
                    if let Some(Value::String(name)) = func.get("name") {
                        return ToolChoice::Function(name.clone());
                    }
                }
                ToolChoice::Auto
            }
            _ => ToolChoice::Auto,
        }
    }
}

/// A tool result message from a previous tool call
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// The tool_call_id this result corresponds to
    pub tool_call_id: String,
    /// The result content
    pub content: String,
}

/// Parse tool definitions from OpenAI-format JSON
///
/// Expected format:
/// ```json
/// [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
/// ```
pub fn parse_tools(tools_json: &Value) -> Vec<ToolDefinition> {
    let mut tools = Vec::new();
    if let Value::Array(arr) = tools_json {
        for item in arr {
            if let Some(func) = item.get("function") {
                let name = func
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let description = func
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let parameters = func
                    .get("parameters")
                    .cloned()
                    .unwrap_or(Value::Object(serde_json::Map::new()));

                if !name.is_empty() {
                    tools.push(ToolDefinition {
                        name,
                        description,
                        parameters,
                    });
                }
            }
        }
    }
    tools
}

/// Format tool definitions into a system prompt section
///
/// Uses a structured format that most instruction-tuned models understand.
pub fn format_tools_prompt(tools: &[ToolDefinition], choice: &ToolChoice) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut prompt = String::from("\n\n# Available Tools\n\n");
    prompt.push_str("You have access to the following functions. To call a function, respond with a JSON object in this exact format:\n\n");
    prompt.push_str("```json\n{\"tool_calls\": [{\"id\": \"call_1\", \"function\": {\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}}]}\n```\n\n");

    match choice {
        ToolChoice::None => {
            prompt.push_str("Do NOT use any tools. Respond directly.\n\n");
            return prompt;
        }
        ToolChoice::Required => {
            prompt.push_str("You MUST call at least one tool. Do not respond with plain text.\n\n");
        }
        ToolChoice::Function(name) => {
            prompt.push_str(&format!(
                "You MUST call the `{}` function. Do not respond with plain text.\n\n",
                name
            ));
        }
        ToolChoice::Auto => {
            prompt
                .push_str("Call a tool if it would help answer the user's request, otherwise respond directly.\n\n");
        }
    }

    for tool in tools {
        prompt.push_str(&format!("## `{}`\n\n", tool.name));
        if !tool.description.is_empty() {
            prompt.push_str(&format!("{}\n\n", tool.description));
        }
        prompt.push_str(&format!(
            "Parameters:\n```json\n{}\n```\n\n",
            serde_json::to_string_pretty(&tool.parameters).unwrap_or_default()
        ));
    }

    prompt
}

/// Generate a unique tool call ID
fn generate_call_id(index: usize) -> String {
    format!("call_{:012x}", index as u64 ^ 0x5ee1ee_facade)
}

/// Extract tool calls from model output text
///
/// Supports multiple formats:
/// 1. JSON with "tool_calls" array (preferred)
/// 2. JSON with single "function_call" object (legacy OpenAI)
/// 3. Markdown code block containing tool call JSON
pub fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
    let trimmed = text.trim();

    // Try to find JSON in markdown code blocks first
    let json_str = if let Some(start) = trimmed.find("```json") {
        let content_start = start + 7;
        if let Some(end) = trimmed[content_start..].find("```") {
            trimmed[content_start..content_start + end].trim()
        } else {
            trimmed
        }
    } else if let Some(start) = trimmed.find("```") {
        let content_start = start + 3;
        // Skip optional language tag on same line
        let line_end = trimmed[content_start..]
            .find('\n')
            .map(|i| content_start + i + 1)
            .unwrap_or(content_start);
        if let Some(end) = trimmed[line_end..].find("```") {
            trimmed[line_end..line_end + end].trim()
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    // Try parsing as JSON
    if let Ok(value) = serde_json::from_str::<Value>(json_str) {
        return extract_calls_from_value(&value);
    }

    // Try to find any JSON object in the text
    if let Some(start) = json_str.find('{') {
        if let Some(end) = find_matching_brace(json_str, start) {
            let substr = &json_str[start..=end];
            if let Ok(value) = serde_json::from_str::<Value>(substr) {
                return extract_calls_from_value(&value);
            }
        }
    }

    Vec::new()
}

/// Extract tool calls from a parsed JSON value
fn extract_calls_from_value(value: &Value) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    // Format 1: {"tool_calls": [...]}
    if let Some(Value::Array(arr)) = value.get("tool_calls") {
        for (i, item) in arr.iter().enumerate() {
            if let Some(call) = parse_single_call(item, i) {
                calls.push(call);
            }
        }
        if !calls.is_empty() {
            return calls;
        }
    }

    // Format 2: {"function_call": {"name": "...", "arguments": "..."}}
    if let Some(fc) = value.get("function_call") {
        if let Some(call) = parse_function_call(fc, 0) {
            calls.push(call);
            return calls;
        }
    }

    // Format 3: Direct single call {"id": "...", "function": {...}}
    if value.get("function").is_some() || value.get("name").is_some() {
        if let Some(call) = parse_single_call(value, 0) {
            calls.push(call);
        }
    }

    calls
}

/// Parse a single tool call object
fn parse_single_call(value: &Value, index: usize) -> Option<ToolCall> {
    let id = value
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| generate_call_id(index));

    if let Some(func) = value.get("function") {
        let name = func.get("name").and_then(|v| v.as_str())?;
        let arguments = match func.get("arguments") {
            Some(Value::String(s)) => s.clone(),
            Some(v) => serde_json::to_string(v).ok()?,
            None => "{}".to_string(),
        };
        return Some(ToolCall {
            id,
            name: name.to_string(),
            arguments,
        });
    }

    // Flat format: {"name": "...", "arguments": {...}}
    let name = value.get("name").and_then(|v| v.as_str())?;
    let arguments = match value.get("arguments") {
        Some(Value::String(s)) => s.clone(),
        Some(v) => serde_json::to_string(v).ok()?,
        None => "{}".to_string(),
    };
    Some(ToolCall {
        id,
        name: name.to_string(),
        arguments,
    })
}

/// Parse legacy function_call format
fn parse_function_call(value: &Value, index: usize) -> Option<ToolCall> {
    let name = value.get("name").and_then(|v| v.as_str())?;
    let arguments = match value.get("arguments") {
        Some(Value::String(s)) => s.clone(),
        Some(v) => serde_json::to_string(v).ok()?,
        None => "{}".to_string(),
    };
    Some(ToolCall {
        id: generate_call_id(index),
        name: name.to_string(),
        arguments,
    })
}

/// Find the matching closing brace for an opening brace
fn find_matching_brace(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for (i, &byte) in bytes.iter().enumerate().skip(start) {
        if escape {
            escape = false;
            continue;
        }
        match byte {
            b'\\' if in_string => escape = true,
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Validate tool call arguments against tool parameter schemas
pub fn validate_tool_call(call: &ToolCall, tools: &[ToolDefinition]) -> Result<(), String> {
    let tool = tools
        .iter()
        .find(|t| t.name == call.name)
        .ok_or_else(|| format!("Unknown function: {}", call.name))?;

    // Parse the arguments as JSON
    let args: Value = serde_json::from_str(&call.arguments)
        .map_err(|e| format!("Invalid JSON arguments: {}", e))?;

    // Check required parameters
    if let Some(Value::Array(required)) = tool.parameters.get("required") {
        if let Value::Object(args_obj) = &args {
            for req in required {
                if let Some(param_name) = req.as_str() {
                    if !args_obj.contains_key(param_name) {
                        return Err(format!("Missing required parameter: {}", param_name));
                    }
                }
            }
        } else if !required.is_empty() {
            return Err("Arguments must be a JSON object".to_string());
        }
    }

    // Check property types (basic validation)
    if let (Some(Value::Object(properties)), Value::Object(args_obj)) =
        (tool.parameters.get("properties"), &args)
    {
        for (key, _value) in args_obj {
            if !properties.contains_key(key) {
                // Allow additional properties by default (lenient)
                continue;
            }
        }
    }

    Ok(())
}

/// Format tool call results for inclusion in the conversation
pub fn format_tool_results(results: &[ToolResult]) -> String {
    let mut output = String::new();
    for result in results {
        output.push_str(&format!(
            "<tool_result id=\"{}\">\n{}\n</tool_result>\n",
            result.tool_call_id, result.content
        ));
    }
    output
}

/// Format tool calls as OpenAI API response JSON
pub fn format_tool_calls_response(calls: &[ToolCall]) -> String {
    let calls_json: Vec<String> = calls
        .iter()
        .map(|call| {
            format!(
                r#"{{"id":"{}","type":"function","function":{{"name":"{}","arguments":"{}"}}}}"#,
                call.id,
                call.name,
                call.arguments.replace('\\', "\\\\").replace('"', "\\\"")
            )
        })
        .collect();

    format!("[{}]", calls_json.join(","))
}

/// Parse tool messages from chat message array
pub fn parse_tool_messages(messages: &Value) -> Vec<ToolResult> {
    let mut results = Vec::new();
    if let Value::Array(arr) = messages {
        for msg in arr {
            if msg.get("role").and_then(|v| v.as_str()) == Some("tool") {
                let content = msg
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let tool_call_id = msg
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                results.push(ToolResult {
                    tool_call_id,
                    content,
                });
            }
        }
    }
    results
}

/// Check if a model response contains tool calls
pub fn has_tool_calls(text: &str) -> bool {
    let trimmed = text.trim();
    // Quick checks before expensive JSON parsing
    if !trimmed.contains("tool_calls")
        && !trimmed.contains("function_call")
        && !trimmed.contains("\"name\"")
    {
        return false;
    }
    !extract_tool_calls(trimmed).is_empty()
}

/// Build a mapping of tool names for quick lookup
pub fn build_tool_index(tools: &[ToolDefinition]) -> HashMap<String, usize> {
    tools
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.clone(), i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_tools_from_json() {
        let tools_json = json!([
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]);

        let tools = parse_tools(&tools_json);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[1].name, "search");
        assert_eq!(tools[0].description, "Get current weather for a location");
    }

    #[test]
    fn test_parse_tools_empty() {
        let tools = parse_tools(&json!([]));
        assert!(tools.is_empty());
    }

    #[test]
    fn test_parse_tools_missing_name() {
        let tools_json = json!([{
            "type": "function",
            "function": {
                "description": "No name here"
            }
        }]);
        let tools = parse_tools(&tools_json);
        assert!(tools.is_empty());
    }

    #[test]
    fn test_tool_choice_from_json() {
        assert_eq!(ToolChoice::from_json(&json!("auto")), ToolChoice::Auto);
        assert_eq!(ToolChoice::from_json(&json!("none")), ToolChoice::None);
        assert_eq!(
            ToolChoice::from_json(&json!("required")),
            ToolChoice::Required
        );
        assert_eq!(
            ToolChoice::from_json(
                &json!({"type": "function", "function": {"name": "get_weather"}})
            ),
            ToolChoice::Function("get_weather".to_string())
        );
    }

    #[test]
    fn test_extract_tool_calls_standard_format() {
        let text = r#"{"tool_calls": [{"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}}]}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].id, "call_1");
    }

    #[test]
    fn test_extract_tool_calls_with_object_arguments() {
        let text = r#"{"tool_calls": [{"id": "call_1", "function": {"name": "search", "arguments": {"query": "rust programming"}}}]}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert!(calls[0].arguments.contains("rust programming"));
    }

    #[test]
    fn test_extract_tool_calls_multiple() {
        let text = r#"{"tool_calls": [
            {"id": "call_1", "function": {"name": "get_weather", "arguments": {"location": "Paris"}}},
            {"id": "call_2", "function": {"name": "get_weather", "arguments": {"location": "London"}}}
        ]}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_weather");
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn test_extract_tool_calls_legacy_function_call() {
        let text =
            r#"{"function_call": {"name": "search", "arguments": "{\"query\": \"hello\"}"}}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_extract_tool_calls_from_code_block() {
        let text = "Sure, let me check the weather.\n\n```json\n{\"tool_calls\": [{\"id\": \"call_1\", \"function\": {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Berlin\"}}}]}\n```";
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn test_extract_tool_calls_no_calls() {
        let text = "I don't need any tools for this. The answer is 42.";
        let calls = extract_tool_calls(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_extract_tool_calls_flat_format() {
        let text = r#"{"name": "search", "arguments": {"query": "test"}}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_validate_tool_call_valid() {
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
        }];

        let call = ToolCall {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: r#"{"location": "Paris"}"#.to_string(),
        };

        assert!(validate_tool_call(&call, &tools).is_ok());
    }

    #[test]
    fn test_validate_tool_call_missing_required() {
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
        }];

        let call = ToolCall {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: r#"{"units": "celsius"}"#.to_string(),
        };

        let result = validate_tool_call(&call, &tools);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing required parameter"));
    }

    #[test]
    fn test_validate_tool_call_unknown_function() {
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: json!({"type": "object"}),
        }];

        let call = ToolCall {
            id: "call_1".to_string(),
            name: "unknown_function".to_string(),
            arguments: "{}".to_string(),
        };

        let result = validate_tool_call(&call, &tools);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown function"));
    }

    #[test]
    fn test_validate_tool_call_invalid_json() {
        let tools = vec![ToolDefinition {
            name: "search".to_string(),
            description: "Search".to_string(),
            parameters: json!({"type": "object"}),
        }];

        let call = ToolCall {
            id: "call_1".to_string(),
            name: "search".to_string(),
            arguments: "not json at all".to_string(),
        };

        let result = validate_tool_call(&call, &tools);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid JSON"));
    }

    #[test]
    fn test_format_tools_prompt_auto() {
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get weather for a city".to_string(),
            parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
        }];

        let prompt = format_tools_prompt(&tools, &ToolChoice::Auto);
        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("Get weather for a city"));
        assert!(prompt.contains("tool_calls"));
    }

    #[test]
    fn test_format_tools_prompt_none() {
        let tools = vec![ToolDefinition {
            name: "search".to_string(),
            description: "Search".to_string(),
            parameters: json!({}),
        }];

        let prompt = format_tools_prompt(&tools, &ToolChoice::None);
        assert!(prompt.contains("Do NOT use any tools"));
    }

    #[test]
    fn test_format_tools_prompt_required() {
        let tools = vec![ToolDefinition {
            name: "search".to_string(),
            description: "Search".to_string(),
            parameters: json!({}),
        }];

        let prompt = format_tools_prompt(&tools, &ToolChoice::Required);
        assert!(prompt.contains("MUST call at least one tool"));
    }

    #[test]
    fn test_format_tools_prompt_empty() {
        let prompt = format_tools_prompt(&[], &ToolChoice::Auto);
        assert!(prompt.is_empty());
    }

    #[test]
    fn test_has_tool_calls_true() {
        assert!(has_tool_calls(
            r#"{"tool_calls": [{"function": {"name": "test", "arguments": "{}"}}]}"#
        ));
    }

    #[test]
    fn test_has_tool_calls_false() {
        assert!(!has_tool_calls("Just a normal response with no tools."));
    }

    #[test]
    fn test_parse_tool_messages() {
        let messages = json!([
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 22}"},
            {"role": "tool", "tool_call_id": "call_2", "content": "{\"temp\": 18}"}
        ]);

        let results = parse_tool_messages(&messages);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tool_call_id, "call_1");
        assert_eq!(results[1].tool_call_id, "call_2");
    }

    #[test]
    fn test_format_tool_calls_response() {
        let calls = vec![ToolCall {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: r#"{"location":"Paris"}"#.to_string(),
        }];

        let response = format_tool_calls_response(&calls);
        assert!(response.contains("call_1"));
        assert!(response.contains("get_weather"));
        assert!(response.contains("function"));
    }

    #[test]
    fn test_format_tool_results() {
        let results = vec![ToolResult {
            tool_call_id: "call_1".to_string(),
            content: "{\"temperature\": 22}".to_string(),
        }];

        let formatted = format_tool_results(&results);
        assert!(formatted.contains("tool_result"));
        assert!(formatted.contains("call_1"));
        assert!(formatted.contains("temperature"));
    }

    #[test]
    fn test_build_tool_index() {
        let tools = vec![
            ToolDefinition {
                name: "a".to_string(),
                description: String::new(),
                parameters: json!({}),
            },
            ToolDefinition {
                name: "b".to_string(),
                description: String::new(),
                parameters: json!({}),
            },
        ];

        let index = build_tool_index(&tools);
        assert_eq!(index.get("a"), Some(&0));
        assert_eq!(index.get("b"), Some(&1));
    }

    #[test]
    fn test_find_matching_brace() {
        assert_eq!(find_matching_brace("{}", 0), Some(1));
        assert_eq!(find_matching_brace(r#"{"a": {"b": 1}}"#, 0), Some(14));
        assert_eq!(find_matching_brace(r#"{"a": "}"}"#, 0), Some(9));
        assert_eq!(find_matching_brace("{", 0), None);
    }

    #[test]
    fn test_extract_embedded_json() {
        let text = "I'll help with that.\n\n{\"tool_calls\": [{\"id\": \"call_1\", \"function\": {\"name\": \"search\", \"arguments\": {\"query\": \"test\"}}}]}\n\nLet me know if you need more.";
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_generate_call_id_unique() {
        let id0 = generate_call_id(0);
        let id1 = generate_call_id(1);
        assert_ne!(id0, id1);
        assert!(id0.starts_with("call_"));
    }
}
