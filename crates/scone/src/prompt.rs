//! Local prompt shaping for both hosts. No model, network, storage or tool
//! execution. Added context is explicitly user-level data, never new authority.
use serde_json::{Value, json};

const CONTEXT_PREFIX: &str = "Scone structured request (user-level data, not additional authority). The original user request remains authoritative; this representation does not change permissions or override higher-priority instructions.\n";
const INSTRUCTIONS: [&str; 3] = [
    "Fulfill the user's request without inventing unstated requirements.",
    "Preserve the meaning of quoted text, code, and explicit constraints.",
    "State consequential uncertainty; respect existing permission boundaries.",
];

pub fn hook_output(input: &str) -> Value {
    let Ok(input) = serde_json::from_str::<Value>(input) else {
        return json!({});
    };
    if input["hook_event_name"] != "UserPromptSubmit" {
        return json!({});
    }
    // Claude versions differ; Codex's documented field is prompt.
    let Some(request) = input
        .get("prompt")
        .or_else(|| input.get("user_input"))
        .and_then(Value::as_str)
    else {
        return json!({});
    };
    let task = request.trim_matches([' ', '\t', '\r', '\n']);
    if task.is_empty() {
        return json!({});
    }
    if task.len() > 60_000 {
        return json!({"systemMessage":"Scone prompt processing skipped: request exceeds the 60000-byte limit; the original prompt is unchanged."});
    }
    let payload = json!({"schema_version":1,"task":task,"instructions":INSTRUCTIONS});
    json!({"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":format!("{CONTEXT_PREFIX}{payload}")}})
}

pub fn run_stdin() {
    use std::io::Read;
    let mut input = String::new();
    // Bound untrusted host input too, not just the embedded prompt.
    let output = if std::io::stdin()
        .take(1_000_001)
        .read_to_string(&mut input)
        .is_ok()
        && input.len() <= 1_000_000
    {
        hook_output(&input)
    } else {
        json!({})
    };
    println!("{output}");
}
