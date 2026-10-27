//! Best-effort routing hints. The parsed value is never used as the wire body.

use serde_json::Value;

pub fn routing_text(body: &[u8], path: &str) -> String {
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        // Backend validation owns malformed JSON and unsupported representations.
        return String::new();
    };
    match path.split('?').next().unwrap_or(path) {
        "/v1/chat/completions" => value
            .pointer("/session_params/session_id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_default()
            .to_owned(),
        "/v1/completions" => text(value.get("prompt")),
        "/generate" => ["text", "prompt", "input_ids"]
            .iter()
            .find_map(|field| value.get(field).filter(|v| !v.is_null()))
            .map(|v| text(Some(v)))
            .unwrap_or_default(),
        "/v1/embeddings" => text(value.get("input")),
        // Keep the existing fallback policy's JSON routing key for Messages,
        // Responses, and other native backend endpoints.
        _ => serde_json::to_string(&value).unwrap_or_default(),
    }
}

fn text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Number(value)) => value.to_string(),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| text(Some(value)))
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}
