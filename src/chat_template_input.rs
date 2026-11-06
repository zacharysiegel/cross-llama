//! The variables both backends give a model's chat template, so the same request formats to the same prompt on each.

use crate::ChatMessage;

pub fn create_messages_json(messages: &[ChatMessage]) -> serde_json::Value {
    let message_jsons: Vec<serde_json::Value> = messages
        .iter()
        .map(|message| {
            serde_json::json!({
                "role": message.role.chat_template_name(),
                "content": message.text,
            })
        })
        .collect();

    serde_json::Value::Array(message_jsons)
}

/// Turns off a reasoning model's thinking.
pub fn create_template_arguments_json() -> serde_json::Value {
    serde_json::json!({
        "enable_thinking": false,
    })
}
