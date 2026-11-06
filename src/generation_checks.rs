use crate::{Error, GenerationRequest};

const JSON_WHITESPACE_CHARACTERS: [char; 4] = [' ', '\t', '\n', '\r'];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplyEnding {
    EndOfGeneration,
    MaximumTokenCount,
}

pub fn check_context_capacity(prompt_token_count: usize, maximum_token_count: usize, context_token_count: usize) -> Result<(), Error> {
    let required_token_count: usize = prompt_token_count.saturating_add(maximum_token_count);

    if required_token_count > context_token_count {
        return Err(Error::new(&format!(
            "the prompt's {prompt_token_count} tokens and up to {maximum_token_count} reply tokens exceed the context of {context_token_count} tokens",
        )));
    }

    Ok(())
}

pub fn check_reply_completes_json_value(request: &GenerationRequest, reply_text: &str, ending: ReplyEnding) -> Result<(), Error> {
    let constrained_by_json_schema: bool = request.json_schema.is_some();

    if !constrained_by_json_schema || ending == ReplyEnding::EndOfGeneration {
        return Ok(());
    }

    if !is_complete_json_value(reply_text) {
        return Err(Error::new(&format!(
            "the reply reached the maximum of {} tokens before completing the JSON value its schema requires",
            request.maximum_token_count,
        )));
    }

    Ok(())
}

/// A schema-constrained reply is a prefix of a matching JSON value, so one that parses is complete, except a number,
/// which only following whitespace shows to have ended.
fn is_complete_json_value(reply_text: &str) -> bool {
    let parsed_reply: Result<serde_json::Value, serde_json::Error> = serde_json::from_str(reply_text);

    match parsed_reply {
        Ok(serde_json::Value::Number(_)) => reply_text.ends_with(JSON_WHITESPACE_CHARACTERS),
        Ok(_) => true,
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::generation_checks::ReplyEnding;
    use crate::{Error, GenerationRequest, generation_checks};

    fn create_request(json_schema: serde_json::Value) -> GenerationRequest {
        GenerationRequest {
            messages: Vec::new(),
            json_schema: Some(json_schema),
            temperature: 0.0,
            seed: 0,
            maximum_token_count: 8,
        }
    }

    #[test]
    fn check_reply_completes_json_value_accepts_complete_value_at_maximum() {
        let request: GenerationRequest = create_request(serde_json::json!({ "type": "object" }));

        let checked: Result<(), Error> = generation_checks::check_reply_completes_json_value(&request, "{\"a\": 1}\n", ReplyEnding::MaximumTokenCount);

        assert!(checked.is_ok());
    }

    #[test]
    fn check_reply_completes_json_value_rejects_number_that_may_continue_at_maximum() {
        let request: GenerationRequest = create_request(serde_json::json!({ "type": "integer" }));

        let checked: Result<(), Error> = generation_checks::check_reply_completes_json_value(&request, "10", ReplyEnding::MaximumTokenCount);

        assert!(checked.is_err());
    }

    #[test]
    fn check_reply_completes_json_value_accepts_number_ended_by_whitespace_at_maximum() {
        let request: GenerationRequest = create_request(serde_json::json!({ "type": "integer" }));

        let checked: Result<(), Error> = generation_checks::check_reply_completes_json_value(&request, "100\n", ReplyEnding::MaximumTokenCount);

        assert!(checked.is_ok());
    }
}
