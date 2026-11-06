use crate::{Error, GenerationRequest};

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

/// A schema-constrained reply is a prefix of a matching JSON value, so one that parses is complete.
pub fn check_reply_completes_json_value(request: &GenerationRequest, reply_text: &str, ending: ReplyEnding) -> Result<(), Error> {
    let constrained_by_json_schema: bool = request.json_schema.is_some();

    if !constrained_by_json_schema || ending == ReplyEnding::EndOfGeneration {
        return Ok(());
    }

    let parsed_reply: Result<serde_json::Value, serde_json::Error> = serde_json::from_str(reply_text);

    if parsed_reply.is_err() {
        return Err(Error::new(&format!(
            "the reply reached the maximum of {} tokens before completing the JSON value its schema requires",
            request.maximum_token_count,
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::generation_checks::ReplyEnding;
    use crate::{Error, GenerationRequest, generation_checks};

    #[test]
    fn check_reply_completes_json_value_accepts_complete_value_at_maximum() {
        let request: GenerationRequest = GenerationRequest {
            messages: Vec::new(),
            json_schema: Some(serde_json::json!({ "type": "object" })),
            temperature: 0.0,
            seed: 0,
            maximum_token_count: 8,
        };

        let checked: Result<(), Error> = generation_checks::check_reply_completes_json_value(&request, "{\"a\": 1}\n", ReplyEnding::MaximumTokenCount);

        assert!(checked.is_ok());
    }
}
