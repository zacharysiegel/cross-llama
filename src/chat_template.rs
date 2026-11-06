//! Formats a conversation with a model's Jinja chat template as Hugging Face's transformers and llama.cpp's server do.

use minijinja::syntax::SyntaxConfig;
use minijinja::value::Serde;
use minijinja::{Environment, ErrorKind, Template, Value};

use crate::{ChatMessage, Error, Role};

const TEMPLATE_NAME: &str = "chat_template";
const RAISE_EXCEPTION_FUNCTION_NAME: &str = "raise_exception";
const SYSTEM_ROLE_PROBE_TEXT: &str = "cross-llama system role probe";
const USER_PROBE_TEXT: &str = "cross-llama user probe";
// llama.cpp's server's separator.
const SYSTEM_MESSAGE_SEPARATOR: &str = "\n";
// llama.cpp's template for a model file that has none.
const CHATML_TEMPLATE_SOURCE: &str = concat!(
    "{%- for message in messages -%}\n",
    "  {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}\n",
    "{%- endfor -%}\n",
    "{%- if add_generation_prompt -%}\n",
    "  {{- '<|im_start|>assistant\\n' -}}\n",
    "{%- endif -%}",
);

minimer::impl_from_error!(Error, minijinja::Error);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceTokens {
    /// Empty when the vocabulary has no beginning-of-sequence token.
    pub beginning_text: String,
    /// Empty when the vocabulary has no end-of-sequence token.
    pub end_text: String,
    pub tokenizer_adds_beginning: bool,
    pub tokenizer_adds_end: bool,
}

pub struct ChatTemplate {
    environment: Environment<'static>,
    sequence_tokens: SequenceTokens,
    system_role_support: SystemRoleSupportKind,
}

impl ChatTemplate {
    /// Formats with ChatML when the model file has no template.
    pub fn new(model_template_source: Option<String>, sequence_tokens: SequenceTokens) -> Result<Self, Error> {
        let source: String = model_template_source.unwrap_or_else(|| CHATML_TEMPLATE_SOURCE.to_string());
        let environment: Environment<'static> = create_environment(source)?;
        let system_role_support: SystemRoleSupportKind = detect_system_role_support(&environment, &sequence_tokens);

        Ok(ChatTemplate {
            environment,
            sequence_tokens,
            system_role_support,
        })
    }

    /// Ends where the model's reply begins.
    pub fn render_prompt(&self, messages: &[ChatMessage]) -> Result<String, Error> {
        let message_jsons: Vec<serde_json::Value> = match self.system_role_support {
            SystemRoleSupportKind::Rendered => messages.iter().map(create_message_json).collect(),
            SystemRoleSupportKind::MergedIntoNextMessage => merge_leading_system_message(messages),
        };

        let rendered: String = render(&self.environment, message_jsons, &self.sequence_tokens)?;

        Ok(self.remove_sequence_tokens_tokenizer_adds(&rendered).to_string())
    }

    /// Tokenizing would otherwise add a second copy of a sequence token the template already wrote.
    fn remove_sequence_tokens_tokenizer_adds<'prompt>(&self, rendered: &'prompt str) -> &'prompt str {
        let mut prompt: &str = rendered;

        if self.sequence_tokens.tokenizer_adds_beginning {
            prompt = prompt.strip_prefix(self.sequence_tokens.beginning_text.as_str()).unwrap_or(prompt);
        }

        if self.sequence_tokens.tokenizer_adds_end {
            prompt = prompt.strip_suffix(self.sequence_tokens.end_text.as_str()).unwrap_or(prompt);
        }

        prompt
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SystemRoleSupportKind {
    Rendered,
    /// The template rejects or drops a system message, so its text joins the next message, as llama.cpp's server does.
    MergedIntoNextMessage,
}

fn detect_system_role_support(environment: &Environment<'static>, sequence_tokens: &SequenceTokens) -> SystemRoleSupportKind {
    let probe_message_jsons: Vec<serde_json::Value> = vec![
        create_message_json(&ChatMessage {
            role: Role::System,
            text: SYSTEM_ROLE_PROBE_TEXT.to_string(),
        }),
        create_message_json(&ChatMessage {
            role: Role::User,
            text: USER_PROBE_TEXT.to_string(),
        }),
    ];
    let rendered: Result<String, Error> = render(environment, probe_message_jsons, sequence_tokens);

    match rendered {
        Ok(prompt) if prompt.contains(SYSTEM_ROLE_PROBE_TEXT) => SystemRoleSupportKind::Rendered,
        _ => SystemRoleSupportKind::MergedIntoNextMessage,
    }
}

/// Joins a leading system message's text to the message after it, or drops it when it is the only message.
fn merge_leading_system_message(messages: &[ChatMessage]) -> Vec<serde_json::Value> {
    let Some((first_message, later_messages)) = messages.split_first()
    else {
        return Vec::new();
    };

    if first_message.role != Role::System {
        return messages.iter().map(create_message_json).collect();
    }

    let Some((next_message, remaining_messages)) = later_messages.split_first()
    else {
        return Vec::new();
    };

    let merged_message: ChatMessage = ChatMessage {
        role: next_message.role,
        text: format!("{}{SYSTEM_MESSAGE_SEPARATOR}{}", first_message.text, next_message.text),
    };

    std::iter::once(&merged_message).chain(remaining_messages).map(create_message_json).collect()
}

fn create_message_json(message: &ChatMessage) -> serde_json::Value {
    serde_json::json!({
        "role": message.role.chat_template_name(),
        "content": message.text,
    })
}

fn render(environment: &Environment<'static>, message_jsons: Vec<serde_json::Value>, sequence_tokens: &SequenceTokens) -> Result<String, Error> {
    let template: Template = environment.get_template(TEMPLATE_NAME)?;
    let context_json: serde_json::Value = serde_json::json!({
        "messages": message_jsons,
        "add_generation_prompt": true,
        "bos_token": sequence_tokens.beginning_text,
        "eos_token": sequence_tokens.end_text,
        // Turns off a reasoning model's thinking.
        "enable_thinking": false,
    });

    let rendered: String = template.render(Value::from(Serde(&context_json)))?;
    Ok(rendered)
}

fn create_environment(source: String) -> Result<Environment<'static>, Error> {
    let mut environment: Environment<'static> = Environment::new();
    let syntax: SyntaxConfig = SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build()?;

    environment.set_syntax(syntax);
    // Chat templates are written for Python's Jinja and call Python's string and dictionary methods.
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    environment.add_function(RAISE_EXCEPTION_FUNCTION_NAME, raise_exception);
    environment.add_template_owned(TEMPLATE_NAME, source)?;

    Ok(environment)
}

/// Templates call this to reject a conversation they cannot format.
fn raise_exception(message: String) -> Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
}

#[cfg(test)]
mod tests {
    use crate::chat_template::{ChatTemplate, SequenceTokens};
    use crate::{ChatMessage, Error, Role};

    // The template in gemma-2's GGUF files.
    const GEMMA_2_TEMPLATE_SOURCE: &str = concat!(
        "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}",
        "{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}",
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}",
        "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}",
        "{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}",
        "{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}",
    );
    fn create_message(role: Role, text: &str) -> ChatMessage {
        ChatMessage {
            role,
            text: text.to_string(),
        }
    }

    fn create_template(model_template_source: Option<&str>) -> ChatTemplate {
        let sequence_tokens: SequenceTokens = SequenceTokens {
            beginning_text: "<s>".to_string(),
            end_text: "</s>".to_string(),
            tokenizer_adds_beginning: true,
            tokenizer_adds_end: false,
        };

        ChatTemplate::new(model_template_source.map(str::to_string), sequence_tokens).unwrap()
    }

    #[test]
    fn render_prompt_turns_thinking_off() {
        let template: ChatTemplate = create_template(Some("{% if enable_thinking is defined and not enable_thinking %}off{% endif %}"));

        let prompt: String = template.render_prompt(&[]).unwrap();

        assert_eq!(prompt, "off");
    }

    #[test]
    fn render_prompt_supports_python_string_and_dictionary_methods() {
        let template: ChatTemplate = create_template(Some("{{ messages[0].get('role') }} {{ messages[0]['content'].strip().upper() }}"));

        let prompt: String = template.render_prompt(&[create_message(Role::User, " hello ")]).unwrap();

        assert_eq!(prompt, "user HELLO");
    }

    #[test]
    fn render_prompt_fails_when_template_raises_exception() {
        let template: ChatTemplate = create_template(Some("{{ raise_exception('roles must alternate') }}"));

        let rendered: Result<String, Error> = template.render_prompt(&[]);

        assert!(rendered.is_err());
    }

    #[test]
    fn render_prompt_formats_with_chatml_when_model_has_no_template() {
        let template: ChatTemplate = create_template(None);

        let prompt: String = template.render_prompt(&[create_message(Role::User, "Olá")]).unwrap();

        assert_eq!(prompt, "<|im_start|>user\nOlá<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn render_prompt_merges_system_message_into_next_when_template_rejects_system_role() {
        let template: ChatTemplate = create_template(Some(GEMMA_2_TEMPLATE_SOURCE));
        let messages: Vec<ChatMessage> = vec![
            create_message(Role::System, "Reply in Portuguese."),
            create_message(Role::User, "Say hello."),
        ];

        let prompt: String = template.render_prompt(&messages).unwrap();

        assert_eq!(prompt, "<start_of_turn>user\nReply in Portuguese.\nSay hello.<end_of_turn>\n<start_of_turn>model\n");
    }

    #[test]
    fn render_prompt_merges_system_message_into_next_when_template_drops_system_role() {
        let template: ChatTemplate = create_template(Some(
            "{% for message in messages %}{% if message.role == 'user' %}{{ '[INST] ' + message.content + ' [/INST]' }}{% endif %}{% endfor %}",
        ));
        let messages: Vec<ChatMessage> = vec![
            create_message(Role::System, "Reply in Portuguese."),
            create_message(Role::User, "Say hello."),
        ];

        let prompt: String = template.render_prompt(&messages).unwrap();

        assert_eq!(prompt, "[INST] Reply in Portuguese.\nSay hello. [/INST]");
    }
}
