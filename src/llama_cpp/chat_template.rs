use std::ffi::NulError;

use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::vocab::LlamaVocab;
use llama_cpp_2::{ApplyChatTemplateError, ChatTemplateError, NewLlamaChatMessageError};
use minijinja::syntax::SyntaxConfig;
use minijinja::value::Serde;
use minijinja::{Environment, ErrorKind, Template, Value};

use crate::{ChatMessage, Error, chat_template_input};

const TEMPLATE_NAME: &str = "chat_template";
// llama.cpp formats with ChatML when a model file has no chat template.
const FALLBACK_BUILT_IN_TEMPLATE_NAME: &str = "chatml";
const RAISE_EXCEPTION_FUNCTION_NAME: &str = "raise_exception";
// llama.cpp's value for a token the vocabulary does not have.
const NULL_TOKEN: LlamaToken = LlamaToken(-1);

minimer::impl_from_error!(Error, minijinja::Error);
minimer::impl_from_error!(Error, ChatTemplateError);
minimer::impl_from_error!(Error, ApplyChatTemplateError);
minimer::impl_from_error!(Error, NewLlamaChatMessageError);
minimer::impl_from_error!(Error, NulError);

pub enum ChatTemplate {
    /// The model file's own template.
    Jinja(JinjaChatTemplate),
    /// One of llama.cpp's built-in templates, for a model file without a template.
    BuiltIn(LlamaChatTemplate),
}

impl ChatTemplate {
    pub fn from_model(model: &LlamaModel) -> Result<Self, Error> {
        let model_template: Result<LlamaChatTemplate, ChatTemplateError> = model.chat_template(None);

        match model_template {
            Ok(model_template) => {
                let sequence_tokens: SequenceTokens = SequenceTokens::from_vocabulary(&model.vocab());
                let jinja_template: JinjaChatTemplate = JinjaChatTemplate::new(model_template.to_string()?, sequence_tokens)?;
                Ok(ChatTemplate::Jinja(jinja_template))
            }
            Err(ChatTemplateError::MissingTemplate) => Ok(ChatTemplate::BuiltIn(LlamaChatTemplate::new(FALLBACK_BUILT_IN_TEMPLATE_NAME)?)),
            Err(template_error) => Err(Error::from(template_error)),
        }
    }

    /// Ends where the model's reply begins.
    pub fn render_prompt(&self, model: &LlamaModel, messages: &[ChatMessage]) -> Result<String, Error> {
        match self {
            ChatTemplate::Jinja(jinja_template) => jinja_template.render_prompt(messages),
            ChatTemplate::BuiltIn(built_in_template) => render_built_in_prompt(model, built_in_template, messages),
        }
    }
}

pub struct SequenceTokens {
    pub beginning_text: String,
    pub end_text: String,
    pub tokenizer_adds_beginning: bool,
    pub tokenizer_adds_end: bool,
}

impl SequenceTokens {
    fn from_vocabulary(vocabulary: &LlamaVocab) -> Self {
        SequenceTokens {
            beginning_text: get_token_text(vocabulary, vocabulary.bos()),
            end_text: get_token_text(vocabulary, vocabulary.eos()),
            tokenizer_adds_beginning: vocabulary.should_add_bos(),
            tokenizer_adds_end: vocabulary.should_add_eos(),
        }
    }
}

/// Renders as Hugging Face's transformers and llama.cpp's server do.
pub struct JinjaChatTemplate {
    environment: Environment<'static>,
    sequence_tokens: SequenceTokens,
}

impl JinjaChatTemplate {
    pub fn new(source: String, sequence_tokens: SequenceTokens) -> Result<Self, Error> {
        let mut environment: Environment<'static> = Environment::new();
        let syntax: SyntaxConfig = SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build()?;

        environment.set_syntax(syntax);
        // Chat templates are written for Python's Jinja and call Python's string and dictionary methods.
        environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function(RAISE_EXCEPTION_FUNCTION_NAME, raise_exception);
        environment.add_template_owned(TEMPLATE_NAME, source)?;

        Ok(JinjaChatTemplate {
            environment,
            sequence_tokens,
        })
    }

    pub fn render_prompt(&self, messages: &[ChatMessage]) -> Result<String, Error> {
        let template: Template = self.environment.get_template(TEMPLATE_NAME)?;
        let context_json: serde_json::Value = self.create_context_json(messages);

        let rendered: String = template.render(Value::from(Serde(&context_json)))?;

        Ok(self.remove_sequence_tokens_tokenizer_adds(&rendered).to_string())
    }

    fn create_context_json(&self, messages: &[ChatMessage]) -> serde_json::Value {
        let mut context_json: serde_json::Value = chat_template_input::create_template_arguments_json();

        context_json["messages"] = chat_template_input::create_messages_json(messages);
        context_json["add_generation_prompt"] = serde_json::Value::Bool(true);
        context_json["bos_token"] = serde_json::Value::String(self.sequence_tokens.beginning_text.clone());
        context_json["eos_token"] = serde_json::Value::String(self.sequence_tokens.end_text.clone());
        context_json
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

/// Templates call this to reject a conversation they cannot format.
fn raise_exception(message: String) -> Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
}

fn get_token_text(vocabulary: &LlamaVocab, token: LlamaToken) -> String {
    if token == NULL_TOKEN {
        return String::new();
    }

    String::from_utf8_lossy(&vocabulary.token_to_piece(token, true, None)).into_owned()
}

fn render_built_in_prompt(model: &LlamaModel, built_in_template: &LlamaChatTemplate, messages: &[ChatMessage]) -> Result<String, Error> {
    let chat_messages: Vec<LlamaChatMessage> = messages
        .iter()
        .map(create_llama_chat_message)
        .collect::<Result<Vec<LlamaChatMessage>, Error>>()?;

    let prompt: String = model.apply_chat_template(built_in_template, &chat_messages, true)?;
    Ok(prompt)
}

fn create_llama_chat_message(message: &ChatMessage) -> Result<LlamaChatMessage, Error> {
    let llama_chat_message: LlamaChatMessage = LlamaChatMessage::new(message.role.chat_template_name().to_string(), message.text.clone())?;
    Ok(llama_chat_message)
}

#[cfg(test)]
mod tests {
    use llama_cpp_2::model::LlamaModel;
    use llama_cpp_2::model::params::LlamaModelParams;

    use crate::llama_cpp::chat_template;
    use crate::llama_cpp::chat_template::{ChatTemplate, JinjaChatTemplate, SequenceTokens};
    use crate::llama_cpp::worker;
    use crate::{ChatMessage, Error, Role, test_model};

    fn load_test_model_vocabulary() -> LlamaModel {
        let model_parameters: LlamaModelParams = LlamaModelParams::default().with_vocab_only(true);
        LlamaModel::load_from_file(worker::get_llama_backend().unwrap(), test_model::get_test_model_path(), &model_parameters).unwrap()
    }

    fn create_message(role: Role, text: &str) -> ChatMessage {
        ChatMessage {
            role,
            text: text.to_string(),
        }
    }

    fn create_template(source: &str) -> JinjaChatTemplate {
        let sequence_tokens: SequenceTokens = SequenceTokens {
            beginning_text: "<s>".to_string(),
            end_text: "</s>".to_string(),
            tokenizer_adds_beginning: true,
            tokenizer_adds_end: false,
        };

        JinjaChatTemplate::new(source.to_string(), sequence_tokens).unwrap()
    }

    #[test]
    fn render_prompt_matches_llama_cpp_built_in_template_for_test_model() {
        let model: LlamaModel = load_test_model_vocabulary();
        let messages: Vec<ChatMessage> = vec![
            create_message(Role::System, "Reply in Portuguese."),
            create_message(Role::User, "Say hello."),
            create_message(Role::Assistant, "Olá!"),
            create_message(Role::User, "Say goodbye."),
        ];
        // llama.cpp recognizes the model's template by its text and formats with its own built-in equivalent.
        let built_in_prompt: String = chat_template::render_built_in_prompt(&model, &model.chat_template(None).unwrap(), &messages).unwrap();

        let chat_template: ChatTemplate = ChatTemplate::from_model(&model).unwrap();
        let prompt: String = chat_template.render_prompt(&model, &messages).unwrap();

        assert!(matches!(chat_template, ChatTemplate::Jinja(_)));
        assert_eq!(prompt, built_in_prompt);
    }

    #[test]
    fn render_prompt_turns_thinking_off() {
        let template: JinjaChatTemplate = create_template("{% if enable_thinking is defined and not enable_thinking %}off{% endif %}");

        let prompt: String = template.render_prompt(&[]).unwrap();

        assert_eq!(prompt, "off");
    }

    #[test]
    fn render_prompt_supports_python_string_and_dictionary_methods() {
        let template: JinjaChatTemplate = create_template("{{ messages[0].get('role') }} {{ messages[0]['content'].strip().upper() }}");

        let prompt: String = template.render_prompt(&[create_message(Role::User, " hello ")]).unwrap();

        assert_eq!(prompt, "user HELLO");
    }

    #[test]
    fn render_prompt_fails_when_template_raises_exception() {
        let template: JinjaChatTemplate = create_template("{{ raise_exception('roles must alternate') }}");

        let rendered: Result<String, Error> = template.render_prompt(&[]);

        assert!(rendered.is_err());
    }
}
