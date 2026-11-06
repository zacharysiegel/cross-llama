use llama_cpp_2::ChatTemplateError;
use llama_cpp_2::model::{LlamaChatTemplate, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::vocab::LlamaVocab;

use crate::Error;
use crate::chat_template::{ChatTemplate, SequenceTokens};

// llama.cpp's value for a token the vocabulary does not have.
const NULL_TOKEN: LlamaToken = LlamaToken(-1);

minimer::impl_from_error!(Error, ChatTemplateError);

pub fn read_chat_template(model: &LlamaModel) -> Result<ChatTemplate, Error> {
    let model_template: Result<LlamaChatTemplate, ChatTemplateError> = model.chat_template(None);
    let model_template_source: Option<String> = match model_template {
        Ok(model_template) => Some(model_template.to_string()?),
        Err(ChatTemplateError::MissingTemplate) => None,
        Err(template_error) => return Err(Error::from(template_error)),
    };

    let vocabulary: LlamaVocab = model.vocab();
    let sequence_tokens: SequenceTokens = SequenceTokens {
        beginning_text: get_token_text(&vocabulary, vocabulary.bos()),
        end_text: get_token_text(&vocabulary, vocabulary.eos()),
        tokenizer_adds_beginning: vocabulary.should_add_bos(),
        tokenizer_adds_end: vocabulary.should_add_eos(),
    };

    ChatTemplate::new(model_template_source, sequence_tokens)
}

fn get_token_text(vocabulary: &LlamaVocab, token: LlamaToken) -> String {
    if token == NULL_TOKEN {
        return String::new();
    }

    String::from_utf8_lossy(&vocabulary.token_to_piece(token, true, None)).into_owned()
}

#[cfg(test)]
mod tests {
    use llama_cpp_2::ApplyChatTemplateError;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};

    use crate::chat_template::ChatTemplate;
    use crate::llama_cpp::{model_chat_template, worker};
    use crate::{ChatMessage, Role, test_model};

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

    /// llama.cpp recognizes the model's template by its text and formats with its own built-in equivalent.
    fn render_built_in_prompt(model: &LlamaModel, messages: &[ChatMessage]) -> Result<String, ApplyChatTemplateError> {
        let chat_messages: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|message| LlamaChatMessage::new(message.role.chat_template_name().to_string(), message.text.clone()).unwrap())
            .collect();

        model.apply_chat_template(&model.chat_template(None).unwrap(), &chat_messages, true)
    }

    #[test]
    fn read_chat_template_renders_test_model_prompt_as_llama_cpp_built_in_template_does() {
        let model: LlamaModel = load_test_model_vocabulary();
        let messages: Vec<ChatMessage> = vec![
            create_message(Role::System, "Reply in Portuguese."),
            create_message(Role::User, "Say hello."),
            create_message(Role::Assistant, "Olá!"),
            create_message(Role::User, "Say goodbye."),
        ];
        let built_in_prompt: String = render_built_in_prompt(&model, &messages).unwrap();

        let chat_template: ChatTemplate = model_chat_template::read_chat_template(&model).unwrap();
        let prompt: String = chat_template.render_prompt(&messages).unwrap();

        assert_eq!(prompt, built_in_prompt);
    }
}
