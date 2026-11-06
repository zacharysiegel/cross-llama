use std::collections::HashMap;

use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory, TokenParser};
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use toktrie::TokEnv;

use crate::Error;

// llguidance reads its JSON compile options from this key of the schema.
const LLGUIDANCE_OPTIONS_KEY: &str = "x-guidance";

/// Compiles each distinct JSON schema once per model.
pub struct SchemaConstraints {
    parser_factory: Option<ParserFactory>,
    compiled_parsers_by_schema_text: HashMap<String, TokenParser>,
}

impl SchemaConstraints {
    pub fn new() -> Self {
        SchemaConstraints {
            parser_factory: None,
            compiled_parsers_by_schema_text: HashMap::new(),
        }
    }

    /// A sampler that allows only tokens which keep the reply a valid prefix of a JSON value matching `json_schema`.
    pub fn create_sampler(&mut self, model: &LlamaModel, json_schema: &serde_json::Value) -> Result<LlamaSampler, Error> {
        let schema_text: String = json_schema.to_string();
        let already_compiled: bool = self.compiled_parsers_by_schema_text.contains_key(&schema_text);

        if !already_compiled {
            let compiled_parser: TokenParser = self.compile(model, json_schema)?;
            self.compiled_parsers_by_schema_text.insert(schema_text.clone(), compiled_parser);
        }

        let parser: TokenParser = self.compiled_parsers_by_schema_text[&schema_text].deep_clone();
        let matcher: Matcher = Matcher::new(Ok(parser));
        Ok(LlamaSampler::from(matcher))
    }

    fn compile(&mut self, model: &LlamaModel, json_schema: &serde_json::Value) -> Result<TokenParser, Error> {
        let parser_factory: &ParserFactory = self.get_parser_factory(model)?;
        let grammar: TopLevelGrammar = TopLevelGrammar::from_json_schema(with_compact_whitespace(json_schema));

        parser_factory
            .create_parser(grammar)
            .map_err(|compile_error| Error::new(&format!("could not compile the JSON schema: {compile_error}")))
    }

    fn get_parser_factory(&mut self, model: &LlamaModel) -> Result<&ParserFactory, Error> {
        match self.parser_factory {
            Some(ref parser_factory) => Ok(parser_factory),
            None => {
                let parser_factory: ParserFactory = create_parser_factory(model)?;
                Ok(self.parser_factory.insert(parser_factory))
            }
        }
    }
}

/// Leaves the caller's own llguidance options in place.
fn with_compact_whitespace(json_schema: &serde_json::Value) -> serde_json::Value {
    let mut compact_json_schema: serde_json::Value = json_schema.clone();

    if let Some(schema_object) = compact_json_schema.as_object_mut() {
        schema_object
            .entry(LLGUIDANCE_OPTIONS_KEY)
            .or_insert_with(|| serde_json::json!({ "whitespace_flexible": false }));
    }

    compact_json_schema
}

fn create_parser_factory(model: &LlamaModel) -> Result<ParserFactory, Error> {
    let token_environment: TokEnv = LlamaSampler::llguidance_tok_env(model);

    ParserFactory::new_simple(&token_environment)
        .map_err(|factory_error| Error::new(&format!("could not prepare the model's vocabulary for JSON schemas: {factory_error}")))
}
