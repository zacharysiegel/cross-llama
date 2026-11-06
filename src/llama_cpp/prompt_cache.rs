use llama_cpp_2::DecodeError;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::kv_cache::KvCacheConversionError;
use llama_cpp_2::llama_batch::{BatchAddError, LlamaBatch};
use llama_cpp_2::token::LlamaToken;

use crate::Error;

const SEQUENCE_ID: u32 = 0;

minimer::impl_from_error!(Error, DecodeError);
minimer::impl_from_error!(Error, BatchAddError);
minimer::impl_from_error!(Error, KvCacheConversionError);

/// The tokens in the context's cache, in position order.
pub struct PromptCache {
    cached_tokens: Vec<LlamaToken>,
}

impl PromptCache {
    pub fn new() -> Self {
        PromptCache {
            cached_tokens: Vec::new(),
        }
    }

    pub fn cached_token_count(&self) -> usize {
        self.cached_tokens.len()
    }

    /// Clears the cache from the first token that differs from `prompt_tokens` and returns how many tokens remain.
    pub fn keep_reusable_prefix(&mut self, context: &mut LlamaContext, prompt_tokens: &[LlamaToken]) -> Result<usize, Error> {
        // A window-only sliding-window cache drops its earliest positions once it fills; a kept prefix needs them all.
        let earliest_cached_position: i32 = context.kv_cache_seq_pos_min(i32::try_from(SEQUENCE_ID)?);
        let reusable_token_count: usize = match earliest_cached_position {
            0 => reusable_prefix_length(&self.cached_tokens, prompt_tokens),
            _ => 0,
        };
        let removed: bool = context.clear_kv_cache_seq(Some(SEQUENCE_ID), Some(u32::try_from(reusable_token_count)?), None)?;

        if !removed {
            self.clear(context);
            return Ok(0);
        }

        self.cached_tokens.truncate(reusable_token_count);
        Ok(reusable_token_count)
    }

    /// Decodes `tokens` at the positions after the cached ones, computing logits for the last of them.
    pub fn decode(&mut self, context: &mut LlamaContext, batch: &mut LlamaBatch, tokens: &[LlamaToken]) -> Result<(), Error> {
        self.fill_batch(batch, tokens)?;
        let decoded: Result<(), DecodeError> = context.decode(batch);

        if let Err(decode_error) = decoded {
            // llama.cpp may leave part of a failed batch in the cache.
            self.clear(context);
            return Err(Error::from(decode_error));
        }

        self.cached_tokens.extend_from_slice(tokens);
        Ok(())
    }

    fn fill_batch(&self, batch: &mut LlamaBatch, tokens: &[LlamaToken]) -> Result<(), Error> {
        batch.clear();
        let sequence_ids: [i32; 1] = [i32::try_from(SEQUENCE_ID)?];

        for (offset, token) in tokens.iter().enumerate() {
            let position: i32 = i32::try_from(self.cached_tokens.len() + offset)?;
            let is_last_token: bool = offset + 1 == tokens.len();
            batch.add(*token, position, &sequence_ids, is_last_token)?;
        }

        Ok(())
    }

    fn clear(&mut self, context: &mut LlamaContext) {
        context.clear_kv_cache();
        self.cached_tokens.clear();
    }
}

/// The longest common prefix, excluding the prompt's last token, whose logits the reply's first token is sampled from.
pub fn reusable_prefix_length(cached_tokens: &[LlamaToken], prompt_tokens: &[LlamaToken]) -> usize {
    let common_prefix_length: usize = cached_tokens
        .iter()
        .zip(prompt_tokens.iter())
        .take_while(|(cached_token, prompt_token)| cached_token == prompt_token)
        .count();

    common_prefix_length.min(prompt_tokens.len().saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use llama_cpp_2::token::LlamaToken;

    use super::reusable_prefix_length;

    fn create_tokens(token_ids: &[i32]) -> Vec<LlamaToken> {
        token_ids.iter().map(|token_id| LlamaToken(*token_id)).collect()
    }

    #[test]
    fn reusable_prefix_length_stops_at_first_differing_token() {
        let cached_tokens: Vec<LlamaToken> = create_tokens(&[1, 2, 3, 4, 5]);
        let prompt_tokens: Vec<LlamaToken> = create_tokens(&[1, 2, 9, 4, 5, 6]);

        assert_eq!(reusable_prefix_length(&cached_tokens, &prompt_tokens), 2);
    }

    #[test]
    fn reusable_prefix_length_leaves_last_prompt_token_when_whole_prompt_is_cached() {
        let cached_tokens: Vec<LlamaToken> = create_tokens(&[1, 2, 3, 4, 5]);
        let prompt_tokens: Vec<LlamaToken> = create_tokens(&[1, 2, 3]);

        assert_eq!(reusable_prefix_length(&cached_tokens, &prompt_tokens), 2);
    }
}
