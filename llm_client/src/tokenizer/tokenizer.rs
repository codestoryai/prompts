//! We are going to run the various tokenizers here, we also make sure to run
//! the tokenizer in a different thread here, because its important that we
//! don't block the main thread from working

use std::collections::HashMap;
use std::str::FromStr;

use thiserror::Error;
use tiktoken_rs::ChatCompletionRequestMessage;
use tokenizers::Tokenizer;

use crate::{
    clients::types::{LLMClientMessage, LLMClientRole, LLMType},
    format::{
        deepseekcoder::DeepSeekCoderFormatting,
        mistral::MistralInstructFormatting,
        mixtral::MixtralInstructFormatting,
        types::{LLMFormatting, TokenizerError},
    },
};

pub struct LLMTokenizer {
    pub tokenizers: HashMap<LLMType, Tokenizer>,
    pub formatters: HashMap<LLMType, Box<dyn LLMFormatting + Send + Sync>>,
}

#[derive(Error, Debug)]
pub enum LLMTokenizerError {
    #[error("Tokenizer not found for model {0}")]
    TokenizerNotFound(LLMType),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("error from tokenizer crate: {0}")]
    TokenizerCrateError(#[from] tokenizers::Error),

    #[error("anyhow error: {0}")]
    AnyhowError(#[from] anyhow::Error),

    #[error("tokenizer error: {0}")]
    TokenizerErrorInternal(#[from] TokenizerError),
}

pub enum LLMTokenizerInput {
    Prompt(String),
    Messages(Vec<LLMClientMessage>),
}

impl LLMTokenizer {
    pub fn new() -> Result<Self, LLMTokenizerError> {
        let tokenizer = Self {
            tokenizers: HashMap::new(),
            formatters: HashMap::new(),
        };
        let updated_tokenizer = tokenizer
            .add_llm_type(
                LLMType::Mixtral,
                Box::new(MixtralInstructFormatting::new()?),
            )
            .add_llm_type(
                LLMType::MistralInstruct,
                Box::new(MistralInstructFormatting::new()?),
            )
            .add_llm_type(
                LLMType::DeepSeekCoder,
                Box::new(DeepSeekCoderFormatting::new()),
            );
        Ok(updated_tokenizer)
    }

    fn add_llm_type(
        mut self,
        llm_type: LLMType,
        formatter: Box<dyn LLMFormatting + Send + Sync>,
    ) -> Self {
        // This can be falliable, since soe llms might have formatting support
        // and if they don't thats fine
        let _ = self.load_tokenizer(&llm_type);
        self.formatters.insert(llm_type, formatter);
        self
    }

    fn to_openai_tokenizer(&self, model: &LLMType) -> Option<String> {
        match model {
            LLMType::GPT3_5_16k => Some("gpt-3.5-turbo-16k-0613".to_owned()),
            LLMType::Gpt4 => Some("gpt-4-0613".to_owned()),
            LLMType::Gpt4Turbo => Some("gpt-4-1106-preview".to_owned()),
            LLMType::Gpt4_32k => Some("gpt-4-32k-0613".to_owned()),
            _ => None,
        }
    }

    pub fn count_tokens(
        &self,
        model: &LLMType,
        input: LLMTokenizerInput,
    ) -> Result<usize, LLMTokenizerError> {
        match input {
            LLMTokenizerInput::Prompt(prompt) => self.count_tokens_using_tokenizer(model, &prompt),
            LLMTokenizerInput::Messages(messages) => {
                // we can't send messages directly to the tokenizer, we have to
                // either make it a message or its an openai prompt in which case
                // its fine
                // so we are going to return an error if its not openai
                if model.is_openai() {
                    // we can use the openai tokenizer
                    let model = self.to_openai_tokenizer(model);
                    match model {
                        Some(model) => Ok(tiktoken_rs::num_tokens_from_messages(
                            &model,
                            messages
                                .into_iter()
                                .map(|message| {
                                    let role = message.role();
                                    let content = message.content();
                                    match role {
                                        LLMClientRole::User => ChatCompletionRequestMessage {
                                            role: "user".to_owned(),
                                            content: Some(content.to_owned()),
                                            name: None,
                                            function_call: None,
                                        },
                                        LLMClientRole::Assistant => ChatCompletionRequestMessage {
                                            role: "assistant".to_owned(),
                                            content: Some(content.to_owned()),
                                            name: None,
                                            function_call: None,
                                        },
                                        LLMClientRole::System => ChatCompletionRequestMessage {
                                            role: "system".to_owned(),
                                            content: Some(content.to_owned()),
                                            name: None,
                                            function_call: None,
                                        },
                                        LLMClientRole::Function => ChatCompletionRequestMessage {
                                            role: "function".to_owned(),
                                            content: Some(content.to_owned()),
                                            name: None,
                                            function_call: None,
                                        },
                                    }
                                })
                                .collect::<Vec<_>>()
                                .as_slice(),
                        )?),
                        None => Err(LLMTokenizerError::TokenizerError(
                            "Only openai models are supported for messages".to_owned(),
                        )),
                    }
                } else {
                    let prompt = self
                        .formatters
                        .get(model)
                        .map(|formatter| formatter.to_prompt(messages));
                    match prompt {
                        Some(prompt) => {
                            let num_tokens = self.tokenizers.get(model).map(|tokenizer| {
                                tokenizer
                                    .encode(prompt, false)
                                    .map(|encoding| encoding.len())
                            });
                            match num_tokens {
                                Some(Ok(num_tokens)) => Ok(num_tokens),
                                _ => Err(LLMTokenizerError::TokenizerError(
                                    "Failed to encode prompt".to_owned(),
                                )),
                            }
                        }
                        None => Err(LLMTokenizerError::TokenizerError(
                            "No formatter found for model".to_owned(),
                        )),
                    }
                }
            }
        }
    }

    pub fn count_tokens_using_tokenizer(
        &self,
        model: &LLMType,
        prompt: &str,
    ) -> Result<usize, LLMTokenizerError> {
        // we have the custom tokenizers already loaded, if this is not the openai loop
        if !model.is_openai() {
            let tokenizer = self.tokenizers.get(model);
            match tokenizer {
                Some(tokenizer) => {
                    // Now over here we will try to figure out how to pass the
                    // values around
                    let result = tokenizer.encode(prompt, false);
                    match result {
                        Ok(encoding) => Ok(encoding.len()),
                        Err(e) => Err(LLMTokenizerError::TokenizerError(format!(
                            "Failed to encode prompt: {}",
                            e
                        ))),
                    }
                }
                None => {
                    return Err(LLMTokenizerError::TokenizerNotFound(model.clone()));
                }
            }
        } else {
            // If we are using openai model, then we have to use the bpe config
            // and count the number of tokens
            let model = self.to_openai_tokenizer(model);
            if let None = model {
                return Err(LLMTokenizerError::TokenizerError(
                    "OpenAI model not found".to_owned(),
                ));
            }
            let model = model.expect("if let None to hold");
            let bpe = tiktoken_rs::get_bpe_from_model(&model)?;
            Ok(bpe.encode_ordinary(prompt).len())
        }
    }

    pub fn load_tokenizer(&mut self, model: &LLMType) -> Result<(), LLMTokenizerError> {
        let tokenizer = match model {
            LLMType::MistralInstruct => {
                let config = include_str!("configs/mistral.json");
                Some(Tokenizer::from_str(config)?)
            }
            LLMType::Mixtral => {
                let config = include_str!("configs/mixtral.json");
                Some(Tokenizer::from_str(config)?)
            }
            LLMType::DeepSeekCoder => {
                let config = include_str!("configs/deepseekcoder.json");
                Some(Tokenizer::from_str(config)?)
            }
            _ => None,
        };
        if let Some(tokenizer) = tokenizer {
            self.tokenizers.insert(model.clone(), tokenizer);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use tokenizers::Tokenizer;

    #[test]
    fn test_loading_deepseek_tokenizer_works() {
        let tokenizer_file = include_str!("configs/deepseekcoder.json");
        let _ = Tokenizer::from_str(tokenizer_file).unwrap();
    }
}
