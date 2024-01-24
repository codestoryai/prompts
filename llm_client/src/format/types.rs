use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use crate::clients::types::LLMClientMessage;

pub trait LLMFormatting {
    fn to_prompt(&self, messages: Vec<LLMClientMessage>) -> String;
}

pub struct DummyLLMFormatting {}

impl DummyLLMFormatting {
    pub fn new() -> Self {
        Self {}
    }
}

impl LLMFormatting for DummyLLMFormatting {
    fn to_prompt(&self, messages: Vec<LLMClientMessage>) -> String {
        messages
            .into_iter()
            .map(|message| message.content().to_owned())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TokenizerConfig {
    add_bos_token: bool,
    add_eos_token: bool,
    added_tokens_decoder: HashMap<String, AddedTokenDecoder>,
    additional_special_tokens: Vec<String>,
    bos_token: String,
    chat_template: String,
    clean_up_tokenization_spaces: bool,
    eos_token: String,
    legacy: bool,
    model_max_length: u128,
    pad_token: Option<String>,
    sp_model_kwargs: HashMap<String, String>,
    spaces_between_special_tokens: bool,
    tokenizer_class: String,
    unk_token: String,
    use_default_system_prompt: bool,
}

impl TokenizerConfig {
    pub fn add_bos_token(&self) -> bool {
        self.add_bos_token
    }

    pub fn add_eos_token(&self) -> bool {
        self.add_eos_token
    }

    pub fn bos_token(&self) -> &str {
        &self.bos_token
    }

    pub fn eos_token(&self) -> &str {
        &self.eos_token
    }

    pub fn chat_template(&self) -> &str {
        &self.chat_template
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddedTokenDecoder {
    content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: bool,
}

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Failed to get response from LLM")]
    FailedToGetResponse,

    #[error("Reqwest error: {0}")]
    ReqwestError(#[from] reqwest::Error),

    #[error("serde failed: {0}")]
    SerdeError(#[from] serde_json::Error),
}
