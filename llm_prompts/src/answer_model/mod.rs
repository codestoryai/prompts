//! We define all the properties for the model configuration related to answering
//! a user question in the chat format here

use std::collections::HashMap;

use llm_client::clients::types::LLMType;

#[derive(Debug)]
pub struct AnswerModel {
    pub llm_type: LLMType,
    /// The number of tokens reserved for the answer
    pub answer_tokens: i64,

    /// The number of tokens reserved for the prompt
    pub prompt_tokens_limit: i64,

    /// The number of tokens reserved for history
    pub history_tokens_limit: i64,

    /// The total number of tokens reserved for the model
    pub total_tokens: i64,
}

// GPT-3.5-16k Turbo has 16,385 tokens
pub const GPT_3_5_TURBO_16K: AnswerModel = AnswerModel {
    llm_type: LLMType::GPT3_5_16k,
    answer_tokens: 1024 * 2,
    prompt_tokens_limit: 2500 * 2,
    history_tokens_limit: 2048 * 2,
    total_tokens: 16385,
};

// GPT-4 has 8,192 tokens
pub const GPT_4: AnswerModel = AnswerModel {
    llm_type: LLMType::Gpt4,
    answer_tokens: 1024,
    // The prompt tokens limit for gpt4 are a bit higher so we can get more context
    // when required
    prompt_tokens_limit: 4500,
    history_tokens_limit: 2048,
    total_tokens: 8192,
};

// GPT4-32k has 32,769 tokens
pub const GPT_4_32K: AnswerModel = AnswerModel {
    llm_type: LLMType::Gpt4_32k,
    answer_tokens: 1024 * 4,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 2048 * 4,
    total_tokens: 32769,
};

// GPT4-Turbo has 128k tokens as input, but let's keep it capped at 32k tokens
// as LLMs exhibit LIM issues which has been frequently documented
pub const GPT_4_TURBO_128K: AnswerModel = AnswerModel {
    llm_type: LLMType::Gpt4Turbo,
    answer_tokens: 1024 * 4,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 2048 * 4,
    total_tokens: 32769,
};

// MistralInstruct has 8k tokens in total
pub const MISTRAL_INSTRUCT: AnswerModel = AnswerModel {
    llm_type: LLMType::MistralInstruct,
    answer_tokens: 1024,
    prompt_tokens_limit: 4500,
    history_tokens_limit: 2048,
    total_tokens: 8000,
};

// Mixtral has 32k tokens in total
pub const MIXTRAL: AnswerModel = AnswerModel {
    llm_type: LLMType::Mixtral,
    answer_tokens: 1024,
    prompt_tokens_limit: 2500 * 4,
    history_tokens_limit: 1024 * 4,
    total_tokens: 32000,
};

pub struct LLMAnswerModelBroker {
    pub models: HashMap<LLMType, AnswerModel>,
}

impl LLMAnswerModelBroker {
    pub fn new() -> Self {
        let broker = Self {
            models: Default::default(),
        };
        broker
            .add_answer_model(GPT_3_5_TURBO_16K)
            .add_answer_model(GPT_4)
            .add_answer_model(GPT_4_32K)
            .add_answer_model(GPT_4_TURBO_128K)
            .add_answer_model(MISTRAL_INSTRUCT)
            .add_answer_model(MIXTRAL)
    }

    pub fn add_answer_model(mut self, model: AnswerModel) -> Self {
        self.models.insert(model.llm_type.clone(), model);
        self
    }

    pub fn get_answer_model(&self, llm_type: &LLMType) -> Option<&AnswerModel> {
        self.models.get(llm_type)
    }
}
