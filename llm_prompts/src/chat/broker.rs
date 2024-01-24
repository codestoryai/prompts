use llm_client::clients::types::LLMType;

use crate::answer_model::{AnswerModel, LLMAnswerModelBroker};

#[derive(thiserror::Error, Debug)]
pub enum ChatModelBrokerErrors {
    #[error("The model {0} is not supported yet")]
    ModelNotSupported(LLMType),
}

pub struct LLMChatModelBroker {
    answer_model_broker: LLMAnswerModelBroker,
}

impl LLMChatModelBroker {
    pub fn init() -> Self {
        let answer_model_broker = LLMAnswerModelBroker::new();
        Self {
            answer_model_broker,
        }
    }

    pub fn get_answer_model(
        &self,
        llm_type: &LLMType,
    ) -> Result<&AnswerModel, ChatModelBrokerErrors> {
        self.answer_model_broker
            .get_answer_model(llm_type)
            .ok_or(ChatModelBrokerErrors::ModelNotSupported(llm_type.clone()))
    }
}
