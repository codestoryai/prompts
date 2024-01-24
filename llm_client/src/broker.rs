//! The llm client broker takes care of getting the right tokenizer formatter etc
//! without us having to worry about the specifics, just pass in the message and the
//! provider we take care of the rest

use std::{collections::HashMap, sync::Arc};

use futures::future::Either;
use sqlx::SqlitePool;

use crate::{
    clients::{
        codestory::CodeStoryClient,
        lmstudio::LMStudioClient,
        ollama::OllamaClient,
        openai::OpenAIClient,
        togetherai::TogetherAIClient,
        types::{
            LLMClient, LLMClientCompletionRequest, LLMClientCompletionResponse,
            LLMClientCompletionStringRequest, LLMClientError,
        },
    },
    config::LLMBrokerConfiguration,
    provider::{CodeStoryLLMType, LLMProvider, LLMProviderAPIKeys},
    sqlite,
};

pub type SqlDb = Arc<SqlitePool>;

pub struct LLMBroker {
    pub providers: HashMap<LLMProvider, Box<dyn LLMClient + Send + Sync>>,
    db: SqlDb,
}

pub type LLMBrokerResponse = Result<String, LLMClientError>;

impl LLMBroker {
    pub async fn new(config: LLMBrokerConfiguration) -> Result<Self, LLMClientError> {
        let sqlite = Arc::new(sqlite::init(config).await?);
        let broker = Self {
            providers: HashMap::new(),
            db: sqlite,
        };
        Ok(broker
            .add_provider(LLMProvider::OpenAI, Box::new(OpenAIClient::new()))
            .add_provider(LLMProvider::Ollama, Box::new(OllamaClient::new()))
            .add_provider(LLMProvider::TogetherAI, Box::new(TogetherAIClient::new()))
            .add_provider(
                LLMProvider::CodeStory(CodeStoryLLMType { llm_type: None }),
                Box::new(CodeStoryClient::new(
                    "https://codestory-provider-dot-anton-390822.ue.r.appspot.com",
                )),
            ))
    }

    pub fn add_provider(
        mut self,
        provider: LLMProvider,
        client: Box<dyn LLMClient + Send + Sync>,
    ) -> Self {
        self.providers.insert(provider, client);
        self
    }

    pub async fn stream_answer(
        &self,
        api_key: LLMProviderAPIKeys,
        provider: LLMProvider,
        request: Either<LLMClientCompletionRequest, LLMClientCompletionStringRequest>,
        metadata: HashMap<String, String>,
        sender: tokio::sync::mpsc::UnboundedSender<LLMClientCompletionResponse>,
    ) -> LLMBrokerResponse {
        match request {
            Either::Left(request) => {
                self.stream_completion(api_key, request, provider, metadata, sender)
                    .await
            }
            Either::Right(request) => {
                self.stream_string_completion(api_key, request, metadata, sender)
                    .await
            }
        }
    }

    pub async fn stream_completion(
        &self,
        api_key: LLMProviderAPIKeys,
        request: LLMClientCompletionRequest,
        provider: LLMProvider,
        metadata: HashMap<String, String>,
        sender: tokio::sync::mpsc::UnboundedSender<LLMClientCompletionResponse>,
    ) -> LLMBrokerResponse {
        let api_key = api_key
            .key(&provider)
            .ok_or(LLMClientError::UnSupportedModel)?;
        let provider_type = match &api_key {
            LLMProviderAPIKeys::Ollama(_) => LLMProvider::Ollama,
            LLMProviderAPIKeys::OpenAI(_) => LLMProvider::OpenAI,
            LLMProviderAPIKeys::OpenAIAzureConfig(_) => LLMProvider::OpenAI,
            LLMProviderAPIKeys::TogetherAI(_) => LLMProvider::TogetherAI,
            LLMProviderAPIKeys::LMStudio(_) => LLMProvider::LMStudio,
            LLMProviderAPIKeys::CodeStory => {
                LLMProvider::CodeStory(CodeStoryLLMType { llm_type: None })
            }
        };
        let provider = self.providers.get(&provider_type);
        if let Some(provider) = provider {
            let result = provider
                .stream_completion(api_key, request.clone(), sender)
                .await;
            if let Ok(result) = result.as_ref() {
                // we write the inputs to the DB so we can keep track of the inputs
                // and the result provided by the LLM
                let llm_type = request.model();
                let temperature = request.temperature();
                let str_metadata = serde_json::to_string(&metadata).unwrap_or_default();
                let llm_type_str = serde_json::to_string(&llm_type)?;
                let messages = serde_json::to_string(&request.messages())?;
                let mut tx = self
                    .db
                    .begin()
                    .await
                    .map_err(|_e| LLMClientError::FailedToStoreInDB)?;
                let _ = sqlx::query! {
                    r#"
                    INSERT INTO llm_data (chat_messages, response, llm_type, temperature, max_tokens, event_type)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    "#,
                    messages,
                    result,
                    llm_type_str,
                    temperature,
                    -1,
                    str_metadata,
                }.execute(&mut *tx).await?;
                let _ = tx
                    .commit()
                    .await
                    .map_err(|_e| LLMClientError::FailedToStoreInDB)?;
            }
            result
        } else {
            Err(LLMClientError::UnSupportedModel)
        }
    }

    pub async fn stream_string_completion(
        &self,
        api_key: LLMProviderAPIKeys,
        request: LLMClientCompletionStringRequest,
        metadata: HashMap<String, String>,
        sender: tokio::sync::mpsc::UnboundedSender<LLMClientCompletionResponse>,
    ) -> LLMBrokerResponse {
        let provider_type = match &api_key {
            LLMProviderAPIKeys::Ollama(_) => LLMProvider::Ollama,
            LLMProviderAPIKeys::OpenAI(_) => LLMProvider::OpenAI,
            LLMProviderAPIKeys::OpenAIAzureConfig(_) => LLMProvider::OpenAI,
            LLMProviderAPIKeys::TogetherAI(_) => LLMProvider::TogetherAI,
            LLMProviderAPIKeys::LMStudio(_) => LLMProvider::LMStudio,
            LLMProviderAPIKeys::CodeStory => {
                LLMProvider::CodeStory(CodeStoryLLMType { llm_type: None })
            }
        };
        let provider = self.providers.get(&provider_type);
        if let Some(provider) = provider {
            let result = provider
                .stream_prompt_completion(api_key, request.clone(), sender)
                .await;
            if let Ok(result) = result.as_ref() {
                // we write the inputs to the DB so we can keep track of the inputs
                // and the result provided by the LLM
                let llm_type = request.model();
                let temperature = request.temperature();
                let str_metadata = serde_json::to_string(&metadata).unwrap_or_default();
                let llm_type_str = serde_json::to_string(&llm_type)?;
                let prompt = request.prompt();
                let mut tx = self
                    .db
                    .begin()
                    .await
                    .map_err(|_e| LLMClientError::FailedToStoreInDB)?;
                let _ = sqlx::query! {
                    r#"
                    INSERT INTO llm_data (prompt, response, llm_type, temperature, max_tokens, event_type)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    "#,
                    prompt,
                    result,
                    llm_type_str,
                    temperature,
                    -1,
                    str_metadata,
                }.execute(&mut *tx).await?;
                let _ = tx
                    .commit()
                    .await
                    .map_err(|_e| LLMClientError::FailedToStoreInDB)?;
            }
            result
        } else {
            Err(LLMClientError::UnSupportedModel)
        }
    }
}
