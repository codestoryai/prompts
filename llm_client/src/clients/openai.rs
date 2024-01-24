//! Client which can help us talk to openai

use async_openai::{
    config::{AzureConfig, OpenAIConfig},
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestMessageArgs,
        CreateChatCompletionRequestArgs, FunctionCall, Role,
    },
    Client,
};
use async_trait::async_trait;
use futures::StreamExt;

use crate::provider::LLMProviderAPIKeys;

use super::types::{
    LLMClient, LLMClientCompletionRequest, LLMClientCompletionResponse, LLMClientError,
    LLMClientMessage, LLMClientRole, LLMType,
};

enum OpenAIClientType {
    AzureClient(Client<AzureConfig>),
    OpenAIClient(Client<OpenAIConfig>),
}

pub struct OpenAIClient {}

impl OpenAIClient {
    pub fn new() -> Self {
        Self {}
    }

    pub fn model(&self, model: &LLMType) -> Option<String> {
        match model {
            LLMType::GPT3_5_16k => Some("gpt-3.5-turbo-16k-0613".to_owned()),
            LLMType::Gpt4 => Some("gpt-4-0613".to_owned()),
            LLMType::Gpt4Turbo => Some("gpt-4-1106-preview".to_owned()),
            LLMType::Gpt4_32k => Some("gpt-4-32k-0613".to_owned()),
            _ => None,
        }
    }

    pub fn messages(
        &self,
        messages: &[LLMClientMessage],
    ) -> Result<Vec<ChatCompletionRequestMessage>, LLMClientError> {
        let formatted_messages = messages
            .into_iter()
            .map(|message| {
                let role = message.role();
                match role {
                    LLMClientRole::User => ChatCompletionRequestMessageArgs::default()
                        .role(Role::User)
                        .content(message.content().to_owned())
                        .build()
                        .map_err(|e| LLMClientError::OpenAPIError(e)),
                    LLMClientRole::System => ChatCompletionRequestMessageArgs::default()
                        .role(Role::System)
                        .content(message.content().to_owned())
                        .build()
                        .map_err(|e| LLMClientError::OpenAPIError(e)),
                    // the assistant is the one which ends up calling the function, so we need to
                    // handle the case where the function is called by the assistant here
                    LLMClientRole::Assistant => match message.get_function_call() {
                        Some(function_call) => ChatCompletionRequestMessageArgs::default()
                            .role(Role::Function)
                            .function_call(FunctionCall {
                                name: function_call.name().to_owned(),
                                arguments: function_call.arguments().to_owned(),
                            })
                            .build()
                            .map_err(|e| LLMClientError::OpenAPIError(e)),
                        None => ChatCompletionRequestMessageArgs::default()
                            .role(Role::Assistant)
                            .content(message.content().to_owned())
                            .build()
                            .map_err(|e| LLMClientError::OpenAPIError(e)),
                    },
                    LLMClientRole::Function => match message.get_function_call() {
                        Some(function_call) => ChatCompletionRequestMessageArgs::default()
                            .role(Role::Function)
                            .content(message.content().to_owned())
                            .function_call(FunctionCall {
                                name: function_call.name().to_owned(),
                                arguments: function_call.arguments().to_owned(),
                            })
                            .build()
                            .map_err(|e| LLMClientError::OpenAPIError(e)),
                        None => Err(LLMClientError::FunctionCallNotPresent),
                    },
                }
            })
            .collect::<Vec<_>>();
        formatted_messages
            .into_iter()
            .collect::<Result<Vec<ChatCompletionRequestMessage>, LLMClientError>>()
    }

    fn generate_openai_client(
        &self,
        api_key: LLMProviderAPIKeys,
    ) -> Result<OpenAIClientType, LLMClientError> {
        match api_key {
            LLMProviderAPIKeys::OpenAI(api_key) => {
                let config = OpenAIConfig::new().with_api_key(api_key.api_key);
                Ok(OpenAIClientType::OpenAIClient(Client::with_config(config)))
            }
            LLMProviderAPIKeys::OpenAIAzureConfig(azure_config) => {
                let config = AzureConfig::new()
                    .with_api_base(azure_config.api_base)
                    .with_api_key(azure_config.api_key)
                    .with_deployment_id(azure_config.deployment_id)
                    .with_api_version(azure_config.api_version);
                Ok(OpenAIClientType::AzureClient(Client::with_config(config)))
            }
            _ => Err(LLMClientError::WrongAPIKeyType),
        }
    }
}

#[async_trait]
impl LLMClient for OpenAIClient {
    fn client(&self) -> &crate::provider::LLMProvider {
        &crate::provider::LLMProvider::OpenAI
    }

    async fn stream_completion(
        &self,
        api_key: LLMProviderAPIKeys,
        request: LLMClientCompletionRequest,
        sender: tokio::sync::mpsc::UnboundedSender<LLMClientCompletionResponse>,
    ) -> Result<String, LLMClientError> {
        let model = self.model(request.model());
        if model.is_none() {
            return Err(LLMClientError::UnSupportedModel);
        }
        let model = model.unwrap();
        let messages = self.messages(request.messages())?;
        let mut request_builder_args = CreateChatCompletionRequestArgs::default();
        let mut request_builder = request_builder_args
            .model(model.to_owned())
            .messages(messages)
            .temperature(request.temperature())
            .stream(true);
        if let Some(frequency_penalty) = request.frequency_penalty() {
            request_builder = request_builder.frequency_penalty(frequency_penalty);
        }
        let request = request_builder.build()?;
        let mut buffer = String::new();
        let client = self.generate_openai_client(api_key)?;

        // TODO(skcd): Bad code :| we are repeating too many things but this
        // just works and we need it right now
        match client {
            OpenAIClientType::AzureClient(client) => {
                let stream_maybe = client.chat().create_stream(request).await;
                if stream_maybe.is_err() {
                    return Err(LLMClientError::OpenAPIError(stream_maybe.err().unwrap()));
                } else {
                    dbg!("no error here");
                }
                let mut stream = stream_maybe.unwrap();
                while let Some(response) = stream.next().await {
                    match response {
                        Ok(response) => {
                            let delta = response
                                .choices
                                .get(0)
                                .map(|choice| choice.delta.content.to_owned())
                                .flatten()
                                .unwrap_or("".to_owned());
                            let _value = response
                                .choices
                                .get(0)
                                .map(|choice| choice.delta.content.as_ref())
                                .flatten();
                            buffer.push_str(&delta);
                            let _ = sender.send(LLMClientCompletionResponse::new(
                                buffer.to_owned(),
                                Some(delta),
                                model.to_owned(),
                            ));
                        }
                        Err(err) => {
                            dbg!(err);
                            break;
                        }
                    }
                }
            }
            OpenAIClientType::OpenAIClient(client) => {
                let mut stream = client.chat().create_stream(request).await?;
                while let Some(response) = stream.next().await {
                    match response {
                        Ok(response) => {
                            let response = response
                                .choices
                                .get(0)
                                .ok_or(LLMClientError::FailedToGetResponse)?;
                            let text = response.delta.content.to_owned();
                            if let Some(text) = text {
                                buffer.push_str(&text);
                                let _ = sender.send(LLMClientCompletionResponse::new(
                                    buffer.to_owned(),
                                    Some(text),
                                    model.to_owned(),
                                ));
                            }
                        }
                        Err(err) => {
                            dbg!(err);
                            break;
                        }
                    }
                }
            }
        }
        Ok(buffer)
    }

    async fn completion(
        &self,
        api_key: LLMProviderAPIKeys,
        request: LLMClientCompletionRequest,
    ) -> Result<String, LLMClientError> {
        let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
        let result = self.stream_completion(api_key, request, sender).await?;
        Ok(result)
    }

    async fn stream_prompt_completion(
        &self,
        _api_key: LLMProviderAPIKeys,
        _request: super::types::LLMClientCompletionStringRequest,
        _sender: tokio::sync::mpsc::UnboundedSender<LLMClientCompletionResponse>,
    ) -> Result<String, LLMClientError> {
        Err(LLMClientError::OpenAIDoesNotSupportCompletion)
    }
}
