use async_openai::types::CreateChatCompletionStreamResponse;
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use tokio::sync::mpsc::UnboundedSender;

use crate::provider::{LLMProvider, LLMProviderAPIKeys};

use super::types::{
    LLMClient, LLMClientCompletionRequest, LLMClientCompletionResponse,
    LLMClientCompletionStringRequest, LLMClientError, LLMClientRole, LLMType,
};

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct LMStudioResponse {
    model: String,
    choices: Vec<Choice>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct Choice {
    text: String,
}

pub struct CodeStoryClient {
    client: reqwest::Client,
    api_base: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct CodeStoryMessage {
    role: String,
    content: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct CodeStoryRequestOptions {
    temperature: f32,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct CodeStoryRequest {
    messages: Vec<CodeStoryMessage>,
    options: CodeStoryRequestOptions,
}

impl CodeStoryRequest {
    fn from_chat_request(request: LLMClientCompletionRequest) -> Self {
        Self {
            messages: request
                .messages()
                .into_iter()
                .map(|message| match message.role() {
                    LLMClientRole::System => CodeStoryMessage {
                        role: "system".to_owned(),
                        content: message.content().to_owned(),
                    },
                    LLMClientRole::User => CodeStoryMessage {
                        role: "user".to_owned(),
                        content: message.content().to_owned(),
                    },
                    LLMClientRole::Function => CodeStoryMessage {
                        role: "function".to_owned(),
                        content: message.content().to_owned(),
                    },
                    LLMClientRole::Assistant => CodeStoryMessage {
                        role: "assistant".to_owned(),
                        content: message.content().to_owned(),
                    },
                })
                .collect(),
            options: CodeStoryRequestOptions {
                temperature: request.temperature(),
            },
        }
    }
}

impl CodeStoryClient {
    pub fn new(api_base: &str) -> Self {
        Self {
            api_base: api_base.to_owned(),
            client: reqwest::Client::new(),
        }
    }

    pub fn gpt3_endpoint(&self, api_base: &str) -> String {
        format!("{api_base}/chat-3")
    }

    pub fn gpt4_endpoint(&self, api_base: &str) -> String {
        format!("{api_base}/chat-4")
    }

    pub fn model_name(&self, model: &LLMType) -> Result<String, LLMClientError> {
        match model {
            LLMType::GPT3_5_16k => Ok("gpt-3.5-turbo-16k-0613".to_owned()),
            LLMType::Gpt4 => Ok("gpt-4-0613".to_owned()),
            _ => Err(LLMClientError::UnSupportedModel),
        }
    }

    pub fn model_endpoint(&self, model: &LLMType) -> Result<String, LLMClientError> {
        match model {
            LLMType::GPT3_5_16k => Ok(self.gpt3_endpoint(&self.api_base)),
            LLMType::Gpt4 => Ok(self.gpt4_endpoint(&self.api_base)),
            _ => Err(LLMClientError::UnSupportedModel),
        }
    }
}

#[async_trait]
impl LLMClient for CodeStoryClient {
    fn client(&self) -> &LLMProvider {
        &LLMProvider::LMStudio
    }

    async fn completion(
        &self,
        api_key: LLMProviderAPIKeys,
        request: LLMClientCompletionRequest,
    ) -> Result<String, LLMClientError> {
        let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
        self.stream_completion(api_key, request, sender).await
    }

    async fn stream_completion(
        &self,
        _api_key: LLMProviderAPIKeys,
        request: LLMClientCompletionRequest,
        sender: UnboundedSender<LLMClientCompletionResponse>,
    ) -> Result<String, LLMClientError> {
        let model = self.model_name(request.model())?;
        let endpoint = self.model_endpoint(request.model())?;

        let request = CodeStoryRequest::from_chat_request(request);
        let mut response_stream = self
            .client
            .post(endpoint)
            .json(&request)
            .send()
            .await?
            .bytes_stream()
            .eventsource();

        let mut buffered_stream = "".to_owned();
        while let Some(event) = response_stream.next().await {
            match event {
                Ok(event) => {
                    if &event.data == "[DONE]" {
                        continue;
                    }
                    // we just proxy back the openai response back here
                    let response =
                        serde_json::from_str::<CreateChatCompletionStreamResponse>(&event.data);
                    match response {
                        Ok(response) => {
                            let delta = response
                                .choices
                                .get(0)
                                .map(|choice| choice.delta.content.to_owned())
                                .flatten()
                                .unwrap_or("".to_owned());
                            buffered_stream.push_str(&delta);
                            println!("{}", &buffered_stream);
                            sender.send(LLMClientCompletionResponse::new(
                                buffered_stream.to_owned(),
                                Some(delta),
                                model.to_owned(),
                            ))?;
                        }
                        Err(e) => {
                            dbg!(e);
                        }
                    }
                }
                Err(e) => {
                    dbg!(e);
                }
            }
        }
        Ok(buffered_stream)
    }

    async fn stream_prompt_completion(
        &self,
        api_key: LLMProviderAPIKeys,
        request: LLMClientCompletionStringRequest,
        sender: UnboundedSender<LLMClientCompletionResponse>,
    ) -> Result<String, LLMClientError> {
        Err(LLMClientError::UnSupportedModel)
    }
}
