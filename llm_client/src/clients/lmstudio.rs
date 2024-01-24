use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use tokio::sync::mpsc::UnboundedSender;

use crate::provider::{LLMProvider, LLMProviderAPIKeys};

use super::types::{
    LLMClient, LLMClientCompletionRequest, LLMClientCompletionResponse,
    LLMClientCompletionStringRequest, LLMClientError, LLMClientRole,
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

pub struct LMStudioClient {
    client: reqwest::Client,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct LLMStudioMessage {
    role: String,
    content: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct LMStudioRequest {
    prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    messages: Option<Vec<LLMStudioMessage>>,
    temperature: f32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    // set the max tokens to -1 so we get as much completion as possible
    max_tokens: i32,
}

impl LMStudioRequest {
    fn from_string_request(request: LLMClientCompletionStringRequest) -> Self {
        Self {
            prompt: Some(request.prompt().to_owned()),
            messages: None,
            temperature: request.temperature(),
            stream: true,
            frequency_penalty: request.frequency_penalty(),
            max_tokens: -1,
        }
    }

    fn from_chat_request(request: LLMClientCompletionRequest) -> Self {
        Self {
            prompt: None,
            messages: Some(
                request
                    .messages()
                    .into_iter()
                    .map(|message| match message.role() {
                        LLMClientRole::System => LLMStudioMessage {
                            role: "system".to_owned(),
                            content: message.content().to_owned(),
                        },
                        LLMClientRole::User => LLMStudioMessage {
                            role: "user".to_owned(),
                            content: message.content().to_owned(),
                        },
                        LLMClientRole::Function => LLMStudioMessage {
                            role: "function".to_owned(),
                            content: message.content().to_owned(),
                        },
                        LLMClientRole::Assistant => LLMStudioMessage {
                            role: "assistant".to_owned(),
                            content: message.content().to_owned(),
                        },
                    })
                    .collect(),
            ),
            temperature: request.temperature(),
            stream: true,
            frequency_penalty: request.frequency_penalty(),
            max_tokens: -1,
        }
    }
}

impl LMStudioClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    pub fn completion_endpoint(&self, base_url: &str) -> String {
        format!("{}/v1/completions", base_url)
    }

    pub fn chat_endpoint(&self, base_url: &str) -> String {
        format!("{}/v1/chat/completions", base_url)
    }

    pub fn generate_base_url(&self, api_key: LLMProviderAPIKeys) -> Result<String, LLMClientError> {
        match api_key {
            LLMProviderAPIKeys::LMStudio(api_key) => Ok(api_key.api_base().to_owned()),
            _ => Err(LLMClientError::UnSupportedModel),
        }
    }
}

#[async_trait]
impl LLMClient for LMStudioClient {
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
        api_key: LLMProviderAPIKeys,
        request: LLMClientCompletionRequest,
        sender: UnboundedSender<LLMClientCompletionResponse>,
    ) -> Result<String, LLMClientError> {
        let base_url = self.generate_base_url(api_key)?;
        let endpoint = self.chat_endpoint(&base_url);

        let request = LMStudioRequest::from_chat_request(request);
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
                    let value = serde_json::from_str::<LMStudioResponse>(&event.data)?;
                    buffered_stream = buffered_stream + &value.choices[0].text;
                    sender.send(LLMClientCompletionResponse::new(
                        buffered_stream.to_owned(),
                        Some(value.choices[0].text.to_owned()),
                        value.model,
                    ))?;
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
        let base_url = self.generate_base_url(api_key)?;
        let endpoint = self.completion_endpoint(&base_url);

        let request = LMStudioRequest::from_string_request(request);
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
                    let value = serde_json::from_str::<LMStudioResponse>(&event.data)?;
                    buffered_stream = buffered_stream + &value.choices[0].text;
                    sender.send(LLMClientCompletionResponse::new(
                        buffered_stream.to_owned(),
                        Some(value.choices[0].text.to_owned()),
                        value.model,
                    ))?;
                }
                Err(e) => {
                    dbg!(e);
                }
            }
        }
        Ok(buffered_stream)
    }
}
