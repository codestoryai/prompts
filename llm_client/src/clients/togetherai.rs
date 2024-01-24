use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use tokio::sync::mpsc::UnboundedSender;

use crate::provider::LLMProviderAPIKeys;

use super::types::LLMClient;
use super::types::LLMClientCompletionRequest;
use super::types::LLMClientCompletionResponse;
use super::types::LLMClientCompletionStringRequest;
use super::types::LLMClientError;
use super::types::LLMType;

pub struct TogetherAIClient {
    pub client: reqwest::Client,
    pub base_url: String,
}

#[derive(serde::Serialize, Debug, Clone)]
struct TogetherAIRequest {
    prompt: String,
    model: String,
    temperature: f32,
    stream_tokens: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct TogetherAIResponse {
    choices: Vec<Choice>,
    // id: String,
    // token: Token,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct Choice {
    text: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct Token {
    id: i32,
    text: String,
    logprob: i32,
    special: bool,
}

impl TogetherAIRequest {
    pub fn from_request(request: LLMClientCompletionRequest) -> Self {
        Self {
            prompt: {
                if request.messages().len() == 1 {
                    request.messages()[0].content().to_owned()
                } else {
                    request
                        .messages()
                        .into_iter()
                        .map(|message| message.content().to_owned())
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            },
            // TODO(skcd): Proper error handling here
            model: TogetherAIClient::model_str(request.model()).expect("to be present"),
            temperature: request.temperature(),
            stream_tokens: true,
            frequency_penalty: request.frequency_penalty(),
        }
    }

    pub fn from_string_request(request: LLMClientCompletionStringRequest) -> Self {
        Self {
            prompt: request.prompt().to_owned(),
            model: TogetherAIClient::model_str(request.model()).expect("to be present"),
            temperature: request.temperature(),
            stream_tokens: true,
            frequency_penalty: request.frequency_penalty(),
        }
    }
}

impl TogetherAIClient {
    pub fn new() -> Self {
        let client = reqwest::Client::new();
        Self {
            client,
            base_url: "https://api.together.xyz".to_owned(),
        }
    }

    pub fn inference_endpoint(&self) -> String {
        format!("{}/inference", self.base_url)
    }

    pub fn completion_endpoint(&self) -> String {
        format!("{}/completions", self.base_url)
    }

    pub fn model_str(model: &LLMType) -> Option<String> {
        match model {
            LLMType::Mixtral => Some("mistralai/Mixtral-8x7B-Instruct-v0.1".to_owned()),
            LLMType::MistralInstruct => Some("mistralai/Mistral-7B-Instruct-v0.1".to_owned()),
            LLMType::Custom(model) => Some(model.to_owned()),
            _ => None,
        }
    }

    fn generate_together_ai_bearer_key(
        &self,
        api_key: LLMProviderAPIKeys,
    ) -> Result<String, LLMClientError> {
        match api_key {
            LLMProviderAPIKeys::TogetherAI(api_key) => Ok(api_key.api_key),
            _ => Err(LLMClientError::WrongAPIKeyType),
        }
    }
}

#[async_trait]
impl LLMClient for TogetherAIClient {
    fn client(&self) -> &crate::provider::LLMProvider {
        &crate::provider::LLMProvider::TogetherAI
    }

    async fn completion(
        &self,
        api_key: LLMProviderAPIKeys,
        request: LLMClientCompletionRequest,
    ) -> Result<String, LLMClientError> {
        let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
        self.stream_completion(api_key, request, sender).await
    }

    async fn stream_prompt_completion(
        &self,
        api_key: LLMProviderAPIKeys,
        request: LLMClientCompletionStringRequest,
        sender: UnboundedSender<LLMClientCompletionResponse>,
    ) -> Result<String, LLMClientError> {
        let model = TogetherAIClient::model_str(request.model());
        if model.is_none() {
            return Err(LLMClientError::FailedToGetResponse);
        }
        let model = model.expect("is_none check above to work");
        let together_ai_request = TogetherAIRequest::from_string_request(request);
        let mut response_stream = self
            .client
            .post(self.inference_endpoint())
            .bearer_auth(self.generate_together_ai_bearer_key(api_key)?.to_owned())
            .json(&together_ai_request)
            .send()
            .await?
            .bytes_stream()
            .eventsource();

        let mut buffered_string = "".to_owned();
        while let Some(event) = response_stream.next().await {
            match event {
                Ok(event) => {
                    if &event.data == "[DONE]" {
                        continue;
                    }
                    let value = serde_json::from_str::<TogetherAIResponse>(&event.data)?;
                    buffered_string = buffered_string + &value.choices[0].text;
                    sender.send(LLMClientCompletionResponse::new(
                        buffered_string.to_owned(),
                        Some(value.choices[0].text.to_owned()),
                        model.to_owned(),
                    ))?;
                }
                Err(e) => {
                    dbg!(e);
                }
            }
        }

        Ok(buffered_string)
    }

    async fn stream_completion(
        &self,
        api_key: LLMProviderAPIKeys,
        request: LLMClientCompletionRequest,
        sender: UnboundedSender<LLMClientCompletionResponse>,
    ) -> Result<String, LLMClientError> {
        let model = TogetherAIClient::model_str(request.model());
        if model.is_none() {
            return Err(LLMClientError::FailedToGetResponse);
        }
        let model = model.expect("is_none check above to work");
        let together_ai_request = TogetherAIRequest::from_request(request);
        let mut response_stream = self
            .client
            .post(self.inference_endpoint())
            .bearer_auth(self.generate_together_ai_bearer_key(api_key)?.to_owned())
            .json(&together_ai_request)
            .send()
            .await?
            .bytes_stream()
            .eventsource();

        let mut buffered_string = "".to_owned();
        while let Some(event) = response_stream.next().await {
            match event {
                Ok(event) => {
                    if &event.data == "[DONE]" {
                        continue;
                    }
                    let value = serde_json::from_str::<TogetherAIResponse>(&event.data)?;
                    buffered_string = buffered_string + &value.choices[0].text;
                    sender.send(LLMClientCompletionResponse::new(
                        buffered_string.to_owned(),
                        Some(value.choices[0].text.to_owned()),
                        model.to_owned(),
                    ))?;
                }
                Err(e) => {
                    dbg!(e);
                }
            }
        }

        Ok(buffered_string)
    }
}
