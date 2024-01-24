use async_openai::{
    config::AzureConfig,
    types::{ChatCompletionRequestMessageArgs, CreateChatCompletionRequestArgs},
    Client,
};
use futures::StreamExt;
use llm_client::clients::{
    openai::OpenAIClient,
    types::{LLMClient, LLMClientCompletionRequest, LLMClientMessage},
};
use llm_client::provider::AzureConfig as ProviderAzureConfig;

#[tokio::main]
async fn main() {
    let openai_client = OpenAIClient::new();
    let api_key =
        llm_client::provider::LLMProviderAPIKeys::OpenAIAzureConfig(ProviderAzureConfig {
            deployment_id: "some_deployment_id".to_string(),
            api_base: "some_base".to_owned(),
            api_key: "some_key".to_owned(),
            api_version: "some_version".to_owned(),
        });
    let request = LLMClientCompletionRequest::new(
        llm_client::clients::types::LLMType::GPT3_5_16k,
        vec![LLMClientMessage::system("message".to_owned())],
        1.0,
        None,
    );
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    let response = openai_client
        .stream_completion(api_key, request, sender)
        .await;
    dbg!(&response);
}
