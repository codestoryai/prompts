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
            deployment_id: "gpt35-turbo-access".to_string(),
            api_base: "https://codestory-gpt4.openai.azure.com".to_owned(),
            api_key: "89ca8a49a33344c9b794b3dabcbbc5d0".to_owned(),
            api_version: "2023-08-01-preview".to_owned(),
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
