//! Call the endpoints of codestory endpoint

use llm_client::{
    clients::{
        codestory::CodeStoryClient,
        types::{LLMClient, LLMClientCompletionRequest, LLMClientMessage, LLMType},
    },
    provider::LLMProviderAPIKeys,
};

#[tokio::main]
async fn main() {
    let codestory_client =
        CodeStoryClient::new("https://codestory-provider-dot-anton-390822.ue.r.appspot.com");
    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let request = LLMClientCompletionRequest::new(
        LLMType::GPT3_5_16k,
        vec![
            LLMClientMessage::system("you are a python expert".to_owned()),
            LLMClientMessage::user(
                "write me a big python function which does a lot of things".to_owned(),
            ),
        ],
        1.0,
        None,
    );
    let response = codestory_client
        .stream_completion(LLMProviderAPIKeys::CodeStory, request, sender)
        .await;
    println!("{:?}", response);
}
