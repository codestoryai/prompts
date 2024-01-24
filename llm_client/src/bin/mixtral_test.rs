use llm_client::clients::togetherai::TogetherAIClient;
use llm_client::{
    clients::types::{LLMClient, LLMClientCompletionRequest, LLMClientMessage},
    provider::TogetherAIProvider,
};

#[tokio::main]
async fn main() {
    let togetherai = TogetherAIClient::new();
    let api_key = llm_client::provider::LLMProviderAPIKeys::TogetherAI(TogetherAIProvider {
        api_key: "cc10d6774e67efef2004b85efdb81a3c9ba0b7682cc33d59c30834183502208d".to_owned(),
    });
    let message = r#"[INST] You are an expert software engineer. You have been given some code context below:

Code Context above the selection:
```rust
// FILEPATH: $/Users/skcd/scratch/dataset/commit_play/src/bin/tokenizers.rs
// BEGIN: abpxx6d04wxr
//! Tokenizers we are going to use

use std::str::FromStr;

use commit_play::llm::tokenizers::mistral::mistral_tokenizer;
use tokenizers::Tokenizer;

#[tokio::main]
// END: abpxx6d04wxr
```

Your task is to rewrite the code below following the instruction: add tracing::info! calls after the call to encode and decode
Code you have to edit:
```rust
// FILEPATH: $/Users/skcd/scratch/dataset/commit_play/src/bin/tokenizers.rs
// BEGIN: ed8c6549bwf9
async fn main() {
    let tokenizer = Tokenizer::from_str(&mistral_tokenizer()).expect("tokenizer error");

    let tokens = tokenizer.encode(
        "[INST] write a function which adds 2 numbers in python [/INST]",
        true,
    );

    let result = tokenizer.decode(
        &[
            733, 16289, 28793, 3324, 264, 908, 690, 13633, 28705, 28750, 5551, 297, 21966, 733,
            28748, 16289, 28793,
        ],
        false,
    );

    dbg!(tokens.expect("to work").get_ids());

    dbg!(result.expect("something"));
}

// END: ed8c6549bwf9
```

Rewrite the code without any explanation [/INST]
```rust
// FILEPATH: /Users/skcd/scratch/dataset/commit_play/src/bin/tokenizers.rs
// BEGIN: ed8c6549bwf9"#;
    let request = LLMClientCompletionRequest::new(
        llm_client::clients::types::LLMType::MistralInstruct,
        vec![LLMClientMessage::user(message.to_owned())],
        1.0,
        None,
    );
    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let response = togetherai.stream_completion(api_key, request, sender).await;
    dbg!(&response);
}
