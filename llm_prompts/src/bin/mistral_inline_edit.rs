//! We want to test the reranking with mistral

use std::path::PathBuf;

use llm_client::{
    broker::LLMBroker,
    clients::types::{LLMClientCompletionStringRequest, LLMType},
    config::LLMBrokerConfiguration,
    provider::{LLMProviderAPIKeys, TogetherAIProvider},
};

#[tokio::main]
async fn main() {
    let prompt = r#"[INST] You are an expert software engineer. You have to perform edits in the selected code snippet following the user instruction in <user_query> tags.
You have been given code context below:

Code Context above the selection:
```rust
// FILEPATH: $/Users/skcd/scratch/dataset/commit_play/src/language/types.rs
// BEGIN: abpxx6d04wxr
use std::{collections::HashMap, fmt::Debug, path::Display, sync::Arc};

use derivative::Derivative;

use crate::git::commit::GitCommit;

use super::config::TreeSitterLanguageParsing;

// These are always 0 indexed
#[derive(
    Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, std::hash::Hash,
)]
#[serde(rename_all = "camelCase")]
pub struct Position {
    line: usize,
    character: usize,
    byte_offset: usize,
}
// END: abpxx6d04wxr
```
Code Context below the selection:
```rust
// FILEPATH: $/Users/skcd/scratch/dataset/commit_play/src/language/types.rs
// BEGIN: be15d9bcejpp
impl Position {
    fn to_tree_sitter(&self) -> tree_sitter::Point {
        tree_sitter::Point::new(self.line, self.character)
    }

    pub fn from_tree_sitter_point(point: &tree_sitter::Point, byte_offset: usize) -> Self {
        Self {
            line: point.row,
            character: point.column,
            byte_offset,
        }
    }

    pub fn to_byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub fn new(line: usize, character: usize, byte_offset: usize) -> Self {
        Self {
            line,
            character,
            byte_offset,
        }
    }

    pub fn line(&self) -> usize {
        self.line
    }

    pub fn column(&self) -> usize {
        self.character
    }

    pub fn set_byte_offset(&mut self, byte_offset: usize) {
        self.byte_offset = byte_offset;
    }

    pub fn from_byte(byte: usize, line_end_indices: &[u32]) -> Self {
        let line = line_end_indices
            .iter()
            .position(|&line_end_byte| (line_end_byte as usize) > byte)
            .unwrap_or(0);

        let column = line
            .checked_sub(1)
            .and_then(|idx| line_end_indices.get(idx))
            .map(|&prev_line_end| byte.saturating_sub(prev_line_end as usize))
            .unwrap_or(byte);

        Self::new(line, column, byte)
    }
}

#[derive(
    Debug, Clone, Copy, serde::Deserialize, serde::Serialize, PartialEq, Eq, std::hash::Hash,
)]
#[serde(rename_all = "camelCase")]
pub struct Range {
    start_position: Position,
    end_position: Position,
}

impl Default for Range {
    fn default() -> Self {
        Self {
            start_position: Position::new(0, 0, 0),
            end_position: Position::new(0, 0, 0),
        }
    }
}

impl Range {
    pub fn new(start_position: Position, end_position: Position) -> Self {
        Self {
            start_position,
            end_position,
        }
    }
// END: be15d9bcejpp
```

Code you have to edit:
```rust
// FILEPATH: $/Users/skcd/scratch/dataset/commit_play/src/language/types.rs
// BEGIN: ed8c6549bwf9
impl Into<tree_sitter::Point> for Position {
    fn into(self) -> tree_sitter::Point {
        self.to_tree_sitter()
    }
}
// END: ed8c6549bwf9
```

Rewrite the code enclosed in // BEGIN: ed8c6549bwf9 and // END: ed8c6549bwf9 following the user query without any explanation [/INST]
The user has instructed me to perform the following edits on the selection:
can you add comments all over the function body?

The edited code is:
```rust
// FILEPATH: /Users/skcd/scratch/dataset/commit_play/src/language/types.rs
// BEGIN: ed8c6549bwf9
"#;
    let llm_broker = LLMBroker::new(LLMBrokerConfiguration::new(PathBuf::from(
        "/Users/skcd/Library/Application Support/ai.codestory.sidecar",
    )))
    .await
    .expect("broker to startup");

    let api_key =
        LLMProviderAPIKeys::TogetherAI(TogetherAIProvider::new("some_key_here".to_owned()));
    let request = LLMClientCompletionStringRequest::new(
        LLMType::MistralInstruct,
        prompt.to_owned(),
        1.0,
        None,
    );
    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let metadata = vec![("event_type".to_owned(), "listwise_reranking".to_owned())]
        .into_iter()
        .collect();
    let result = llm_broker
        .stream_string_completion(api_key.clone(), request, metadata, sender)
        .await;
    println!("Mistral:");
    println!("{:?}", result);
    let mixtral_request =
        LLMClientCompletionStringRequest::new(LLMType::Mixtral, prompt.to_owned(), 0.7, None);
    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let metadata = vec![("event_type".to_owned(), "listwise_reranking".to_owned())]
        .into_iter()
        .collect();
    let result = llm_broker
        .stream_string_completion(api_key, mixtral_request, metadata, sender)
        .await;
    println!("Mixtral:");
    println!("{:?}", result);
}
