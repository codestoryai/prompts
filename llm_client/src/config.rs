//! The configuration which will be passed to the llm broker

use std::path::PathBuf;

pub struct LLMBrokerConfiguration {
    pub data_dir: PathBuf,
}

impl LLMBrokerConfiguration {
    pub fn new(data_dir: PathBuf) -> Self {
        Self { data_dir }
    }
}
