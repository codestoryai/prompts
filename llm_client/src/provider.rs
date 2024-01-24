//! Contains types for setting the provider for the LLM, we are going to support
//! 3 things for now:
//! - CodeStory
//! - OpenAI
//! - Ollama
//! - Azure
//! - together.ai

use crate::clients::types::LLMType;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Hash, PartialEq, Eq)]
pub struct AzureOpenAIDeploymentId {
    pub deployment_id: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Hash, PartialEq, Eq)]
pub struct CodeStoryLLMType {
    // shoe horning the llm type here so we can provide the correct api keys
    pub llm_type: Option<LLMType>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Hash, PartialEq, Eq)]
pub enum LLMProvider {
    OpenAI,
    TogetherAI,
    Ollama,
    LMStudio,
    CodeStory(CodeStoryLLMType),
    Azure(AzureOpenAIDeploymentId),
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum LLMProviderAPIKeys {
    OpenAI(OpenAIProvider),
    TogetherAI(TogetherAIProvider),
    Ollama(OllamaProvider),
    OpenAIAzureConfig(AzureConfig),
    LMStudio(LMStudioConfig),
    CodeStory,
}

impl LLMProviderAPIKeys {
    pub fn is_openai(&self) -> bool {
        matches!(self, LLMProviderAPIKeys::OpenAI(_))
    }

    pub fn provider_type(&self) -> LLMProvider {
        match self {
            LLMProviderAPIKeys::OpenAI(_) => LLMProvider::OpenAI,
            LLMProviderAPIKeys::TogetherAI(_) => LLMProvider::TogetherAI,
            LLMProviderAPIKeys::Ollama(_) => LLMProvider::Ollama,
            LLMProviderAPIKeys::OpenAIAzureConfig(_) => {
                LLMProvider::Azure(AzureOpenAIDeploymentId {
                    deployment_id: "".to_owned(),
                })
            }
            LLMProviderAPIKeys::LMStudio(_) => LLMProvider::LMStudio,
            LLMProviderAPIKeys::CodeStory => {
                LLMProvider::CodeStory(CodeStoryLLMType { llm_type: None })
            }
        }
    }

    // Gets the relevant key from the llm provider
    pub fn key(&self, llm_provider: &LLMProvider) -> Option<Self> {
        match llm_provider {
            LLMProvider::OpenAI => {
                if let LLMProviderAPIKeys::OpenAI(key) = self {
                    Some(LLMProviderAPIKeys::OpenAI(key.clone()))
                } else {
                    None
                }
            }
            LLMProvider::TogetherAI => {
                if let LLMProviderAPIKeys::TogetherAI(key) = self {
                    Some(LLMProviderAPIKeys::TogetherAI(key.clone()))
                } else {
                    None
                }
            }
            LLMProvider::Ollama => {
                if let LLMProviderAPIKeys::Ollama(key) = self {
                    Some(LLMProviderAPIKeys::Ollama(key.clone()))
                } else {
                    None
                }
            }
            LLMProvider::LMStudio => {
                if let LLMProviderAPIKeys::LMStudio(key) = self {
                    Some(LLMProviderAPIKeys::LMStudio(key.clone()))
                } else {
                    None
                }
            }
            // Azure is weird, so we are have to copy the config which we get
            // from the provider keys and then set the deployment id of it
            // properly for the azure provider, if its set to "" that means
            // we do not have a deployment key and we should be returning quickly
            // here.
            // NOTE: We should change this to using the codestory configuration
            // and make calls appropriately, for now this is fine
            LLMProvider::Azure(deployment_id) => {
                if deployment_id.deployment_id == "" {
                    return None;
                }
                if let LLMProviderAPIKeys::OpenAIAzureConfig(key) = self {
                    let mut azure_config = key.clone();
                    azure_config.deployment_id = deployment_id.deployment_id.to_owned();
                    Some(LLMProviderAPIKeys::OpenAIAzureConfig(azure_config))
                } else {
                    None
                }
            }
            LLMProvider::CodeStory(_) => Some(LLMProviderAPIKeys::CodeStory),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct OpenAIProvider {
    pub api_key: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TogetherAIProvider {
    pub api_key: String,
}

impl TogetherAIProvider {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct OllamaProvider {}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct AzureConfig {
    pub deployment_id: String,
    pub api_base: String,
    pub api_key: String,
    pub api_version: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LMStudioConfig {
    pub api_base: String,
}

impl LMStudioConfig {
    pub fn api_base(&self) -> &str {
        &self.api_base
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CodeStoryConfig {
    pub llm_type: LLMType,
}

#[cfg(test)]
mod tests {
    use super::{AzureOpenAIDeploymentId, LLMProvider, LLMProviderAPIKeys};

    #[test]
    fn test_reading_from_string_for_provider() {
        let provider = LLMProvider::Azure(AzureOpenAIDeploymentId {
            deployment_id: "testing".to_owned(),
        });
        let string_provider = serde_json::to_string(&provider).expect("to work");
        assert_eq!(
            string_provider,
            "{\"Azure\":{\"deployment_id\":\"testing\"}}"
        );
        let provider = LLMProvider::Ollama;
        let string_provider = serde_json::to_string(&provider).expect("to work");
        assert_eq!(string_provider, "\"Ollama\"");
    }

    #[test]
    fn test_reading_provider_keys() {
        let provider_keys = LLMProviderAPIKeys::OpenAI(super::OpenAIProvider {
            api_key: "testing".to_owned(),
        });
        let string_provider_keys = serde_json::to_string(&provider_keys).expect("to work");
        assert_eq!(string_provider_keys, "",);
    }
}
