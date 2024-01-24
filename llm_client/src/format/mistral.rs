use crate::clients::types::LLMClientMessage;

use super::types::{LLMFormatting, TokenizerConfig, TokenizerError};

pub struct MistralInstructFormatting {
    tokenizer_config: TokenizerConfig,
}

impl MistralInstructFormatting {
    pub fn new() -> Result<Self, TokenizerError> {
        let config = include_str!("tokenizer_config/mistral.json");
        let tokenizer_config = serde_json::from_str::<TokenizerConfig>(config)?;
        Ok(Self { tokenizer_config })
    }
}

impl LLMFormatting for MistralInstructFormatting {
    fn to_prompt(&self, messages: Vec<LLMClientMessage>) -> String {
        // we want to convert the message to mistral format
        // persent here: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json#L31-L34
        // {{ bos_token }}
        // {% for message in messages %}
        // {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        // {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
        // {% endif %}
        // {% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}
        // {% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}
        // {% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        // First the messages have to be alternating, if that's not enforced then we run into problems
        // but since thats the case, we can do something better, which is to to just send consecutive messages
        // from human and assistant together
        let formatted_message = messages
            .into_iter()
            .skip_while(|message| message.role().is_assistant())
            .map(|message| {
                let content = message.content();
                let eos_token = self.tokenizer_config.eos_token();
                if message.role().is_system() || message.role().is_user() {
                    format!("[INST] {content} [/INST]")
                } else if message.role().is_function() {
                    // This will be formatted as a function call as well
                    match message.get_function_call() {
                        Some(function_call) => {
                            let function_call = serde_json::to_string(function_call)
                                .expect("serde deserialize to not fail");
                            format!("[INST] {function_call} [/INST]")
                        }
                        None => {
                            // not entirely correct, we will make it better with more testing
                            format!("[INST] {content} [/INST]")
                        }
                    }
                } else {
                    // we are in an assistant message now, so we can have a function
                    // call which we have to format
                    match message.get_function_call() {
                        Some(function_call) => {
                            let function_call = serde_json::to_string(function_call)
                                .expect("serde deserialize to not fail");
                            format!("{content}{function_call}{eos_token} ")
                        }
                        None => {
                            format!("{content}{eos_token} ")
                        }
                    }
                }
            })
            .collect::<Vec<_>>()
            .join("");
        format!("<s>{formatted_message}")
    }
}

#[cfg(test)]
mod tests {

    use crate::clients::types::LLMClientMessage;

    use super::LLMFormatting;
    use super::MistralInstructFormatting;

    #[test]
    fn test_formatting_works() {
        let messages = vec![
            LLMClientMessage::user("user_msg1".to_owned()),
            LLMClientMessage::assistant("assistant_msg1".to_owned()),
        ];
        let mistral_formatting = MistralInstructFormatting::new().unwrap();
        assert_eq!(
            mistral_formatting.to_prompt(messages),
            "<s>[INST] user_msg1 [/INST]assistant_msg1</s> ",
        );
    }
}
