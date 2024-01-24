use crate::clients::types::LLMClientMessage;

use super::types::LLMFormatting;

pub struct DeepSeekCoderFormatting {}

impl DeepSeekCoderFormatting {
    pub fn new() -> Self {
        Self {}
    }
}

impl LLMFormatting for DeepSeekCoderFormatting {
    fn to_prompt(&self, messages: Vec<LLMClientMessage>) -> String {
        // we want to convert the message to deepseekcoder format
        // present here: https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/blob/main/tokenizer_config.json#L34
        // {% if not add_generation_prompt is defined %}
        // {% set add_generation_prompt = false %}
        // {% endif %}
        // {%- set ns = namespace(found=false) -%}
        // {%- for message in messages -%}
        //     {%- if message['role'] == 'system' -%}
        //         {%- set ns.found = true -%}
        //     {%- endif -%}
        // {%- endfor -%}
        // {{bos_token}}{%- if not ns.found -%}
        // {{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n'}}
        // {%- endif %}
        // {%- for message in messages %}
        //     {%- if message['role'] == 'system' %}
        // {{ message['content'] }}
        //     {%- else %}
        //         {%- if message['role'] == 'user' %}
        // {{'### Instruction:\n' + message['content'] + '\n'}}
        //         {%- else %}
        // {{'### Response:\n' + message['content'] + '\n<|EOT|>\n'}}
        //         {%- endif %}
        //     {%- endif %}
        // {%- endfor %}
        // {% if add_generation_prompt %}
        // {{'### Response:'}}
        // {% endif %}
        let formatted_message = messages.into_iter().skip_while(|message| message.role().is_assistant())
        .map(|message| {
            let content = message.content();
            if message.role().is_system() {
                "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n".to_owned()
            } else if message.role().is_user() {
                format!("### Instruction:\n{}\n", content)
            } else {
                format!("### Response:\n{}\n<|EOT|>\n", content)
            }
        }).collect::<Vec<_>>().join("");
        formatted_message
    }
}

#[cfg(test)]
mod tests {
    use crate::clients::types::LLMClientMessage;

    use super::DeepSeekCoderFormatting;
    use super::LLMFormatting;

    #[test]
    fn test_formatting_works() {
        let messages = vec![
            LLMClientMessage::system("system_message_not_show_up".to_owned()),
            LLMClientMessage::user("user_message1".to_owned()),
            LLMClientMessage::assistant("assistant_message1".to_owned()),
            LLMClientMessage::user("user_message2".to_owned()),
        ];
        let deepseek_formatting = DeepSeekCoderFormatting::new();
        assert_eq!(deepseek_formatting.to_prompt(messages), "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\nuser_message1\n### Response:\nassistant_message1\n<|EOT|>\n### Instruction:\nuser_message2\n");
    }
}
