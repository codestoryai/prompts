use llm_client::clients::types::LLMClientMessage;

use crate::in_line_edit::doc_helpers::document_symbol_metadata;

use super::types::InLineDocRequest;
use super::types::InLineEditPrompt;
use super::types::InLineEditRequest;
use super::types::InLineFixRequest;
use super::types::InLinePromptResponse;

pub struct OpenAILineEditPrompt {}

impl OpenAILineEditPrompt {
    pub fn new() -> Self {
        Self {}
    }
}

impl OpenAILineEditPrompt {
    fn system_message_inline_edit(&self, language: &str) -> String {
        format!(
            r#"You are an AI programming assistant.
When asked for your name, you must respond with "Aide".
Follow the user's requirements carefully & to the letter.
- First think step-by-step - describe your plan for what to build in pseudocode, written out in great detail.
- Then output the code in a single code block.
- Minimize any other prose.
- Each code block starts with ``` and // FILEPATH.
- If you suggest to run a terminal command, use a code block that starts with ```bash.
- You always answer with {language} code.
- Modify the code or create new code.
- Unless directed otherwise, the user is expecting for you to edit their selected code.
- Make sure to ALWAYS INCLUDE the BEGIN and END markers in your generated code with // BEGIN and then // END which is present in the code selection given by the user
You must decline to answer if the question is not related to a developer.
If the question is related to a developer, you must respond with content related to a developer."#
        )
    }

    fn system_message_fix(&self, language: &str) -> String {
        format!(
            r#"You are an AI programming assistant.
When asked for your name, you must respond with "Aide".
Follow the user's requirements carefully & to the letter.
- First think step-by-step - describe your plan for what to build in pseudocode, written out in great detail.
- Then output the code in a single code block.
- Minimize any other prose.
- Each code block starts with ``` and // FILEPATH.
- If you suggest to run a terminal command, use a code block that starts with ```bash.
- You always answer with {language} code.
- Modify the code or create new code.
- Unless directed otherwise, the user is expecting for you to edit their selected code.
You must decline to answer if the question is not related to a developer.
If the question is related to a developer, you must respond with content related to a developer."#
        )
    }

    fn documentation_system_prompt(&self, language: &str, is_identifier_node: bool) -> String {
        if is_identifier_node {
            let system_prompt = format!(
                r#"You are an AI programming assistant.
When asked for your name, you must respond with "Aide".
Follow the user's requirements carefully & to the letter.
- Each code block must ALWAYS STARTS and include ```{language} and // FILEPATH
- You always answer with {language} code.
- When the user asks you to document something, you must answer in the form of a {language} code block.
- Your documentation should not include just the name of the function, think about what the function is really doing.
- When generating the documentation, be sure to understand what the function is doing and include that as part of the documentation and then generate the documentation.
- DO NOT modify the code which you will be generating"#
            );
            system_prompt.to_owned()
        } else {
            let system_prompt = format!(
                r#"You are an AI programming assistant.
When asked for your name, you must respond with "Aide".
Follow the user's requirements carefully & to the letter.
- Each code block must ALWAYS STARTS and include ```{language} and // FILEPATH
- You always answer with {language} code.
- When the user asks you to document something, you must answer in the form of a {language} code block.
- Your documentation should not include just the code selection, think about what the selection is really doing.
- When generating the documentation, be sure to understand what the selection is doing and include that as part of the documentation and then generate the documentation.
- DO NOT modify the code which you will be generating"#
            );
            system_prompt.to_owned()
        }
    }

    fn above_selection(&self, above_context: Option<&String>) -> Option<String> {
        if let Some(above_context) = above_context {
            Some(format!(
                r#"I have the following code above:
{above_context}"#
            ))
        } else {
            None
        }
    }

    fn below_selection(&self, below_context: Option<&String>) -> Option<String> {
        if let Some(below_context) = below_context {
            Some(format!(
                r#"I have the following code below:
{below_context}"#
            ))
        } else {
            None
        }
    }
}

impl InLineEditPrompt for OpenAILineEditPrompt {
    fn inline_edit(&self, request: InLineEditRequest) -> InLinePromptResponse {
        // Here we create the messages for the openai, since we have flexibility
        // and the llms are in general smart we can just send the chat messages
        // instead of the completion(which has been deprecated)
        let above = request.above();
        let below = request.below();
        let in_range = request.in_range();
        let language = request.language();

        let mut messages = vec![];
        messages.push(LLMClientMessage::system(
            self.system_message_inline_edit(language),
        ));
        if let Some(above) = self.above_selection(above) {
            messages.push(LLMClientMessage::user(above));
        }
        if let Some(below) = self.below_selection(below) {
            messages.push(LLMClientMessage::user(below));
        }
        if let Some(in_range) = in_range {
            messages.push(LLMClientMessage::user(in_range.to_owned()));
        }
        messages.push(LLMClientMessage::user(request.user_query().to_owned()));
        // Add an additional message about keeping the // FILEPATH and the markers
        messages.push(LLMClientMessage::system(format!(
            r#"Make sure to ALWAYS INCLUDE the BEGIN and END markers in your generated code with // BEGIN and then // END which is present in the code selection given by me"#
        )));
        InLinePromptResponse::Chat(messages)
    }

    fn inline_fix(&self, request: InLineFixRequest) -> InLinePromptResponse {
        let above = request.above();
        let below = request.below();
        let in_range = request.in_range();
        let language = request.language();

        let mut messages = vec![];
        messages.push(LLMClientMessage::system(self.system_message_fix(language)));
        if let Some(above) = self.above_selection(above) {
            messages.push(LLMClientMessage::user(above));
        }
        if let Some(below) = self.below_selection(below) {
            messages.push(LLMClientMessage::user(below));
        }
        messages.push(LLMClientMessage::user(in_range.to_owned()));
        messages.extend(
            request
                .diagnostics_prompts()
                .into_iter()
                .map(|diagnostic_prompt| LLMClientMessage::user(diagnostic_prompt.to_owned())),
        );
        messages.push(
            LLMClientMessage::user("Do not forget to include the // BEGIN and // END markers in your generated code. Only change the code inside of the selection, delimited by the markers: // BEGIN: ed8c6549bwf9 and // END: ed8c6549bwf9".to_owned())
        );
        InLinePromptResponse::Chat(messages)
    }

    fn inline_doc(&self, request: InLineDocRequest) -> InLinePromptResponse {
        let system_prompt =
            self.documentation_system_prompt(request.language(), request.is_identifier_node());
        let mut messages = vec![];
        messages.push(LLMClientMessage::system(system_prompt));
        messages.push(LLMClientMessage::user(request.in_range().to_owned()));
        messages.push(LLMClientMessage::user(document_symbol_metadata(&request)));
        messages.push(LLMClientMessage::user("Do not forget to the include the // BEGIN and // END markers in your generated code. Only change the code provided to you in the selection".to_owned()));
        InLinePromptResponse::Chat(messages)
    }
}
