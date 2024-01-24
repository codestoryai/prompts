use std::collections::HashMap;

use llm_client::clients::types::LLMClientCompletionStringRequest;

use super::types::{
    CodeSpan, CodeSpanDigest, ReRankCodeSpan, ReRankCodeSpanError, ReRankCodeSpanRequest,
    ReRankCodeSpanResponse, ReRankListWiseResponse, ReRankPointWisePrompt, ReRankStrategy,
};

#[derive(Default)]
pub struct MistralReRank {}

impl MistralReRank {
    pub fn new() -> Self {
        Default::default()
    }
}

impl MistralReRank {
    pub fn pointwise_reranking(&self, request: ReRankCodeSpanRequest) -> ReRankCodeSpanResponse {
        let code_span_digests = CodeSpan::to_digests(request.code_spans().to_vec());
        // Now we query the LLM for the pointwise reranking here
        let user_query = request.user_query().to_owned();
        let prompts = code_span_digests
            .into_iter()
            .map(|code_span_digest| {
                let user_query = user_query.to_owned();
                let hash = code_span_digest.hash();
                let data = code_span_digest.data();
                let prompt = format!(r#"<s>[INST] You are an expert software developer responsible for helping detect whether the retrieved snippet of code is relevant to the query. For a given input, you need to output a single word: "Yes" or "No" indicating the retrieved snippet is relevant to the query.
Query: Where is the client for OpenAI defined?
Code Snippet:
```/Users/skcd/client/openai.rs
pub struct OpenAIClient {{}}

impl OpenAIClient {{
    pub fn new() -> Self {{
        Self {{}}
    }}
```
Relevant: Yes

Query: Where do we handle the errors in the webview?
Snippet:
```/Users/skcd/algorithm/dfs.rs
pub fn dfs(graph: &Graph, start: NodeId) -> Vec<NodeId> {{
    let mut visited = HashSet::new();
    let mut stack = vec![start];
    let mut result = vec![];
    while let Some(node) = stack.pop() {{
        if visited.contains(&node) {{
            continue;
        }}
        visited.insert(node);
        result.push(node);
        for neighbor in graph.neighbors(node) {{
            stack.push(neighbor);
        }}
    }}
    result
}}
```
Relevant: No

Query: {user_query}
Snippet:
```{hash}
{data}
``` [/INST]
Relevant: "#);
                let prompt = LLMClientCompletionStringRequest::new(
                    request.llm_type().clone(),
                    prompt,
                    0.0,
                    None,
                );
                ReRankPointWisePrompt::new_string_completion(prompt, code_span_digest)
            })
            .collect();

        ReRankCodeSpanResponse::PointWise(prompts)
    }

    pub fn listwise_reranking(&self, request: ReRankCodeSpanRequest) -> ReRankCodeSpanResponse {
        // First we get the code spans which are present here cause they are important
        let code_spans = request.code_spans().to_vec();
        let user_query = request.user_query().to_owned();
        // Now we need to generate the prompt for this
        let code_span_digests = CodeSpan::to_digests(code_spans);
        let code_snippets = code_span_digests
            .iter()
            .map(|code_span_digest| {
                let identifier = code_span_digest.hash();
                let data = code_span_digest.data();
                let span_identifier = code_span_digest.get_span_identifier();
                format!(
                    "<id>\n{identifier}\n</id>\n<code_snippet>\n```\n{span_identifier}\n{data}\n```\n<code_snippet>\n"
                )
            })
            .collect::<Vec<String>>()
            .join("\n");
        // Now we create the prompt for this reranking
        let prompt = format!(
            r#"<s>[INST] You are an expert at ordering the code snippets from the most relevant to the least relevant for the user query. You have the order the list of code snippets from the most relevant to the least relevant. As an example
<code_snippets>
<id>
subtract.rs::0
</id>
<snippet>
```
fn subtract(a: i32, b: i32) -> i32 {{
    a - b
}}
```
</snippet>

<id>
add.rs::0
</id>
<snippet>
```
fn add(a: i32, b: i32) -> i32 {{
    a + b
}}
```
</snippet>
</code_snippets>

And if you thought the code snippet with id add.rs::0 is more relevant than subtract.rs::0 then you would rank it as:
<ranking>
<id>
add.rs::0
</id>
<id>
subtract.rs::0
</id>
</ranking>

Now for the actual query.
The user has asked the following query:
<user_query>
{user_query}
</user_query>

The code snippets along with their ids are given below:
<code_snippets>
{code_snippets}
</code_snippets>

As a reminder the user question is:
<user_query>
{user_query}
</user_query>
You have to order all the code snippets from the most relevant to the least relevant to the user query, all the code snippet ids should be present in your final reordered list. Only output the ids of the code snippets.
[/INST]<ranking>
<id>
"#
        );
        let prompt =
            LLMClientCompletionStringRequest::new(request.llm_type().clone(), prompt, 0.0, None);
        ReRankCodeSpanResponse::listwise_completion(prompt, code_span_digests)
    }

    fn parse_listwise_output(
        &self,
        output: &str,
        code_span_digests: Vec<CodeSpanDigest>,
    ) -> Result<Vec<CodeSpanDigest>, ReRankCodeSpanError> {
        // The output is generally in the format of
        // <id>
        // {id}
        // </id>
        // ...
        // </reranking>
        // This is not guaranteed with mistral instruct (but always by mixtral)
        // so we split the string on \n and ignore the values which are <id> or
        // </id> and only parse until we get the </reranking> tag
        let mut output = output.split("\n");
        let mut code_span_digests_mapping: HashMap<String, CodeSpanDigest> = code_span_digests
            .into_iter()
            .map(|code_span_digest| (code_span_digest.hash().to_owned(), code_span_digest))
            .collect();
        let mut code_spans_reordered_list = Vec::new();
        while let Some(line) = output.next() {
            if line.contains("</reranking>") {
                break;
            }
            if line.contains("<id>") || line.contains("</id>") {
                continue;
            }
            let possible_id = line.trim();
            if let Some(code_span) = code_span_digests_mapping.remove(possible_id) {
                code_spans_reordered_list.push(code_span)
            }
        }

        // Add all the remaining code spans to the end of the list
        code_span_digests_mapping
            .into_iter()
            .for_each(|(_, code_span)| {
                code_spans_reordered_list.push(code_span);
            });

        // Now that we have the possible ids in the list, we get the list of ranked
        // code span digests in the same manner
        Ok(code_spans_reordered_list)
    }
}

impl ReRankCodeSpan for MistralReRank {
    fn rerank_prompt(
        &self,
        request: ReRankCodeSpanRequest,
    ) -> Result<ReRankCodeSpanResponse, ReRankCodeSpanError> {
        Ok(match request.strategy() {
            ReRankStrategy::ListWise => self.listwise_reranking(request),
            ReRankStrategy::PointWise => {
                // We need to generate the prompt for this
                self.pointwise_reranking(request)
            }
        })
    }

    fn parse_listwise_output(
        &self,
        llm_output: String,
        rerank_request: ReRankListWiseResponse,
    ) -> Result<Vec<CodeSpanDigest>, ReRankCodeSpanError> {
        self.parse_listwise_output(&llm_output, rerank_request.code_span_digests)
    }
}
