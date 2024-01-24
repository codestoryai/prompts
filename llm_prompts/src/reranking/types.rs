use std::collections::HashMap;

use futures::future::Either;
use llm_client::{
    clients::types::{
        LLMClientCompletionRequest, LLMClientCompletionStringRequest, LLMClientError, LLMType,
    },
    tokenizer::tokenizer::LLMTokenizerError,
};

#[derive(Clone, Debug, PartialEq)]
pub struct CodeSpan {
    file_path: String,
    start_line: u64,
    end_line: u64,
    data: String,
}

impl CodeSpan {
    pub fn to_prompt(&self) -> String {
        format!(
            // TODO(skcd): Pass the language here for more accurate token counting
            "FILEPATH: {}-{}:{}\n```language\n{}```",
            self.file_path, self.start_line, self.end_line, self.data
        )
    }

    pub fn merge_consecutive_spans(code_spans: Vec<Self>) -> Vec<Self> {
        const CHUNK_MERGE_DISTANCE: usize = 0;
        let mut file_to_code_snippets: HashMap<String, Vec<CodeSpan>> = Default::default();

        code_spans.into_iter().for_each(|code_span| {
            let file_path = code_span.file_path.clone();
            let code_spans = file_to_code_snippets
                .entry(file_path)
                .or_insert_with(Vec::new);
            code_spans.push(code_span);
        });

        // We want to sort the code snippets in increasing order of the start line
        file_to_code_snippets
            .iter_mut()
            .for_each(|(_, code_snippets)| {
                code_snippets.sort_by(|a, b| a.start_line.cmp(&b.start_line));
            });

        // Now we will merge chunks which are in the range of CHUNK_MERGE_DISTANCE
        let results = file_to_code_snippets
            .into_iter()
            .map(|(file_path, mut code_snippets)| {
                let mut final_code_snippets = Vec::new();
                let mut current_code_snippet = code_snippets.remove(0);
                for code_snippet in code_snippets {
                    if code_snippet.end_line - current_code_snippet.start_line
                        <= CHUNK_MERGE_DISTANCE as u64
                    {
                        // We can merge these two code snippets
                        current_code_snippet.end_line = code_snippet.end_line;
                        current_code_snippet.data =
                            format!("{}{}", current_code_snippet.data, code_snippet.data);
                    } else {
                        // We cannot merge these two code snippets
                        final_code_snippets.push(current_code_snippet);
                        current_code_snippet = code_snippet;
                    }
                }
                final_code_snippets.push(current_code_snippet);
                final_code_snippets
                    .into_iter()
                    .map(|code_snippet| CodeSpan {
                        file_path: file_path.clone(),
                        data: code_snippet.data,
                        start_line: code_snippet.start_line,
                        end_line: code_snippet.end_line,
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();
        results
    }
}

/// This is the digest of the code span, we create a unique id for the code span
/// always and use that for passing it to the prompt
#[derive(Clone)]
pub struct CodeSpanDigest {
    code_span: CodeSpan,
    hash: String,
}

impl CodeSpanDigest {
    pub fn new(code_span: CodeSpan, file_path: &str, index: usize) -> Self {
        // TODO(skcd): Add proper error handling here
        let base_name = std::path::Path::new(file_path)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap();
        Self {
            code_span,
            hash: format!("{}::{}", base_name, index),
        }
    }

    pub fn hash(&self) -> &str {
        &self.hash
    }

    pub fn data(&self) -> &str {
        self.code_span.data()
    }

    pub fn file_path(&self) -> &str {
        self.code_span.file_path()
    }

    pub fn get_code_span(self) -> CodeSpan {
        self.code_span
    }

    pub fn get_span_identifier(&self) -> String {
        format!(
            "// FILEPATH: {}:{}-{}",
            self.file_path(),
            self.code_span.start_line(),
            self.code_span.end_line()
        )
    }
}

impl CodeSpan {
    pub fn new(file_path: String, start_line: u64, end_line: u64, data: String) -> Self {
        Self {
            file_path,
            start_line,
            end_line,
            data,
        }
    }

    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    pub fn start_line(&self) -> u64 {
        self.start_line
    }

    pub fn end_line(&self) -> u64 {
        self.end_line
    }

    pub fn data(&self) -> &str {
        &self.data
    }

    pub fn to_digests(code_spans: Vec<Self>) -> Vec<CodeSpanDigest> {
        // Naming the digests should happen using the filepath and creating a
        // numbered alias on top of it.
        let mut file_paths_counter: HashMap<String, usize> = Default::default();
        code_spans
            .into_iter()
            .map(|code_span| {
                let file_path = code_span.file_path().to_owned();
                let mut index = 0;
                if let Some(value) = file_paths_counter.get_mut(&file_path) {
                    *value += 1;
                    index = *value;
                } else {
                    file_paths_counter.insert(file_path.to_string(), 0);
                }
                CodeSpanDigest::new(code_span, &file_path, index)
            })
            .collect()
    }
}

/// We support both listwise and pairwise reranking strategies
/// Going further we will add more strategies to this, right now
/// these are the best ones
/// list wise reading material here: https://arxiv.org/pdf/2312.02724.pdf
/// point wise reading material here: https://cookbook.openai.com/examples/search_reranking_with_cross-encoders
#[derive(Clone)]
pub enum ReRankStrategy {
    ListWise,
    // This works best with logits enabled, if logits are not provied by the
    // underlying infra, then this is not that great tbh
    PointWise,
}

pub struct ReRankCodeSpanRequest {
    user_query: String,
    answer_snippets: usize,
    answer_limit_tokens: i64,
    code_spans: Vec<CodeSpan>,
    strategy: ReRankStrategy,
    llm_type: LLMType,
}

impl ReRankCodeSpanRequest {
    pub fn new(
        user_query: String,
        answer_snippets: usize,
        answer_limit_tokens: i64,
        code_spans: Vec<CodeSpan>,
        strategy: ReRankStrategy,
        llm_type: LLMType,
    ) -> Self {
        Self {
            user_query,
            answer_snippets,
            answer_limit_tokens,
            code_spans,
            strategy,
            llm_type,
        }
    }

    pub fn user_query(&self) -> &str {
        &self.user_query
    }

    pub fn limit(&self) -> usize {
        self.answer_snippets
    }

    pub fn token_limit(&self) -> i64 {
        self.answer_limit_tokens
    }

    pub fn strategy(&self) -> &ReRankStrategy {
        &self.strategy
    }

    pub fn code_spans(&self) -> &[CodeSpan] {
        self.code_spans.as_slice()
    }

    pub fn llm_type(&self) -> &LLMType {
        &self.llm_type
    }
}

pub struct ReRankListWiseResponse {
    pub prompt: Either<LLMClientCompletionRequest, LLMClientCompletionStringRequest>,
    pub code_span_digests: Vec<CodeSpanDigest>,
}

pub struct ReRankPointWisePrompt {
    pub prompt: Either<LLMClientCompletionRequest, LLMClientCompletionStringRequest>,
    pub code_span_digest: CodeSpanDigest,
}

impl ReRankPointWisePrompt {
    pub fn new_message_request(
        prompt: LLMClientCompletionRequest,
        code_span_digest: CodeSpanDigest,
    ) -> Self {
        Self {
            prompt: Either::Left(prompt),
            code_span_digest,
        }
    }

    pub fn new_string_completion(
        prompt: LLMClientCompletionStringRequest,
        code_span_digest: CodeSpanDigest,
    ) -> Self {
        Self {
            prompt: Either::Right(prompt),
            code_span_digest,
        }
    }
}

pub enum ReRankCodeSpanResponse {
    ListWise(ReRankListWiseResponse),
    PointWise(Vec<ReRankPointWisePrompt>),
}

impl ReRankCodeSpanResponse {
    pub fn listwise_message(
        request: LLMClientCompletionRequest,
        code_span_digests: Vec<CodeSpanDigest>,
    ) -> Self {
        Self::ListWise(ReRankListWiseResponse {
            prompt: Either::Left(request),
            code_span_digests,
        })
    }

    pub fn listwise_completion(
        request: LLMClientCompletionStringRequest,
        code_span_digests: Vec<CodeSpanDigest>,
    ) -> Self {
        Self::ListWise(ReRankListWiseResponse {
            prompt: Either::Right(request),
            code_span_digests,
        })
    }

    pub fn pointwise(prompts: Vec<ReRankPointWisePrompt>) -> Self {
        Self::PointWise(prompts)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ReRankCodeSpanError {
    #[error("Model not found")]
    ModelNotFound,

    #[error("tokenizer errors: {0}")]
    TokenizerError(#[from] LLMTokenizerError),

    #[error("Wrong rerank strategy returned")]
    WrongReRankStrategy,

    #[error("LLMClientError: {0}")]
    LLMClientError(#[from] LLMClientError),
}

/// The rerank code span will take in a list of code spans and generate a prompt
/// for it, but I do think reranking by itself is pretty interesting, should we
/// make it its own trait?
pub trait ReRankCodeSpan {
    fn rerank_prompt(
        &self,
        request: ReRankCodeSpanRequest,
    ) -> Result<ReRankCodeSpanResponse, ReRankCodeSpanError>;

    fn parse_listwise_output(
        &self,
        llm_output: String,
        rerank_request: ReRankListWiseResponse,
    ) -> Result<Vec<CodeSpanDigest>, ReRankCodeSpanError>;
}
