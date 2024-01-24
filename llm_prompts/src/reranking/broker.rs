use std::{
    cmp::{max, min},
    collections::HashMap,
    sync::Arc,
};

use futures::stream;
use futures::StreamExt;
use llm_client::{
    broker::LLMBroker,
    clients::types::LLMType,
    provider::{LLMProvider, LLMProviderAPIKeys},
    tokenizer::tokenizer::LLMTokenizer,
};

use super::{
    mistral::MistralReRank,
    openai::OpenAIReRank,
    types::{
        CodeSpan, CodeSpanDigest, ReRankCodeSpan, ReRankCodeSpanError, ReRankCodeSpanRequest,
        ReRankCodeSpanResponse, ReRankListWiseResponse, ReRankStrategy,
    },
};

const SLIDING_WINDOW: i64 = 10;
const TOP_K: i64 = 5;

pub struct ReRankBroker {
    rerankers: HashMap<LLMType, Box<dyn ReRankCodeSpan + Send + Sync>>,
}

impl ReRankBroker {
    pub fn new() -> Self {
        let mut rerankers: HashMap<LLMType, Box<dyn ReRankCodeSpan + Send + Sync>> = HashMap::new();
        rerankers.insert(LLMType::GPT3_5_16k, Box::new(OpenAIReRank::new()));
        rerankers.insert(LLMType::Gpt4, Box::new(OpenAIReRank::new()));
        rerankers.insert(LLMType::Gpt4_32k, Box::new(OpenAIReRank::new()));
        rerankers.insert(LLMType::MistralInstruct, Box::new(MistralReRank::new()));
        rerankers.insert(LLMType::Mixtral, Box::new(MistralReRank::new()));
        Self { rerankers }
    }

    pub fn rerank_prompt(
        &self,
        request: ReRankCodeSpanRequest,
    ) -> Result<ReRankCodeSpanResponse, ReRankCodeSpanError> {
        let reranker = self.rerankers.get(&request.llm_type()).unwrap();
        reranker.rerank_prompt(request)
    }

    fn measure_tokens(
        &self,
        llm_type: &LLMType,
        code_digests: &[CodeSpanDigest],
        tokenizer: Arc<LLMTokenizer>,
    ) -> Result<usize, ReRankCodeSpanError> {
        let total_tokens: usize = code_digests
            .into_iter()
            .map(|code_digest| {
                let file_path = code_digest.file_path();
                let data = code_digest.data();
                let prompt = format!(
                    r#"FILEPATH: {file_path}
```
{data}
```"#
                );
                tokenizer.count_tokens_using_tokenizer(llm_type, &prompt)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum();
        Ok(total_tokens)
    }

    fn order_code_digests_listwise(
        &self,
        llm_type: &LLMType,
        response: String,
        rerank_list_request: ReRankListWiseResponse,
    ) -> Result<Vec<CodeSpanDigest>, ReRankCodeSpanError> {
        if let Some(reranker) = self.rerankers.get(llm_type) {
            let mut reranked_code_spans =
                reranker.parse_listwise_output(response, rerank_list_request)?;
            reranked_code_spans.reverse();
            // revers it here since we want the most relevant one to be at the right and not the left
            Ok(reranked_code_spans)
        } else {
            Err(ReRankCodeSpanError::ModelNotFound)
        }
    }

    pub async fn listwise_reranking(
        &self,
        api_keys: LLMProviderAPIKeys,
        request: ReRankCodeSpanRequest,
        provider: LLMProvider,
        client_broker: Arc<LLMBroker>,
        tokenizer: Arc<LLMTokenizer>,
    ) -> Result<Vec<CodeSpan>, ReRankCodeSpanError> {
        // We are given a list of code spans, we are going to do the following:
        // - implement a sliding window algorithm which goes over the snippets
        // and keeps ranking them until we have the list of top k snippets
        let code_spans = request.code_spans().to_vec();
        let mut digests = CodeSpan::to_digests(code_spans);
        // First we check if we need to do a sliding window here by measuring
        // against the token limit we have
        if request.token_limit()
            >= self.measure_tokens(request.llm_type(), &digests, tokenizer)? as i64
        {
            return Ok(digests
                .into_iter()
                .map(|digest| digest.get_code_span())
                .collect());
        }
        let mut end_index: i64 = (min(
            SLIDING_WINDOW,
            digests.len().try_into().expect("conversion to not fail"),
        ) - 1)
            .try_into()
            .expect("conversion to work");
        while end_index < digests.len() as i64 {
            // Now that we are in the window, we have to take the elements from
            // (end_index - SLIDING_WINDOW)::(end_index)
            // and rank them, once we have these ranked
            // we move our window forward by TOP_K and repeat the process
            let llm_type = request.llm_type().clone();
            let index_start: usize = max(end_index - SLIDING_WINDOW, 0).try_into().unwrap();
            let end_index_usize = end_index.try_into().expect("to work");
            let code_spans = digests[index_start..=end_index_usize]
                .iter()
                .map(|digest| digest.clone().get_code_span())
                .collect::<Vec<_>>();
            let request = ReRankCodeSpanRequest::new(
                request.user_query().to_owned(),
                request.limit(),
                request.token_limit(),
                code_spans,
                request.strategy().clone(),
                llm_type.clone(),
            );
            let prompt = self.rerank_prompt(request)?;
            if let ReRankCodeSpanResponse::ListWise(listwise_request) = prompt {
                let prompt = listwise_request.prompt.to_owned();
                let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
                let response = client_broker
                    .stream_answer(
                        api_keys.clone(),
                        provider.clone(),
                        prompt,
                        vec![("event_type".to_owned(), "listwise_reranking".to_owned())]
                            .into_iter()
                            .collect(),
                        sender,
                    )
                    .await?;

                // We have the updated list
                let updated_list =
                    self.order_code_digests_listwise(&llm_type, response, listwise_request)?;
                // Now we will in place replace the code spans from the digests from our start position
                // with the elements in this list
                for (index, code_span_digest) in updated_list.into_iter().enumerate() {
                    let index_i64: i64 = index.try_into().expect("to work");
                    let new_index: usize = (max(end_index - SLIDING_WINDOW, 0) + index_i64)
                        .try_into()
                        .expect("to work");
                    digests[new_index] = code_span_digest;
                }

                // Now move the window forward
                end_index += TOP_K;
            } else {
                return Err(ReRankCodeSpanError::WrongReRankStrategy);
            }
            // let response = client_broker.stream_completion(api_key, request, metadata, sender)
        }

        // At the end of this iteration we have our updated list of answers

        // First reverse the list so its ordered from the most relevant to the least
        digests.reverse();
        // Only take the request.limit() number of answers
        digests.truncate(request.limit());
        // convert back to the code span
        Ok(digests
            .into_iter()
            .map(|digest| digest.get_code_span())
            .collect())
    }

    pub async fn pointwise_reranking(
        &self,
        api_keys: LLMProviderAPIKeys,
        provider: LLMProvider,
        request: ReRankCodeSpanRequest,
        client_broker: Arc<LLMBroker>,
        tokenizer: Arc<LLMTokenizer>,
    ) -> Result<Vec<CodeSpan>, ReRankCodeSpanError> {
        // This approach uses the logits generated for yes and no to get the final
        // answer, since we are not use if we can logits yet on various platforms
        // we assume 1.0 for yes if thats the case or 0.0 for no otherwise
        let code_spans = request.code_spans().to_vec();
        let digests = CodeSpan::to_digests(code_spans);
        let answer_snippets = request.limit();

        // We first measure if we are within the token limit
        if request.token_limit()
            >= self.measure_tokens(request.llm_type(), &digests, tokenizer)? as i64
        {
            return Ok(digests
                .into_iter()
                .map(|digest| digest.get_code_span())
                .collect());
        }

        let request = ReRankCodeSpanRequest::new(
            request.user_query().to_owned(),
            request.limit(),
            request.token_limit(),
            digests
                .into_iter()
                .map(|digest| digest.get_code_span())
                .collect(),
            request.strategy().clone(),
            request.llm_type().clone(),
        );

        let prompt = self.rerank_prompt(request)?;

        if let ReRankCodeSpanResponse::PointWise(pointwise_prompts) = prompt {
            let response_with_code_digests = stream::iter(pointwise_prompts.into_iter())
                .map(|pointwise_prompt| async {
                    let prompt = pointwise_prompt.prompt;
                    let code_digest = pointwise_prompt.code_span_digest;
                    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
                    client_broker
                        .stream_answer(
                            api_keys.clone(),
                            provider.clone(),
                            prompt,
                            vec![("event_type".to_owned(), "pointwise_reranking".to_owned())]
                                .into_iter()
                                .collect(),
                            sender,
                        )
                        .await
                        .map(|response| (response, code_digest))
                })
                .buffer_unordered(25)
                .filter_map(|response| {
                    if let Ok((response, code_digest)) = response {
                        if response.trim().to_lowercase() == "yes" {
                            futures::future::ready(Some(code_digest))
                        } else {
                            futures::future::ready(None)
                        }
                    } else {
                        futures::future::ready(None)
                    }
                })
                .collect::<Vec<_>>()
                .await;
            // Now we only keep the code spans from the start until the length
            // of the limit we have
            let mut response_with_code_digests = response_with_code_digests
                .into_iter()
                .map(|code_digest| code_digest.get_code_span())
                .collect::<Vec<_>>();
            // Only keep until the answer snippets which are limited in this case
            response_with_code_digests.truncate(answer_snippets);
            return Ok(response_with_code_digests);
        } else {
            return Err(ReRankCodeSpanError::WrongReRankStrategy);
        }
    }

    pub async fn rerank(
        &self,
        api_keys: LLMProviderAPIKeys,
        provider: LLMProvider,
        request: ReRankCodeSpanRequest,
        // we need the broker here to get the right client
        client_broker: Arc<LLMBroker>,
        // we need the tokenizer here to count the tokens properly
        tokenizer_broker: Arc<LLMTokenizer>,
    ) -> Result<Vec<CodeSpan>, ReRankCodeSpanError> {
        let strategy = request.strategy();
        match strategy {
            ReRankStrategy::ListWise => {
                self.listwise_reranking(
                    api_keys,
                    request,
                    provider,
                    client_broker,
                    tokenizer_broker,
                )
                .await
            }
            ReRankStrategy::PointWise => {
                // We need to generate the prompt for this
                self.pointwise_reranking(
                    api_keys,
                    provider,
                    request,
                    client_broker,
                    tokenizer_broker,
                )
                .await
            }
        }
    }
}
