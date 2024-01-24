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
    let prompt = r#"<s>[INST] You are an expert at ordering the code snippets from the most relevant to the least relevant for the user query. You have the order the list of code snippets from the most relevant to the least relevant. As an example
<code_snippets>
<id>
subtract.rs::0
</id>
<snippet>
```
fn subtract(a: i32, b: i32) -> i32 {
    a - b
}
```
</snippet>

<id>
add.rs::0
</id>
<snippet>
```
fn add(a: i32, b: i32) -> i32 {
    a + b
}
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
User: [#file:client.rs:1-635](values:file:client.rs:1-635) where do we initialize the language server?
</user_query>

The code snippets along with their ids are given below:
<code_snippets>
<id>
client.rs::4
</id>
<snippet>
```
pub struct LanguageServerRef<W: AsyncWriteExt>(Arc<Mutex<LanguageServer<W>>>);

//FIXME: this is hacky, and prevents good error propogation,
fn number_from_id(id: Option<&Value>) -> usize {
    let id = id.expect("response missing id field");
    let id = match id {
        &Value::Number(ref n) => n.as_u64().expect("failed to take id as u64"),
        &Value::String(ref s) => {
            u64::from_str_radix(s, 10).expect("failed to convert string id to u64")
        }
        other => panic!("unexpected value for id field: {:?}", other),
    };

    id as usize
}

fn fetch_ts_files_recursively(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
```
</snippet>

<id>
client.rs::0
</id>
<snippet>
```
use anyhow::Result;
use lsp_types::{
    ClientCapabilities, CodeActionClientCapabilities, CodeLensClientCapabilities,
    DynamicRegistrationClientCapabilities, ExecuteCommandClientCapabilities, GotoCapability,
    GotoDefinitionParams, GotoDefinitionResponse, InitializeParams, ReferenceParams,
    RenameClientCapabilities, SignatureHelpClientCapabilities, TextDocumentClientCapabilities,
    WorkDoneProgressParams, WorkspaceClientCapabilities, WorkspaceFolder,
};
use std::str::FromStr;
use tokio::fs;
use tokio::task::JoinHandle;
use tracing::info;

use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::oneshot;
use url::Url;

use serde_json::value::Value;
use serde_json::{self, json};

use jsonrpc_lite::{Error, Id, JsonRpc};

use super::parsing;

// this to get around some type system pain related to callbacks. See:
// https://doc.rust-lang.org/beta/book/trait-objects.html,
// http://stackoverflow.com/questions/41081240/idiomatic-callbacks-in-rust
trait Callable: Send {
    fn call(self: Box<Self>, result: Result<Value, Value>);
}

impl<F: Send + FnOnce(Result<Value, Value>)> Callable for F {
    fn call(self: Box<F>, result: Result<Value, Value>) {
        (*self)(result)
    }
}

type Callback = Box<dyn Callable>;

/// Represents (and mediates communcation with) a Language Server.
///
```
</snippet>

<id>
client.rs::1
</id>
<snippet>
```
/// LanguageServer should only ever be instantiated or accessed through an instance of
/// LanguageServerRef, which mediates access to a single shared LanguageServer through a Mutex.
struct LanguageServer<W: AsyncWriteExt> {
    peer: W,
    pending: HashMap<usize, Callback>,
    next_id: usize,
}

/// Generates a Language Server Protocol compliant message.
fn prepare_lsp_json(msg: &Value) -> Result<String, serde_json::error::Error> {
    let request = serde_json::to_string(&msg)?;
    Ok(format!(
        "Content-Length: {}\r\n\r\n{}",
        request.len(),
        request
    ))
}

impl<W: AsyncWriteExt + Unpin> LanguageServer<W> {
```
</snippet>

<id>
client.rs::2
</id>
<snippet>
```
    async fn write(&mut self, msg: &str) {
        self.peer
            .write_all(msg.as_bytes())
            .await
            .expect("error writing to stdin");
        self.peer.flush().await.expect("error flushing child stdin");
    }

    async fn send_request(&mut self, method: &str, params: &Value, completion: Callback) {
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": method,
            "params": params
        });

        self.pending.insert(self.next_id, completion);
        self.next_id += 1;
        self.send_rpc(&request).await;
    }

    async fn send_notification(&mut self, method: &str, params: &Value) {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        self.send_rpc(&notification).await;
    }

    fn handle_response(&mut self, id: usize, result: Value) {
        let callback = self
            .pending
            .remove(&id)
            .expect(&format!("id {} missing from request table", id));
        callback.call(Ok(result));
    }

    fn handle_error(&mut self, id: usize, error: Error) {
        let callback = self
            .pending
            .remove(&id)
            .expect(&format!("id {} missing from request table", id));
        callback.call(Err(error.data.unwrap_or(serde_json::Value::Null)));
    }

    async fn send_rpc(&mut self, rpc: &Value) {
        let rpc = match prepare_lsp_json(&rpc) {
```
</snippet>

<id>
client.rs::3
</id>
<snippet>
```
            Ok(r) => r,
            Err(err) => panic!("error encoding rpc {:?}", err),
        };
        self.write(&rpc).await;
    }
}

/// Access control and convenience wrapper around a shared LanguageServer instance.
```
</snippet>

<id>
client.rs::5
</id>
<snippet>
```
    // If the path starts with `file://` then we need to remove it. Make sure the 3rd slash is not removed.
    // Example: `file:///Users/nareshr/github/codestory/ide/src` -> `/Users/nareshr/github/codestory/ide/src`
    // Don't use `strip_prefix` because it removes the 3rd slash.
    // let dir = dir.to_str().unwrap().replace("file://", "");

    match std::fs::read_dir(dir) {
        Ok(entries) => {
            println!("Successfully read directory: {:?}", dir);
            for entry in entries {
                match entry {
                    Ok(entry) => {
                        let path = entry.path();
                        if path.is_dir() {
                            match fetch_ts_files_recursively(&path, files) {
                                Ok(_) => (),
                                Err(e) => eprintln!("Error reading directory: {:?}", e),
                            }
                        } else if path.extension() == Some(OsStr::new("ts")) {
                            files.push(path.clone());
                            println!("Added file: {:?}", path);
                        }
                    }
                    Err(e) => eprintln!("Error reading entry: {:?}", e),
                }
            }
        }
        Err(e) => eprintln!("Error reading directory: {:?}", e),
    }
    Ok(())
}

impl<W: AsyncWriteExt + Unpin> LanguageServerRef<W> {
    fn new(peer: W) -> Self {
        LanguageServerRef(Arc::new(Mutex::new(LanguageServer {
            peer: peer,
```
</snippet>

<id>
client.rs::6
</id>
<snippet>
```
            pending: HashMap::new(),
            next_id: 1,
        })))
    }

    fn handle_msg(&self, val: &str) {
        let parsed_value = JsonRpc::parse(val);
        if let Err(err) = parsed_value {
            println!("error parsing json: {:?}", err);
            return;
        }
        let parsed_value = parsed_value.expect("to be present");
        let id = parsed_value.get_id();
        let response = parsed_value.get_result();
        let error = parsed_value.get_error();
        match (id, response, error) {
            (Some(Id::Num(id)), Some(response), None) => {
                let mut inner = self.0.lock().unwrap();
                inner.handle_response(id.try_into().unwrap(), response.clone());
            }
            (Some(Id::Num(id)), None, Some(error)) => {
                let mut inner = self.0.lock().unwrap();
                inner.handle_error(id.try_into().unwrap(), error.clone());
            }
            (Some(Id::Num(id)), Some(response), Some(error)) => {
                panic!("We got both response and error.. what even??");
            }
            _ => {}
        }
    }

    /// Sends a JSON-RPC request message with the provided method and parameters.
    /// `completion` should be a callback which will be executed with the server's response.
    pub async fn send_request<CB>(&self, method: &str, params: &Value, completion: CB)
    where
        CB: 'static + Send + FnOnce(Result<Value, Value>),
    {
        let mut inner = self.0.lock().unwrap();
```
</snippet>

<id>
client.rs::7
</id>
<snippet>
```
        inner
            .send_request(method, params, Box::new(completion))
            .await;
    }

    /// Sends a JSON-RPC notification message with the provided method and parameters.
    pub async fn send_notification(&self, method: &str, params: &Value) {
        let mut inner = self.0.lock().unwrap();
        inner.send_notification(method, params).await;
    }

    pub async fn initialize(&self, working_directory: &PathBuf) {
        info!(
            event_name = "initialize_lsp",
            event_type = "start",
            working_directory = ?working_directory
        );

```
</snippet>

<id>
client.rs::8
</id>
<snippet>
```
        let working_directory_path =
            format!("file://{}", working_directory.to_str().expect("to work"));
        let start = std::time::Instant::now();

        let init_params = InitializeParams {
            process_id: None, // Super important to set it to NONE https://github.com/typescript-language-server/typescript-language-server/issues/262
```
</snippet>

<id>
client.rs::9
</id>
<snippet>
```
            root_uri: Some(Url::parse(&working_directory_path).unwrap()),
            root_path: None,
            initialization_options: Some(serde_json::json!({
                "hostInfo": "vscode",
                "maxTsServerMemory": 4096 * 2,
                "tsserver": {
                    "logDirectory": "/tmp/tsserver",
                    "logVerbosity": "verbose",
                    "maxTsServerMemory": 4096 * 2,
                    // sending the same path as the vscode extension
                    // "path": "/Users/skcd/.aide/extensions/ms-vscode.vscode-typescript-next-5.3.20231102/node_modules/typescript/lib/tsserver.js"
                },
                "preferences": {
                    "providePrefixAndSuffixTextForRename": true,
                    "allowRenameOfImportPath": true,
                    "includePackageJsonAutoImports": "auto",
                    "excludeLibrarySymbolsInNavTo": true
                }
            })),
            capabilities: ClientCapabilities {
                text_document: Some(TextDocumentClientCapabilities {
                    declaration: Some(GotoCapability {
                        dynamic_registration: Some(true),
```
</snippet>
</code_snippets>

As a reminder the user question is: [#file:client.rs:1-635](values:file:client.rs:1-635) where do we initialize the language server?
You have to order all the code snippets from the most relevant to the least relevant to the user query, all the code snippet ids should be present in your final reordered list. Only output the ids of the code snippets.
[/INST]<ranking>
<id>
"#;
    let llm_broker = LLMBroker::new(LLMBrokerConfiguration::new(PathBuf::from(
        "/Users/skcd/Library/Application Support/ai.codestory.sidecar",
    )))
    .await
    .expect("broker to startup");

    let api_key = LLMProviderAPIKeys::TogetherAI(TogetherAIProvider::new(
        "cc10d6774e67efef2004b85efdb81a3c9ba0b7682cc33d59c30834183502208d".to_owned(),
    ));
    let request = LLMClientCompletionStringRequest::new(
        LLMType::MistralInstruct,
        prompt.to_owned(),
        0.9,
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
