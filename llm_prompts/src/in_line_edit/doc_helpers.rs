use super::types::InLineDocRequest;

pub fn documentation_type(identifier_node: &InLineDocRequest) -> String {
    let language = identifier_node.language();
    let is_identifier = identifier_node.is_identifier_node();
    let comment_type = match language {
        "typescript" | "typescriptreact" => match is_identifier {
            true => "a TSDoc comment".to_owned(),
            false => "TSDoc comment".to_owned(),
        },
        "javascript" | "javascriptreact" => match is_identifier {
            true => "a JSDoc comment".to_owned(),
            false => "JSDoc comment".to_owned(),
        },
        "python" => "docstring".to_owned(),
        "rust" => "Rustdoc comment".to_owned(),
        _ => "documentation comment".to_owned(),
    };
    comment_type
}

pub fn selection_type(identifier_node: &InLineDocRequest) -> String {
    let identifier_node_str = identifier_node.identifier_node_str();
    match identifier_node_str {
        Some(identifier_node) => identifier_node.to_owned(),
        None => "the selection".to_owned(),
    }
}

pub fn document_symbol_metadata(identifier_node: &InLineDocRequest) -> String {
    let is_identifier = identifier_node.is_identifier_node();
    let language = identifier_node.language();
    let comment_type = documentation_type(identifier_node);
    let identifier_node_str = identifier_node.identifier_node_str();
    match identifier_node_str {
        Some(identifier_node) => {
            format!("Please add {comment_type} for {identifier_node}")
        }
        None => format!("Please add {comment_type} for the selection"),
    }
}
