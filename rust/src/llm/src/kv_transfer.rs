use serde_json::Value;

/// Whether a `return_inline` value requests connector inline return, mirroring
/// Python truthiness (`if not kv_transfer_params.get("return_inline")`).
///
/// Empty arrays and objects are falsy in Python, so they are treated as "not
/// requested" here as well.
pub fn requests_inline_kv_return(return_inline: Option<&Value>) -> bool {
    match return_inline {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64() != Some(0.0),
        Some(Value::String(string)) => !string.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(map)) => !map.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn truthy_values_request_inline_return() {
        assert!(requests_inline_kv_return(Some(&json!(true))));
        assert!(requests_inline_kv_return(Some(&json!(1))));
        assert!(requests_inline_kv_return(Some(&json!("yes"))));
        assert!(requests_inline_kv_return(Some(&json!([0]))));
        assert!(requests_inline_kv_return(Some(&json!({"a": 1}))));
    }

    #[test]
    fn falsy_values_do_not_request_inline_return() {
        assert!(!requests_inline_kv_return(None));
        assert!(!requests_inline_kv_return(Some(&json!(null))));
        assert!(!requests_inline_kv_return(Some(&json!(false))));
        assert!(!requests_inline_kv_return(Some(&json!(0))));
        assert!(!requests_inline_kv_return(Some(&json!(""))));
    }

    #[test]
    fn empty_containers_match_python_falsiness() {
        // Empty arrays/objects are falsy in Python; the Rust check must agree.
        assert!(!requests_inline_kv_return(Some(&json!([]))));
        assert!(!requests_inline_kv_return(Some(&json!({}))));
    }
}
