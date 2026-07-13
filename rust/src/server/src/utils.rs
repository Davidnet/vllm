use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::HeaderMap;
use serde_json::Value;
use thiserror_ext::AsReport;
use uuid::Uuid;
use vllm_llm::requests_inline_kv_return;

use crate::error::{ApiError, bail_invalid_request};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResolvedRequestContext {
    pub request_id: String,
    pub data_parallel_rank: Option<u32>,
}

/// Return the current Unix timestamp in seconds for OpenAI response objects.
pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

/// Construct an API error for a failed utility call to the engine core.
pub fn utility_call_error(method: &str, error: impl AsReport) -> ApiError {
    ApiError::server_error(format!("failed to call {method}: {}", error.as_report()))
}

/// Merge `kv_transfer_params` into the `vllm_xargs` map, mirroring the Python
/// vLLM behavior where `kv_transfer_params` is injected into `extra_args` for
/// engine-core consumption.
pub fn merge_kv_transfer_params(
    mut xargs: Option<HashMap<String, Value>>,
    kv_transfer_params: Option<&HashMap<String, Value>>,
) -> Option<HashMap<String, Value>> {
    if let Some(kv_params) = kv_transfer_params {
        let map = xargs.get_or_insert_with(HashMap::new);
        map.insert(
            "kv_transfer_params".to_string(),
            // This is safe because we know that `kv_params` is already valid JSON.
            serde_json::to_value(kv_params).unwrap(),
        );
    }
    xargs
}

/// Reject connector inline return (`return_inline`) requests: the Rust frontend
/// forwards `kv_transfer_params` but does not decode the out-of-band inline
/// tensor frames, so accepting one would silently drop the returned tensors.
///
/// `kv_transfer_params` may also be smuggled in nested under `vllm_xargs`. This
/// mirrors [`merge_kv_transfer_params`]: the top-level field wins when present;
/// otherwise a value nested in `vllm_xargs` is the effective payload forwarded
/// to EngineCore. Validate whichever one is effective so the rejection cannot
/// be bypassed via `vllm_xargs`.
pub fn reject_inline_kv_return(
    kv_transfer_params: Option<&HashMap<String, Value>>,
    vllm_xargs: Option<&HashMap<String, Value>>,
) -> Result<(), ApiError> {
    let return_inline = match kv_transfer_params {
        Some(kv) => kv.get("return_inline"),
        None => vllm_xargs
            .and_then(|xargs| xargs.get("kv_transfer_params"))
            .and_then(|kv| kv.get("return_inline")),
    };
    if requests_inline_kv_return(return_inline) {
        bail_invalid_request!(
            param = "kv_transfer_params",
            "Connector inline return (`return_inline`) is not supported by the \
             Rust frontend; use the Python frontend."
        );
    }
    Ok(())
}

/// Convert OpenAI-style `logit_bias` with string token-ID keys into the
/// internal `HashMap<u32, f32>` representation, validating that every key
/// parses as a `u32`.
pub fn convert_logit_bias(
    logit_bias: Option<HashMap<String, f32>>,
) -> Result<Option<HashMap<u32, f32>>, ApiError> {
    logit_bias
        .map(|bias| {
            bias.into_iter()
                .map(|(key, value)| {
                    key.parse().map(|k| (k, value)).map_err(|_| {
                        ApiError::invalid_request(
                            format!(
                                "Invalid key in 'logit_bias': '{key}' is not a valid token ID. \
                                 Token IDs must be non-negative integers."
                            ),
                            Some("logit_bias"),
                        )
                    })
                })
                .collect()
        })
        .transpose()
}

/// Extract common request metadata from HTTP headers: the external request ID
/// and the optional data-parallel rank used for engine routing.
pub fn resolve_request_context(
    headers: &HeaderMap,
    request_id: Option<&str>,
) -> ResolvedRequestContext {
    // `None` when the header is absent or cannot be parsed as a `u32`.
    let data_parallel_rank = headers
        .get("X-data-parallel-rank")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse().ok());

    // Extract request id from header.
    let request_id_header = headers.get("X-Request-Id").and_then(|value| value.to_str().ok());
    let request_id = resolve_base_request_id(request_id_header, request_id);

    ResolvedRequestContext {
        request_id,
        data_parallel_rank,
    }
}

/// Resolve the base external request ID before API-specific prefixes such as
/// `chatcmpl-`.
pub fn resolve_base_request_id(
    request_id_header: Option<&str>,
    request_id: Option<&str>,
) -> String {
    request_id_header.or(request_id).map(ToOwned::to_owned).unwrap_or_else(|| {
        let mut id = Uuid::new_v4().simple().to_string();
        id.truncate(8);
        id
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn kv(params: Value) -> HashMap<String, Value> {
        match params {
            Value::Object(map) => map.into_iter().collect(),
            other => panic!("expected object, got {other:?}"),
        }
    }

    #[test]
    fn reject_inline_kv_return_rejects_truthy_return_inline() {
        assert!(reject_inline_kv_return(Some(&kv(json!({"return_inline": true}))), None).is_err());
    }

    #[test]
    fn reject_inline_kv_return_rejects_nested_vllm_xargs() {
        // return_inline smuggled inside vllm_xargs, with no top-level field.
        let xargs = kv(json!({"kv_transfer_params": {"return_inline": true}}));
        assert!(reject_inline_kv_return(None, Some(&xargs)).is_err());
    }

    #[test]
    fn reject_inline_kv_return_top_level_wins_over_nested() {
        // Top-level kv_transfer_params overrides the nested one during merge, so
        // a benign top-level value makes the effective payload benign.
        let top = kv(json!({}));
        let xargs = kv(json!({"kv_transfer_params": {"return_inline": true}}));
        assert!(reject_inline_kv_return(Some(&top), Some(&xargs)).is_ok());
    }

    #[test]
    fn reject_inline_kv_return_allows_absent_or_falsy() {
        assert!(reject_inline_kv_return(None, None).is_ok());
        assert!(reject_inline_kv_return(Some(&kv(json!({}))), None).is_ok());
        assert!(reject_inline_kv_return(Some(&kv(json!({"return_inline": false}))), None).is_ok());
        assert!(reject_inline_kv_return(Some(&kv(json!({"other": "value"}))), None).is_ok());
        let xargs = kv(json!({"kv_transfer_params": {"return_inline": false}}));
        assert!(reject_inline_kv_return(None, Some(&xargs)).is_ok());
    }
}
