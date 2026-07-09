//! Request/response payloads for the HTTP API.

use serde::{Deserialize, Serialize};

use crate::verify::forge;

#[derive(Deserialize)]
pub(crate) struct OptimizeRequest {
    pub(crate) contract_source: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct OptimizeResponse {
    pub(crate) analysis: String,
    pub(crate) suggested_patterns: Vec<String>,
    pub(crate) optimized_code: String,
    /// Construction gas of the original contract (set only when forge verified).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) gas_before: Option<u64>,
    /// Construction gas of the optimized contract (set only when forge verified).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) gas_after: Option<u64>,
    /// Gas saved (original − optimized). Positive = improvement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) gas_saved: Option<i64>,
    /// Real per-function runtime gas (original vs optimized), from forge's gas
    /// report. Empty when forge is unavailable or no rewrite was verified.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(crate) per_function_gas: Vec<forge::FunctionGas>,
}

#[derive(Deserialize)]
pub(crate) struct IngestLocalRequest {
    pub(crate) directory_paths: Vec<String>,
}

#[derive(Serialize)]
pub(crate) struct IngestLocalResponse {
    pub(crate) successful_patterns: Vec<String>,
    pub(crate) failed_patterns: Vec<(String, String)>,
}
