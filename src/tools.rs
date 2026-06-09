//! `FunctionForgeTool` — a rig `Tool` the per-function agent calls to verify a
//! single optimized function. It splices the candidate function into the
//! original contract at the function's byte range and compiles the result, so
//! the agent can refine non-compiling YUL in its loop. Verification is
//! compile-level (per-function runtime gas via call-tests comes later).

use std::sync::Arc;

use rig_core::completion::ToolDefinition;
use rig_core::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::forge::run_forge_sandbox_async;
use crate::utils::strip_code_fences;

#[derive(Deserialize)]
pub struct FnForgeArgs {
    /// The complete optimized function (signature + body), no contract wrapper.
    pub optimized_function: String,
}

#[derive(Serialize)]
pub struct ForgeResult {
    pub compiles: bool,
    /// True only when it compiled AND a construction-gas number was parsed.
    /// NOTE: not a behavioural-equivalence check, and the gas is whole-contract
    /// *construction* gas — not this function's runtime gas.
    pub gas_measured: bool,
    pub gas_original: Option<u64>,
    pub gas_optimized: Option<u64>,
    pub gas_saved: Option<i64>,
    pub errors: Vec<String>,
    /// Truncated forge stdout/stderr so it fits the model's context window.
    pub forge_excerpt: String,
}

#[derive(Debug, thiserror::Error)]
#[error("forge tool error: {0}")]
pub struct ForgeError(String);

/// Verifies one optimized function by splicing it into the original contract.
pub struct FunctionForgeTool {
    /// The full original contract source.
    pub original: Arc<str>,
    /// Byte range of the target function within `original`.
    pub start: usize,
    pub end: usize,
}

impl FunctionForgeTool {
    pub fn new(original: Arc<str>, start: usize, end: usize) -> Self {
        Self { original, start, end }
    }
}

impl Tool for FunctionForgeTool {
    const NAME: &'static str = "forge_verify";

    type Error = ForgeError;
    type Args = FnForgeArgs;
    type Output = ForgeResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Compile-check your optimized function. Pass ONLY the complete optimized \
                          function (signature + body); it is spliced into the original contract and \
                          compiled on a Mantle fork. Returns whether it compiles and any compiler \
                          errors. If it does not compile, fix the function using the errors and call \
                          again."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "optimized_function": {
                        "type": "string",
                        "description": "The complete optimized function, no contract wrapper"
                    }
                },
                "required": ["optimized_function"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let optimized_fn = strip_code_fences(&args.optimized_function);

        // Splice the candidate function into the original contract.
        let mut spliced = self.original.to_string();
        if self.end <= spliced.len() {
            spliced.replace_range(self.start..self.end, &optimized_fn);
        } else {
            return Err(ForgeError("function byte range out of bounds".to_string()));
        }

        let res = run_forge_sandbox_async(self.original.to_string(), spliced)
            .await
            .map_err(ForgeError)?;

        let gas_measured = res.compiles && res.gas_optimized.is_some();
        let forge_excerpt: String = res.forge_output.chars().take(2000).collect();

        Ok(ForgeResult {
            compiles: res.compiles,
            gas_measured,
            gas_original: res.gas_original,
            gas_optimized: res.gas_optimized,
            gas_saved: res.gas_saved,
            errors: res.errors,
            forge_excerpt,
        })
    }
}
