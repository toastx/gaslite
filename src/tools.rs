//! `ForgeTool` — a rig `Tool` that compiles and gas-tests an optimized contract
//! against the original in a Foundry sandbox on a Mantle fork. The refinement
//! agent calls this to close the loop: on a compile failure or gas regression it
//! reads the returned errors and tries again.

use rig_core::completion::ToolDefinition;
use rig_core::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::forge::run_forge_sandbox_async;

#[derive(Deserialize)]
pub struct ForgeArgs {
    /// Full original contract source.
    pub original_code: String,
    /// Full optimized contract source.
    pub optimized_code: String,
}

#[derive(Serialize)]
pub struct ForgeResult {
    pub compiles: bool,
    /// True only when the contract compiled AND a construction-gas number was
    /// parsed. NOTE: this is *not* a behavioural-equivalence check — the sandbox
    /// only compiles and measures constructor gas, it does not prove the
    /// optimized contract behaves identically to the original.
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

pub struct ForgeTool;

impl Tool for ForgeTool {
    const NAME: &'static str = "forge_verify";

    type Error = ForgeError;
    type Args = ForgeArgs;
    type Output = ForgeResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Compile and gas-test an optimized Solidity contract against the original \
                          on a Mantle fork. Returns whether it compiles, gas saved (negative = \
                          regression), and any compiler/test errors. Call this after producing an \
                          optimized contract. If it does not compile or gas does not improve, fix \
                          the code using the returned errors and call again."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "original_code":  { "type": "string", "description": "Full original contract source" },
                    "optimized_code": { "type": "string", "description": "Full optimized contract source" }
                },
                "required": ["original_code", "optimized_code"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let res = run_forge_sandbox_async(args.original_code, args.optimized_code)
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
