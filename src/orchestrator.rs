//! The routing orchestrator. A single, lightweight DeepSeek call decides — from the
//! contract *skeleton* only (signatures + sizes + declarations, NO bodies) — whether
//! to optimize the whole contract in one shot or to decompose it into scoped
//! semantic tasks.
//!
//! The decision is delivered via native tool-calling (`submit_plan`), never raw JSON
//! holding code: the only data crossing the wire is short routing metadata (a mode
//! string and, for decompose, task names + function names). We capture the tool-call
//! arguments in a hook and terminate the loop immediately (same trick as the forge
//! early-exit) — the tool body never has to run.

use std::sync::{Arc, Mutex};

use rig_core::agent::{PromptHook, ToolCallHookAction};
use rig_core::client::CompletionClient;
use rig_core::completion::{CompletionModel, Prompt, ToolDefinition};
use rig_core::providers::deepseek;
use rig_core::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use tracing::info;

const ROUTER_PROMPT: &str = "You are the routing planner for a Solidity gas optimizer.\n\
    You are given only a STRUCTURAL SKELETON of a contract (function signatures, body sizes, state \
    variables, and declarations) — never the function bodies.\n\
    \n\
    Decide how to optimize it and call the `submit_plan` tool with your decision:\n\
    - mode = \"oneshot\": optimize the whole contract in a single pass. PREFER THIS for small or \
      simple contracts (few functions, small bodies). It is the fast path.\n\
    - mode = \"decompose\": only for large/complex contracts. Provide `tasks`, each grouping related \
      functions (by exact name from the skeleton) into one unit of work.\n\
    \n\
    When in doubt, choose oneshot. Always call submit_plan exactly once; do not write prose.";

/// The routing decision parsed from the `submit_plan` tool call.
#[derive(Debug)]
pub enum Route {
    Oneshot,
    Decompose(Vec<Task>),
}

/// A scoped unit of work for the decompose path.
#[derive(Debug, Clone, Deserialize)]
pub struct Task {
    #[allow(dead_code)]
    pub title: String,
    /// Exact function names (from the skeleton) this task owns.
    #[serde(default)]
    pub target_fns: Vec<String>,
    /// Optional pattern-id hints to seed retrieval. Reserved: parsed from the router
    /// but not yet threaded into per-function retrieval (fan-out retrieves per
    /// function from its own source). Kept so the router contract stays stable.
    #[serde(default)]
    #[allow(dead_code)]
    pub pattern_hints: Vec<String>,
}

/// Raw shape of the `submit_plan` arguments. Parsed by us from the captured arg
/// string, so the tool body never runs.
#[derive(Deserialize)]
struct PlanArgs {
    #[serde(default)]
    mode: String,
    #[serde(default)]
    tasks: Vec<Task>,
}

/// Route a contract from its skeleton. On any failure (no tool call, unparseable
/// args), returns `Err` so the caller can fall back to full per-function fan-out.
pub async fn route(client: &deepseek::Client, skeleton: &str) -> Result<Route, String> {
    let slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let hook = CaptureHook { slot: slot.clone() };

    // The loop terminates inside the hook the moment the tool is called, so the
    // result is a cancellation error — the decision lives in `slot`.
    let _ = client
        .agent(deepseek::DEEPSEEK_V4_FLASH)
        .preamble(ROUTER_PROMPT)
        .context(skeleton)
        .temperature(0.0)
        .max_tokens(1024)
        .tool(SubmitPlanTool)
        .build()
        .prompt("Analyze the contract skeleton and call submit_plan with your routing decision.")
        .with_hook(hook)
        .max_turns(2)
        .await;

    let args = slot
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| "router did not call submit_plan".to_string())?;

    let parsed: PlanArgs =
        serde_json::from_str(&args).map_err(|e| format!("router args parse failed: {e}"))?;

    if parsed.mode.eq_ignore_ascii_case("decompose") && !parsed.tasks.is_empty() {
        info!("  router: decompose ({} task(s))", parsed.tasks.len());
        Ok(Route::Decompose(parsed.tasks))
    } else {
        info!("  router: oneshot");
        Ok(Route::Oneshot)
    }
}

// ── hook: capture the tool-call args, then terminate ────────────────────────────
#[derive(Clone)]
struct CaptureHook {
    slot: Arc<Mutex<Option<String>>>,
}

impl<M: CompletionModel> PromptHook<M> for CaptureHook {
    async fn on_tool_call(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
    ) -> ToolCallHookAction {
        *self.slot.lock().unwrap() = Some(args.to_string());
        ToolCallHookAction::terminate("plan captured")
    }
}

// ── the routing tool (offered to the model; body never executes) ────────────────
#[derive(Debug, thiserror::Error)]
#[error("submit_plan error: {0}")]
pub struct PlanError(String);

struct SubmitPlanTool;

impl Tool for SubmitPlanTool {
    const NAME: &'static str = "submit_plan";

    type Error = PlanError;
    type Args = PlanArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Submit the optimization routing decision for this contract.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["oneshot", "decompose"],
                        "description": "oneshot = optimize whole contract in one pass (prefer for small contracts); decompose = split into tasks (only for large/complex contracts)"
                    },
                    "tasks": {
                        "type": "array",
                        "description": "Required only when mode=decompose. Each task groups related functions.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": { "type": "string", "description": "Short task label" },
                                "target_fns": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "Exact function names (from the skeleton) this task owns"
                                },
                                "pattern_hints": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "Optional pattern-id hints"
                                }
                            },
                            "required": ["title", "target_fns"]
                        }
                    }
                },
                "required": ["mode"]
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Unreachable in practice: the hook terminates the loop before the tool body
        // executes. Present only so the model is offered the tool.
        Ok("ok".to_string())
    }
}
