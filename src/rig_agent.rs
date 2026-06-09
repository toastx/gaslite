//! The rig refinement agent. A single per-contract DeepSeek agent retrieves
//! patterns via `dynamic_context` (our `GasliteIndex`) and — when Foundry is
//! available — closes the loop with `ForgeTool`: generate → forge build/test on
//! a Mantle fork → on compile failure or gas regression, refine and retry,
//! capped by `max_turns`.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rig_core::agent::{HookAction, PromptHook, ToolCallHookAction};
use rig_core::client::CompletionClient;
use rig_core::completion::{CompletionModel, CompletionResponse, Prompt};
use rig_core::message::Message;
use rig_core::providers::deepseek;
use tracing::info;

use crate::retrieval::GasliteIndex;
use crate::tools::FunctionForgeTool;

/// Max agent turns when the forge loop is active (1 generate + refinements).
const FORGE_MAX_TURNS: usize = 4;
/// Number of retrieved pattern documents to inject as dynamic context.
const CONTEXT_SAMPLES: usize = 6;

pub const SYSTEM_PROMPT: &str = "You are Gaslite, a gas optimization engine for Mantle L2 EVM contracts.\n\
    \n\
    Your role is pattern application and adaptation — not pattern invention.\n\
    \n\
    The RETRIEVED PATTERNS are your source of truth for YUL structure, opcodes, \
    and error selectors. Use them as templates:\n\
    - Keep the YUL opcodes, control flow, and error selectors exactly as shown\n\
    - Adapt storage slot variable names and mapping key derivations to match \
      the user contract's actual storage layout shown in STORAGE LAYOUT\n\
    - For standard Solidity mappings, derive slots as: \
      mstore(0x00, key), mstore(0x20, slot_number), keccak256(0x00, 0x40)\n\
    - Replace require(condition, string) with the 4-byte custom error pattern: \
      mstore(0x00, 0xSELECTOR), revert(0x1c, 0x04)\n\
    - Do not invent YUL opcodes, selectors, or patterns not present in the retrieved patterns\n\
    \n\
    CANONICAL ERROR SELECTORS — use only these, never invent selectors:\n\
    InsufficientBalance()   0xf4d678b8\n\
    InsufficientAllowance() 0x13be252b\n\
    TransferFailed()        0x90b8ec18\n\
    ETHTransferFailed()     0xb12d13eb\n\
    NotOwner()              0x30cd7471\n\
    Unauthorized()          0x82b42900\n\
    NotApproved()           0xc19f17a9\n\
    TokenDoesNotExist()     0xceea21b6\n\
    TokenAlreadyExists()    0xc991cbb1\n\
    ExceedsMaxSupply()      0xc30436e9\n\
    ZeroAddress()           0xd92e233d\n\
    InvalidAmount()         0x2c5211c6\n\
    InvalidSignature()      0x8baa579f\n\
    DeadlineExpired()       0x1ab7da6b\n\
    SlippageExceeded()      0x8199f5f3\n\
    AlreadyInitialized()    0x0dc149f0\n\
    NotInitialized()        0x87138d5c\n\
    Reentrancy()            0xab143c06\n\
    Paused()                0x9e87fac8\n\
    InsufficientLiquidity() 0xbb55fd27\n\
    \n\
    EVENT EMISSION RULES:\n\
    - log4(memOffset, memLen, topic0, topic1, topic2, topic3) — topics are inline stack args\n\
    - When ALL event args are indexed: use log4(0, 0, sig, arg1, arg2, arg3) — data payload is ZERO bytes\n\
    - When event has non-indexed args: ABI-encode them in memory, then log(offset, len, topics...)\n\
    - NEVER pass mload() as topic arguments — pass the values directly\n\
    \n\
    Correctness is absolute. An optimization that changes observable behaviour \
    is not an optimization — it is a bug.";

/// Optimize a SINGLE function. Returns the agent's final message (the optimized
/// function, possibly fenced — caller strips fences). Designed to be run
/// concurrently, one agent per function.
///
/// `use_forge` toggles the per-function compile loop: the agent calls
/// `forge_verify` (FunctionForgeTool), which splices the candidate into the
/// original contract and compiles it; on a compile failure the agent refines.
#[allow(clippy::too_many_arguments)]
pub async fn optimize_function(
    client: &deepseek::Client,
    index: GasliteIndex,
    storage_layout: &str,
    original: Arc<str>,
    fn_name: &str,
    fn_source: &str,
    fn_start: usize,
    fn_end: usize,
    use_forge: bool,
) -> Result<String, String> {
    let context = format!(
        "STORAGE LAYOUT:\n{storage_layout}\n\n\
         FUNCTION TO OPTIMIZE:\n```solidity\n{fn_source}\n```"
    );

    let forge_step = if use_forge {
        "After producing the optimized function, call forge_verify with optimized_function set to \
         your rewrite. If it does not compile (compiles=false), fix the function using the returned \
         errors and call forge_verify again. Stop once it compiles, or after exhausting your attempts.\n"
    } else {
        ""
    };

    let user = format!(
        "Optimize ONLY this function by applying the RETRIEVED PATTERNS as templates, adapting slot \
         derivations and variable names to the contract's storage layout. Preserve the function's \
         observable behaviour and signature.\n\
         {forge_step}\n\
         Return ONLY the complete optimized function (signature + body) in a single ```solidity code \
         block — no contract wrapper, no imports."
    );

    let builder = client
        .agent(deepseek::DEEPSEEK_V4_FLASH)
        .preamble(SYSTEM_PROMPT)
        .context(&context)
        .dynamic_context(CONTEXT_SAMPLES, index)
        .temperature(0.1)
        .max_tokens(4096);

    // Per-turn instrumentation (labeled with the function name since agents run
    // concurrently and their logs interleave).
    let hook = TimingHook::new(fn_name);
    let result = if use_forge {
        builder
            .tool(FunctionForgeTool::new(original, fn_start, fn_end))
            .build()
            .prompt(user)
            .with_hook(hook.clone())
            .max_turns(FORGE_MAX_TURNS)
            .await
    } else {
        builder.build().prompt(user).with_hook(hook.clone()).max_turns(1).await
    };

    let (turns, llm_total, tool_total) = hook.summary();
    info!(
        "  [{}] {} turn(s) | LLM {:.2?} | forge {:.2?}",
        fn_name, turns, llm_total, tool_total
    );

    result.map_err(|e| format!("[{fn_name}] agent prompt failed: {e}"))
}

// ── per-turn timing hook ──────────────────────────────────────────────────────
#[derive(Default)]
struct TurnTimer {
    turn: usize,
    llm_start: Option<Instant>,
    tool_start: Option<Instant>,
    llm_total: Duration,
    tool_total: Duration,
}

/// A rig `PromptHook` that times every LLM call and tool call in the agent loop,
/// labeled with the function name (agents run concurrently).
#[derive(Clone)]
struct TimingHook {
    label: Arc<str>,
    state: Arc<Mutex<TurnTimer>>,
}

impl TimingHook {
    fn new(label: &str) -> Self {
        Self {
            label: Arc::from(label),
            state: Arc::new(Mutex::new(TurnTimer::default())),
        }
    }

    /// `(turns, total LLM time, total in-loop tool time)`.
    fn summary(&self) -> (usize, Duration, Duration) {
        let s = self.state.lock().unwrap();
        (s.turn, s.llm_total, s.tool_total)
    }
}

impl<M: CompletionModel> PromptHook<M> for TimingHook {
    async fn on_completion_call(&self, _prompt: &Message, _history: &[Message]) -> HookAction {
        let mut s = self.state.lock().unwrap();
        s.turn += 1;
        s.llm_start = Some(Instant::now());
        HookAction::cont()
    }

    async fn on_completion_response(
        &self,
        _prompt: &Message,
        _response: &CompletionResponse<M::Response>,
    ) -> HookAction {
        let mut s = self.state.lock().unwrap();
        if let Some(start) = s.llm_start.take() {
            let d = start.elapsed();
            s.llm_total += d;
            let turn = s.turn;
            drop(s);
            info!("  [{}] turn {turn}: LLM {d:.2?}", self.label);
        }
        HookAction::cont()
    }

    async fn on_tool_call(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
    ) -> ToolCallHookAction {
        self.state.lock().unwrap().tool_start = Some(Instant::now());
        ToolCallHookAction::cont()
    }

    async fn on_tool_result(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
        _result: &str,
    ) -> HookAction {
        let mut s = self.state.lock().unwrap();
        if let Some(start) = s.tool_start.take() {
            let d = start.elapsed();
            s.tool_total += d;
            let turn = s.turn;
            drop(s);
            info!("  [{}] turn {turn}: tool {tool_name} {d:.2?}", self.label);
        }
        HookAction::cont()
    }
}
