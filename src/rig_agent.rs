//! The rig refinement agent. A single per-contract DeepSeek agent retrieves
//! patterns via `dynamic_context` (our `GasliteIndex`) and — when Foundry is
//! available — closes the loop with `ForgeTool`: generate → forge build/test on
//! a Mantle fork → on compile failure or gas regression, refine and retry,
//! capped by `max_turns`.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use rig_core::{
    agent::{HookAction, PromptHook, ToolCallHookAction},
    client::CompletionClient,
    completion::{CompletionModel, CompletionResponse, Prompt},
    message::Message,
    providers::deepseek,
    tool::Tool,
};
use tracing::info;

use crate::{retrieval::GasliteIndex, tools::FunctionForgeTool, utils::strip_code_fences};

/// The single model id used by every agent (router, oneshot, per-function, verify).
/// `deepseek-chat` is the **non-thinking** mode of `deepseek-v4-flash` — it skips the
/// chain-of-thought that made generation 3-5x slower. One switch point for all agents.
#[allow(deprecated)]
pub const MODEL: &str = deepseek::DEEPSEEK_CHAT;

/// Max agent turns when the forge loop is active (1 generate + refinements).
const FORGE_MAX_TURNS: usize = 4;
/// Number of retrieved pattern documents to inject as dynamic context. Now actually
/// honored by `GasliteIndex::top_n` (it ranks by score and truncates to this many).
/// 4 balances coverage (a multi-function oneshot needs several patterns) against
/// prefill cost; 1 starves the model, the old ~6 bloated the prompt.
const CONTEXT_SAMPLES: usize = 4;

/// Appended to generation prompts to cut  output tokens (generation time scales with
/// emitted tokens): no comments, minimal whitespace, no prose around the code.
const TERSE_OUTPUT: &str = "OUTPUT STYLE — emit TERSE code: NO comments anywhere in the \
    assembly or Solidity, and minimal whitespace. Do not annotate, label, or explain the Yul. \
    Correctness and the exact opcodes/selectors still matter — only drop comments and blank lines.";

pub const SYSTEM_PROMPT: &str = "You are Gaslite, a gas optimization engine for Mantle L2 EVM contracts.\n\
    \n\
    Your role is pattern application and adaptation — not pattern invention.\n\
    \n\
    The RETRIEVED PATTERNS are your source of truth for YUL structure, opcodes, \
    and error selectors. Use them as templates:\n\
    - Keep the YUL opcodes, control flow, and error selectors exactly as shown\n\
    - The retrieved patterns may assume a DIFFERENT storage layout (e.g. packed \
      ERC721A/Solady). Take ONLY the technique from them (custom errors, inline log4 \
      emission, read caching) — NEVER copy their storage slot scheme.\n\
    - For EVERY storage access, use the EXACT SLOT recipes given under STORAGE LAYOUT \
      verbatim — they use the `.slot` accessor and keccak256(0x00, 0x40). Do not invent \
      slot arithmetic, offsets, or packed layouts the contract does not declare.\n\
    - `public` state variables have compiler-generated getters that read the STANDARD \
      slots. Any assembly write to a different slot (packed, offset, relocated) makes \
      those getters silently return wrong data — an automatic behavioural failure. \
      Keep every declaration and its layout exactly as-is; optimize only the access code.\n\
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
    file_decls: &str,
    original: Arc<str>,
    fn_name: &str,
    fn_source: &str,
    fn_start: usize,
    fn_end: usize,
    use_forge: bool,
) -> Result<String, String> {
    let context = format!(
        "STORAGE LAYOUT:\n{storage_layout}\n\n{}\
         FUNCTION TO OPTIMIZE:\n```solidity\n{fn_source}\n```",
        decls_block(file_decls)
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
         block — no contract wrapper, no imports.\n\
         {TERSE_OUTPUT}"
    );

    let builder = client
        .agent(MODEL)
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
            .tool(FunctionForgeTool::new(
                original, fn_start, fn_end,
            ))
            .build()
            .prompt(user)
            .with_hook(hook.clone())
            .max_turns(FORGE_MAX_TURNS)
            .await
    } else {
        builder
            .build()
            .prompt(user)
            .with_hook(hook.clone())
            .max_turns(1)
            .await
    };

    let captured = hook.captured();
    let (turns, llm_total, tool_total) = hook.summary();
    info!(
        "  [{}] {} turn(s) | LLM {:.2?} | forge {:.2?}{}",
        fn_name,
        turns,
        llm_total,
        tool_total,
        if captured.is_some() {
            " | early-exit"
        } else {
            ""
        }
    );

    // When the forge loop verified a compiling candidate, the hook captured it
    // from the tool-call args and terminated the loop early — skipping a redundant
    // LLM turn that would only re-emit the same function. Prefer that candidate
    // over the (cancelled) prompt result.
    if let Some(verified) = captured {
        return Ok(verified);
    }

    result.map_err(|e| format!("[{fn_name}] agent prompt failed: {e}"))
}

/// Render the file-level declarations block for agent context (empty when none).
fn decls_block(file_decls: &str) -> String {
    if file_decls
        .trim()
        .is_empty()
    {
        String::new()
    } else {
        format!(
            "CONTRACT DECLARATIONS (structs / enums / custom errors / events / modifiers the code \
             may reference — do not redefine or rename these):\n{file_decls}\n\n"
        )
    }
}

/// One-shot whole-contract optimization: a single LLM call that rewrites the entire
/// contract. No `forge_verify` tool loop — the caller's whole-contract forge gate is
/// the authority. This is the fast path the orchestrator picks for simple contracts.
pub async fn optimize_oneshot(
    client: &deepseek::Client,
    index: GasliteIndex,
    storage_layout: &str,
    file_decls: &str,
    contract_source: &str,
) -> Result<String, String> {
    let context = format!(
        "STORAGE LAYOUT:\n{storage_layout}\n\n{}\
         CONTRACT TO OPTIMIZE:\n```solidity\n{contract_source}\n```",
        decls_block(file_decls)
    );

    let user = format!(
        "Optimize the ENTIRE contract by applying the RETRIEVED PATTERNS as templates, \
         adapting slot derivations and variable names to the contract's storage layout. Preserve \
         every function's observable behaviour and signature, and keep all existing declarations. \
         Return ONLY the complete optimized contract in a single ```solidity code block — no \
         commentary, no explanations.\n\
         {TERSE_OUTPUT}"
    );

    let hook = TimingHook::new("oneshot");
    let result = client
        .agent(MODEL)
        .preamble(SYSTEM_PROMPT)
        .context(&context)
        .dynamic_context(CONTEXT_SAMPLES, index)
        .temperature(0.1)
        .max_tokens(8192)
        .build()
        .prompt(user)
        .with_hook(hook.clone())
        .max_turns(1)
        .await;

    let (turns, llm_total, tool_total) = hook.summary();
    info!(
        "  [oneshot] {} turn(s) | LLM {:.2?} | forge {:.2?}",
        turns, llm_total, tool_total
    );

    result.map_err(|e| format!("[oneshot] agent prompt failed: {e}"))
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
    /// Set to the verified function (from the `forge_verify` args) the first time
    /// the tool reports `compiles:true`; signals the loop was terminated early.
    captured: Arc<Mutex<Option<String>>>,
}

impl TimingHook {
    fn new(label: &str) -> Self {
        Self {
            label: Arc::from(label),
            state: Arc::new(Mutex::new(
                TurnTimer::default(),
            )),
            captured: Arc::new(Mutex::new(None)),
        }
    }

    /// `(turns, total LLM time, total in-loop tool time)`.
    fn summary(&self) -> (usize, Duration, Duration) {
        let s = self
            .state
            .lock()
            .unwrap();
        (
            s.turn,
            s.llm_total,
            s.tool_total,
        )
    }

    /// The verified candidate captured on early exit, if any.
    fn captured(&self) -> Option<String> {
        self.captured
            .lock()
            .unwrap()
            .clone()
    }
}

impl<M: CompletionModel> PromptHook<M> for TimingHook {
    async fn on_completion_call(
        &self,
        _prompt: &Message,
        _history: &[Message],
    ) -> HookAction {
        let mut s = self
            .state
            .lock()
            .unwrap();
        s.turn += 1;
        s.llm_start = Some(Instant::now());
        HookAction::cont()
    }

    async fn on_completion_response(
        &self,
        _prompt: &Message,
        _response: &CompletionResponse<M::Response>,
    ) -> HookAction {
        let mut s = self
            .state
            .lock()
            .unwrap();
        if let Some(start) = s
            .llm_start
            .take()
        {
            let d = start.elapsed();
            s.llm_total += d;
            let turn = s.turn;
            drop(s);
            info!(
                "  [{}] turn {turn}: LLM {d:.2?}",
                self.label
            );
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
        self.state
            .lock()
            .unwrap()
            .tool_start = Some(Instant::now());
        ToolCallHookAction::cont()
    }

    async fn on_tool_result(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
        result: &str,
    ) -> HookAction {
        {
            let mut s = self
                .state
                .lock()
                .unwrap();
            if let Some(start) = s
                .tool_start
                .take()
            {
                let d = start.elapsed();
                s.tool_total += d;
                let turn = s.turn;
                drop(s);
                info!(
                    "  [{}] turn {turn}: tool {tool_name} {d:.2?}",
                    self.label
                );
            }
        }

        // Once forge_verify confirms the candidate compiles, we have everything we
        // need: the verified function is in the tool-call args. Capture it and
        // terminate the loop, skipping the redundant LLM turn that would only
        // re-emit the same code. On a compile failure we continue so the model can
        // refine using the returned errors.
        if tool_name == FunctionForgeTool::NAME {
            let result_json: Option<serde_json::Value> = serde_json::from_str(result).ok();
            let compiles = result_json
                .as_ref()
                .and_then(|v| v.get("compiles"))
                .and_then(|c| c.as_bool())
                .unwrap_or(false);
            if compiles {
                let args_json: Option<serde_json::Value> = serde_json::from_str(args).ok();
                let func = args_json
                    .as_ref()
                    .and_then(|v| v.get("optimized_function"))
                    .and_then(|x| x.as_str());
                if let Some(f) = func {
                    *self
                        .captured
                        .lock()
                        .unwrap() = Some(strip_code_fences(f));
                    return HookAction::terminate("forge_verify compiled — skipping final turn");
                }
            }
        }

        HookAction::cont()
    }
}
