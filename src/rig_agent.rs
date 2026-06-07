//! The rig refinement agent. A single per-contract DeepSeek agent retrieves
//! patterns via `dynamic_context` (our `GasliteIndex`) and — when Foundry is
//! available — closes the loop with `ForgeTool`: generate → forge build/test on
//! a Mantle fork → on compile failure or gas regression, refine and retry,
//! capped by `max_turns`.

use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use rig_core::providers::deepseek;

use crate::retrieval::GasliteIndex;
use crate::tools::ForgeTool;

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

/// Run the refinement agent over a whole contract. Returns the agent's final
/// message (the optimized contract, possibly fenced — caller strips fences).
///
/// `use_forge` toggles the closed loop: when false (Foundry unavailable) the
/// agent runs one-shot with no tool, preserving the old non-verified behaviour.
pub async fn optimize_contract(
    client: &deepseek::Client,
    index: GasliteIndex,
    storage_layout: &str,
    original_contract: &str,
    use_forge: bool,
) -> Result<String, String> {
    let context = format!(
        "STORAGE LAYOUT:\n{storage_layout}\n\n\
         ORIGINAL CONTRACT:\n```solidity\n{original_contract}\n```"
    );

    let forge_step = if use_forge {
        "After producing the optimized contract, call forge_verify with original_code set to the \
         ORIGINAL CONTRACT exactly as given and optimized_code set to your rewrite. If it does not \
         compile (compiles=false) or gas does not improve (gas_saved <= 0), fix the code using the \
         returned errors and call forge_verify again. Stop once it compiles and gas_saved > 0, or \
         after you have exhausted your attempts.\n"
    } else {
        ""
    };

    let user = format!(
        "Optimize the gas-heavy functions in the ORIGINAL CONTRACT by applying the RETRIEVED \
         PATTERNS as templates, adapting slot derivations and variable names to this contract's \
         storage layout. Preserve every function's observable behaviour and signature.\n\
         {forge_step}\n\
         Return the COMPLETE optimized contract in a single ```solidity code block — the full \
         source, compilable as-is, with the same contract name and imports."
    );

    let builder = client
        .agent(deepseek::DEEPSEEK_V4_FLASH)
        .preamble(SYSTEM_PROMPT)
        .context(&context)
        .dynamic_context(CONTEXT_SAMPLES, index)
        .temperature(0.1)
        .max_tokens(8192);

    let result = if use_forge {
        builder
            .tool(ForgeTool)
            .build()
            .prompt(user)
            .max_turns(FORGE_MAX_TURNS)
            .await
    } else {
        builder.build().prompt(user).max_turns(1).await
    };

    result.map_err(|e| format!("agent prompt failed: {e}"))
}
