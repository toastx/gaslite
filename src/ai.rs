//! AI surface: text embeddings (FastEmbed) and the DeepSeek optimization call.

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client as OpenAIClient,
};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::{Arc, Mutex};

// ── embeddings ──────────────────────────────────────────────────────────────
/// Wraps the FastEmbed model behind a mutex so embed calls are serialised.
pub struct Embedder(Mutex<TextEmbedding>);

impl Embedder {
    /// Loads the BGE-Small-EN v1.5 model (384-dim vectors).
    pub fn new() -> Result<Arc<Self>, Box<dyn std::error::Error>> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true),
        )?;
        Ok(Arc::new(Self(Mutex::new(model))))
    }

    /// Embeds a single string, offloading the blocking model call to a worker thread.
    pub async fn embed(self: Arc<Self>, text: &str) -> Result<Vec<f32>, String> {
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            let mut model = self.0.lock().unwrap();
            let mut embeddings = model
                .embed(vec![text.as_str()], None)
                .map_err(|e| format!("Embed error: {e}"))?;
            embeddings
                .pop()
                .ok_or_else(|| "Embedding returned empty results".to_string())
        })
        .await
        .map_err(|e| format!("Embedding task panicked: {e}"))?
    }
}

// ── DeepSeek optimization ────────────────────────────────────────────────────
pub async fn call_deepseek(
    client: &OpenAIClient<OpenAIConfig>,
    storage_layout: &str,
    function_source: &str,
    context: &str,
) -> Result<String, String> {
    let system = "You are Gaslite, a gas optimization engine for Mantle L2 EVM contracts.\n\
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

    let user = format!(
        "STORAGE LAYOUT:\n{storage_layout}\n\n\
        FUNCTION TO OPTIMIZE:\n```solidity\n{function_source}\n```\n\n\
        RETRIEVED PATTERNS:\n{context}\n\n\
        TASK:\n\
        Optimize ONLY this function by applying the retrieved patterns as templates, \
        adapting slot derivations and variable names to this contract's storage layout.\n\
        Return ONLY the complete optimized function — no contract wrapper, no imports.\n\
        After the function, add one line per change: pattern ID applied + estimated gas saved on Mantle.\n\
        If a pattern genuinely cannot apply even with adaptation, say why in one line and skip it."
    );

    let request = CreateChatCompletionRequestArgs::default()
        .model("deepseek-v4-flash")
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system)
                .build()
                .map_err(|e| e.to_string())?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user.as_str())
                .build()
                .map_err(|e| e.to_string())?
                .into(),
        ])
        .temperature(0.1_f32)
        .build()
        .map_err(|e| format!("Failed to build request: {e}"))?;

    let response = client
        .chat()
        .create(request)
        .await
        .map_err(|e| format!("DeepSeek request failed: {e}"))?;

    response
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .ok_or_else(|| "Empty response from DeepSeek".to_string())
}
