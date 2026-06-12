//! The verify agent: generates a **differential** equivalence test for one function.
//!
//! The original contract is the behavioural oracle. For each function we ask the LLM
//! to write a Foundry `test_eq_<fn>()` that calls the function identically on the
//! original (`o`) and optimized (`p`) instances and asserts every observable output
//! matches (return values + public getters), plus a revert-parity check. The forge
//! harness then executes these; any mismatch fails the gate. This is what catches a
//! wrong storage-slot derivation that construction-gas measurement cannot see.
//!
//! One agent per function — these are generated concurrently (one thread each).

use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use rig_core::providers::deepseek;

use crate::utils::strip_code_fences;

const VERIFY_SYSTEM_PROMPT: &str = "You write Foundry differential tests in Solidity. The ORIGINAL \
    contract is the source of truth; the OPTIMIZED contract must behave identically. You produce \
    exactly one test function that proves it, or finds a divergence.\n\
    \n\
    Your test is also executed in a SANITY harness where both instances are the ORIGINAL contract. \
    A correct test always passes there; if yours would not, it is broken and gets discarded. So \
    before emitting the test, mentally execute the ORIGINAL contract's code line by line with your \
    chosen literal values and confirm every happy-path call succeeds and every assertion holds.";

/// Generate the `test_eq_<fn_name>()` body for one function. `orig_type`/`opt_type`
/// are the two contract type names already deployed as `o` and `p` in the harness's
/// `setUp()`. `prev_attempt` is `(previous test source, sanity failure line)` when a
/// prior attempt failed the original-vs-original sanity suite — the agent is asked
/// to diagnose and fix its own test. Returns the Solidity function text (fences
/// stripped).
#[allow(clippy::too_many_arguments)]
pub async fn gen_equivalence_test(
    client: &deepseek::Client,
    original_contract: &str,
    storage_layout: &str,
    orig_type: &str,
    opt_type: &str,
    fn_name: &str,
    fn_signature: &str,
    prev_attempt: Option<(&str, &str)>,
) -> Result<String, String> {
    let context = format!(
        "ORIGINAL CONTRACT (the behavioural spec):\n```solidity\n{original_contract}\n```\n\n\
         PUBLIC STATE (these generate getters you can read for assertions):\n{storage_layout}\n\n\
         The test harness already deployed two instances with IDENTICAL interfaces:\n\
         - `o` of type {orig_type} (original / oracle)\n\
         - `p` of type {opt_type} (optimized / under test)\n\n\
         TARGET FUNCTION: {fn_signature}"
    );

    // When a prior attempt failed the original-vs-original sanity run, hand the
    // model its own broken test plus the revert reason so it fixes the actual bug
    // instead of regenerating blind.
    let feedback = match prev_attempt {
        Some((code, fail)) => format!(
            "\n\nYOUR PREVIOUS ATTEMPT WAS BROKEN — it failed even when both `o` and `p` were the \
             ORIGINAL contract, so the fault is in the TEST's preconditions or arithmetic, not in \
             either contract.\n\
             Failure: {fail}\n\
             Previous test:\n```solidity\n{code}\n```\n\
             Trace the ORIGINAL contract's requires and arithmetic with the exact literals that \
             test used, find the line that reverts or the assertion that cannot hold, and return \
             a corrected test."
        ),
        None => String::new(),
    };

    let user = format!(
        "Write exactly ONE Solidity function `test_eq_{fn_name}()` (public, no args) that proves \
         `{fn_name}` behaves identically on `o` and `p`.\n\
         \n\
         HARD CONSTRAINTS (violating any of these makes the test unusable):\n\
         - Your output is pasted VERBATIM inside an existing test contract. Do NOT declare a \
           contract, imports, `o`, `p`, `setUp`, or any state variable — only the one function. \
           NO forge-std, NO vm/cheatcodes, NO console, NO comments.\n\
         - OWNERSHIP: every call runs with msg.sender = THIS TEST CONTRACT (address(this)). \
           For any function gated on ownership/balance of msg.sender (e.g. `require(ownerOf[id] == \
           msg.sender)`, or that decrements the caller's balance), the caller must own/hold the \
           thing FIRST — so mint to `address(this)`, NOT to an external address. Minting the target \
           token to 0xBEEF and then calling transfer/approve from the test will revert with the \
           owner check. Use external addresses only as the destination/recipient, never as the \
           owner the gated call needs.\n\
         - PRECONDITIONS FIRST: read the ORIGINAL function body and trace its requires and \
           arithmetic with your exact literals before you write a single call. If a call needs \
           owned tokens, balance, allowance, or supply, create that state first by calling setup \
           functions IDENTICALLY on both `o` and `p` — and create ENOUGH of it. If the body \
           subtracts N from a balance, mint to that holder N times first (mint adds 1 each call). \
           A happy-path call that reverts on the original is a broken test and will be thrown away.\n\
         - Happy-path calls are NEVER wrapped in try/catch — if they revert, the test should fail \
           loudly. try/catch is only for the deliberate revert-parity probe at the end.\n\
         - Every variable you declare must be used. No dead code.\n\
         \n\
         WHAT TO ASSERT (coverage is the goal):\n\
         - Do the SAME operations on `o` and on `p`, in the same order, with the SAME literal \
           arguments. msg.sender is this test contract for every call on both instances.\n\
         - After each state-changing call, assert equality across instances of: every return \
           value, AND every public getter the function could have touched — including ones it \
           should NOT have changed (e.g. after approve, also check ownerOf and balanceOf are \
           still equal). Wrong-storage-slot bugs hide exactly there.\n\
         - MANDATORY balance coverage: after any call that moves or changes balances, assert \
           `o.balanceOf(X) == p.balanceOf(X)` for EVERY address involved — the CALLER \
           (address(this)) FIRST, then every recipient/other party — and assert `o.totalSupply() \
           == p.totalSupply()`. The single most-missed bug is an off-by-one or non-conserving \
           change to the CALLER's own balance (e.g. debiting 6 while crediting 5); checking only \
           the recipient lets it slip through. Never skip the caller's balance.\n\
         - Use distinct literals so swapped values can't cancel out (e.g. two different \
           addresses, token ids 0 and 1) and exercise the function at least twice when cheap.\n\
         - One `require(a == b, \"label\")` per checked value, with a short unique label.\n\
         - End with ONE revert-parity probe: a call that should revert per the original's \
           requires, wrapped in try/catch on both, then `require(ro == rp, \"revert parity\")`.\n\
         \n\
         Example shape (adapt to the real function, getters, and preconditions):\n\
         ```solidity\n\
         function test_eq_mint() public {{\n\
             address a1 = address(0xBEEF);\n\
             address a2 = address(0xCAFE);\n\
             uint256 r1o = o.mint(a1);\n\
             uint256 r1p = p.mint(a1);\n\
             require(r1o == r1p, \"ret1\");\n\
             uint256 r2o = o.mint(a2);\n\
             uint256 r2p = p.mint(a2);\n\
             require(r2o == r2p, \"ret2\");\n\
             require(o.ownerOf(r1o) == p.ownerOf(r1p), \"owner1\");\n\
             require(o.ownerOf(r2o) == p.ownerOf(r2p), \"owner2\");\n\
             require(o.balanceOf(a1) == p.balanceOf(a1), \"bal1\");\n\
             require(o.balanceOf(a2) == p.balanceOf(a2), \"bal2\");\n\
             require(o.totalSupply() == p.totalSupply(), \"supply\");\n\
             bool ro; try o.transfer(a1, r1o) {{ ro = false; }} catch {{ ro = true; }}\n\
             bool rp; try p.transfer(a1, r1p) {{ rp = false; }} catch {{ rp = true; }}\n\
             require(ro == rp, \"revert parity\");\n\
         }}\n\
         ```\n\
         Owner-gated example (note: mints to address(this), and mints ENOUGH for a balance the body \
         decrements — adapt the count to the original's arithmetic):\n\
         ```solidity\n\
         function test_eq_transfer() public {{\n\
             address self = address(this);\n\
             address dest = address(0xD15);\n\
             // original transfer subtracts 5 from the caller's balance, so hold >= 5 first\n\
             uint256 idO; uint256 idP;\n\
             for (uint256 i = 0; i < 5; i++) {{ idO = o.mint(self); idP = p.mint(self); }}\n\
             o.transfer(dest, idO);\n\
             p.transfer(dest, idP);\n\
             require(o.ownerOf(idO) == p.ownerOf(idP), \"owner\");\n\
             require(o.balanceOf(self) == p.balanceOf(self), \"balSelf\");\n\
             require(o.balanceOf(dest) == p.balanceOf(dest), \"balDest\");\n\
             require(o.getApproved(idO) == p.getApproved(idP), \"approvalCleared\");\n\
             bool ro; try o.transfer(dest, 999) {{ ro = false; }} catch {{ ro = true; }}\n\
             bool rp; try p.transfer(dest, 999) {{ rp = false; }} catch {{ rp = true; }}\n\
             require(ro == rp, \"revert parity\");\n\
         }}\n\
         ```\n\
         Return ONLY the function in a single ```solidity code block.{feedback}"
    );

    let result = client
        .agent(crate::rig_agent::MODEL)
        .preamble(VERIFY_SYSTEM_PROMPT)
        .context(&context)
        .temperature(0.0)
        .max_tokens(2048)
        .build()
        .prompt(user)
        .await
        .map_err(|e| format!("[verify {fn_name}] agent prompt failed: {e}"))?;

    Ok(strip_code_fences(&result).to_string())
}
