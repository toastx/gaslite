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

use rig_core::{client::CompletionClient, completion::Prompt, providers::deepseek};

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
        "Write exactly ONE Solidity function named `test_eq_{fn_name}` (public) that proves \
         `{fn_name}` behaves identically on `o` and `p`. It MAY take fuzz parameters (preferred — \
         see DIFFERENTIAL FUZZING) or take no args; either way keep the exact name `test_eq_{fn_name}`.\n\
         \n\
         HARD CONSTRAINTS (violating any of these makes the test unusable):\n\
         - Your output is pasted VERBATIM inside an existing test contract. Do NOT declare a \
           contract, imports, `o`, `p`, `setUp`, or any state variable — only the one function. \
           NO forge-std, NO vm/cheatcodes, NO console, NO comments.\n\
         - CALLER IS address(this): every call runs with msg.sender = THIS TEST CONTRACT. For any \
           function gated on the caller's ownership/balance/allowance (e.g. `require(ownerOf[id] == \
           msg.sender)`, or one that decrements the caller's balance), first put address(this) into \
           that state using the ORIGINAL contract's OWN functions — never an external address as the \
           gated owner. Use external addresses only as destinations/recipients.\n\
         - PRECONDITIONS FIRST, USING THE CONTRACT'S REAL FUNCTIONS: read the ORIGINAL function body \
           and trace its requires and arithmetic with your exact literals before writing a single \
           call. If a call needs prior state (a balance, allowance, supply, a registered entry), \
           create it first by calling the contract's ACTUAL state-changing functions IDENTICALLY on \
           `o` and `p`, and create ENOUGH of it for the arithmetic the body performs. Do NOT \
           hallucinate generic helpers like `mint()`, `balanceOf()`, `totalSupply()`, `ownerOf()` or \
           `approve()` unless they actually exist in the ORIGINAL contract — call only functions and \
           getters that appear in the provided source. If the contract offers no path to reach a \
           required state, do not test that path. A happy-path call that reverts on the original is a \
           broken test and will be thrown away.\n\
         - NO CHEATCODES, ONE CALLER, NO ETH: there is no `vm`, so you cannot advance time, change \
           msg.sender, or fund the test with ETH. If a function needs elapsed time, a different \
           caller, or msg.value the test contract does not have, do not try to force it — pick a \
           happy path address(this) can reach right now; if none exists, emit ONLY the revert-parity \
           probe. Never call a payable function with a value the test contract cannot pay.\n\
         - Happy-path calls are NEVER wrapped in try/catch — if they revert, the test should fail \
           loudly. try/catch is only for the deliberate revert-parity probe at the end.\n\
         - PUBLIC GETTERS OF STRUCTS RETURN TUPLES, NOT STRUCTS. A `mapping(...) public stakes` \
           whose value is `struct Stake {{ uint256 amount; uint256 since; bool active; }}` generates \
           a getter returning `(uint256, uint256, bool)` — you CANNOT write `o.stakes(a).amount`. \
           Destructure positionally, naming only the fields you compare: \
           `(uint256 amtO,,) = o.stakes(a); (uint256 amtP,,) = p.stakes(a); require(amtO == amtP, \"amt\");`. \
           The same applies to any `public` struct or array-of-struct state variable.\n\
         - Every variable you declare must be used. No dead code.\n\
         \n\
         WHAT TO ASSERT (coverage is the goal):\n\
         - Do the SAME operations on `o` and on `p`, in the same order, with the SAME literal \
           arguments. msg.sender is this test contract for every call on both instances.\n\
         - After each state-changing call, assert equality across instances of: every return \
           value, AND every public getter the function could have touched — including ones it \
           should NOT have changed (e.g. after approve, also check ownerOf and balanceOf are \
           still equal). Wrong-storage-slot bugs hide exactly there.\n\
         - CONSERVATION COVERAGE: if the contract exposes balance/supply getters (e.g. `balanceOf`, \
           `totalSupply`), then after any call that moves or changes balances assert \
           `o.balanceOf(X) == p.balanceOf(X)` for EVERY address involved — the CALLER \
           (address(this)) FIRST, then every recipient/other party — and `o.totalSupply() == \
           p.totalSupply()`. The single most-missed bug is a non-conserving change to the CALLER's \
           own balance (e.g. debiting 6 while crediting 5); checking only the recipient lets it slip \
           through, so never skip the caller's balance. If the contract has no such getters, assert \
           equality of whatever public state the call actually touches instead.\n\
         - Use distinct literals so swapped values can't cancel out (e.g. two different \
           addresses, token ids 0 and 1) and exercise the function at least twice when cheap.\n\
         - One `require(a == b, \"label\")` per checked value, with a short unique label.\n\
         - End with ONE revert-parity probe: a call that should revert per the original's \
           requires, wrapped in try/catch on both, then `require(ro == rp, \"revert parity\")`.\n\
         \n\
         DIFFERENTIAL FUZZING (PREFERRED — far stronger than fixed literals):\n\
         - You MAY declare the test with typed parameters, e.g. \
           `function test_eq_{fn_name}(uint256 amt, address a) public`. Foundry fuzzes them, running \
           the SAME random inputs on `o` and `p` many times. Keep the name `test_eq_{fn_name}` (do \
           NOT rename to `testFuzz_`).\n\
         - There is NO `vm.assume` and NO forge-std `bound`, so you CANNOT reject an input. You must \
           MAP every possible fuzz value into a valid domain with plain arithmetic at the very top of \
           the function, so the ORIGINAL never reverts for ANY input — otherwise the fuzzer finds a \
           reverting case, the sanity suite fails, and your test is discarded. Bound conservatively:\n\
           · amounts / ids / counts: `amt = amt % 1000000 + 1;` (non-zero, comfortably within any \
             supply or balance the body needs — keep upper bounds small so repeated calls can't \
             overflow the original's arithmetic).\n\
           · recipient addresses: `a = address(uint160(a) | 1);` to avoid address(0); if the body \
             rejects the caller as recipient, also push it off address(this).\n\
         - Fuzz ONLY free value/amount/recipient inputs. A parameter that must hold a fixed role — \
           above all the caller/owner, which is ALWAYS address(this) — must stay a fixed literal, \
           never a fuzz arg.\n\
         - Every fuzzed call is a happy-path call on BOTH instances with the SAME bounded value; \
           assert equality of returns and touched getters exactly as above. The revert-parity probe \
           at the end still uses FIXED out-of-domain literals (not the fuzz inputs).\n\
         - If a function takes no fuzzable input (e.g. a no-arg owner setter), just write it with no \
           parameters and fixed literals — fuzzing is optional, correctness is not.\n\
         \n\
         The two examples below are for ERC721-style TOKEN contracts and are illustrations of SHAPE \
         only — the `mint`/`ownerOf`/`balanceOf`/`transfer` calls and the 999 bad-id probe are \
         token-specific. Adapt the structure to the ACTUAL functions and getters of the provided \
         ORIGINAL contract; if it is not a token, do not use any of these names.\n\
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
         Fuzzed, NON-token example (note: bounds the input first so the original never reverts, \
         destructures the struct-tuple getter, and keeps the caller as address(this)):\n\
         ```solidity\n\
         function test_eq_stake(uint256 amount) public {{\n\
             amount = amount % 1000000 + 1;\n\
             o.stake(amount);\n\
             p.stake(amount);\n\
             require(o.totalStaked() == p.totalStaked(), \"staked\");\n\
             (uint256 amtO,,) = o.stakes(address(this));\n\
             (uint256 amtP,,) = p.stakes(address(this));\n\
             require(amtO == amtP, \"amt\");\n\
             bool ro; try o.stake(0) {{ ro = false; }} catch {{ ro = true; }}\n\
             bool rp; try p.stake(0) {{ rp = false; }} catch {{ rp = true; }}\n\
             require(ro == rp, \"revert parity\");\n\
         }}\n\
         ```\n\
         Return ONLY the function in a single ```solidity code block.{feedback}"
    );

    let result = client
        .agent(super::rig_agent::MODEL)
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
