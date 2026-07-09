//! Forge verification: compiles original vs. optimized contracts in a temp
//! sandbox and measures construction gas via a Mantle fork.

use std::{
    fs,
    io::Read,
    path::Path,
    process::{Command, Output, Stdio},
    sync::OnceLock,
    time::{Duration, Instant},
};

use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tracing::{info, warn};
use uuid::Uuid;

// ── DTOs ──────────────────────────────────────────────────────────────────────
#[derive(Deserialize)]
pub struct VerifyRequest {
    original_code: String,
    optimized_code: String,
}

#[derive(Serialize)]
pub struct VerifyResponse {
    pub(crate) compiles: bool,
    pub(crate) errors: Vec<String>,
    pub(crate) gas_original: Option<u64>,
    pub(crate) gas_optimized: Option<u64>,
    pub(crate) gas_saved: Option<i64>,
    pub(crate) forge_output: String,
}

/// Real per-function runtime gas, original vs optimized, parsed from forge's
/// `--gas-report` (avg over the differential test calls — both instances are
/// exercised with identical arguments, so the comparison is apples-to-apples).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FunctionGas {
    pub name: String,
    pub gas_original: Option<u64>,
    pub gas_optimized: Option<u64>,
    pub gas_saved: Option<i64>,
}

// ── handler ───────────────────────────────────────────────────────────────────
pub async fn verify_contract(
    Json(payload): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, (axum::http::StatusCode, String)> {
    info!(
        "POST /api/verify — {} + {} bytes",
        payload.original_code.len(),
        payload.optimized_code.len()
    );
    run_forge_sandbox_async(payload.original_code, payload.optimized_code)
        .await
        .map(Json)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))
}

// ── concurrency-bounded async entrypoint ──────────────────────────────────────
/// Global cap on concurrent forge sandboxes so N in-flight requests (each of
/// which may run forge several times in the agent loop plus a final check) don't
/// fork unbounded subprocesses. Override with `FORGE_MAX_CONCURRENCY`.
fn forge_semaphore() -> &'static Semaphore {
    static SEM: OnceLock<Semaphore> = OnceLock::new();
    SEM.get_or_init(|| {
        let permits = std::env::var("FORGE_MAX_CONCURRENCY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
            .max(1);
        Semaphore::new(permits)
    })
}

/// Run the (blocking) forge sandbox on a worker thread, bounded by the global
/// concurrency limit. The single entrypoint for every forge invocation —
/// `/api/verify`, `ForgeTool`, and the optimize handler's final check.
pub(crate) async fn run_forge_sandbox_async(
    original: String,
    optimized: String,
) -> Result<VerifyResponse, String> {
    let _permit = forge_semaphore()
        .acquire()
        .await
        .map_err(|e| format!("forge semaphore closed: {e}"))?;
    tokio::task::spawn_blocking(move || run_forge_sandbox(&original, &optimized))
        .await
        .map_err(|e| format!("forge task panicked: {e}"))?
}

// ── helpers ───────────────────────────────────────────────────────────────────
/// Run a command with a wall-clock timeout, killing it on expiry. Stdout/stderr
/// are drained on threads so a chatty child can't deadlock on a full pipe buffer.
fn run_with_timeout(mut cmd: Command, timeout: Duration) -> std::io::Result<Output> {
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;

    let mut out_pipe = child.stdout.take().expect("piped stdout");
    let mut err_pipe = child.stderr.take().expect("piped stderr");
    let out_h = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = out_pipe.read_to_end(&mut b);
        b
    });
    let err_h = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = err_pipe.read_to_end(&mut b);
        b
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            warn!(
                "  forge: killing subprocess after {}s timeout",
                timeout.as_secs()
            );
            let _ = child.kill();
            let _ = child.wait();
            let _ = out_h.join();
            let _ = err_h.join();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("forge subprocess timed out after {}s", timeout.as_secs()),
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    };

    Ok(Output {
        status,
        stdout: out_h.join().unwrap_or_default(),
        stderr: err_h.join().unwrap_or_default(),
    })
}

/// Timeout (seconds) for a forge step. `build` is local; `test` hits a Mantle fork.
fn forge_timeout(var: &str, default_secs: u64) -> Duration {
    let secs = std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default_secs);
    Duration::from_secs(secs)
}

/// Whether a usable `forge` binary is present — gates the closed-loop refinement.
pub(crate) fn forge_available() -> bool {
    std::process::Command::new(forge_binary())
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn forge_binary() -> String {
    if let Ok(home) = std::env::var("HOME") {
        let p = format!("{home}/.foundry/bin/forge");
        if Path::new(&p).exists() {
            return p;
        }
    }
    "forge".to_string()
}

pub(crate) fn extract_sol_contract_name(source: &str) -> Option<String> {
    for line in source.lines() {
        if let Some(rest) = line.trim().strip_prefix("contract ")
            && let Some(name) = rest
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
            && !name.is_empty()
        {
            return Some(name.to_string());
        }
    }
    None
}

fn build_gas_test(orig_name: &str, opt_name: &str, ctor_args: &str) -> String {
    format!(
        "// SPDX-License-Identifier: MIT\n\
         pragma solidity ^0.8.0;\n\
         import \"../src/Original.sol\";\n\
         import \"../src/Optimized.sol\";\n\n\
         contract GasCompareTest {{\n\
             function test_original() external {{ new {orig_name}({ctor_args}); }}\n\
             function test_optimized() external {{ new {opt_name}({ctor_args}); }}\n\
         }}\n"
    )
}

/// Synthesize a Solidity constructor-argument list (no surrounding parens) of
/// default literals, so the sandbox can instantiate contracts whose constructor
/// takes parameters (e.g. `constructor(address _token)` → `address(0)`). Returns
/// "" for a parameterless or absent constructor. The optimizer never changes the
/// constructor signature, so the original's args work for both contracts.
fn synthesize_constructor_args(source: &str) -> String {
    match extract_constructor_params(source) {
        Some(params) => params
            .iter()
            .map(|t| default_literal_for_type(t))
            .collect::<Vec<_>>()
            .join(", "),
        None => String::new(),
    }
}

/// Extract the constructor's parameter TYPES (data location + name stripped) from
/// the source. `None` = no constructor; `Some(vec![])` = constructor with no params.
fn extract_constructor_params(source: &str) -> Option<Vec<String>> {
    // Find a `constructor` keyword that is immediately followed (modulo whitespace)
    // by `(` — i.e. the definition, not a mention in a comment/string.
    let mut search = 0;
    let open = loop {
        let rel = source[search..].find("constructor")?;
        let kw = search + rel;
        let after = kw + "constructor".len();
        let before_boundary = kw == 0
            || !matches!(source.as_bytes()[kw - 1], b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_');
        let rest = source[after..].trim_start();
        if before_boundary && rest.starts_with('(') {
            break after + source[after..].find('(')?;
        }
        search = after;
    };

    // Walk to the matching close paren.
    let mut depth = 0i32;
    let mut close = None;
    for (i, ch) in source[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            },
            _ => {},
        }
    }
    let inner = &source[open + 1..close?];
    if inner.trim().is_empty() {
        return Some(vec![]);
    }

    // Split on top-level commas (ignoring nested generics/arrays/tuples).
    let mut params: Vec<String> = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in inner.chars() {
        match ch {
            '(' | '[' | '<' => {
                depth += 1;
                cur.push(ch);
            },
            ')' | ']' | '>' => {
                depth -= 1;
                cur.push(ch);
            },
            ',' if depth == 0 => {
                params.push(cur.trim().to_string());
                cur.clear();
            },
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        params.push(cur.trim().to_string());
    }

    Some(params.iter().map(|p| param_type(p)).collect())
}

/// Reduce a constructor parameter (`address payable _to`) to just its type
/// (`address payable`) by dropping the data location and the parameter name.
fn param_type(param: &str) -> String {
    let toks: Vec<&str> = param
        .split_whitespace()
        .filter(|t| !matches!(*t, "memory" | "calldata" | "storage"))
        .collect();
    // The trailing identifier is the parameter name when more than the type remains.
    if toks.len() > 1 {
        toks[..toks.len() - 1].join(" ")
    } else {
        toks.join(" ")
    }
}

/// A default literal for a Solidity type, used to fill a constructor call. Covers
/// the common primitives, arrays and contract/interface types; falls back to `0`.
fn default_literal_for_type(ty: &str) -> String {
    let t = ty.trim();
    if let Some(inner) = t.strip_suffix("[]") {
        return format!("new {}[](0)", inner.trim());
    }
    if t.contains("payable") {
        return "payable(address(0))".to_string();
    }
    match t {
        "address" => return "address(0)".to_string(),
        "bool" => return "false".to_string(),
        "string" | "bytes" => return "\"\"".to_string(),
        _ => {},
    }
    if t.starts_with("uint") || t.starts_with("int") {
        return "0".to_string();
    }
    // Fixed-size bytesN.
    if let Some(n) = t.strip_prefix("bytes")
        && n.parse::<u32>().is_ok()
    {
        return format!("{t}(0)");
    }
    // Contract/interface types (PascalCase) — cast the zero address.
    if t.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
        return format!("{t}(address(0))");
    }
    "0".to_string()
}

// Strip markdown artifacts that DeepSeek sometimes embeds in optimized output:
// ``` fence markers, **bold** lines, *(italic notes)*, and bullet-point explanations.
fn clean_for_forge(code: &str) -> String {
    code.lines()
        .filter(|line| {
            let t = line.trim();
            if t.starts_with("```") {
                return false;
            }
            if t.starts_with("**") {
                return false;
            }
            if t.starts_with("*(") {
                return false;
            }
            // Bullet points that start with an uppercase word are English prose, not Solidity
            if let Some(rest) = t.strip_prefix("- ")
                && rest
                    .trim()
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
            {
                return false;
            }
            true
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn collect_forge_errors(stderr: &str) -> Vec<String> {
    stderr
        .lines()
        .filter(|l| {
            let lo = l.to_lowercase();
            lo.contains("error") || lo.contains("undeclared") || lo.contains("not found")
        })
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .take(20)
        .collect()
}

fn parse_test_gas(output: &str, fn_suffix: &str) -> Option<u64> {
    for line in output.lines() {
        if line.contains(fn_suffix)
            && line.contains("gas:")
            && let Some(g) = line.split("gas:").nth(1)
        {
            let s = g.trim().trim_end_matches(')').trim();
            if let Ok(n) = s.parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

pub(crate) fn run_forge_sandbox(original: &str, optimized: &str) -> Result<VerifyResponse, String> {
    let forge = forge_binary();
    let root = std::env::temp_dir().join(format!("gaslite_{}", Uuid::new_v4()));
    let res = forge_sandbox_inner(&forge, &root, original, optimized);
    let _ = fs::remove_dir_all(&root);
    res
}

/// Write the two-contract sandbox project (src/Original.sol, src/Optimized.sol,
/// foundry.toml). The optimized contract is renamed `{orig}Optimized` so both can be
/// imported into one test. Returns `(orig_name, opt_name, mantle_rpc)`. Shared by the
/// construction-gas and behavioral-equivalence runners.
fn write_sandbox_project(
    root: &Path,
    original: &str,
    optimized: &str,
) -> Result<(String, String, String), String> {
    fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    fs::create_dir_all(root.join("test")).map_err(|e| e.to_string())?;

    let orig_name =
        extract_sol_contract_name(original).unwrap_or_else(|| "OriginalContract".to_string());
    let opt_src_name = extract_sol_contract_name(optimized).unwrap_or_else(|| orig_name.clone());
    // Rename optimized contract to avoid symbol collision with original.
    let opt_name = format!("{orig_name}Optimized");
    let opt_code = optimized.replacen(
        &format!("contract {opt_src_name}"),
        &format!("contract {opt_name}"),
        1,
    );

    fs::write(root.join("src/Original.sol"), clean_for_forge(original))
        .map_err(|e| e.to_string())?;
    fs::write(root.join("src/Optimized.sol"), clean_for_forge(&opt_code))
        .map_err(|e| e.to_string())?;

    let mantle_rpc =
        std::env::var("MANTLE_RPC_URL").unwrap_or_else(|_| "https://rpc.mantle.xyz".to_string());

    // Differential tests may be fuzzed (parameterized `test_eq_*`); cap runs so a
    // Mantle-fork suite stays bounded. Override with FORGE_FUZZ_RUNS.
    let fuzz_runs = std::env::var("FORGE_FUZZ_RUNS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(64)
        .max(1);
    fs::write(
        root.join("foundry.toml"),
        format!(
            "[profile.default]\nsrc=\"src\"\ntest=\"test\"\nevm_version=\"paris\"\n\
                 [fuzz]\nruns={fuzz_runs}\n\
                 [rpc_endpoints]\nmantle=\"{mantle_rpc}\"\n"
        ),
    )
    .map_err(|e| e.to_string())?;

    Ok((orig_name, opt_name, mantle_rpc))
}

fn forge_sandbox_inner(
    forge: &str,
    root: &Path,
    original: &str,
    optimized: &str,
) -> Result<VerifyResponse, String> {
    let (orig_name, opt_name, mantle_rpc) = write_sandbox_project(root, original, optimized)?;

    // ── build ─────────────────────────────────────────────────────────────────
    info!("  forge build: {}", root.display());
    let mut build_cmd = Command::new(forge);
    build_cmd.args(["build", "--root", root.to_str().unwrap()]);
    let build = run_with_timeout(build_cmd, forge_timeout("FORGE_BUILD_TIMEOUT_SECS", 90))
        .map_err(|e| format!("forge build failed (not installed or timed out): {e}"))?;

    if !build.status.success() {
        let stderr = String::from_utf8_lossy(&build.stderr).to_string();
        let stdout = String::from_utf8_lossy(&build.stdout).to_string();
        info!("  forge build: FAILED");
        return Ok(VerifyResponse {
            compiles: false,
            errors: collect_forge_errors(&stderr),
            gas_original: None,
            gas_optimized: None,
            gas_saved: None,
            forge_output: format!("{stdout}{stderr}"),
        });
    }
    info!("  forge build: OK");

    // ── test (gas measurement via Mantle fork) ─────────────────────────────────
    let ctor_args = synthesize_constructor_args(original);
    fs::write(
        root.join("test/GasCompare.t.sol"),
        build_gas_test(&orig_name, &opt_name, &ctor_args),
    )
    .map_err(|e| e.to_string())?;

    info!("  forge test: fork={}", mantle_rpc);
    let mut test_cmd = Command::new(forge);
    test_cmd.args([
        "test",
        "--root",
        root.to_str().unwrap(),
        "--fork-url",
        &mantle_rpc,
        "-vv",
    ]);
    let test_run = run_with_timeout(test_cmd, forge_timeout("FORGE_TEST_TIMEOUT_SECS", 240))
        .map_err(|e| format!("forge test failed or timed out: {e}"))?;

    let stdout = String::from_utf8_lossy(&test_run.stdout).to_string();
    let stderr = String::from_utf8_lossy(&test_run.stderr).to_string();

    let gas_original = parse_test_gas(&stdout, "test_original");
    let gas_optimized = parse_test_gas(&stdout, "test_optimized");
    let gas_saved = match (gas_original, gas_optimized) {
        (Some(b), Some(a)) => Some(b as i64 - a as i64),
        _ => None,
    };

    info!(
        "  gas original={:?} optimized={:?} saved={:?}",
        gas_original, gas_optimized, gas_saved
    );

    Ok(VerifyResponse {
        compiles: true,
        errors: vec![],
        gas_original,
        gas_optimized,
        gas_saved,
        forge_output: format!("{stdout}{stderr}"),
    })
}

// ── behavioral equivalence ──────────────────────────────────────────────────────
/// Result of a differential equivalence run: per-function PASS/FAIL plus the
/// (kept) construction-gas comparison.
///
/// Every generated test runs in TWO suites: `EquivalenceTest` (original `o` vs
/// optimized `p`) and `SanityTest` (original vs a second original). A test that
/// fails the sanity suite is a broken test (bad preconditions, wrong arithmetic),
/// not a finding — it is reported in `invalid` and excluded from gating, so a
/// buggy generated test can never falsely reject a good optimization.
///
/// A function is only PROVEN equivalent when it has a test that passed both
/// suites. `invalid` (broken test) and `missing` (no test at all) are both
/// *absence of proof*, not proof of correctness — [`EquivResult::unverified`]
/// unions them, and the caller must surface that set rather than silently
/// counting only the tests that happened to exist.
#[derive(Debug, Default)]
pub struct EquivResult {
    pub compiles: bool,
    pub errors: Vec<String>,
    /// True only when it compiled, at least one valid test ran, and every valid
    /// test passed. Says NOTHING about coverage — check `unverified()` too.
    pub all_passed: bool,
    /// Function names with a GENUINE behavioural divergence (sanity passed,
    /// equivalence failed).
    pub failed: Vec<String>,
    /// Function names whose test was itself broken (failed against original-vs-
    /// original) — excluded from gating.
    pub invalid: Vec<String>,
    /// For each broken test, the sanity-suite `[FAIL...]` line (revert reason) —
    /// fed back to the verify agent when regenerating the test.
    pub invalid_reasons: std::collections::HashMap<String, String>,
    /// Target functions for which no equivalence test was ever generated (the
    /// verify agent errored or returned an empty body). Filled in by the caller:
    /// `equivalence_inner` only ever sees the tests that DO exist, so it cannot
    /// know what is absent.
    pub missing: Vec<String>,
    /// Number of tests that were valid (passed sanity) and therefore counted.
    pub valid_count: usize,
    pub gas_original: Option<u64>,
    pub gas_optimized: Option<u64>,
    pub gas_saved: Option<i64>,
    /// Real per-function runtime gas (original vs optimized), from `--gas-report`.
    pub per_function_gas: Vec<FunctionGas>,
    pub forge_output: String,
}

impl EquivResult {
    /// Target functions with no valid passing equivalence test — broken tests
    /// (`invalid`) plus never-generated ones (`missing`). Both mean "we did not
    /// prove this function equivalent"; the two sets are disjoint by construction.
    pub fn unverified(&self) -> Vec<String> {
        let mut out = self.invalid.clone();
        out.extend(self.missing.iter().cloned());
        out.sort();
        out.dedup();
        out
    }
}

/// Assemble the differential test file. Two suites share the same generated
/// `test_eq_*` bodies (they reference only the `o`/`p` instance variables):
/// - `EquivalenceTest`: `o` = original, `p` = optimized — the real comparison, plus the
///   `test_gas_*` construction-gas pair.
/// - `SanityTest`: `o` and `p` are BOTH the original — a test that fails here is broken by
///   construction and must not gate the optimization.
fn build_equivalence_test(
    orig_name: &str,
    opt_name: &str,
    test_bodies: &[String],
    ctor_args: &str,
) -> String {
    let joined = test_bodies.join("\n\n");
    format!(
        "// SPDX-License-Identifier: MIT\n\
         pragma solidity ^0.8.0;\n\
         import \"../src/Original.sol\";\n\
         import \"../src/Optimized.sol\";\n\n\
         contract EquivalenceTest {{\n\
         \x20   {orig_name} o;\n\
         \x20   {opt_name} p;\n\
         \x20   function setUp() public {{ o = new {orig_name}({ctor_args}); p = new {opt_name}({ctor_args}); }}\n\
         \x20   function test_gas_original() external {{ new {orig_name}({ctor_args}); }}\n\
         \x20   function test_gas_optimized() external {{ new {opt_name}({ctor_args}); }}\n\n\
         {joined}\n\
         }}\n\n\
         contract SanityTest {{\n\
         \x20   {orig_name} o;\n\
         \x20   {orig_name} p;\n\
         \x20   function setUp() public {{ o = new {orig_name}({ctor_args}); p = new {orig_name}({ctor_args}); }}\n\n\
         {joined}\n\
         }}\n"
    )
}

/// Run differential equivalence (concurrency-bounded, on a worker thread).
/// `test_fns` is `(fn_name, test_function_body)` — the per-function `test_eq_*`
/// Solidity functions generated by the verify agent.
pub(crate) async fn run_equivalence_async(
    original: String,
    optimized: String,
    test_fns: Vec<(String, String)>,
) -> Result<EquivResult, String> {
    let _permit = forge_semaphore()
        .acquire()
        .await
        .map_err(|e| format!("forge semaphore closed: {e}"))?;
    tokio::task::spawn_blocking(move || run_equivalence(&original, &optimized, &test_fns))
        .await
        .map_err(|e| format!("forge task panicked: {e}"))?
}

fn run_equivalence(
    original: &str,
    optimized: &str,
    test_fns: &[(String, String)],
) -> Result<EquivResult, String> {
    let forge = forge_binary();
    let root = std::env::temp_dir().join(format!("gaslite_eq_{}", Uuid::new_v4()));
    let res = equivalence_inner(&forge, &root, original, optimized, test_fns);
    let _ = fs::remove_dir_all(&root);
    res
}

fn equivalence_inner(
    forge: &str,
    root: &Path,
    original: &str,
    optimized: &str,
    test_fns: &[(String, String)],
) -> Result<EquivResult, String> {
    let (orig_name, opt_name, mantle_rpc) = write_sandbox_project(root, original, optimized)?;

    // 1. Build the contracts ALONE first, so a compile failure here is unambiguously the optimized
    //    contract's fault (accurate rejection), not the generated tests'.
    let mut build_cmd = Command::new(forge);
    build_cmd.args(["build", "--root", root.to_str().unwrap()]);
    let build = run_with_timeout(build_cmd, forge_timeout("FORGE_BUILD_TIMEOUT_SECS", 90))
        .map_err(|e| format!("forge build failed (not installed or timed out): {e}"))?;
    if !build.status.success() {
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&build.stdout),
            String::from_utf8_lossy(&build.stderr)
        );
        info!("  forge equivalence: contract compile FAILED");
        return Ok(EquivResult {
            compiles: false,
            errors: collect_forge_errors(&combined),
            forge_output: combined,
            ..Default::default()
        });
    }

    // 2. Now add the generated tests. A compile failure from here on is the TESTS' fault → Err, so
    //    the caller reports "could not verify" instead of wrongly blaming the contract.
    let bodies: Vec<String> = test_fns.iter().map(|(_, body)| body.clone()).collect();
    let ctor_args = synthesize_constructor_args(original);
    fs::write(
        root.join("test/Equivalence.t.sol"),
        build_equivalence_test(&orig_name, &opt_name, &bodies, &ctor_args),
    )
    .map_err(|e| e.to_string())?;

    info!(
        "  forge equivalence: {} test(s) x2 suites, fork={}",
        test_fns.len(),
        mantle_rpc
    );
    let mut test_cmd = Command::new(forge);
    test_cmd.args([
        "test",
        "--root",
        root.to_str().unwrap(),
        "--fork-url",
        &mantle_rpc,
        // --gas-report adds a per-function gas table for BOTH contracts. Each
        // function is called identically on `o` (original) and `p` (optimized)
        // by the differential tests, so the avg columns are directly comparable.
        "--gas-report",
        "-vv",
    ]);
    let test_run = run_with_timeout(test_cmd, forge_timeout("FORGE_TEST_TIMEOUT_SECS", 240))
        .map_err(|e| format!("forge test failed or timed out: {e}"))?;

    let stdout = String::from_utf8_lossy(&test_run.stdout).to_string();
    let stderr = String::from_utf8_lossy(&test_run.stderr).to_string();
    let combined = format!("{stdout}{stderr}");

    // Contracts compiled in step 1, so no test results here means the GENERATED
    // TESTS broke compilation.
    let ran_tests = stdout.contains("[PASS]") || stdout.contains("[FAIL");
    if !ran_tests {
        info!("  forge equivalence: generated tests failed to compile");
        return Err(format!(
            "generated equivalence tests failed to compile: {}",
            collect_forge_errors(&combined).join("; ")
        ));
    }

    // 3. Classify per function from the per-suite results:
    //    - fails SanityTest (original vs original)        → broken test → invalid
    //    - passes SanityTest, fails EquivalenceTest        → genuine divergence
    //    - passes both                                     → equivalent
    let suites = suite_result_lines(&stdout);
    let empty: Vec<String> = Vec::new();
    let eq_lines = suites.get("EquivalenceTest").unwrap_or(&empty);
    let sanity_lines = suites.get("SanityTest").unwrap_or(&empty);

    let mut failed: Vec<String> = Vec::new();
    let mut invalid: Vec<String> = Vec::new();
    let mut invalid_reasons: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for (name, _) in test_fns {
        if !suite_test_passed(sanity_lines, name) {
            let needle = format!("test_eq_{name}(");
            let reason = sanity_lines
                .iter()
                .find(|l| l.starts_with("[FAIL") && l.contains(&needle))
                .cloned()
                .unwrap_or_else(|| "test produced no result".to_string());
            invalid_reasons.insert(name.clone(), reason);
            invalid.push(name.clone());
        } else if !suite_test_passed(eq_lines, name) {
            failed.push(name.clone());
        }
    }
    let valid_count = test_fns.len() - invalid.len();

    let gas_original = parse_test_gas(&stdout, "test_gas_original");
    let gas_optimized = parse_test_gas(&stdout, "test_gas_optimized");
    let gas_saved = match (gas_original, gas_optimized) {
        (Some(b), Some(a)) => Some(b as i64 - a as i64),
        _ => None,
    };

    // Real per-function runtime gas from the --gas-report table (original vs the
    // renamed optimized contract).
    let per_function_gas = parse_gas_report(&stdout, &orig_name, &opt_name);
    if !per_function_gas.is_empty() {
        info!(
            "  forge equivalence: per-function gas for {} function(s)",
            per_function_gas.len()
        );
    }

    let all_passed = failed.is_empty() && valid_count > 0;
    info!(
        "  forge equivalence: {} | valid {}/{} | genuine failures: {:?} | broken tests: {:?} | gas saved={:?}",
        if all_passed { "PASS" } else { "NOT PROVEN" },
        valid_count,
        test_fns.len(),
        failed,
        invalid,
        gas_saved
    );

    Ok(EquivResult {
        compiles: true,
        errors: vec![],
        all_passed,
        failed,
        invalid,
        invalid_reasons,
        // Only the caller knows the full target list; it fills this in.
        missing: Vec::new(),
        valid_count,
        gas_original,
        gas_optimized,
        gas_saved,
        per_function_gas,
        forge_output: combined,
    })
}

/// Split one gas-report line into trimmed, non-empty cells. Foundry has shipped
/// several table styles (old `│`/`┆` box-drawing, newer `|`/`+`); normalizing both
/// vertical separators to `|` makes the parser version-tolerant.
fn gas_report_cells(line: &str) -> Vec<String> {
    line.replace('┆', "|")
        .replace('│', "|")
        .split('|')
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect()
}

/// Parse `forge test --gas-report` into per-function avg gas for the original and
/// optimized contracts, paired by function name. Functions appear only if the
/// differential tests called them; getters called for assertions are included too —
/// the caller filters to the real optimization targets.
fn parse_gas_report(output: &str, orig_name: &str, opt_name: &str) -> Vec<FunctionGas> {
    use std::collections::HashMap;
    let mut orig: HashMap<String, u64> = HashMap::new();
    let mut opt: HashMap<String, u64> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    // Which contract's section we are currently inside (matched against the two
    // names we care about; other contracts are ignored).
    let mut current: Option<String> = None;

    for line in output.lines() {
        let cells = gas_report_cells(line);
        if cells.is_empty() {
            continue;
        }

        // Contract header row: a cell like "src/Original.sol:RewardPool contract".
        if let Some(h) = cells
            .iter()
            .find(|c| c.contains(".sol:") && c.ends_with("contract"))
        {
            let name = h
                .rsplit(':')
                .next()
                .unwrap_or("")
                .trim_end_matches("contract")
                .trim()
                .to_string();
            current = Some(name);
            continue;
        }

        // Function row: [name, min, avg, median, max, #calls]. Require the four
        // stat columns to be numeric so headers/borders/deployment rows are skipped.
        if cells.len() >= 6 && !cells[0].eq_ignore_ascii_case("Function Name") {
            let stats: Vec<Option<u64>> = cells[1..5]
                .iter()
                .map(|c| c.replace(',', "").parse::<u64>().ok())
                .collect();
            if stats.iter().all(|n| n.is_some()) {
                let fname = cells[0].clone();
                let avg = stats[1].unwrap(); // cells[2] = avg column
                match current.as_deref() {
                    Some(c) if c == orig_name => {
                        if !orig.contains_key(&fname) && !opt.contains_key(&fname) {
                            order.push(fname.clone());
                        }
                        orig.insert(fname, avg);
                    },
                    Some(c) if c == opt_name => {
                        if !orig.contains_key(&fname) && !opt.contains_key(&fname) {
                            order.push(fname.clone());
                        }
                        opt.insert(fname, avg);
                    },
                    _ => {},
                }
            }
        }
    }

    order
        .into_iter()
        .map(|name| {
            let go = orig.get(&name).copied();
            let gp = opt.get(&name).copied();
            let saved = match (go, gp) {
                (Some(b), Some(a)) => Some(b as i64 - a as i64),
                _ => None,
            };
            FunctionGas {
                name,
                gas_original: go,
                gas_optimized: gp,
                gas_saved: saved,
            }
        })
        .collect()
}

/// Group forge's `[PASS]`/`[FAIL...]` result lines by test suite. Suite headers
/// look like `Ran 5 tests for test/Equivalence.t.sol:EquivalenceTest`.
fn suite_result_lines(output: &str) -> std::collections::HashMap<String, Vec<String>> {
    let mut map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    let mut current: Option<String> = None;
    for line in output.lines() {
        if line.starts_with("Ran ")
            && let Some(rest) = line.split(" for ").nth(1)
            && let Some(suite) = rest.split(':').nth(1)
        {
            current = Some(suite.trim().to_string());
            continue;
        }
        if (line.starts_with("[PASS]") || line.starts_with("[FAIL"))
            && let Some(suite) = &current
        {
            map.entry(suite.clone()).or_default().push(line.to_string());
        }
    }
    map
}

/// Whether `test_eq_<name>` passed within one suite's result lines. Matches on
/// `test_eq_<name>(` so `transfer` cannot collide with `transferFrom`.
fn suite_test_passed(lines: &[String], fn_name: &str) -> bool {
    let needle = format!("test_eq_{fn_name}(");
    lines
        .iter()
        .any(|l| l.starts_with("[PASS]") && l.contains(&needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctor_args_none_when_no_constructor() {
        let src = "contract C { function f() public {} }";
        assert_eq!(synthesize_constructor_args(src), "");
    }

    #[test]
    fn ctor_args_empty_constructor() {
        let src = "contract C { constructor() { } }";
        assert_eq!(synthesize_constructor_args(src), "");
    }

    #[test]
    fn ctor_args_single_address() {
        let src = "contract C { constructor(address _token) { } }";
        assert_eq!(synthesize_constructor_args(src), "address(0)");
    }

    #[test]
    fn ctor_args_mixed_primitives() {
        let src =
            "contract C { constructor(address _o, uint256 _r, bool _b, string memory _n) { } }";
        assert_eq!(
            synthesize_constructor_args(src),
            "address(0), 0, false, \"\""
        );
    }

    #[test]
    fn ctor_args_array_and_payable_and_contract() {
        let src =
            "contract C { constructor(address[] memory xs, address payable to, IERC20 t) { } }";
        assert_eq!(
            synthesize_constructor_args(src),
            "new address[](0), payable(address(0)), IERC20(address(0))"
        );
    }

    #[test]
    fn ctor_args_fixed_bytes() {
        let src = "contract C { constructor(bytes32 root) { } }";
        assert_eq!(synthesize_constructor_args(src), "bytes32(0)");
    }

    #[test]
    fn ctor_keyword_in_identifier_is_ignored() {
        // No real constructor; a function whose name contains "constructor".
        let src = "contract C { function reconstructor() public {} }";
        assert_eq!(synthesize_constructor_args(src), "");
    }

    #[test]
    fn unverified_unions_broken_and_missing_tests() {
        let er = EquivResult {
            invalid: vec!["transfer".into()],
            missing: vec!["approve".into(), "mint".into()],
            ..Default::default()
        };
        assert_eq!(er.unverified(), vec!["approve", "mint", "transfer"]);
    }

    #[test]
    fn unverified_is_empty_when_every_function_was_tested() {
        assert!(EquivResult::default().unverified().is_empty());
    }

    #[test]
    fn gas_report_parses_per_function_avg() {
        let report = "\
| src/Original.sol:RewardPool contract |                 |       |        |       |         |
| Function Name                        | min             | avg   | median | max   | # calls |
| distribute                           | 12000           | 23456 | 23456  | 34000 | 2       |
| stake                                | 40000           | 41000 | 41000  | 42000 | 1       |
| src/Optimized.sol:RewardPoolOptimized contract |       |       |        |       |         |
| Function Name                        | min             | avg   | median | max   | # calls |
| distribute                           | 8000            | 15000 | 15000  | 22000 | 2       |
| stake                                | 30000           | 31000 | 31000  | 32000 | 1       |";
        let out = parse_gas_report(report, "RewardPool", "RewardPoolOptimized");
        let dist = out.iter().find(|f| f.name == "distribute").unwrap();
        assert_eq!(dist.gas_original, Some(23456));
        assert_eq!(dist.gas_optimized, Some(15000));
        assert_eq!(dist.gas_saved, Some(8456));
        let stake = out.iter().find(|f| f.name == "stake").unwrap();
        assert_eq!(stake.gas_saved, Some(10000));
    }
}
