//! Forge verification: compiles original vs. optimized contracts in a temp
//! sandbox and measures construction gas via a Mantle fork.

use axum::Json;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Read,
    path::Path,
    process::{Command, Output, Stdio},
    sync::OnceLock,
    time::{Duration, Instant},
};
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

// ── handler ───────────────────────────────────────────────────────────────────
pub async fn verify_contract(
    Json(payload): Json<VerifyRequest>
) -> Result<Json<VerifyResponse>, (axum::http::StatusCode, String)> {
    info!(
        "POST /api/verify — {} + {} bytes",
        payload
            .original_code
            .len(),
        payload
            .optimized_code
            .len()
    );
    run_forge_sandbox_async(
        payload.original_code,
        payload.optimized_code,
    )
    .await
    .map(Json)
    .map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            e,
        )
    })
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
            .and_then(|v| {
                v.parse::<usize>()
                    .ok()
            })
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
fn run_with_timeout(
    mut cmd: Command,
    timeout: Duration,
) -> std::io::Result<Output> {
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut out_pipe = child
        .stdout
        .take()
        .expect("piped stdout");
    let mut err_pipe = child
        .stderr
        .take()
        .expect("piped stderr");
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
                format!(
                    "forge subprocess timed out after {}s",
                    timeout.as_secs()
                ),
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    };

    Ok(Output {
        status,
        stdout: out_h
            .join()
            .unwrap_or_default(),
        stderr: err_h
            .join()
            .unwrap_or_default(),
    })
}

/// Timeout (seconds) for a forge step. `build` is local; `test` hits a Mantle fork.
fn forge_timeout(
    var: &str,
    default_secs: u64,
) -> Duration {
    let secs = std::env::var(var)
        .ok()
        .and_then(|v| {
            v.parse::<u64>()
                .ok()
        })
        .unwrap_or(default_secs);
    Duration::from_secs(secs)
}

/// Whether a usable `forge` binary is present — gates the closed-loop refinement.
pub(crate) fn forge_available() -> bool {
    std::process::Command::new(forge_binary())
        .arg("--version")
        .output()
        .map(|o| {
            o.status
                .success()
        })
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
        if let Some(rest) = line
            .trim()
            .strip_prefix("contract ")
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

fn build_gas_test(
    orig_name: &str,
    opt_name: &str,
) -> String {
    format!(
        "// SPDX-License-Identifier: MIT\n\
         pragma solidity ^0.8.0;\n\
         import \"../src/Original.sol\";\n\
         import \"../src/Optimized.sol\";\n\n\
         contract GasCompareTest {{\n\
             function test_original() external {{ new {orig_name}(); }}\n\
             function test_optimized() external {{ new {opt_name}(); }}\n\
         }}\n"
    )
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
        .map(|l| {
            l.trim()
                .to_string()
        })
        .filter(|l| !l.is_empty())
        .take(20)
        .collect()
}

fn parse_test_gas(
    output: &str,
    fn_suffix: &str,
) -> Option<u64> {
    for line in output.lines() {
        if line.contains(fn_suffix)
            && line.contains("gas:")
            && let Some(g) = line
                .split("gas:")
                .nth(1)
        {
            let s = g
                .trim()
                .trim_end_matches(')')
                .trim();
            if let Ok(n) = s.parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

pub(crate) fn run_forge_sandbox(
    original: &str,
    optimized: &str,
) -> Result<VerifyResponse, String> {
    let forge = forge_binary();
    let root = std::env::temp_dir().join(format!(
        "gaslite_{}",
        Uuid::new_v4()
    ));
    let res = forge_sandbox_inner(
        &forge, &root, original, optimized,
    );
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

    fs::write(
        root.join("src/Original.sol"),
        clean_for_forge(original),
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        root.join("src/Optimized.sol"),
        clean_for_forge(&opt_code),
    )
    .map_err(|e| e.to_string())?;

    let mantle_rpc =
        std::env::var("MANTLE_RPC_URL").unwrap_or_else(|_| "https://rpc.mantle.xyz".to_string());

    fs::write(
        root.join("foundry.toml"),
        format!(
            "[profile.default]\nsrc=\"src\"\ntest=\"test\"\nevm_version=\"paris\"\n\
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
    info!(
        "  forge build: {}",
        root.display()
    );
    let mut build_cmd = Command::new(forge);
    build_cmd.args([
        "build",
        "--root",
        root.to_str()
            .unwrap(),
    ]);
    let build = run_with_timeout(
        build_cmd,
        forge_timeout("FORGE_BUILD_TIMEOUT_SECS", 90),
    )
    .map_err(|e| format!("forge build failed (not installed or timed out): {e}"))?;

    if !build
        .status
        .success()
    {
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
    fs::write(
        root.join("test/GasCompare.t.sol"),
        build_gas_test(&orig_name, &opt_name),
    )
    .map_err(|e| e.to_string())?;

    info!(
        "  forge test: fork={}",
        mantle_rpc
    );
    let mut test_cmd = Command::new(forge);
    test_cmd.args([
        "test",
        "--root",
        root.to_str()
            .unwrap(),
        "--fork-url",
        &mantle_rpc,
        "-vv",
    ]);
    let test_run = run_with_timeout(
        test_cmd,
        forge_timeout("FORGE_TEST_TIMEOUT_SECS", 240),
    )
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
#[derive(Debug, Default)]
pub struct EquivResult {
    pub compiles: bool,
    pub errors: Vec<String>,
    /// True only when it compiled, at least one valid test ran, and every valid
    /// test passed.
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
    /// Number of tests that were valid (passed sanity) and therefore counted.
    pub valid_count: usize,
    pub gas_original: Option<u64>,
    pub gas_optimized: Option<u64>,
    pub gas_saved: Option<i64>,
    pub forge_output: String,
}

/// Assemble the differential test file. Two suites share the same generated
/// `test_eq_*` bodies (they reference only the `o`/`p` instance variables):
/// - `EquivalenceTest`: `o` = original, `p` = optimized — the real comparison,
///   plus the `test_gas_*` construction-gas pair.
/// - `SanityTest`: `o` and `p` are BOTH the original — a test that fails here is
///   broken by construction and must not gate the optimization.
fn build_equivalence_test(
    orig_name: &str,
    opt_name: &str,
    test_bodies: &[String],
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
         \x20   function setUp() public {{ o = new {orig_name}(); p = new {opt_name}(); }}\n\
         \x20   function test_gas_original() external {{ new {orig_name}(); }}\n\
         \x20   function test_gas_optimized() external {{ new {opt_name}(); }}\n\n\
         {joined}\n\
         }}\n\n\
         contract SanityTest {{\n\
         \x20   {orig_name} o;\n\
         \x20   {orig_name} p;\n\
         \x20   function setUp() public {{ o = new {orig_name}(); p = new {orig_name}(); }}\n\n\
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
    let root = std::env::temp_dir().join(format!(
        "gaslite_eq_{}",
        Uuid::new_v4()
    ));
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

    // 1. Build the contracts ALONE first, so a compile failure here is
    //    unambiguously the optimized contract's fault (accurate rejection), not
    //    the generated tests'.
    let mut build_cmd = Command::new(forge);
    build_cmd.args([
        "build",
        "--root",
        root.to_str()
            .unwrap(),
    ]);
    let build = run_with_timeout(
        build_cmd,
        forge_timeout("FORGE_BUILD_TIMEOUT_SECS", 90),
    )
    .map_err(|e| format!("forge build failed (not installed or timed out): {e}"))?;
    if !build
        .status
        .success()
    {
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

    // 2. Now add the generated tests. A compile failure from here on is the
    //    TESTS' fault → Err, so the caller reports "could not verify" instead of
    //    wrongly blaming the contract.
    let bodies: Vec<String> = test_fns
        .iter()
        .map(|(_, body)| body.clone())
        .collect();
    fs::write(
        root.join("test/Equivalence.t.sol"),
        build_equivalence_test(&orig_name, &opt_name, &bodies),
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
        root.to_str()
            .unwrap(),
        "--fork-url",
        &mantle_rpc,
        "-vv",
    ]);
    let test_run = run_with_timeout(
        test_cmd,
        forge_timeout("FORGE_TEST_TIMEOUT_SECS", 240),
    )
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
    let eq_lines = suites
        .get("EquivalenceTest")
        .unwrap_or(&empty);
    let sanity_lines = suites
        .get("SanityTest")
        .unwrap_or(&empty);

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
        valid_count,
        gas_original,
        gas_optimized,
        gas_saved,
        forge_output: combined,
    })
}

/// Group forge's `[PASS]`/`[FAIL...]` result lines by test suite. Suite headers
/// look like `Ran 5 tests for test/Equivalence.t.sol:EquivalenceTest`.
fn suite_result_lines(output: &str) -> std::collections::HashMap<String, Vec<String>> {
    let mut map: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    let mut current: Option<String> = None;
    for line in output.lines() {
        if line.starts_with("Ran ")
            && let Some(rest) = line
                .split(" for ")
                .nth(1)
            && let Some(suite) = rest
                .split(':')
                .nth(1)
        {
            current = Some(
                suite
                    .trim()
                    .to_string(),
            );
            continue;
        }
        if (line.starts_with("[PASS]") || line.starts_with("[FAIL"))
            && let Some(suite) = &current
        {
            map.entry(suite.clone())
                .or_default()
                .push(line.to_string());
        }
    }
    map
}

/// Whether `test_eq_<name>` passed within one suite's result lines. Matches on
/// `test_eq_<name>(` so `transfer` cannot collide with `transferFrom`.
fn suite_test_passed(
    lines: &[String],
    fn_name: &str,
) -> bool {
    let needle = format!("test_eq_{fn_name}(");
    lines
        .iter()
        .any(|l| l.starts_with("[PASS]") && l.contains(&needle))
}