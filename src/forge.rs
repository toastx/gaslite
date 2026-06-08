//! Forge verification: compiles original vs. optimized contracts in a temp
//! sandbox and measures construction gas via a Mantle fork.

use axum::Json;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
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
    Json(payload): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, (axum::http::StatusCode, String)> {
    info!("POST /api/verify — {} + {} bytes", payload.original_code.len(), payload.optimized_code.len());
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
            warn!("  forge: killing subprocess after {}s timeout", timeout.as_secs());
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
        if Path::new(&p).exists() { return p; }
    }
    "forge".to_string()
}

fn extract_sol_contract_name(source: &str) -> Option<String> {
    for line in source.lines() {
        if let Some(rest) = line.trim().strip_prefix("contract ") {
            if let Some(name) = rest.split(|c: char| !c.is_alphanumeric() && c != '_').next() {
                if !name.is_empty() { return Some(name.to_string()); }
            }
        }
    }
    None
}

fn build_gas_test(orig_name: &str, opt_name: &str) -> String {
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
            if t.starts_with("```") { return false; }
            if t.starts_with("**") { return false; }
            if t.starts_with("*(") { return false; }
            // Bullet points that start with an uppercase word are English prose, not Solidity
            if let Some(rest) = t.strip_prefix("- ") {
                if rest.trim().chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    return false;
                }
            }
            true
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn collect_forge_errors(stderr: &str) -> Vec<String> {
    stderr.lines()
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
        if line.contains(fn_suffix) && line.contains("gas:") {
            if let Some(g) = line.split("gas:").nth(1) {
                let s = g.trim().trim_end_matches(')').trim();
                if let Ok(n) = s.parse::<u64>() { return Some(n); }
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

fn forge_sandbox_inner(forge: &str, root: &Path, original: &str, optimized: &str) -> Result<VerifyResponse, String> {
    fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    fs::create_dir_all(root.join("test")).map_err(|e| e.to_string())?;

    let orig_name = extract_sol_contract_name(original).unwrap_or_else(|| "OriginalContract".to_string());
    let opt_src_name = extract_sol_contract_name(optimized).unwrap_or_else(|| orig_name.clone());
    // Rename optimized contract to avoid symbol collision with original
    let opt_name = format!("{orig_name}Optimized");
    let opt_code = optimized.replacen(
        &format!("contract {opt_src_name}"),
        &format!("contract {opt_name}"),
        1,
    );

    let original_clean = clean_for_forge(original);
    let opt_code_clean = clean_for_forge(&opt_code);

    fs::write(root.join("src/Original.sol"), &original_clean).map_err(|e| e.to_string())?;
    fs::write(root.join("src/Optimized.sol"), &opt_code_clean).map_err(|e| e.to_string())?;

    let mantle_rpc = std::env::var("MANTLE_RPC_URL")
        .unwrap_or_else(|_| "https://rpc.mantle.xyz".to_string());

    fs::write(
        root.join("foundry.toml"),
        format!("[profile.default]\nsrc=\"src\"\ntest=\"test\"\nevm_version=\"paris\"\n\
                 [rpc_endpoints]\nmantle=\"{mantle_rpc}\"\n"),
    ).map_err(|e| e.to_string())?;

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
    fs::write(root.join("test/GasCompare.t.sol"), build_gas_test(&orig_name, &opt_name))
        .map_err(|e| e.to_string())?;

    info!("  forge test: fork={}", mantle_rpc);
    let mut test_cmd = Command::new(forge);
    test_cmd.args(["test", "--root", root.to_str().unwrap(),
                   "--fork-url", &mantle_rpc, "-vv"]);
    let test_run = run_with_timeout(test_cmd, forge_timeout("FORGE_TEST_TIMEOUT_SECS", 240))
        .map_err(|e| format!("forge test failed or timed out: {e}"))?;

    let stdout = String::from_utf8_lossy(&test_run.stdout).to_string();
    let stderr = String::from_utf8_lossy(&test_run.stderr).to_string();

    let gas_original  = parse_test_gas(&stdout, "test_original");
    let gas_optimized = parse_test_gas(&stdout, "test_optimized");
    let gas_saved = match (gas_original, gas_optimized) {
        (Some(b), Some(a)) => Some(b as i64 - a as i64),
        _ => None,
    };

    info!("  gas original={:?} optimized={:?} saved={:?}", gas_original, gas_optimized, gas_saved);

    Ok(VerifyResponse {
        compiles: true,
        errors: vec![],
        gas_original,
        gas_optimized,
        gas_saved,
        forge_output: format!("{stdout}{stderr}"),
    })
}
