//! Forge verification: compiles original vs. optimized contracts in a temp
//! sandbox and measures construction gas via a Mantle fork.

use axum::Json;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tracing::info;
use uuid::Uuid;

// ── DTOs ──────────────────────────────────────────────────────────────────────
#[derive(Deserialize)]
pub struct VerifyRequest {
    original_code: String,
    optimized_code: String,
}

#[derive(Serialize)]
pub struct VerifyResponse {
    compiles: bool,
    errors: Vec<String>,
    gas_original: Option<u64>,
    gas_optimized: Option<u64>,
    gas_saved: Option<i64>,
    forge_output: String,
}

// ── handler ───────────────────────────────────────────────────────────────────
pub async fn verify_contract(
    Json(payload): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, (axum::http::StatusCode, String)> {
    info!("POST /api/verify — {} + {} bytes", payload.original_code.len(), payload.optimized_code.len());
    tokio::task::spawn_blocking(move || run_forge_sandbox(&payload.original_code, &payload.optimized_code))
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map(Json)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))
}

// ── helpers ───────────────────────────────────────────────────────────────────
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

fn run_forge_sandbox(original: &str, optimized: &str) -> Result<VerifyResponse, String> {
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
    let build = std::process::Command::new(forge)
        .args(["build", "--root", root.to_str().unwrap()])
        .output()
        .map_err(|e| format!("forge not found — is Foundry installed? ({e})"))?;

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
    let test_run = std::process::Command::new(forge)
        .args(["test", "--root", root.to_str().unwrap(),
               "--fork-url", &mantle_rpc, "-vv"])
        .output()
        .map_err(|e| format!("forge test failed: {e}"))?;

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
