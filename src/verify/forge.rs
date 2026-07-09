//! Forge verification: compiles original vs. optimized contracts in a temp
//! sandbox and measures construction gas via a Mantle fork.

use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Read,
    path::Path,
    process::{Command, Output, Stdio},
    sync::OnceLock,
    time::{Duration, Instant},
};

use axum::Json;
use serde::{Deserialize, Serialize};
use solang_parser::pt::{
    CodeLocation, ContractPart, ContractTy, FunctionTy, Loc, SourceUnitPart, StructDefinition,
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

/// The contract the sandbox should instantiate and measure.
///
/// Interfaces, libraries and `abstract contract`s are never instantiable, so they
/// are excluded outright. Among the concrete contracts, one that another contract
/// inherits from is a base class, not the subject — so those are excluded too, which
/// leaves the *most-derived* contract. When several survive (independent helpers, no
/// inheritance) the last one wins: Solidity convention is dependencies first, subject
/// last. Falls back to a text scan when the source does not parse (LLM output can be
/// mangled).
pub(crate) fn extract_sol_contract_name(source: &str) -> Option<String> {
    let Ok((su, _)) = solang_parser::parse(source, 0) else {
        return scan_first_contract_name(source);
    };

    let contracts: Vec<&str> = su
        .0
        .iter()
        .filter_map(|part| match part {
            SourceUnitPart::ContractDefinition(def)
                if matches!(def.ty, ContractTy::Contract(_)) =>
            {
                def.name.as_ref().map(|n| n.name.as_str())
            },
            _ => None,
        })
        .collect();

    // Every name used as a base by some contract, at any inheritance depth.
    let bases: HashSet<&str> = su
        .0
        .iter()
        .filter_map(|part| match part {
            SourceUnitPart::ContractDefinition(def) => Some(&def.base),
            _ => None,
        })
        .flatten()
        .filter_map(|b| b.name.identifiers.last().map(|id| id.name.as_str()))
        .collect();

    contracts
        .iter()
        .rev()
        .find(|c| !bases.contains(*c))
        .or_else(|| contracts.last())
        .map(|c| (*c).to_string())
        .or_else(|| scan_first_contract_name(source))
}

/// Text fallback for sources that do not parse.
fn scan_first_contract_name(source: &str) -> Option<String> {
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

/// Rename a contract *declaration* by splicing over its name identifier, located via
/// the AST. A plain `source.replace("contract Foo", …)` is not safe here: it also
/// matches the prefix of `contract FooHelper`, and it can hit the phrase inside a
/// comment. Falls back to that replacement only when the source does not parse.
fn rename_contract(source: &str, from: &str, to: &str) -> String {
    if let Ok((su, _)) = solang_parser::parse(source, 0) {
        for part in &su.0 {
            if let SourceUnitPart::ContractDefinition(def) = part
                && let Some(name) = &def.name
                && name.name == from
                && let Loc::File(_, s, e) = name.loc
            {
                let mut out = String::with_capacity(source.len() + to.len());
                out.push_str(&source[..s]);
                out.push_str(to);
                out.push_str(&source[e..]);
                return out;
            }
        }
    }
    source.replacen(&format!("contract {from}"), &format!("contract {to}"), 1)
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
    // Enums/structs declared in the source, so a user-defined type resolves to a
    // valid literal (`Enum(0)`, `Struct(field defaults…)`) instead of a bogus
    // `Type(address(0))` cast that fails to compile.
    let types = collect_sol_types(source);
    match extract_constructor_params(source) {
        Some(params) => params
            .iter()
            .map(|t| default_literal_for_type(t, &types, 0))
            .collect::<Vec<_>>()
            .join(", "),
        None => String::new(),
    }
}

/// User-defined types declared in the contract source, so constructor-arg
/// synthesis can emit a compilable literal for enum/struct parameters.
#[derive(Default)]
struct SolTypes {
    /// Enum type names (any member → the `Enum(0)` int-conversion literal).
    enums: HashSet<String>,
    /// Struct name → its field TYPES (in declaration order), for `Struct(a, b, …)`.
    structs: HashMap<String, Vec<String>>,
}

/// Parse the source once and index every enum/struct declared at file or contract
/// level. Best-effort: a parse failure yields an empty registry (falls back to the
/// contract/interface literal path).
fn collect_sol_types(source: &str) -> SolTypes {
    let mut t = SolTypes::default();
    let Ok((su, _)) = solang_parser::parse(source, 0) else {
        return t;
    };
    for part in &su.0 {
        match part {
            SourceUnitPart::EnumDefinition(e) => {
                if let Some(n) = &e.name {
                    t.enums
                        .insert(n.name.clone());
                }
            }
            SourceUnitPart::StructDefinition(s) => collect_struct(source, s, &mut t),
            SourceUnitPart::ContractDefinition(def) => {
                for cp in &def.parts {
                    match cp {
                        ContractPart::EnumDefinition(e) => {
                            if let Some(n) = &e.name {
                                t.enums
                                    .insert(n.name.clone());
                            }
                        }
                        ContractPart::StructDefinition(s) => collect_struct(source, s, &mut t),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    t
}

/// Record a struct's field types (as source text) under its name.
fn collect_struct(
    source: &str,
    s: &StructDefinition,
    t: &mut SolTypes,
) {
    let Some(name) = &s.name else { return };
    let fields = s
        .fields
        .iter()
        .filter_map(|f| type_text_of(source, f.ty.loc()))
        .collect();
    t.structs
        .insert(name.name.clone(), fields);
}

/// The trimmed source text spanned by `loc` (a type expression), if it is a file span.
fn type_text_of(
    source: &str,
    loc: Loc,
) -> Option<String> {
    if let Loc::File(_, s, e) = loc {
        source
            .get(s..e)
            .map(|x| {
                x.trim()
                    .to_string()
            })
    } else {
        None
    }
}

/// Extract the constructor's parameter TYPES (data location + name stripped) from
/// the source via the AST. `None` = no constructor anywhere; `Some(vec![])` =
/// constructor with no params.
///
/// Working from the parsed tree (not a text scan) means a `constructor(...)`
/// mention inside a comment or string can never be misread as the real definition.
/// The target contract (the one the sandbox instantiates — see
/// [`extract_sol_contract_name`]) is preferred; any other contract's constructor is
/// a fallback, which is also the right choice when the target inherits its ctor.
fn extract_constructor_params(source: &str) -> Option<Vec<String>> {
    let (su, _) = solang_parser::parse(source, 0).ok()?;
    let target = extract_sol_contract_name(source);
    let mut fallback: Option<Vec<String>> = None;

    for part in &su.0 {
        let SourceUnitPart::ContractDefinition(def) = part else {
            continue;
        };
        let Some(ctor) = def
            .parts
            .iter()
            .find_map(|cp| {
                let ContractPart::FunctionDefinition(f) = cp else {
                    return None;
                };
                matches!(f.ty, FunctionTy::Constructor).then(|| {
                    f.params
                        .iter()
                        .filter_map(|(loc, p)| {
                            p.as_ref()?;
                            type_text_of(source, *loc).map(|txt| param_type(&txt))
                        })
                        .collect::<Vec<String>>()
                })
            })
        else {
            continue;
        };

        let is_target = matches!(
            (&def.name, &target),
            (Some(n), Some(t)) if &n.name == t
        );
        if is_target {
            return Some(ctor);
        }
        fallback.get_or_insert(ctor);
    }

    fallback
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

/// Bound on struct-nesting expansion, so a (pathological) recursive/deeply nested
/// struct can't blow the stack or emit an enormous literal.
const MAX_STRUCT_DEPTH: usize = 4;

/// Split a fixed-size array type `T[N]` into `(T, N)`. Dynamic arrays (`T[]`) and
/// non-arrays return `None`.
fn parse_fixed_array(t: &str) -> Option<(&str, usize)> {
    let open = t.rfind('[')?;
    let close = t
        .strip_suffix(']')?
        .len();
    let n: usize = t[open + 1..close]
        .trim()
        .parse()
        .ok()?;
    Some((t[..open].trim(), n))
}

/// A single element literal for a fixed-size array. Numeric elements are cast to
/// their declared type (`uint256(0)`) so the array literal infers the right element
/// type rather than defaulting to `uint8`.
fn array_element_literal(
    inner: &str,
    types: &SolTypes,
    depth: usize,
) -> String {
    let inner = inner.trim();
    // Guard on `[`: `uint256[2]` also starts with "uint", and `uint256[2](0)` is not
    // a cast — a nested array element must recurse into its own array literal.
    let is_plain_numeric =
        !inner.contains('[') && (inner.starts_with("uint") || inner.starts_with("int"));
    if is_plain_numeric {
        format!("{inner}(0)")
    } else {
        default_literal_for_type(inner, types, depth)
    }
}

/// A default literal for a Solidity type, used to fill a constructor call. Covers
/// the common primitives, arrays, enums, structs and contract/interface types;
/// falls back to `0`. `types` resolves user-defined enum/struct names; `depth`
/// bounds struct-field recursion.
fn default_literal_for_type(
    ty: &str,
    types: &SolTypes,
    depth: usize,
) -> String {
    let t = ty.trim();
    if let Some(inner) = t.strip_suffix("[]") {
        return format!(
            "new {}[](0)",
            inner.trim()
        );
    }
    // Fixed-size array `T[N]` → an N-element array literal of default elements.
    if let Some((inner, n)) = parse_fixed_array(t) {
        let elem = array_element_literal(inner, types, depth);
        return format!(
            "[{}]",
            vec![elem; n].join(", ")
        );
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
    // User-defined enum → explicit int-to-enum conversion (every enum has member 0).
    if types
        .enums
        .contains(t)
    {
        return format!("{t}(0)");
    }
    // User-defined struct → construct it from per-field default literals.
    if let Some(fields) = types
        .structs
        .get(t)
        && depth < MAX_STRUCT_DEPTH
    {
        let args = fields
            .iter()
            .map(|f| default_literal_for_type(f, types, depth + 1))
            .collect::<Vec<_>>()
            .join(", ");
        return format!("{t}({args})");
    }
    // Contract/interface types (PascalCase) — cast the zero address.
    if t.chars()
        .next()
        .is_some_and(|c| c.is_ascii_uppercase())
    {
        return format!("{t}(address(0))");
    }
    "0".to_string()
}

/// Drop ``` fence markers only. A fence line is never valid Solidity, so this is
/// safe to run over any source, including code the user wrote.
fn strip_fence_lines(code: &str) -> String {
    code.lines()
        .filter(|line| !line.trim().starts_with("```"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// [`strip_fence_lines`] plus the prose heuristics needed for LLM output: `**bold**`
/// lines, `*(italic notes)*`, and bullet-point explanations.
///
/// Only ever apply this to model-generated code. The heuristics are not sound over
/// Solidity — a wrapped expression can legitimately begin a line with `**` (the
/// exponent operator) or `- ` (subtraction), and deleting that line silently changes
/// the contract's meaning. The original contract is the behavioural oracle, so it is
/// written to the sandbox with fences stripped and nothing else touched.
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

    // Strip the model's markdown BEFORE renaming, so `rename_contract` sees source
    // that parses and can splice the name identifier precisely.
    let opt_clean = clean_for_forge(optimized);
    let opt_src_name = extract_sol_contract_name(&opt_clean).unwrap_or_else(|| orig_name.clone());
    // Rename optimized contract to avoid symbol collision with original.
    let opt_name = format!("{orig_name}Optimized");
    let opt_code = rename_contract(&opt_clean, &opt_src_name, &opt_name);

    // The original is user-supplied Solidity and the behavioural oracle — never run
    // the prose heuristics over it (see `clean_for_forge`).
    fs::write(root.join("src/Original.sol"), strip_fence_lines(original))
        .map_err(|e| e.to_string())?;
    fs::write(root.join("src/Optimized.sol"), opt_code).map_err(|e| e.to_string())?;

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
fn parse_gas_report(
    output: &str,
    orig_name: &str,
    opt_name: &str,
) -> Vec<FunctionGas> {
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
    fn ctor_keyword_in_comment_is_ignored() {
        // A `constructor(...)` mention in a comment must not be parsed as the real
        // definition — the AST only sees the actual constructor.
        let src = "contract C {\n // fake note: constructor(address evil)\n \
                   constructor(uint256 x) {} }";
        assert_eq!(synthesize_constructor_args(src), "0");
    }

    #[test]
    fn ctor_args_enum_uses_int_literal() {
        let src = "contract C { enum Status { Active, Paused } constructor(Status s) {} }";
        assert_eq!(
            synthesize_constructor_args(src),
            "Status(0)"
        );
    }

    #[test]
    fn ctor_args_struct_expands_to_field_defaults() {
        let src = "contract C { struct P { uint256 a; address b; bool c; } \
                   constructor(P memory p) {} }";
        assert_eq!(
            synthesize_constructor_args(src),
            "P(0, address(0), false)"
        );
    }

    #[test]
    fn ctor_args_fixed_size_array() {
        let src = "contract C { constructor(uint256[3] memory xs) {} }";
        assert_eq!(
            synthesize_constructor_args(src),
            "[uint256(0), uint256(0), uint256(0)]"
        );
    }

    #[test]
    fn ctor_args_nested_struct_and_enum() {
        let src = "contract C { enum E { A } struct S { E e; uint8 n; } \
                   constructor(S memory s) {} }";
        assert_eq!(
            synthesize_constructor_args(src),
            "S(E(0), 0)"
        );
    }

    #[test]
    fn ctor_args_nested_fixed_array_recurses_instead_of_casting() {
        // `uint256[2]` starts with "uint", but `uint256[2](0)` is not a cast.
        let src = "contract C { constructor(uint256[2][2] memory xs) {} }";
        assert_eq!(
            synthesize_constructor_args(src),
            "[[uint256(0), uint256(0)], [uint256(0), uint256(0)]]"
        );
    }

    // ── contract selection (C3): the sandbox must instantiate the SUBJECT ──────────
    #[test]
    fn contract_name_skips_helpers_and_takes_the_subject() {
        let src = "contract Helper { function h() public {} }\n\
                   contract Main { function m() public {} }";
        assert_eq!(extract_sol_contract_name(src).as_deref(), Some("Main"));
    }

    #[test]
    fn contract_name_skips_interfaces_libraries_and_abstract() {
        let src = "interface IThing { function t() external; }\n\
                   library L { function l() internal pure {} }\n\
                   abstract contract Base { function b() public virtual; }\n\
                   contract Impl is Base { function b() public override {} }";
        assert_eq!(extract_sol_contract_name(src).as_deref(), Some("Impl"));
    }

    #[test]
    fn contract_name_prefers_most_derived_even_when_base_declared_last() {
        // "last concrete contract" alone would wrongly pick ERC20 here.
        let src = "contract Token is ERC20 { function t() public {} }\n\
                   contract ERC20 { function e() public {} }";
        assert_eq!(extract_sol_contract_name(src).as_deref(), Some("Token"));
    }

    #[test]
    fn contract_name_falls_back_to_text_scan_when_source_does_not_parse() {
        let src = "contract Broken { function f( <<<not solidity>>> }";
        assert_eq!(extract_sol_contract_name(src).as_deref(), Some("Broken"));
    }

    // ── renaming (C3): must not match a prefix of a longer contract name ───────────
    #[test]
    fn rename_contract_does_not_hit_a_longer_prefixed_name() {
        let src = "contract MainHelper { function h() public {} }\n\
                   contract Main { function m() public {} }";
        let out = rename_contract(src, "Main", "MainOptimized");
        assert!(
            out.contains("contract MainHelper {"),
            "helper was renamed: {out}"
        );
        assert!(
            out.contains("contract MainOptimized {"),
            "subject not renamed: {out}"
        );
    }

    // ── cleaning (C3): prose heuristics must not touch user Solidity ───────────────
    #[test]
    fn strip_fence_lines_preserves_solidity_that_looks_like_prose() {
        // `**` is exponentiation and `- ` is subtraction on a wrapped line.
        let src = "contract C {\n  uint256 a = 2\n    ** 8;\n  uint256 b = x\n    - Y;\n}";
        assert_eq!(strip_fence_lines(src), src);
    }

    #[test]
    fn clean_for_forge_would_have_corrupted_that_same_source() {
        // Documents exactly why the original is no longer run through this.
        let src = "contract C {\n  uint256 a = 2\n    ** 8;\n  uint256 b = x\n    - Y;\n}";
        assert_ne!(clean_for_forge(src), src);
    }

    #[test]
    fn strip_fence_lines_still_drops_fences() {
        assert_eq!(
            strip_fence_lines("```solidity\ncontract C {}\n```"),
            "contract C {}"
        );
    }

    #[test]
    fn clean_for_forge_drops_model_prose() {
        let src = "**Optimized version:**\ncontract C {}\n- Uses custom errors\n*(note)*";
        assert_eq!(clean_for_forge(src), "contract C {}");
    }

    // ── verification coverage (C1) ────────────────────────────────────────────────
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
