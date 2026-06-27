//! `POST /api/optimize` — the core pipeline: cache lookup, parse, route
//! (oneshot/decompose/fallback), optimize functions concurrently, then gate the
//! result on behavioural equivalence + a proven construction-gas win.

use std::sync::Arc;

use axum::{Json, extract::State};
use tracing::{info, warn};

use crate::analyze::{FunctionInfo, analyze_contract};
use crate::dto::{OptimizeRequest, OptimizeResponse};
use crate::embedding::FastembedAdapter;
use crate::retrieval::GasliteIndex;
use crate::state::{
    AppState, MAX_PARALLEL_FUNCS, ONESHOT_MAX_BYTES, ONESHOT_MAX_FUNCS, db_cache_get, db_cache_put,
};
use crate::{forge, logging, normalize, orchestrator, rig_agent, utils, verify_agent};

/// Per-function optimization result: `(start, end, original_fn, optimized, pattern_ids)`.
type FnOptResult = (
    usize,
    usize,
    String,
    Result<String, String>,
    Vec<String>,
);

/// Render an optional gas figure for user-facing strings ("n/a" when absent).
fn fmt_gas(g: Option<u64>) -> String {
    g.map_or_else(
        || "n/a".to_string(),
        |v| v.to_string(),
    )
}

/// Non-cryptographic identity hash of the contract source, for the run log.
fn hash_source(src: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Optimize a set of functions concurrently (one scoped agent each, bounded by a
/// semaphore) and splice the accepted rewrites back into the original source.
/// Returns `(optimized_code, optimized_count, deduped_pattern_ids)`. Shared by the
/// decompose and fallback paths.
async fn fan_out_functions(
    state: &Arc<AppState>,
    functions: Vec<FunctionInfo>,
    original: Arc<str>,
    storage: Arc<str>,
    file_decls: Arc<str>,
    category: Option<&'static str>,
    original_source: &str,
) -> (String, usize, Vec<String>) {
    let sem = Arc::new(tokio::sync::Semaphore::new(
        MAX_PARALLEL_FUNCS,
    ));
    let mut set: tokio::task::JoinSet<FnOptResult> = tokio::task::JoinSet::new();
    for func in functions {
        let state = state.clone();
        let permit_sem = sem.clone();
        let original = original.clone();
        let storage = storage.clone();
        let file_decls = file_decls.clone();
        let FunctionInfo {
            name,
            source: fsrc,
            start,
            end,
        } = func;
        set.spawn(async move {
            let _permit = permit_sem
                .acquire()
                .await
                .expect("semaphore closed");
            let adapter = FastembedAdapter::new(
                state
                    .embedder
                    .clone(),
            );
            let matcher = state
                .pattern_matcher
                .read()
                .unwrap()
                .clone();
            let index = GasliteIndex::new(
                state
                    .qdrant
                    .clone(),
                state
                    .db
                    .clone(),
                adapter,
                category,
                fsrc.clone(),
                matcher,
                name.clone(),
            );
            let pattern_ids = index
                .pattern_ids()
                .await
                .unwrap_or_default();
            let optimized = rig_agent::optimize_function(
                &state.deepseek,
                index,
                &storage,
                &file_decls,
                original.clone(),
                &name,
                &fsrc,
                start,
                end,
                state.forge_available,
            )
            .await;
            (
                start,
                end,
                fsrc,
                optimized,
                pattern_ids,
            )
        });
    }

    let mut results = Vec::new();
    while let Some(joined) = set
        .join_next()
        .await
    {
        match joined {
            Ok(tuple) => results.push(tuple),
            Err(e) => warn!("  ! function task panicked: {e}"),
        }
    }

    // Splice descending by start offset so earlier replacements don't shift later ones.
    results.sort_by(|a, b| {
        b.0.cmp(&a.0)
    });
    let mut optimized_code = original_source.to_string();
    let mut optimized_count = 0usize;
    let mut all_patterns: Vec<String> = Vec::new();
    for (start, end, fsrc, optimized, pattern_ids) in &results {
        all_patterns.extend(
            pattern_ids
                .iter()
                .cloned(),
        );
        match optimized {
            Ok(opt) => {
                let opt = utils::strip_code_fences(opt);
                if &opt != fsrc && *end <= optimized_code.len() {
                    optimized_code.replace_range(*start..*end, &opt);
                    optimized_count += 1;
                }
            }
            Err(e) => warn!("  ! {e}"),
        }
    }
    all_patterns.sort();
    all_patterns.dedup();
    (
        optimized_code,
        optimized_count,
        all_patterns,
    )
}

/// Behavioural verification: generate a differential equivalence test per function
/// (one thread each), then run them all in one forge harness. The optimized contract
/// is accepted only if it compiles AND every function behaves identically to the
/// original — which construction-gas measurement alone cannot prove.
/// How many times broken (sanity-failing) tests are regenerated with feedback
/// before we give up and report those functions as unverified.
const VERIFY_REGEN_ROUNDS: usize = 1;

/// Generate `test_eq_*` bodies for `targets` concurrently (one task each).
/// `feedback` maps a function name to `(previous test source, sanity failure)`
/// for regeneration rounds. Returns `(name, body)` pairs for the successes.
async fn gen_equiv_tests(
    state: &Arc<AppState>,
    original_source: &str,
    storage_layout: &str,
    orig_type: &str,
    opt_type: &str,
    targets: &[(String, String)],
    feedback: &std::collections::HashMap<String, (String, String)>,
) -> Vec<(String, String)> {
    let sem = Arc::new(tokio::sync::Semaphore::new(MAX_PARALLEL_FUNCS));
    let mut set: tokio::task::JoinSet<(String, Result<String, String>)> =
        tokio::task::JoinSet::new();
    for (name, sig) in targets {
        let state = state.clone();
        let permit_sem = sem.clone();
        let original_source = original_source.to_string();
        let storage = storage_layout.to_string();
        let orig_type = orig_type.to_string();
        let opt_type = opt_type.to_string();
        let name = name.clone();
        let sig = sig.clone();
        let prev = feedback
            .get(&name)
            .cloned();
        set.spawn(async move {
            let _permit = permit_sem
                .acquire()
                .await
                .expect("semaphore closed");
            let body = verify_agent::gen_equivalence_test(
                &state.deepseek,
                &original_source,
                &storage,
                &orig_type,
                &opt_type,
                &name,
                &sig,
                prev.as_ref()
                    .map(|(c, f)| (c.as_str(), f.as_str())),
            )
            .await;
            (name, body)
        });
    }

    let mut out: Vec<(String, String)> = Vec::new();
    while let Some(joined) = set
        .join_next()
        .await
    {
        match joined {
            Ok((name, Ok(body))) if !body
                .trim()
                .is_empty() =>
            {
                out.push((name, body))
            }
            Ok((name, Ok(_))) => warn!("  ! verify-test gen produced empty body for {name}"),
            Ok((name, Err(e))) => warn!("  ! verify-test gen failed for {name}: {e}"),
            Err(e) => warn!("  ! verify-test task panicked: {e}"),
        }
    }
    out
}

async fn behavioral_verify(
    state: &Arc<AppState>,
    original_source: &str,
    optimized_code: &str,
    targets: &[(String, String)],
    storage_layout: &str,
    // Pre-generated `test_eq_*` bodies — produced concurrently with the
    // optimization itself (they depend only on the original contract).
    mut test_fns: Vec<(String, String)>,
) -> Result<forge::EquivResult, String> {
    let orig_type = forge::extract_sol_contract_name(original_source)
        .unwrap_or_else(|| "OriginalContract".to_string());
    let opt_type = format!("{orig_type}Optimized");

    // 1. Fallback: if the concurrent pre-generation produced nothing (task failed
    //    or returned empty), generate here so verification can still proceed.
    if test_fns.is_empty() {
        test_fns = gen_equiv_tests(
            state,
            original_source,
            storage_layout,
            &orig_type,
            &opt_type,
            targets,
            &std::collections::HashMap::new(),
        )
        .await;
    }
    if test_fns.is_empty() {
        return Err("no equivalence tests could be generated".to_string());
    }

    // 2. Run the differential harness (build + all test_eq_* on a Mantle fork).
    let mut er = forge::run_equivalence_async(
        original_source.to_string(),
        optimized_code.to_string(),
        test_fns.clone(),
    )
    .await?;

    // 3. Tests that failed the original-vs-original sanity suite are bugs in the
    //    TEST. Regenerate just those, feeding each agent its own broken test plus
    //    the failure line, and re-run. A failed regen round keeps the previous
    //    result, so retrying can only improve coverage, never lose it.
    for round in 1..=VERIFY_REGEN_ROUNDS {
        if er
            .invalid
            .is_empty()
        {
            break;
        }
        info!(
            "  verify: regenerating {} broken test(s) with failure feedback (round {round})",
            er.invalid
                .len()
        );

        let mut feedback: std::collections::HashMap<String, (String, String)> =
            std::collections::HashMap::new();
        for name in &er.invalid {
            let prev_body = test_fns
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, b)| b.clone())
                .unwrap_or_default();
            let reason = er
                .invalid_reasons
                .get(name)
                .cloned()
                .unwrap_or_default();
            feedback.insert(name.clone(), (prev_body, reason));
        }
        let regen_targets: Vec<(String, String)> = targets
            .iter()
            .filter(|(n, _)| {
                er.invalid
                    .contains(n)
            })
            .cloned()
            .collect();

        let regenerated = gen_equiv_tests(
            state,
            original_source,
            storage_layout,
            &orig_type,
            &opt_type,
            &regen_targets,
            &feedback,
        )
        .await;
        if regenerated.is_empty() {
            warn!("  verify: regen produced no tests — keeping previous result");
            break;
        }
        for (name, body) in regenerated {
            if let Some(slot) = test_fns
                .iter_mut()
                .find(|(n, _)| *n == name)
            {
                slot.1 = body;
            }
        }

        match forge::run_equivalence_async(
            original_source.to_string(),
            optimized_code.to_string(),
            test_fns.clone(),
        )
        .await
        {
            Ok(new_er) => er = new_er,
            Err(e) => {
                warn!("  verify: regen round failed ({e}) — keeping previous result");
                break;
            }
        }
    }

    Ok(er)
}

pub(crate) async fn optimize_contract(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<OptimizeRequest>,
) -> Result<Json<OptimizeResponse>, (axum::http::StatusCode, String)> {
    let t0 = std::time::Instant::now();

    // 0. Result cache — keyed on the NORMALIZED source (comments/whitespace stripped), so
    //    formatting-only differences still hit. L1 = in-memory, L2 = Turso (durable across
    //    restarts).
    let cache_key = normalize::lexical_key(&payload.contract_source);
    if let Some(hit) = state
        .cache
        .lock()
        .unwrap()
        .get(&cache_key)
        .cloned()
    {
        info!(
            "optimize: cache HIT (L1 memory) → returned in {:.2?}",
            t0.elapsed()
        );
        return Ok(Json(hit));
    }
    if let Some(hit) = db_cache_get(&state.db, &cache_key).await {
        info!(
            "optimize: cache HIT (L2 turso) → returned in {:.2?}",
            t0.elapsed()
        );
        // Warm L1 so subsequent hits are instant.
        state
            .cache
            .lock()
            .unwrap()
            .insert(cache_key.clone(), hit.clone());
        return Ok(Json(hit));
    }

    // 1. Parse the contract into its skeleton: category, functions, storage, decls.
    let skeleton = analyze_contract(&payload.contract_source);
    let category = skeleton.category;
    let t_parse = std::time::Instant::now();
    let category_str = category.unwrap_or("general");

    info!("=== OPTIMIZE REQUEST ===");
    info!(
        "  contract : {} bytes",
        payload
            .contract_source
            .len()
    );
    info!(
        "  detected : {}",
        category_str
    );
    info!(
        "  functions: {}",
        skeleton
            .functions
            .len()
    );
    info!(
        "  forge    : {}",
        if state.forge_available {
            "closed-loop"
        } else {
            "one-shot"
        }
    );
    info!("========================");

    if skeleton
        .functions
        .is_empty()
    {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "No optimizable functions found — ensure the contract parses correctly".to_string(),
        ));
    }

    // Shared inputs for whichever optimization path the router picks.
    let original: Arc<str> = Arc::from(
        payload
            .contract_source
            .as_str(),
    );
    // Storage context for the optimization agents = raw declarations + the
    // deterministic per-contract slot-derivation guide, so the model uses THIS
    // layout's slots rather than a retrieved pattern's incompatible scheme.
    let storage: Arc<str> = Arc::from(
        format!(
            "{}\n\n{}",
            skeleton.storage_layout, skeleton.slot_guide
        )
        .as_str(),
    );
    let file_decls: Arc<str> = Arc::from(
        skeleton
            .file_decls
            .as_str(),
    );
    let skeleton_text = skeleton.render();
    // Captured before `functions` is moved into the routing arms — used by the
    // behavioural verifier to generate a differential test per function.
    let verify_targets: Vec<(String, String)> = skeleton
        .signatures
        .iter()
        .map(|s| {
            (
                s.name
                    .clone(),
                s.signature
                    .clone(),
            )
        })
        .collect();
    let all_functions = skeleton.functions;

    // 2. Start generating the equivalence tests NOW, concurrently with the
    //    optimization itself: they are derived from the ORIGINAL contract only, so
    //    the verify agents' LLM calls overlap the optimizer's instead of running
    //    serially after it. The handle is joined (or aborted) at the verify stage.
    let pregen_tests: Option<tokio::task::JoinHandle<Vec<(String, String)>>> =
        if state.forge_available {
            let state = state.clone();
            let original_source = payload
                .contract_source
                .clone();
            let storage = storage.to_string();
            let targets = verify_targets.clone();
            Some(tokio::spawn(async move {
                let orig_type = forge::extract_sol_contract_name(&original_source)
                    .unwrap_or_else(|| "OriginalContract".to_string());
                let opt_type = format!("{orig_type}Optimized");
                gen_equiv_tests(
                    &state,
                    &original_source,
                    &storage,
                    &orig_type,
                    &opt_type,
                    &targets,
                    &std::collections::HashMap::new(),
                )
                .await
            }))
        } else {
            None
        };

    // 3. Route. Small contracts skip the router LLM call entirely — the answer is
    //    always oneshot, so a deterministic gate saves the round-trip and removes a
    //    failure surface. Bigger contracts get the orchestrator decision; any
    //    routing failure falls back to full per-function fan-out, so robustness
    //    never regresses.
    let mode: &'static str;
    let mut optimized_code: String;
    let suggested_patterns: Vec<String>;
    let route = if all_functions.len() <= ONESHOT_MAX_FUNCS
        && payload
            .contract_source
            .len()
            <= ONESHOT_MAX_BYTES
    {
        info!("  router: oneshot (heuristic — small contract, no LLM call)");
        Ok(orchestrator::Route::Oneshot)
    } else {
        orchestrator::route(
            &state.deepseek,
            &skeleton_text,
        )
        .await
    };
    match route {
        Ok(orchestrator::Route::Oneshot) => {
            mode = "oneshot";
            info!("=== OPTIMIZING WHOLE CONTRACT (one-shot) ===");
            let adapter = FastembedAdapter::new(
                state
                    .embedder
                    .clone(),
            );
            let matcher = state
                .pattern_matcher
                .read()
                .unwrap()
                .clone();
            let index = GasliteIndex::new(
                state
                    .qdrant
                    .clone(),
                state
                    .db
                    .clone(),
                adapter,
                category,
                payload
                    .contract_source
                    .clone(),
                matcher,
                "oneshot",
            );
            let mut pattern_ids = index
                .pattern_ids()
                .await
                .unwrap_or_default();
            optimized_code = match rig_agent::optimize_oneshot(
                &state.deepseek,
                index,
                &storage,
                &file_decls,
                &payload.contract_source,
            )
            .await
            {
                Ok(c) => utils::strip_code_fences(&c).to_string(),
                Err(e) => {
                    warn!("  ! one-shot failed: {e} — keeping original");
                    payload
                        .contract_source
                        .clone()
                }
            };
            pattern_ids.sort();
            pattern_ids.dedup();
            suggested_patterns = pattern_ids;
        }
        Ok(orchestrator::Route::Decompose(tasks)) => {
            mode = "decompose";
            let mut wanted: Vec<String> = tasks
                .iter()
                .flat_map(|t| {
                    t.target_fns
                        .iter()
                        .cloned()
                })
                .collect();
            wanted.sort();
            wanted.dedup();
            let selected: Vec<FunctionInfo> = if all_functions
                .iter()
                .any(|f| wanted.contains(&f.name))
            {
                all_functions
                    .into_iter()
                    .filter(|f| wanted.contains(&f.name))
                    .collect()
            } else {
                warn!("  router named no known functions — fanning out all");
                all_functions
            };
            info!(
                "=== OPTIMIZING {} FUNCTION(S) (decompose) ===",
                selected.len()
            );
            let (code, count, patterns) = fan_out_functions(
                &state,
                selected,
                original.clone(),
                storage.clone(),
                file_decls.clone(),
                category,
                &payload.contract_source,
            )
            .await;
            info!("  functions optimized: {count}");
            optimized_code = code;
            suggested_patterns = patterns;
        }
        Err(e) => {
            mode = "fallback";
            warn!("  router failed ({e}) — falling back to per-function fan-out");
            info!(
                "=== OPTIMIZING {} FUNCTIONS (fallback fan-out) ===",
                all_functions.len()
            );
            let (code, count, patterns) = fan_out_functions(
                &state,
                all_functions,
                original.clone(),
                storage.clone(),
                file_decls.clone(),
                category,
                &payload.contract_source,
            )
            .await;
            info!("  functions optimized: {count}");
            optimized_code = code;
            suggested_patterns = patterns;
        }
    }
    let t_agent = std::time::Instant::now();

    // 4. Final authoritative gate: behavioural equivalence (differential tests vs
    //    the original on a Mantle fork) + a proven construction-gas win.
    let analysis: String;
    // Whether the result is worth caching: a real optimization or a clean
    // one-shot. Transient failures (compile error, regression, forge error) are
    // NOT cached, so an identical request can be retried.
    let cacheable: bool;
    // Gas figures captured for the run log (set only when forge measured them).
    let mut run_gas_original: Option<u64> = None;
    let mut run_gas_optimized: Option<u64> = None;
    let mut run_gas_saved: Option<i64> = None;
    // Real per-function runtime gas (set only on an accepted, forge-verified rewrite).
    let mut run_fn_gas: Vec<forge::FunctionGas> = Vec::new();
    if optimized_code == payload.contract_source {
        // No rewrite was produced (agent failure / nothing changed) — verifying
        // the original against itself would waste ~seconds of forge + LLM time.
        if let Some(h) = pregen_tests {
            h.abort();
        }
        warn!("  verify: skipped — no rewrite produced, returning original");
        analysis = "No optimized rewrite produced — original returned unchanged.".to_string();
        cacheable = false;
    } else if state.forge_available {
        // Join the tests that were generated concurrently with the optimization.
        let test_fns: Vec<(String, String)> = match pregen_tests {
            Some(h) => h
                .await
                .unwrap_or_else(|e| {
                    warn!("  ! verify-test pregen task failed: {e}");
                    Vec::new()
                }),
            None => Vec::new(),
        };
        match behavioral_verify(
            &state,
            &payload.contract_source,
            &optimized_code,
            &verify_targets,
            storage.as_ref(),
            test_fns,
        )
        .await
        {
            // Behaviourally equivalent AND a construction-gas win → accept.
            Ok(er)
                if er.compiles
                    && er.all_passed
                    && er
                        .gas_saved
                        .unwrap_or(0)
                        > 0 =>
            {
                let saved = er
                    .gas_saved
                    .unwrap_or(0);
                run_gas_original = er.gas_original;
                run_gas_optimized = er.gas_optimized;
                run_gas_saved = er.gas_saved;
                // Keep only the real optimization targets (drop getters the
                // differential tests called for assertions).
                let targets: std::collections::HashSet<&str> = verify_targets
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect();
                run_fn_gas = er
                    .per_function_gas
                    .iter()
                    .filter(|f| {
                        targets.contains(
                            f.name
                                .as_str(),
                        )
                    })
                    .cloned()
                    .collect();
                if !er
                    .invalid
                    .is_empty()
                {
                    warn!(
                        "  verify: {:?} had broken tests (failed sanity) — those functions are UNVERIFIED",
                        er.invalid
                    );
                }
                info!(
                    "  verify ACCEPTED: {}/{} equivalence test(s) passed | construction gas {} → {} (saved {})",
                    er.valid_count,
                    verify_targets.len(),
                    fmt_gas(er.gas_original),
                    fmt_gas(er.gas_optimized),
                    saved
                );
                analysis = format!(
                    "Behaviourally equivalent to the original on {} differential test(s) on a \
                     Mantle fork{}. Construction gas {} → {} (saved {}).",
                    er.valid_count,
                    if er
                        .invalid
                        .is_empty()
                    {
                        String::new()
                    } else {
                        format!(
                            " (unverified — test generation failed: {})",
                            er.invalid
                                .join(", ")
                        )
                    },
                    fmt_gas(er.gas_original),
                    fmt_gas(er.gas_optimized),
                    saved
                );
                cacheable = true;
            }
            // Equivalent but no construction-gas win → keep original.
            Ok(er) if er.compiles && er.all_passed => {
                warn!("  verify: equivalent but no gas improvement — keeping original");
                optimized_code = payload
                    .contract_source
                    .clone();
                analysis = "Rewrite rejected — behaviourally equivalent but no construction-gas \
                     improvement. Kept original."
                    .to_string();
                cacheable = false;
            }
            // Compiled but a genuine behavioural mismatch (the test passed against
            // original-vs-original, so the divergence is real) → reject.
            Ok(er) if er.compiles && !er.failed.is_empty() => {
                warn!(
                    "  verify: BEHAVIOURAL MISMATCH in {:?} (broken tests excluded: {:?}) — keeping original",
                    er.failed, er.invalid
                );
                warn!(
                    "  verify forge output (truncated):\n{}",
                    er.forge_output
                        .chars()
                        .take(1500)
                        .collect::<String>()
                );
                optimized_code = payload
                    .contract_source
                    .clone();
                analysis = format!(
                    "Rewrite rejected — behavioural mismatch vs original in: {}. Kept original.",
                    er.failed
                        .join(", ")
                );
                cacheable = false;
            }
            // Compiled, no genuine failures, but no valid test ran either (every
            // generated test was broken) → unverified, don't ship.
            Ok(er) if er.compiles => {
                warn!(
                    "  verify: no valid equivalence tests (all broken: {:?}) — keeping original",
                    er.invalid
                );
                optimized_code = payload
                    .contract_source
                    .clone();
                analysis = "Rewrite rejected — equivalence could not be established (test \
                     generation produced no valid tests). Kept original."
                    .to_string();
                cacheable = false;
            }
            // Did not compile → keep original.
            Ok(er) => {
                warn!("  verify: optimized did not compile — keeping original");
                optimized_code = payload
                    .contract_source
                    .clone();
                analysis = format!(
                    "Rewrite rejected — did not compile. Kept original. Errors: {}",
                    er.errors
                        .join("; ")
                );
                cacheable = false;
            }
            // Could not run verification at all → don't ship.
            Err(e) => {
                warn!("  verify failed: {e} — keeping original (could not verify)");
                optimized_code = payload
                    .contract_source
                    .clone();
                analysis = format!("Rewrite rejected — could not verify ({e}). Kept original.");
                cacheable = false;
            }
        }
    } else {
        analysis = "Optimized one-shot — forge unavailable, not verified.".to_string();
        cacheable = true;
    }
    let t_verify = std::time::Instant::now();

    info!("=== OPTIMIZE COMPLETE ===");
    info!("  mode     : {}", mode);
    info!(
        "  patterns : {}",
        suggested_patterns.len()
    );
    info!("  cached   : {}", cacheable);
    info!(
        "  timing   : parse {:.2?} | route+agents {:.2?} | final-verify {:.2?}",
        t_parse - t0,
        t_agent - t_parse,
        t_verify - t_agent,
    );
    info!(
        "  total    : {:.2?}",
        t0.elapsed()
    );
    info!("=========================");

    // Record the run (stub sink → tracing; the seam for on-chain Mantle logging).
    let run = logging::RunLog {
        contract_hash: hash_source(&payload.contract_source),
        mode,
        gas_original: run_gas_original,
        gas_optimized: run_gas_optimized,
        gas_saved: run_gas_saved,
        pattern_ids: suggested_patterns.clone(),
        ts: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    if let Err(e) = state
        .logging
        .log_run(&run)
        .await
    {
        warn!("  run-log sink failed: {e}");
    }

    let response = OptimizeResponse {
        analysis,
        suggested_patterns,
        optimized_code,
        gas_before: run_gas_original,
        gas_after: run_gas_optimized,
        gas_saved: run_gas_saved,
        per_function_gas: run_fn_gas,
    };

    if cacheable {
        // L1: in-memory, bounded so a flood of distinct inputs can't grow it.
        {
            let mut cache = state
                .cache
                .lock()
                .unwrap();
            if cache.len() < 1024 {
                cache.insert(
                    cache_key.clone(),
                    response.clone(),
                );
            }
        }
        // L2: Turso (durable). Best-effort — a write failure doesn't fail the request.
        if let Err(e) = db_cache_put(
            &state.db, &cache_key, &response,
        )
        .await
        {
            warn!("cache: L2 turso write failed: {e}");
        }
    }

    Ok(Json(response))
}
