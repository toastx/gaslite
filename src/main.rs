//TODO erc1155

mod ai;
mod db;
mod embedding;
mod forge;
mod logging;
mod normalize;
mod orchestrator;
mod retrieval;
mod rig_agent;
mod tools;
mod utils;
mod verify_agent;

use ai::Embedder;
use db::{Turso, TursoArg};
use embedding::FastembedAdapter;
use retrieval::GasliteIndex;

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use qdrant_client::{
    Payload, Qdrant,
    qdrant::{
        CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance, FieldType,
        PointStruct, UpsertPointsBuilder, VectorParamsBuilder,
    },
};
use rig_core::{client::ProviderClient, providers::deepseek};
use serde::{Deserialize, Serialize};
use solang_parser::pt::{ContractPart, FunctionTy, Loc, SourceUnitPart};
use std::{fs, path::Path, sync::Arc};
use tracing::{error, info, warn};
use uuid::Uuid;

// ── constants ─────────────────────────────────────────────────────────────────
pub const COLLECTION: &str = "gaslite_patterns";
const VECTOR_DIM: u64 = 384;
/// Max functions optimized concurrently (bounds in-flight DeepSeek requests).
const MAX_PARALLEL_FUNCS: usize = 6;
/// Contracts at or below BOTH thresholds skip the router LLM call — they are
/// always routed oneshot, so the round-trip is pure latency.
const ONESHOT_MAX_FUNCS: usize = 4;
const ONESHOT_MAX_BYTES: usize = 4096;

// ── app state ─────────────────────────────────────────────────────────────────
struct AppState {
    db: Arc<Turso>,
    qdrant: Arc<Qdrant>,
    deepseek: deepseek::Client,
    embedder: Arc<Embedder>,
    forge_available: bool,
    /// Result cache keyed on the *normalized* contract source (comments/whitespace
    /// stripped) — see `normalize::lexical_key`. A hit skips the whole agent +
    /// forge pipeline. Only successful/one-shot results are cached so transient
    /// failures can be retried. Cleared on Qdrant reset.
    cache: std::sync::Mutex<std::collections::HashMap<String, OptimizeResponse>>,
    /// Deterministic structural pattern matcher (the "Seeker"), rebuilt from the
    /// knowledge base on startup and after ingest/reset. Read-snapshotted per
    /// request so reads never block writes.
    pattern_matcher: std::sync::RwLock<Arc<normalize::PatternMatcher>>,
    /// Where finished runs are recorded (stubbed: `NoopSink` → tracing). This is the
    /// seam for on-chain Mantle logging — see [`logging`].
    logging: Arc<dyn logging::LoggingSink>,
}

/// L2 cache read — fetch a stored optimization from Turso by normalized key.
async fn db_cache_get(
    db: &Turso,
    key: &str,
) -> Option<OptimizeResponse> {
    let rows = db
        .query(
            "SELECT response FROM optimize_cache WHERE cache_key = ?",
            vec![TursoArg::Text(key.to_string())],
        )
        .await
        .ok()?;
    let json = rows
        .first()?
        .get("response")?
        .as_str()?;
    serde_json::from_str::<OptimizeResponse>(json).ok()
}

/// L2 cache write — persist an optimization to Turso (write-through).
async fn db_cache_put(
    db: &Turso,
    key: &str,
    resp: &OptimizeResponse,
) -> Result<(), String> {
    let json = serde_json::to_string(resp).map_err(|e| e.to_string())?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    db.execute(
        "INSERT OR REPLACE INTO optimize_cache (cache_key, response, created_at) VALUES (?,?,?)",
        vec![
            TursoArg::Text(key.to_string()),
            TursoArg::Text(json),
            TursoArg::Integer(now.to_string()),
        ],
    )
    .await
}

/// Load the structural pattern matcher from the knowledge base (Turso).
async fn load_pattern_matcher(db: &Turso) -> normalize::PatternMatcher {
    let rows = match db
        .query(
            "SELECT id, solidity_before FROM optimization_patterns WHERE solidity_before != ''",
            vec![],
        )
        .await
    {
        Ok(rows) => rows,
        Err(e) => {
            warn!("pattern matcher: KB query failed: {e}");
            return normalize::PatternMatcher::default();
        }
    };
    let pairs = rows
        .into_iter()
        .filter_map(|row| {
            let id = row
                .get("id")?
                .as_str()?
                .to_string();
            let before = row
                .get("solidity_before")?
                .as_str()?
                .to_string();
            Some((id, before))
        });
    normalize::PatternMatcher::build(pairs)
}

// ── DTOs ──────────────────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct OptimizeRequest {
    contract_source: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct OptimizeResponse {
    analysis: String,
    suggested_patterns: Vec<String>,
    optimized_code: String,
}

#[derive(Deserialize)]
struct IngestLocalRequest {
    directory_paths: Vec<String>,
}

#[derive(Serialize)]
struct IngestLocalResponse {
    successful_patterns: Vec<String>,
    failed_patterns: Vec<(String, String)>,
}

// ── entry point ───────────────────────────────────────────────────────────────
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let _ = rustls::crypto::ring::default_provider().install_default();

    let deepseek = deepseek::Client::from_env()
        .expect("DEEPSEEK_API_KEY required to build the rig DeepSeek client");
    let qdrant_api_key = std::env::var("QDRANT_API_KEY").expect("QDRANT_API_KEY required");
    let qdrant_url = std::env::var("QDRANT_CLUSTER_URL").expect("QDRANT_CLUSTER_URL required");
    let turso_url = std::env::var("TURSO_DATABASE_URL").expect("TURSO_DATABASE_URL required");
    let turso_token = std::env::var("TURSO_AUTH_TOKEN").expect("TURSO_AUTH_TOKEN required");

    let http = reqwest::Client::new();
    let embedder = Embedder::new()?;

    // qdrant
    let qdrant = Qdrant::from_url(&qdrant_url)
        .api_key(qdrant_api_key)
        .build()
        .expect("Failed to connect to Qdrant");

    let existing = qdrant
        .list_collections()
        .await
        .expect("Failed to list Qdrant collections");

    if !existing
        .collections
        .iter()
        .any(|c| c.name == COLLECTION)
    {
        qdrant
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION).vectors_config(VectorParamsBuilder::new(
                    VECTOR_DIM,
                    Distance::Cosine,
                )),
            )
            .await
            .expect("Failed to create Qdrant collection");
    }

    let forge_available = forge::forge_available();
    if forge_available {
        info!("forge detected — closed-loop refinement enabled");
    } else {
        warn!("forge not found — closed-loop refinement disabled (one-shot mode)");
    }

    let state = Arc::new(AppState {
        db: Arc::new(Turso::new(
            http,
            turso_url,
            turso_token,
        )),
        qdrant: Arc::new(qdrant),
        deepseek,
        embedder,
        forge_available,
        cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        pattern_matcher: std::sync::RwLock::new(Arc::new(
            normalize::PatternMatcher::default(),
        )),
        logging: Arc::new(logging::NoopSink),
    });

    // run migration via HTTP
    state
        .db
        .execute(
            "CREATE TABLE IF NOT EXISTS optimization_patterns (
                id                TEXT PRIMARY KEY,
                category          TEXT,
                version           TEXT,
                title             TEXT,
                source            TEXT,
                source_file       TEXT,
                difficulty        TEXT,
                mantle_specific   INTEGER,
                evm_version       TEXT,
                trigger_patterns  TEXT,
                solidity_before   TEXT,
                yul_optimized     TEXT,
                patterns_used     TEXT,
                explanation       TEXT,
                risk_level        TEXT,
                when_to_apply     TEXT,
                when_not_to_apply TEXT
            )",
            vec![],
        )
        .await
        .expect("Migration failed");

    // Durable result cache (survives restarts) — L2 behind the in-memory L1.
    state
        .db
        .execute(
            "CREATE TABLE IF NOT EXISTS optimize_cache (
                cache_key  TEXT PRIMARY KEY,
                response   TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )",
            vec![],
        )
        .await
        .expect("optimize_cache migration failed");

    create_qdrant_indexes(&state)
        .await
        .expect("Failed to create Qdrant indexes");

    // Build the structural "Seeker" matcher from the knowledge base.
    {
        let matcher = load_pattern_matcher(&state.db).await;
        if matcher.is_empty() {
            warn!("structural matcher: 0 templates (knowledge base empty — ingest patterns first)");
        } else {
            info!(
                "structural matcher: {} pattern templates loaded",
                matcher.len()
            );
        }
        *state
            .pattern_matcher
            .write()
            .unwrap() = Arc::new(matcher);
    }

    let router = Router::new()
        .route("/health", get(health_check))
        .route(
            "/api/optimize",
            post(optimize_contract),
        )
        .route(
            "/api/verify",
            post(forge::verify_contract),
        )
        .route(
            "/api/admin/ingest-local",
            post(ingest_local_files),
        )
        .route(
            "/api/admin/qdrant/reset",
            post(reset_collection),
        )
        .with_state(state);

    spawn_pinger();

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await?;
    info!(
        "Gaslite listening on {}",
        listener.local_addr()?
    );
    axum::serve(listener, router).await?;
    Ok(())
}

fn spawn_pinger() {
    const DEFAULT_URL: &str = "https://gaslite-analytics.onrender.com/api/status";
    const DEFAULT_INTERVAL_SECS: u64 = 300;

    let url = std::env::var("PING_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
    let interval_secs = std::env::var("PING_INTERVAL_SECS")
        .ok()
        .and_then(|v| {
            v.parse()
                .ok()
        })
        .unwrap_or(DEFAULT_INTERVAL_SECS);

    tokio::spawn(async move {
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                error!("pinger: failed to build HTTP client: {e}");
                return;
            }
        };

        info!("pinger: targeting {url} every {interval_secs}s");
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        loop {
            ticker
                .tick()
                .await;
            let started = std::time::Instant::now();
            match client
                .get(&url)
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status();
                    let ms = started
                        .elapsed()
                        .as_millis();
                    if status.is_success() {
                        info!("pinger: OK {status} in {ms}ms");
                    } else {
                        warn!("pinger: non-2xx {status} in {ms}ms");
                    }
                }
                Err(e) => warn!("pinger: request failed: {e}"),
            }
        }
    });
}

// ── handlers ──────────────────────────────────────────────────────────────────

// ── health check ──────────────────────────────────────────────────────────────
#[derive(Serialize)]
struct ComponentHealth {
    status: &'static str, // "ok" | "down"
    latency_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct HealthChecks {
    turso: ComponentHealth,
    qdrant: ComponentHealth,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str, // "ok" | "degraded"
    service: &'static str,
    server: &'static str, // "ok" — if this handler runs, the server is up
    checks: HealthChecks,
}

async fn health_check(
    State(state): State<Arc<AppState>>
) -> (
    axum::http::StatusCode,
    Json<HealthResponse>,
) {
    info!("GET /health");

    // Turso (structured store) — cheapest possible round-trip.
    let t = std::time::Instant::now();
    let turso = match state
        .db
        .query("SELECT 1", vec![])
        .await
    {
        Ok(_) => ComponentHealth {
            status: "ok",
            latency_ms: t
                .elapsed()
                .as_millis(),
            error: None,
        },
        Err(e) => {
            warn!("health: turso check failed: {e}");
            ComponentHealth {
                status: "down",
                latency_ms: t
                    .elapsed()
                    .as_millis(),
                error: Some(e),
            }
        }
    };

    // Qdrant (vector store) — listing collections is a lightweight connectivity probe.
    let q = std::time::Instant::now();
    let qdrant = match state
        .qdrant
        .list_collections()
        .await
    {
        Ok(_) => ComponentHealth {
            status: "ok",
            latency_ms: q
                .elapsed()
                .as_millis(),
            error: None,
        },
        Err(e) => {
            warn!("health: qdrant check failed: {e}");
            ComponentHealth {
                status: "down",
                latency_ms: q
                    .elapsed()
                    .as_millis(),
                error: Some(e.to_string()),
            }
        }
    };

    let healthy = turso.status == "ok" && qdrant.status == "ok";
    let code = if healthy {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    };

    (
        code,
        Json(HealthResponse {
            status: if healthy { "ok" } else { "degraded" },
            service: "gaslite",
            server: "ok",
            checks: HealthChecks { turso, qdrant },
        }),
    )
}

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

async fn optimize_contract(
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

// ── Contract analysis: category + per-function extraction + storage layout ──
// Each named function (with a body) is extracted with its exact source text and
// byte range, so we can optimize functions concurrently and splice the results
// back at their original offsets.
struct FunctionInfo {
    name: String,
    source: String,
    start: usize,
    end: usize,
}

/// A function header + body size, for the router's skeleton view.
struct FnSig {
    name: String,
    signature: String,
    size: usize,
}

/// The structural view of a parsed contract. `functions` + `storage_layout` drive
/// the per-function/oneshot agents; `file_decls` + `signatures` are the lightweight
/// "skeleton" the orchestrator routes on (no function bodies).
struct ContractSkeleton {
    category: Option<&'static str>,
    functions: Vec<FunctionInfo>,
    /// State-variable declarations, newline-joined (agent slot-derivation context).
    storage_layout: String,
    /// File-level declarations a function depends on to compile: structs, enums,
    /// custom errors, events, modifiers, user types. Injected into scoped agents so
    /// they don't reference or invent the wrong definitions.
    file_decls: String,
    /// Deterministic, per-contract storage-slot derivations (using `.slot` accessors),
    /// so the model uses THIS contract's actual layout instead of copying a retrieved
    /// pattern's (possibly packed/incompatible) slot scheme.
    slot_guide: String,
    signatures: Vec<FnSig>,
}

impl ContractSkeleton {
    /// Render the skeleton (signatures + sizes + decl summary) for the router — no
    /// function bodies, to keep the routing prompt small and its TTFT low.
    fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("FUNCTIONS (name — body bytes):\n");
        for s in &self.signatures {
            out.push_str(&format!(
                "- {} [{} bytes]: {}\n",
                s.name, s.size, s.signature
            ));
        }
        if !self
            .storage_layout
            .is_empty()
        {
            out.push_str("\nSTATE VARIABLES:\n");
            out.push_str(&self.storage_layout);
            out.push('\n');
        }
        if !self
            .file_decls
            .is_empty()
        {
            out.push_str("\nFILE-LEVEL DECLARATIONS:\n");
            out.push_str(&self.file_decls);
            out.push('\n');
        }
        out
    }
}

fn analyze_contract(source: &str) -> ContractSkeleton {
    let empty = || ContractSkeleton {
        category: detect_category_fallback(source),
        functions: vec![],
        storage_layout: String::new(),
        file_decls: String::new(),
        slot_guide: String::new(),
        signatures: vec![],
    };
    let Ok((su, _)) = solang_parser::parse(source, 0) else {
        return empty();
    };

    let mut category: Option<&'static str> = None;
    let mut functions: Vec<FunctionInfo> = Vec::new();
    let mut signatures: Vec<FnSig> = Vec::new();
    let mut storage_vars: Vec<String> = Vec::new();
    // (name, declaration_text) for state vars — drives the slot-derivation guide.
    let mut state_var_defs: Vec<(String, String)> = Vec::new();
    let mut decls: Vec<String> = Vec::new();

    // Append the source text spanned by `loc` to `decls` (trimmed).
    let push_decl = |loc: Loc, decls: &mut Vec<String>| {
        if let Loc::File(_, s, e) = loc
            && let Some(t) = source.get(s..e)
        {
            decls.push(
                t.trim()
                    .to_string(),
            );
        }
    };

    for part in su.0 {
        let SourceUnitPart::ContractDefinition(def) = part else {
            continue;
        };

        // Inheritance → category
        for base in &def.base {
            let base_name = base
                .name
                .identifiers
                .iter()
                .map(|id| {
                    id.name
                        .to_lowercase()
                })
                .collect::<Vec<_>>()
                .join(".");
            if category.is_none() {
                category = match base_name.as_str() {
                    s if s.contains("erc721") => Some("erc721"),
                    s if s.contains("erc1155") => Some("erc1155"),
                    s if s.contains("erc20") => Some("erc20"),
                    s if s.contains("erc2981") => Some("erc2981"),
                    _ => None,
                };
            }
        }

        for cp in &def.parts {
            match cp {
                ContractPart::VariableDefinition(var) => {
                    if let Loc::File(_, start, end) = var.loc
                        && let Some(text) = source.get(start..end)
                    {
                        let decl = text
                            .trim()
                            .to_string();
                        if let Some(name) = &var.name {
                            state_var_defs.push((
                                name.name
                                    .clone(),
                                decl.clone(),
                            ));
                        }
                        storage_vars.push(decl);
                    }
                }
                // File-level dependencies a function may reference — context only.
                ContractPart::StructDefinition(d) => push_decl(d.loc, &mut decls),
                ContractPart::EnumDefinition(d) => push_decl(d.loc, &mut decls),
                ContractPart::EventDefinition(d) => push_decl(d.loc, &mut decls),
                ContractPart::ErrorDefinition(d) => push_decl(d.loc, &mut decls),
                ContractPart::TypeDefinition(d) => push_decl(d.loc, &mut decls),
                ContractPart::FunctionDefinition(func) => {
                    // Modifiers are dependencies, not optimization targets.
                    if matches!(func.ty, FunctionTy::Modifier) {
                        push_decl(func.loc, &mut decls);
                        continue;
                    }
                    // Optimize only named, bodied `function`s (skip constructor/
                    // fallback/receive and abstract declarations).
                    if !matches!(func.ty, FunctionTy::Function) {
                        continue;
                    }
                    let Some(name_ident) = &func.name else {
                        continue;
                    };
                    let Some(_) = &func.body else { continue };
                    let Loc::File(_, start, end) = func.loc else {
                        continue;
                    };
                    let Some(func_text) = source.get(start..end) else {
                        continue;
                    };
                    // Signature = header up to the body's opening brace.
                    let signature = func_text
                        .split_once('{')
                        .map(|(h, _)| {
                            h.trim()
                                .to_string()
                        })
                        .unwrap_or_else(|| {
                            func_text
                                .trim()
                                .to_string()
                        });
                    signatures.push(FnSig {
                        name: name_ident
                            .name
                            .clone(),
                        signature,
                        size: end.saturating_sub(start),
                    });
                    functions.push(FunctionInfo {
                        name: name_ident
                            .name
                            .clone(),
                        source: func_text.to_string(),
                        start,
                        end,
                    });
                }
                _ => {}
            }
        }
    }

    ContractSkeleton {
        category,
        functions,
        storage_layout: storage_vars.join("\n"),
        file_decls: decls.join("\n\n"),
        slot_guide: build_slot_guide(&state_var_defs),
        signatures,
    }
}

/// Deterministic per-contract storage-slot derivations. Each state variable is
/// emitted with the EXACT inline-assembly recipe for its slot, using `.slot`
/// accessors (so solc resolves the slot index — no hardcoded numbers) and the
/// canonical `keccak256(0x00, 0x40)` mapping derivation. Mapping depth is read from
/// the declaration text. This is what stops a non-reasoning model from copying a
/// retrieved pattern's incompatible (e.g. packed ERC721A) slot scheme.
fn build_slot_guide(vars: &[(String, String)]) -> String {
    if vars.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "EXACT STORAGE SLOTS FOR THIS CONTRACT — derive every slot with these recipes \
         VERBATIM (use the `.slot` accessor; ignore any slot scheme from the retrieved patterns):\n",
    );
    for (name, decl) in vars {
        let depth = decl
            .matches("mapping(")
            .count();
        let line = match depth {
            0 => format!(
                "- {name} (value type): read sload({name}.slot), write sstore({name}.slot, v)"
            ),
            1 => format!(
                "- {name}[k]: mstore(0x00, k); mstore(0x20, {name}.slot); let s := keccak256(0x00, 0x40)"
            ),
            2 => format!(
                "- {name}[k1][k2]: mstore(0x00, k1); mstore(0x20, {name}.slot); let inner := keccak256(0x00, 0x40); mstore(0x00, k2); mstore(0x20, inner); let s := keccak256(0x00, 0x40)"
            ),
            _ => format!("- {name}: deep mapping — derive each level as keccak256(key ++ parentSlot)"),
        };
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn detect_category_fallback(source: &str) -> Option<&'static str> {
    let s = source.to_lowercase();
    if s.contains("is erc721") || s.contains(": erc721") {
        return Some("erc721");
    }
    if s.contains("is erc1155") || s.contains(": erc1155") {
        return Some("erc1155");
    }
    if s.contains("is erc2981") || s.contains(": erc2981") {
        return Some("erc2981");
    }
    if s.contains("is erc20") || s.contains(": erc20") {
        return Some("erc20");
    }
    None
}

async fn create_qdrant_indexes(state: &AppState) -> Result<(), String> {
    for field in ["category", "type"] {
        state
            .qdrant
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    COLLECTION,
                    field,
                    FieldType::Keyword,
                ),
            )
            .await
            .map_err(|e| format!("Failed to create Qdrant index on '{field}': {e}"))?;
        info!("  index created: {}", field);
    }
    Ok(())
}

async fn reset_collection(
    State(state): State<Arc<AppState>>
) -> Result<&'static str, (axum::http::StatusCode, String)> {
    info!("=== RESET COLLECTION ===");

    state
        .qdrant
        .delete_collection(COLLECTION)
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string(),
            )
        })?;
    info!("  deleted : {}", COLLECTION);

    state
        .qdrant
        .create_collection(
            CreateCollectionBuilder::new(COLLECTION).vectors_config(VectorParamsBuilder::new(
                VECTOR_DIM,
                Distance::Cosine,
            )),
        )
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string(),
            )
        })?;
    info!(
        "  created : {} ({} dims, cosine)",
        COLLECTION, VECTOR_DIM
    );

    create_qdrant_indexes(&state)
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                e,
            )
        })?;

    // The knowledge base changed — cached optimizations are now stale (L1 + L2).
    state
        .cache
        .lock()
        .unwrap()
        .clear();
    if let Err(e) = state
        .db
        .execute(
            "DELETE FROM optimize_cache",
            vec![],
        )
        .await
    {
        warn!("  cache   : L2 clear failed: {e}");
    }
    info!("  cache   : cleared (L1 + L2)");

    info!("========================");
    Ok("Collection reset successfully")
}

async fn ingest_local_files(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<IngestLocalRequest>,
) -> Result<Json<IngestLocalResponse>, (axum::http::StatusCode, String)> {
    let mut successful = Vec::new();
    let mut failed = Vec::new();

    info!("=== INGEST START ===");
    info!(
        "  directories: {}",
        payload
            .directory_paths
            .len()
    );

    for dir_path in payload.directory_paths {
        let dir = Path::new(&dir_path);

        if !dir.is_dir() {
            warn!(
                "  ! Not a directory: {}",
                dir_path
            );
            failed.push((
                dir_path,
                "Not a valid directory".to_string(),
            ));
            continue;
        }

        info!("  scanning: {}", dir_path);

        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                error!(
                    "  ! Cannot read directory {}: {}",
                    dir_path, e
                );
                failed.push((
                    dir_path,
                    format!("Cannot read directory: {e}"),
                ));
                continue;
            }
        };

        for entry in entries.flatten() {
            let file_path = entry.path();
            if file_path
                .extension()
                .and_then(|s| s.to_str())
                != Some("json")
            {
                continue;
            }

            let file_name = file_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned();

            let content = match fs::read_to_string(&file_path) {
                Ok(c) => c,
                Err(e) => {
                    warn!(
                        "    ! read error {}: {}",
                        file_name, e
                    );
                    failed.push((
                        file_name,
                        format!("Read error: {e}"),
                    ));
                    continue;
                }
            };

            let meta: serde_json::Value = match serde_json::from_str(&content) {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        "    ! invalid JSON {}: {}",
                        file_name, e
                    );
                    failed.push((
                        file_name,
                        format!("Invalid JSON: {e}"),
                    ));
                    continue;
                }
            };

            // Use clean let-else syntax for the critical ID extraction
            let Some(id) = meta["id"]
                .as_str()
                .map(String::from)
            else {
                failed.push((
                    file_name,
                    "Missing 'id' field".to_string(),
                ));
                continue;
            };

            // Extract core fields exactly once to clean up database injection
            let title = meta["title"]
                .as_str()
                .unwrap_or("title");
            let category = meta["category"]
                .as_str()
                .unwrap_or("general");
            let triggers = meta["trigger_patterns"].to_string();
            let sol_before = meta["solidity_before"]
                .as_str()
                .or(meta["pattern_before"].as_str())
                .or(meta["wrong_code"].as_str())
                .unwrap_or("");

            let entry_type = meta["type"]
                .as_str()
                .unwrap_or("pattern");

            let embed_text = if entry_type == "antipattern" {
                let wrong = meta["wrong_code"]
                    .as_str()
                    .unwrap_or("");
                let why = meta["why_wrong"]
                    .as_str()
                    .unwrap_or("");
                format!(
                    "TOKEN_STANDARD_NAMESPACE: {}\n\
                    // Antipattern to avoid: {}\n\
                    // Triggers: {}\n\
                    // Wrong code: {}\n\
                    // Why wrong: {}",
                    category.to_uppercase(),
                    title,
                    triggers,
                    wrong,
                    why
                )
            } else {
                // existing pattern embed text
                format!(
                    "TOKEN_STANDARD_NAMESPACE: {}\n// Optimization: {}\n// Keywords: {}\n{}",
                    category.to_uppercase(),
                    title,
                    triggers,
                    sol_before
                )
            };

            let vector = match state
                .embedder
                .clone()
                .embed(&embed_text)
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    failed.push((
                        id,
                        format!("Embedding error: {e}"),
                    ));
                    continue;
                }
            };

            // Turso SQL Insert
            let sql = "INSERT OR REPLACE INTO optimization_patterns \
                (id,category,version,title,source,source_file,difficulty,mantle_specific,\
                 evm_version,trigger_patterns,solidity_before,yul_optimized,patterns_used,\
                 explanation,risk_level,when_to_apply,when_not_to_apply) \
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";

            let args = vec![
                TursoArg::Text(id.clone()),
                TursoArg::Text(category.to_string()),
                TursoArg::Text(
                    meta["version"]
                        .as_str()
                        .unwrap_or("1.0")
                        .to_string(),
                ),
                TursoArg::Text(title.to_string()),
                TursoArg::Text(
                    meta["source"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                ),
                TursoArg::Text(
                    meta["source_file"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                ),
                TursoArg::Text(
                    meta["difficulty"]
                        .as_str()
                        .unwrap_or("medium")
                        .to_string(),
                ),
                TursoArg::Integer(
                    (meta["mantle_specific"]
                        .as_bool()
                        .unwrap_or(false) as i64)
                        .to_string(),
                ),
                TursoArg::Text(
                    meta["evm_version"]
                        .as_str()
                        .unwrap_or("paris")
                        .to_string(),
                ),
                TursoArg::Text(triggers),
                TursoArg::Text(sol_before.to_string()),
                TursoArg::Text(
                    meta["yul_optimized"]
                        .as_str()
                        .or(meta["pattern_after"].as_str())
                        .or(meta["correct_code"].as_str())
                        .unwrap_or("")
                        .to_string(),
                ),
                TursoArg::Text(meta["patterns_used"].to_string()),
                TursoArg::Text(meta["explanation"].to_string()),
                TursoArg::Text(
                    meta["risk_level"]
                        .as_str()
                        .unwrap_or("low")
                        .to_string(),
                ),
                TursoArg::Text(meta["when_to_apply"].to_string()),
                TursoArg::Text(
                    meta["when_not_to_apply"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                ),
            ];

            if let Err(e) = state
                .db
                .execute(sql, args)
                .await
            {
                failed.push((
                    id,
                    format!("Turso error: {e}"),
                ));
                continue;
            }

            // Clean Qdrant Payload Construction
            let qdrant_payload: Payload = serde_json::json!({
                "pattern_id": id.clone(),
                "category": category,
                "type": entry_type,
            })
            .try_into()
            .expect("Failed to parse JSON into Qdrant Payload");

            let point = PointStruct::new(
                Uuid::new_v4().to_string(),
                vector,
                qdrant_payload,
            );

            if let Err(e) = state
                .qdrant
                .upsert_points(UpsertPointsBuilder::new(
                    COLLECTION,
                    vec![point],
                ))
                .await
            {
                warn!(
                    "    ! Qdrant upsert failed {}: {}",
                    id, e
                );
                failed.push((
                    id,
                    format!("Qdrant error: {e}"),
                ));
                continue;
            }

            info!(
                "    + {} ({}, {})",
                id, category, entry_type
            );
            successful.push(id);
        }
    }

    info!("=== INGEST COMPLETE ===");
    info!(
        "  ok     : {}",
        successful.len()
    );
    info!("  failed : {}", failed.len());
    for (id, reason) in &failed {
        warn!("    ! {} — {}", id, reason);
    }
    info!("======================");

    // Refresh the structural matcher with the newly ingested patterns.
    {
        let matcher = load_pattern_matcher(&state.db).await;
        info!(
            "  structural matcher: {} templates (rebuilt)",
            matcher.len()
        );
        *state
            .pattern_matcher
            .write()
            .unwrap() = Arc::new(matcher);
    }

    Ok(Json(IngestLocalResponse {
        successful_patterns: successful,
        failed_patterns: failed,
    }))
}