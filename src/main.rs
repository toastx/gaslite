//TODO erc1155

mod ai;
mod db;
mod embedding;
mod forge;
mod normalize;
mod retrieval;
mod rig_agent;
mod tools;
mod utils;

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
use solang_parser::pt::{ContractPart, Loc, SourceUnitPart};
use std::{fs, path::Path, sync::Arc};
use tracing::{error, info, warn};
use uuid::Uuid;

// ── constants ─────────────────────────────────────────────────────────────────
pub const COLLECTION: &str = "gaslite_patterns";
const VECTOR_DIM: u64 = 384;
/// Max functions optimized concurrently (bounds in-flight DeepSeek requests).
const MAX_PARALLEL_FUNCS: usize = 6;

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

    // 1. Parse the contract: detect category, extract functions, storage layout.
    let (category, functions, storage_layout) = analyze_contract(&payload.contract_source);
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
        functions.len()
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

    if functions.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "No optimizable functions found — ensure the contract parses correctly".to_string(),
        ));
    }

    // 2. Optimize every function concurrently — one rig agent per function, bounded by a semaphore.
    //    Each agent does its own retrieval and (when forge is available) its own compile loop
    //    against the original contract.
    info!(
        "=== OPTIMIZING {} FUNCTIONS CONCURRENTLY ===",
        functions.len()
    );
    let original: Arc<str> = Arc::from(
        payload
            .contract_source
            .as_str(),
    );
    let storage: Arc<str> = Arc::from(storage_layout.as_str());
    let sem = Arc::new(tokio::sync::Semaphore::new(
        MAX_PARALLEL_FUNCS,
    ));

    let mut set: tokio::task::JoinSet<FnOptResult> = tokio::task::JoinSet::new();
    for func in functions {
        let state = state.clone();
        let permit_sem = sem.clone();
        let original = original.clone();
        let storage = storage.clone();
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
    let t_agent = std::time::Instant::now();

    // 3. Splice optimized functions back (descending start keeps offsets valid), and aggregate
    //    pattern ids for the response.
    results.sort_by(|a, b| {
        b.0.cmp(&a.0)
    });
    let mut optimized_code = payload
        .contract_source
        .clone();
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
    let suggested_patterns = all_patterns;
    info!(
        "  functions optimized: {}/{}",
        optimized_count,
        results.len()
    );

    // 4. Final authoritative forge check. We only return the rewrite when it compiles AND
    //    demonstrably saves construction gas; otherwise we keep the original. Note this proves
    //    "compiles + cheaper constructor", NOT behavioural equivalence — the sandbox does not test
    //    runtime semantics.
    let analysis: String;
    // Whether the result is worth caching: a real optimization or a clean
    // one-shot. Transient failures (compile error, regression, forge error) are
    // NOT cached, so an identical request can be retried.
    let cacheable: bool;
    if state.forge_available {
        match forge::run_forge_sandbox_async(
            payload
                .contract_source
                .clone(),
            optimized_code.clone(),
        )
        .await
        {
            // Accept only on a proven gas improvement.
            Ok(vr)
                if vr.compiles
                    && vr
                        .gas_saved
                        .unwrap_or(0)
                        > 0 =>
            {
                let saved = vr
                    .gas_saved
                    .unwrap_or(0);
                info!(
                    "  forge accepted: original={} optimized={} saved={}",
                    fmt_gas(vr.gas_original),
                    fmt_gas(vr.gas_optimized),
                    saved
                );
                analysis = format!(
                    "Compiled on a Mantle fork; construction gas {} → {} (saved {}). \
                     Behavioural equivalence is not tested.",
                    fmt_gas(vr.gas_original),
                    fmt_gas(vr.gas_optimized),
                    saved
                );
                cacheable = true;
            }
            // Compiles but no proven improvement (regression, zero, or unmeasured) → keep original.
            Ok(vr) if vr.compiles => {
                warn!(
                    "  forge: no proven gas improvement (saved={:?}) — keeping original",
                    vr.gas_saved
                );
                optimized_code = payload
                    .contract_source
                    .clone();
                analysis = match vr.gas_saved {
                    Some(s) => format!(
                        "Rewrite rejected — no gas improvement (construction gas saved {s}). Kept original."
                    ),
                    None => {
                        "Rewrite rejected — construction gas could not be measured. Kept original."
                            .to_string()
                    }
                };
                cacheable = false;
            }
            // Did not compile → keep original.
            Ok(vr) => {
                warn!("  forge: optimized did not compile — keeping original");
                optimized_code = payload
                    .contract_source
                    .clone();
                analysis = format!(
                    "Rewrite rejected — did not compile. Kept original. Errors: {}",
                    vr.errors
                        .join("; ")
                );
                cacheable = false;
            }
            // Forge errored, timed out, or panicked — can't verify, so don't ship it.
            Err(e) => {
                warn!("  forge check failed: {e} — keeping original (could not verify)");
                optimized_code = payload
                    .contract_source
                    .clone();
                analysis =
                    format!("Rewrite rejected — could not verify (forge: {e}). Kept original.");
                cacheable = false;
            }
        }
    } else {
        analysis = "Optimized one-shot — forge unavailable, not verified.".to_string();
        cacheable = true;
    }
    let t_verify = std::time::Instant::now();

    info!("=== OPTIMIZE COMPLETE ===");
    info!(
        "  patterns : {}",
        suggested_patterns.len()
    );
    info!("  cached   : {}", cacheable);
    info!(
        "  timing   : parse {:.2?} | functions {:.2?} | final-verify {:.2?}",
        t_parse - t0,
        t_agent - t_parse,
        t_verify - t_agent,
    );
    info!(
        "  total    : {:.2?}",
        t0.elapsed()
    );
    info!("=========================");

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

fn analyze_contract(
    source: &str
) -> (
    Option<&'static str>,
    Vec<FunctionInfo>,
    String,
) {
    let Ok((su, _)) = solang_parser::parse(source, 0) else {
        return (
            detect_category_fallback(source),
            vec![],
            String::new(),
        );
    };

    let mut category: Option<&'static str> = None;
    let mut functions: Vec<FunctionInfo> = Vec::new();
    let mut storage_vars: Vec<String> = Vec::new();

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
                        storage_vars.push(
                            text.trim()
                                .to_string(),
                        );
                    }
                }
                ContractPart::FunctionDefinition(func) => {
                    // Extract only named functions with a body (skip constructor/
                    // fallback/receive and abstract declarations).
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

    (
        category,
        functions,
        storage_vars.join("\n"),
    )
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
