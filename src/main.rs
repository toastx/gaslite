//TODO erc1155

mod ai;
mod db;
mod forge;
mod utils;

use ai::Embedder;
use db::{Turso, TursoArg};

use tracing::{info, warn, error};
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use qdrant_client::{Payload, qdrant::{
    Condition, CreateCollectionBuilder, Distance, Filter, PointStruct, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder
}};
use qdrant_client::qdrant::{CreateFieldIndexCollectionBuilder, FieldType};
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;
use solang_parser::pt::{ContractPart, Loc, SourceUnitPart};
use async_openai::{
    Client as OpenAIClient,
    config::OpenAIConfig,
};

// ── constants ─────────────────────────────────────────────────────────────────
const COLLECTION: &str = "gaslite_patterns";
const VECTOR_DIM: u64 = 384;

// ── app state ─────────────────────────────────────────────────────────────────
struct AppState {
    db: Turso,
    qdrant: Qdrant,
    deepseek: OpenAIClient<OpenAIConfig>,
    embedder: Arc<Embedder>,
}

// ── DTOs ──────────────────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct OptimizeRequest {
    contract_source: String,
}

#[derive(Serialize)]
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

// ── per-function analysis ─────────────────────────────────────────────────────
struct FunctionInfo {
    name: String,
    source: String,  // exact source text extracted via byte offsets
    start: usize,    // byte offset in original contract source
    end: usize,
    ops: Vec<String>,
}

// ── entry point ───────────────────────────────────────────────────────────────
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
        )
        .init();

    let _ = rustls::crypto::ring::default_provider().install_default();

    let deepseek_api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY required");
    let deepseek_base = std::env::var("DEEPSEEK_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com/v1".to_string());
    let deepseek = OpenAIClient::with_config(
        OpenAIConfig::new()
            .with_api_base(deepseek_base)
            .with_api_key(&deepseek_api_key),
    );
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

    if !existing.collections.iter().any(|c| c.name == COLLECTION) {
        qdrant
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION)
                    .vectors_config(VectorParamsBuilder::new(VECTOR_DIM, Distance::Cosine)),
            )
            .await
            .expect("Failed to create Qdrant collection");
    }

    let state = Arc::new(AppState {
        db: Turso::new(http, turso_url, turso_token),
        qdrant,
        deepseek,
        embedder,
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

    create_qdrant_indexes(&state)
        .await
        .expect("Failed to create Qdrant indexes");

    let router = Router::new()
        .route("/health", get(health_check))
        .route("/api/optimize", post(optimize_contract))
        .route("/api/verify", post(forge::verify_contract))
        .route("/api/admin/ingest-local", post(ingest_local_files))
        .route("/api/admin/qdrant/reset", post(reset_collection))
        .with_state(state);

    spawn_pinger();

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await?;
    info!("Gaslite listening on {}", listener.local_addr()?);
    axum::serve(listener, router).await?;
    Ok(())
}

fn spawn_pinger() {
    const DEFAULT_URL: &str = "https://gaslite-analytics.onrender.com/api/status";
    const DEFAULT_INTERVAL_SECS: u64 = 300;

    let url = std::env::var("PING_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
    let interval_secs = std::env::var("PING_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
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
            ticker.tick().await;
            let started = std::time::Instant::now();
            match client.get(&url).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    let ms = started.elapsed().as_millis();
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
    State(state): State<Arc<AppState>>,
) -> (axum::http::StatusCode, Json<HealthResponse>) {
    info!("GET /health");

    // Turso (structured store) — cheapest possible round-trip.
    let t = std::time::Instant::now();
    let turso = match state.db.query("SELECT 1", vec![]).await {
        Ok(_) => ComponentHealth { status: "ok", latency_ms: t.elapsed().as_millis(), error: None },
        Err(e) => {
            warn!("health: turso check failed: {e}");
            ComponentHealth { status: "down", latency_ms: t.elapsed().as_millis(), error: Some(e) }
        }
    };

    // Qdrant (vector store) — listing collections is a lightweight connectivity probe.
    let q = std::time::Instant::now();
    let qdrant = match state.qdrant.list_collections().await {
        Ok(_) => ComponentHealth { status: "ok", latency_ms: q.elapsed().as_millis(), error: None },
        Err(e) => {
            warn!("health: qdrant check failed: {e}");
            ComponentHealth { status: "down", latency_ms: q.elapsed().as_millis(), error: Some(e.to_string()) }
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

async fn optimize_contract(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<OptimizeRequest>,
) -> Result<Json<OptimizeResponse>, (axum::http::StatusCode, String)> {
    let t0 = std::time::Instant::now();

    // 1. Parse contract into individual functions
    let (category, functions, storage_layout) = analyze_contract(&payload.contract_source);
    let category_str = category.unwrap_or("general");

    info!("=== OPTIMIZE REQUEST ===");
    info!("  contract : {} bytes", payload.contract_source.len());
    info!("  detected : {}", category_str);
    info!("  functions: {}", functions.len());
    info!("========================");

    if functions.is_empty() {
        return Err((axum::http::StatusCode::BAD_REQUEST,
            "No optimizable functions found — ensure the contract parses correctly".to_string()));
    }

    // 2. Sequential: embed + retrieve per function (mutex on embedding model serialises embed calls)
    struct FuncWithContext {
        info: FunctionInfo,
        context: String,
        pattern_ids: Vec<String>,
    }

    let mut func_contexts: Vec<FuncWithContext> = Vec::new();

    for func in functions {
        let query_vec = match state.embedder.clone().embed(&func.source).await {
            Ok(v) => v,
            Err(e) => {
                warn!("  ! Embed failed for {}: {}", func.name, e);
                func_contexts.push(FuncWithContext { info: func, context: String::new(), pattern_ids: vec![] });
                continue;
            }
        };

        let (context, pattern_ids) = retrieve_function_context(&state, &query_vec, category)
            .await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;

        info!("  {} → {} patterns retrieved", func.name, pattern_ids.len());
        func_contexts.push(FuncWithContext { info: func, context, pattern_ids });
    }

    // 3. Parallel: one tokio task per function → DeepSeek call
    info!("=== SPAWNING {} DEEPSEEK TASKS ===", func_contexts.len());

    let handles: Vec<tokio::task::JoinHandle<Result<String, String>>> = func_contexts
        .iter()
        .map(|fc| {
            let client   = state.deepseek.clone();
            let storage  = storage_layout.clone();
            let func_src = fc.info.source.clone();
            let func_name = fc.info.name.clone();
            let context  = fc.context.clone();

            tokio::spawn(async move {
                if context.is_empty() {
                    info!("  {} → no patterns, kept unchanged", func_name);
                    return Ok(func_src);
                }
                info!("  {} → calling DeepSeek ({} ctx chars)", func_name, context.len());
                ai::call_deepseek(&client, &storage, &func_src, &context).await
            })
        })
        .collect();

    // Await all in order (tasks already run in parallel since all spawned before any await)
    let mut results: Vec<Result<Result<String, String>, tokio::task::JoinError>> = Vec::new();
    for h in handles { results.push(h.await); }

    // 4. Splice optimized functions back — reverse order preserves earlier byte offsets
    let mut optimized_source = payload.contract_source.clone();
    let mut optimized_count = 0usize;
    let mut all_pattern_ids: Vec<String> = Vec::new();

    for (fc, task_result) in func_contexts.iter().zip(results.iter()).rev() {
        all_pattern_ids.extend(fc.pattern_ids.iter().cloned());

        let optimized_func = match task_result {
            Ok(Ok(s)) => utils::strip_code_fences(s),
            Ok(Err(e)) => { error!("  ! DeepSeek failed for {}: {}", fc.info.name, e); continue; }
            Err(e)     => { error!("  ! Task panicked for {}: {}", fc.info.name, e); continue; }
        };

        if optimized_func == fc.info.source { continue; } // unchanged, skip splice

        if fc.info.end <= optimized_source.len() {
            optimized_source.replace_range(fc.info.start..fc.info.end, &optimized_func);
            optimized_count += 1;
        }
    }

    all_pattern_ids.sort();
    all_pattern_ids.dedup();

    info!("=== OPTIMIZE COMPLETE ===");
    info!("  functions optimized: {}/{}", optimized_count, func_contexts.len());
    info!("  patterns used      : {}", all_pattern_ids.len());
    info!("  elapsed            : {:.2?}", t0.elapsed());
    info!("=========================");

    Ok(Json(OptimizeResponse {
        analysis: format!(
            "Optimized {}/{} functions using {} patterns.",
            optimized_count, func_contexts.len(), all_pattern_ids.len()
        ),
        suggested_patterns: all_pattern_ids,
        optimized_code: optimized_source,
    }))
}


// ── Contract analysis: category + per-function extraction ────────────────────
fn analyze_contract(source: &str) -> (Option<&'static str>, Vec<FunctionInfo>, String) {
    let Ok((su, _)) = solang_parser::parse(source, 0) else {
        return (detect_category_fallback(source), vec![], String::new());
    };

    let mut category: Option<&'static str> = None;
    let mut functions: Vec<FunctionInfo> = Vec::new();
    let mut storage_vars: Vec<String> = Vec::new();

    for part in su.0 {
        let SourceUnitPart::ContractDefinition(def) = part else { continue };

        // Inheritance → category
        for base in &def.base {
            let base_name = base.name.identifiers.iter()
                .map(|id| id.name.to_lowercase()).collect::<Vec<_>>().join(".");
            if category.is_none() {
                category = match base_name.as_str() {
                    s if s.contains("erc721")  => Some("erc721"),
                    s if s.contains("erc1155") => Some("erc1155"),
                    s if s.contains("erc20")   => Some("erc20"),
                    s if s.contains("erc2981") => Some("erc2981"),
                    _ => None,
                };
            }
        }

        for cp in &def.parts {
            match cp {
                ContractPart::VariableDefinition(var) => {
                    if let Loc::File(_, start, end) = var.loc {
                        if let Some(text) = source.get(start..end) {
                            storage_vars.push(text.trim().to_string());
                        }
                    }
                }
                ContractPart::FunctionDefinition(func) => {
                    // Skip unnamed (constructor/fallback/receive) and abstract (no body)
                    let Some(name_ident) = &func.name else { continue };
                    let Some(_) = &func.body else { continue };
                    let Loc::File(_, start, end) = func.loc else { continue };
                    let Some(func_text) = source.get(start..end) else { continue };

                    let name = name_ident.name.clone();

                    // Per-function ops from debug-printed body
                    let body_str = format!("{:?}", func.body);
                    let mut ops = vec![name.clone()];
                    if body_str.contains("Assembly")
                        { ops.push("inline_assembly".into()); }
                    if body_str.contains("For(") || body_str.contains("While(")
                        { ops.push("loop_iteration".into()); }
                    if body_str.contains("Emit(")
                        { ops.push("event_emission".into()); }
                    if body_str.contains("transferFrom")
                        { ops.push("erc20_transfer_from".into()); }
                    if body_str.contains("msg.value")
                        { ops.push("eth_handling".into()); }
                    if body_str.contains("rewardDebt") || body_str.contains("rewardPerToken")
                        { ops.push("reward_calculation".into()); }
                    ops.sort();
                    ops.dedup();

                    functions.push(FunctionInfo { name, source: func_text.to_string(), start, end, ops });
                }
                _ => {}
            }
        }
    }

    (category, functions, storage_vars.join("\n"))
}

fn detect_category_fallback(source: &str) -> Option<&'static str> {
    let s = source.to_lowercase();
    if s.contains("is erc721") || s.contains(": erc721")  { return Some("erc721"); }
    if s.contains("is erc1155") || s.contains(": erc1155") { return Some("erc1155"); }
    if s.contains("is erc2981") || s.contains(": erc2981") { return Some("erc2981"); }
    if s.contains("is erc20") || s.contains(": erc20")    { return Some("erc20"); }
    None
}



// Returns (context_string, pattern_ids) for a single function's query vector
async fn retrieve_function_context(
    state: &Arc<AppState>,
    query_vec: &[f32],
    category: Option<&'static str>,
) -> Result<(String, Vec<String>), String> {
    let mut pattern_contexts: Vec<String> = Vec::new();
    let mut anti_contexts: Vec<String> = Vec::new();
    let mut pattern_ids: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    let token_cats = ["erc20", "erc721", "erc1155", "erc2981", "accounts"];
    let is_token = category.map(|c| token_cats.contains(&c)).unwrap_or(false);

    let pattern_hits = if is_token {
        let cat = category.unwrap();
        let cat_r = state.qdrant.search_points(
            SearchPointsBuilder::new(COLLECTION, query_vec.to_vec(), 2)
                .with_payload(true)
                .filter(Filter::must([Condition::matches("category", cat.to_string())]))
        ).await.map_err(|e| e.to_string())?;

        let gen_r = state.qdrant.search_points(
            SearchPointsBuilder::new(COLLECTION, query_vec.to_vec(), 1)
                .with_payload(true)
                .filter(Filter::must_not(
                    token_cats.iter().map(|c| Condition::matches("category", c.to_string())).collect::<Vec<_>>()
                ))
        ).await.map_err(|e| e.to_string())?;

        let mut combined = cat_r.result;
        combined.extend(gen_r.result);
        combined
    } else {
        state.qdrant.search_points(
            SearchPointsBuilder::new(COLLECTION, query_vec.to_vec(), 3)
                .with_payload(true)
        ).await.map_err(|e| e.to_string())?.result
    };

    for hit in pattern_hits {
        let id = match hit.payload.get("pattern_id") {
            Some(v) => v.to_string().trim().replace('"', ""),
            None => continue,
        };
        if !seen.insert(id.clone()) { continue; }
        pattern_ids.push(id.clone());
        let rows = state.db.query(
            "SELECT title, explanation, yul_optimized, risk_level, when_not_to_apply FROM optimization_patterns WHERE id = ?",
            vec![TursoArg::Text(id.clone())],
        ).await?;
        if let Some(row) = rows.first() {
            let get = |k: &str| row.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
            pattern_contexts.push(format!(
                "PATTERN ID: {}\nTitle: {}\nExplanation: {}\nOptimized YUL:\n{}\nRisk: {}\nDo NOT apply when: {}",
                id, get("title"), get("explanation"), get("yul_optimized"), get("risk_level"), get("when_not_to_apply"),
            ));
        }
    }

    let anti_hits = state.qdrant.search_points(
        SearchPointsBuilder::new(COLLECTION, query_vec.to_vec(), 2)
            .with_payload(true)
            .filter(Filter::must([Condition::matches("type", "antipattern".to_string())]))
    ).await.map_err(|e| e.to_string())?.result;

    for hit in anti_hits {
        let id = match hit.payload.get("pattern_id") {
            Some(v) => v.to_string().trim().replace('"', ""),
            None => continue,
        };
        if !seen.insert(id.clone()) { continue; }
        pattern_ids.push(id.clone());
        let rows = state.db.query(
            "SELECT title, explanation, solidity_before, yul_optimized FROM optimization_patterns WHERE id = ?",
            vec![TursoArg::Text(id.clone())],
        ).await?;
        if let Some(row) = rows.first() {
            let get = |k: &str| row.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
            anti_contexts.push(format!(
                "ANTIPATTERN ID: {}\nTitle: {}\nExplanation: {}\nWrong:\n{}\nCorrect:\n{}",
                id, get("title"), get("explanation"), get("solidity_before"), get("yul_optimized"),
            ));
        }
    }

    let mut parts = Vec::new();
    if !pattern_contexts.is_empty() {
        parts.push(format!("PATTERNS TO APPLY:\n\n{}", pattern_contexts.join("\n\n---\n\n")));
    }
    if !anti_contexts.is_empty() {
        parts.push(format!("ANTIPATTERNS TO AVOID:\n\n{}", anti_contexts.join("\n\n---\n\n")));
    }

    Ok((parts.join("\n\n===\n\n"), pattern_ids))
}

async fn create_qdrant_indexes(state: &AppState) -> Result<(), String> {
    for field in ["category", "type"] {
        state.qdrant
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(COLLECTION, field, FieldType::Keyword)
            )
            .await
            .map_err(|e| format!("Failed to create Qdrant index on '{field}': {e}"))?;
        info!("  index created: {}", field);
    }
    Ok(())
}

async fn reset_collection(
    State(state): State<Arc<AppState>>,
) -> Result<&'static str, (axum::http::StatusCode, String)> {
    info!("=== RESET COLLECTION ===");

    state.qdrant
        .delete_collection(COLLECTION)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    info!("  deleted : {}", COLLECTION);

    state.qdrant
        .create_collection(
            CreateCollectionBuilder::new(COLLECTION)
                .vectors_config(VectorParamsBuilder::new(VECTOR_DIM, Distance::Cosine)),
        )
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    info!("  created : {} ({} dims, cosine)", COLLECTION, VECTOR_DIM);

    create_qdrant_indexes(&state).await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;

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
    info!("  directories: {}", payload.directory_paths.len());

    for dir_path in payload.directory_paths {
        let dir = Path::new(&dir_path);

        if !dir.is_dir() {
            warn!("  ! Not a directory: {}", dir_path);
            failed.push((dir_path, "Not a valid directory".to_string()));
            continue;
        }

        info!("  scanning: {}", dir_path);

        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                error!("  ! Cannot read directory {}: {}", dir_path, e);
                failed.push((dir_path, format!("Cannot read directory: {e}")));
                continue;
            }
        };

        for entry in entries.flatten() {
            let file_path = entry.path();
            if file_path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }

            let file_name = file_path.file_name().unwrap().to_string_lossy().into_owned();

            let content = match fs::read_to_string(&file_path) {
                Ok(c) => c,
                Err(e) => {
                    warn!("    ! read error {}: {}", file_name, e);
                    failed.push((file_name, format!("Read error: {e}"))); continue;
                }
            };

            let meta: serde_json::Value = match serde_json::from_str(&content) {
                Ok(v) => v,
                Err(e) => {
                    warn!("    ! invalid JSON {}: {}", file_name, e);
                    failed.push((file_name, format!("Invalid JSON: {e}"))); continue;
                }
            };

            // Use clean let-else syntax for the critical ID extraction
            let Some(id) = meta["id"].as_str().map(String::from) else {
                failed.push((file_name, "Missing 'id' field".to_string()));
                continue;
            };

            // Extract core fields exactly once to clean up database injection
            let title = meta["title"].as_str().unwrap_or("title");
            let category = meta["category"].as_str().unwrap_or("general");
            let triggers = meta["trigger_patterns"].to_string();
            let sol_before = meta["solidity_before"].as_str()
                .or(meta["pattern_before"].as_str())
                .or(meta["wrong_code"].as_str())
                .unwrap_or("");

            let entry_type = meta["type"].as_str().unwrap_or("pattern");

            let embed_text = if entry_type == "antipattern" {
                let wrong = meta["wrong_code"].as_str().unwrap_or("");
                let why = meta["why_wrong"].as_str().unwrap_or("");
                format!(
                    "TOKEN_STANDARD_NAMESPACE: {}\n\
                    // Antipattern to avoid: {}\n\
                    // Triggers: {}\n\
                    // Wrong code: {}\n\
                    // Why wrong: {}",
                    category.to_uppercase(), title, triggers, wrong, why
                )
            } else {
                // existing pattern embed text
                format!(
                    "TOKEN_STANDARD_NAMESPACE: {}\n// Optimization: {}\n// Keywords: {}\n{}",
                    category.to_uppercase(), title, triggers, sol_before
                )
            };

            let vector = match state.embedder.clone().embed(&embed_text).await {
                Ok(v) => v,
                Err(e) => { failed.push((id, format!("Embedding error: {e}"))); continue; }
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
                TursoArg::Text(meta["version"].as_str().unwrap_or("1.0").to_string()),
                TursoArg::Text(title.to_string()),
                TursoArg::Text(meta["source"].as_str().unwrap_or("").to_string()),
                TursoArg::Text(meta["source_file"].as_str().unwrap_or("").to_string()),
                TursoArg::Text(meta["difficulty"].as_str().unwrap_or("medium").to_string()),
                TursoArg::Integer((meta["mantle_specific"].as_bool().unwrap_or(false) as i64).to_string()),
                TursoArg::Text(meta["evm_version"].as_str().unwrap_or("paris").to_string()),
                TursoArg::Text(triggers),
                TursoArg::Text(sol_before.to_string()),
                TursoArg::Text(meta["yul_optimized"].as_str().or(meta["pattern_after"].as_str()).or(meta["correct_code"].as_str()).unwrap_or("").to_string()),
                TursoArg::Text(meta["patterns_used"].to_string()),
                TursoArg::Text(meta["explanation"].to_string()),
                TursoArg::Text(meta["risk_level"].as_str().unwrap_or("low").to_string()),
                TursoArg::Text(meta["when_to_apply"].to_string()),
                TursoArg::Text(meta["when_not_to_apply"].as_str().unwrap_or("").to_string()),
            ];

            if let Err(e) = state.db.execute(sql, args).await {
                failed.push((id, format!("Turso error: {e}")));
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

            let point = PointStruct::new(Uuid::new_v4().to_string(), vector, qdrant_payload);

            if let Err(e) = state.qdrant.upsert_points(UpsertPointsBuilder::new(COLLECTION, vec![point])).await {
                warn!("    ! Qdrant upsert failed {}: {}", id, e);
                failed.push((id, format!("Qdrant error: {e}")));
                continue;
            }

            info!("    + {} ({}, {})", id, category, entry_type);
            successful.push(id);
        }
    }

    info!("=== INGEST COMPLETE ===");
    info!("  ok     : {}", successful.len());
    info!("  failed : {}", failed.len());
    for (id, reason) in &failed {
        warn!("    ! {} — {}", id, reason);
    }
    info!("======================");

    Ok(Json(IngestLocalResponse { successful_patterns: successful, failed_patterns: failed }))
}
