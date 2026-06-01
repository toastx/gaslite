//TODO erc1155

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
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;
use solang_parser::pt::{ContractPart, Loc, SourceUnitPart};
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::sync::Mutex;

// ── constants ─────────────────────────────────────────────────────────────────
const COLLECTION: &str = "gaslite_patterns";
const VECTOR_DIM: u64 = 384;

// ── app state ─────────────────────────────────────────────────────────────────
struct AppState {
    http: reqwest::Client,
    turso_url: String,
    turso_token: String,
    qdrant: Qdrant,
    deepseek_api_key: String,
    embedding_model: Mutex<TextEmbedding>,
}

// ── turso HTTP types ──────────────────────────────────────────────────────────
#[derive(Serialize)]
struct TursoRequest {
    requests: Vec<TursoStatement>,
}

#[derive(Serialize)]
struct TursoStatement {
    #[serde(rename = "type")]
    stmt_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    stmt: Option<TursoStmtInner>, // Now points to an inner struct
}

#[derive(Serialize)]
struct TursoStmtInner {
    sql: String,
    args: Vec<TursoArg>,
}

// Hrana requires args to look like {"type": "text", "value": "foo"}
// This exact serde macro configuration achieves that automatically.
#[derive(Serialize, Clone)]
#[serde(tag = "type", content = "value")]
enum TursoArg {
    #[serde(rename = "text")]
    Text(String),
    #[serde(rename = "integer")]
    Integer(String),
    #[serde(rename = "null")]
    Null,
}

#[derive(Deserialize, Debug)]
struct TursoResponse {
    results: Vec<TursoResult>,
}

#[derive(Deserialize, Debug)]
struct TursoResult {
    #[serde(rename = "type")]
    result_type: String,
    response: Option<TursoResultResponse>,
    error: Option<TursoError>,
}

#[derive(Deserialize, Debug)]
struct TursoResultResponse {
    #[serde(rename = "type")]
    response_type: String,
    result: Option<TursoRows>,
}

#[derive(Deserialize, Debug)]
struct TursoRows {
    cols: Vec<TursoCol>,
    rows: Vec<Vec<serde_json::Value>>,
}

#[derive(Deserialize, Debug)]
struct TursoCol {
    name: String,
}

#[derive(Deserialize, Debug)]
struct TursoError {
    message: String,
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

#[derive(Deserialize)]
struct VerifyRequest {
    original_code: String,
    optimized_code: String,
}

#[derive(Serialize)]
struct VerifyResponse {
    compiles: bool,
    errors: Vec<String>,
    gas_original: Option<u64>,
    gas_optimized: Option<u64>,
    gas_saved: Option<i64>,
    forge_output: String,
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

// ── turso HTTP client ─────────────────────────────────────────────────────────
impl AppState {
    async fn turso_execute(
        &self,
        sql: &str,
        args: Vec<TursoArg>,
    ) -> Result<(), String> {
        let stmts = vec![
            TursoStatement {
                stmt_type: "execute".to_string(),
                stmt: Some(TursoStmtInner {
                    sql: sql.to_string(),
                    args,
                }),
            },
            TursoStatement {
                stmt_type: "close".to_string(),
                stmt: None,
            },
        ];

        let res = self
            .http
            .post(format!("{}/v2/pipeline", self.turso_url))
            .bearer_auth(&self.turso_token)
            .json(&TursoRequest { requests: stmts })
            .send()
            .await
            .map_err(|e| format!("Turso request failed: {e}"))?;

        if !res.status().is_success() {
            return Err(format!("Turso returned status {}", res.status()));
        }

        let body: TursoResponse = res
            .json()
            .await
            .map_err(|e| format!("Turso parse error: {e}"))?;

        // check for errors in results
        for result in &body.results {
            if let Some(err) = &result.error {
                return Err(format!("Turso SQL error: {}", err.message));
            }
        }

        Ok(())
    }

    async fn turso_query(
        &self,
        sql: &str,
        args: Vec<TursoArg>,
    ) -> Result<Vec<HashMap<String, serde_json::Value>>, String> {
        let stmts = vec![
            TursoStatement {
                stmt_type: "execute".to_string(),
                stmt: Some(TursoStmtInner {
                    sql: sql.to_string(),
                    args,
                }),
            },
            TursoStatement {
                stmt_type: "close".to_string(),
                stmt: None,
            },
        ];

        let res = self
            .http
            .post(format!("{}/v2/pipeline", self.turso_url))
            .bearer_auth(&self.turso_token)
            .json(&TursoRequest { requests: stmts })
            .send()
            .await
            .map_err(|e| format!("Turso request failed: {e}"))?;

        if !res.status().is_success() {
            return Err(format!("Turso returned status {}", res.status()));
        }

        let body: TursoResponse = res
            .json()
            .await
            .map_err(|e| format!("Turso parse error: {e}"))?;

        // extract rows from first execute result
        for result in &body.results {
            if let Some(err) = &result.error {
                return Err(format!("Turso SQL error: {}", err.message));
            }
            if result.result_type == "ok" {
                if let Some(resp) = &result.response {
                    if let Some(rows_data) = &resp.result {
                        let col_names: Vec<&str> =
                            rows_data.cols.iter().map(|c| c.name.as_str()).collect();

                        let rows = rows_data
                            .rows
                            .iter()
                            .map(|row| {
                                col_names
                                    .iter()
                                    .zip(row.iter())
                                    .map(|(col, val)| (col.to_string(), val.clone()))
                                    .collect::<HashMap<String, serde_json::Value>>()
                            })
                            .collect();

                        return Ok(rows);
                    }
                }
            }
        }

        Ok(vec![])
    }
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
    let qdrant_api_key = std::env::var("QDRANT_API_KEY").expect("QDRANT_API_KEY required");
    let qdrant_url = std::env::var("QDRANT_CLUSTER_URL").expect("QDRANT_CLUSTER_URL required");
    let turso_url = std::env::var("TURSO_DATABASE_URL").expect("TURSO_DATABASE_URL required");
    let turso_token = std::env::var("TURSO_AUTH_TOKEN").expect("TURSO_AUTH_TOKEN required");


    let http = reqwest::Client::new();
    let embedding_model = TextEmbedding::try_new(
    InitOptions::new(EmbeddingModel::BGESmallENV15)
        .with_show_download_progress(true)
    )?;
    

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
        http: http.clone(),
        turso_url: turso_url.clone(),
        turso_token: turso_token.clone(),
        qdrant,
        deepseek_api_key,
        embedding_model:Mutex::new(embedding_model)
    });

    // run migration via HTTP
    state
        .turso_execute(
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
        .route("/api/verify", post(verify_contract))
        .route("/api/admin/ingest-local", post(ingest_local_files))
        .route("/api/admin/qdrant/reset", post(reset_collection))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await?;
    info!("Gaslite listening on {}", listener.local_addr()?);
    axum::serve(listener, router).await?;
    Ok(())
}

// ── handlers ──────────────────────────────────────────────────────────────────

async fn health_check() -> &'static str {
    info!("GET /health");
    "Gaslite Engine is online."
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
        let query_vec = match get_embedding(state.clone(), &func.source).await {
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
            let http       = state.http.clone();
            let api_key    = state.deepseek_api_key.clone();
            let storage    = storage_layout.clone();
            let func_src   = fc.info.source.clone();
            let func_name  = fc.info.name.clone();
            let context    = fc.context.clone();

            tokio::spawn(async move {
                if context.is_empty() {
                    info!("  {} → no patterns, kept unchanged", func_name);
                    return Ok(func_src);
                }
                info!("  {} → calling DeepSeek ({} ctx chars)", func_name, context.len());
                call_deepseek(&http, &storage, &func_src, &context, &api_key).await
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
            Ok(Ok(s)) => strip_code_fences(s),
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
        let rows = state.turso_query(
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
        let rows = state.turso_query(
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

fn strip_code_fences(s: &str) -> String {
    let s = s.trim();
    if !s.contains("```") {
        return s.to_string();
    }
    // Collect only lines that appear inside code fence blocks.
    // Handles multi-block responses where DeepSeek wraps each function separately.
    let mut result: Vec<&str> = Vec::new();
    let mut in_fence = false;
    let mut found_fence = false;
    for line in s.lines() {
        let t = line.trim();
        if t == "```" || t.starts_with("```solidity") || t.starts_with("```yul") {
            found_fence = true;
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            result.push(line);
        }
    }
    if found_fence && !result.is_empty() {
        result.join("\n").trim().to_string()
    } else {
        s.to_string()
    }
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

            let vector = match get_embedding(state.clone(), &embed_text).await {
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

            if let Err(e) = state.turso_execute(sql, args).await {
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
// ── helpers ───────────────────────────────────────────────────────────────────

async fn get_embedding(state: Arc<AppState>, text: &str) -> Result<Vec<f32>, String> {
    let text = text.to_string();
    tokio::task::spawn_blocking(move || {
        let mut model = state.embedding_model.lock().unwrap();
        let mut embeddings = model
            .embed(vec![text.as_str()], None)
            .map_err(|e| format!("Embed error: {e}"))?;
        embeddings.pop().ok_or_else(|| "Embedding returned empty results".to_string())
    })
    .await
    .map_err(|e| format!("Embedding task panicked: {e}"))?
}

// ── Forge verification ────────────────────────────────────────────────────────

async fn verify_contract(
    Json(payload): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, (axum::http::StatusCode, String)> {
    info!("POST /api/verify — {} + {} bytes", payload.original_code.len(), payload.optimized_code.len());
    tokio::task::spawn_blocking(move || run_forge_sandbox(&payload.original_code, &payload.optimized_code))
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map(Json)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))
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

async fn call_deepseek(
    client: &reqwest::Client,
    storage_layout: &str,
    function_source: &str,
    context: &str,
    api_key: &str,
) -> Result<String, String> {
    let system =
        "You are Gaslite, a gas optimization engine for Mantle L2 EVM contracts.\n\
        \n\
        Your role is pattern application and adaptation — not pattern invention.\n\
        \n\
        The RETRIEVED PATTERNS are your source of truth for YUL structure, opcodes, \
        and error selectors. Use them as templates:\n\
        - Keep the YUL opcodes, control flow, and error selectors exactly as shown\n\
        - Adapt storage slot variable names and mapping key derivations to match \
          the user contract's actual storage layout shown in STORAGE LAYOUT\n\
        - For standard Solidity mappings, derive slots as: \
          mstore(0x00, key), mstore(0x20, slot_number), keccak256(0x00, 0x40)\n\
        - Replace require(condition, string) with the 4-byte custom error pattern: \
          mstore(0x00, 0xSELECTOR), revert(0x1c, 0x04)\n\
        - Do not invent YUL opcodes, selectors, or patterns not present in the retrieved patterns\n\
        \n\
        Correctness is absolute. An optimization that changes observable behaviour \
        is not an optimization — it is a bug.";

    let user = format!(
        "STORAGE LAYOUT:\n{storage_layout}\n\n\
        FUNCTION TO OPTIMIZE:\n```solidity\n{function_source}\n```\n\n\
        RETRIEVED PATTERNS:\n{context}\n\n\
        TASK:\n\
        Optimize ONLY this function by applying the retrieved patterns as templates, \
        adapting slot derivations and variable names to this contract's storage layout.\n\
        Return ONLY the complete optimized function — no contract wrapper, no imports.\n\
        After the function, add one line per change: pattern ID applied + estimated gas saved on Mantle.\n\
        If a pattern genuinely cannot apply even with adaptation, say why in one line and skip it."
    );

    let res = client
        .post("https://api.deepseek.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&serde_json::json!({
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            "temperature": 0.1
        }))
        .send()
        .await
        .map_err(|e| format!("DeepSeek request failed: {e}"))?;

    if !res.status().is_success() {
        return Err(format!("DeepSeek returned status {}", res.status()));
    }

    let json: serde_json::Value = res
        .json()
        .await
        .map_err(|e| format!("DeepSeek parse error: {e}"))?;

    json["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or("Missing content in DeepSeek response".to_string())
}