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
use solang_parser::pt::{ContractPart, SourceUnitPart};
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

#[derive(Serialize)]
struct IngestLocalResponse {
    successful_patterns: Vec<String>,
    failed_patterns: Vec<(String, String)>,
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
#[shuttle_runtime::main]
async fn main(
    #[shuttle_runtime::Secrets] secrets: shuttle_runtime::SecretStore,
) -> shuttle_axum::ShuttleAxum {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let deepseek_api_key = secrets
        .get("DEEPSEEK_API_KEY")
        .expect("DEEPSEEK_API_KEY required");
    let qdrant_api_key = secrets
        .get("QDRANT_API_KEY")
        .expect("QDRANT_API_KEY required");
    let qdrant_url = secrets
        .get("QDRANT_CLUSTER_URL")
        .expect("QDRANT_CLUSTER_URL required");
    let turso_url = secrets
        .get("TURSO_DATABASE_URL")
        .expect("TURSO_DATABASE_URL required");
    let turso_token = secrets
        .get("TURSO_AUTH_TOKEN")
        .expect("TURSO_AUTH_TOKEN required");


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

    let router = Router::new()
        .route("/health", get(health_check))
        .route("/api/optimize", post(optimize_contract))
        .route("/api/admin/ingest-local", post(ingest_local_files))
        .route("/api/admin/qdrant/reset", post(reset_collection))
        .with_state(state);

    Ok(router.into())
}

// ── handlers ──────────────────────────────────────────────────────────────────

async fn health_check() -> &'static str {
    "Gaslite Engine is online."
}

async fn optimize_contract(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<OptimizeRequest>,
) -> Result<Json<OptimizeResponse>, (axum::http::StatusCode, String)> {

    // 1. Use Solang AST to extract true intent and operations (0ms latency, 0 cost)
    let (detected, key_ops) = analyze_contract_ast(&payload.contract_source);
    let category_str = detected.unwrap_or("general");

    // Build a semantic embedding query instead of embedding the raw, noisy code
    let query_text = format!(
        "TOKEN_STANDARD_NAMESPACE: {}\n// Operations detected: {}",
        category_str.to_uppercase(),
        key_ops.join(", ")
    );

    // 2. Embed incoming clean string
    let query_vec = get_embedding(State(state.clone()),&query_text.as_str())
        .await
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e))?;

    // 3. Search Qdrant with category filter
    let results = match detected {
        Some(cat) => {
            let cat_results = state.qdrant.search_points(
                SearchPointsBuilder::new(COLLECTION, query_vec.clone(), 2)
                    .with_payload(true)
                    .filter(Filter::must([
                        Condition::matches("category", cat.to_string())
                    ]))
            ).await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

            let gen_results = state.qdrant.search_points(
                SearchPointsBuilder::new(COLLECTION, query_vec.clone(), 5)
                    .with_payload(true)
                    .filter(Filter::must_not([
                        Condition::matches("category", "erc20".to_string()),
                        Condition::matches("category", "erc721".to_string()),
                        Condition::matches("category", "erc1155".to_string()),
                    ]))
            ).await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

            let mut combined = cat_results.result;
            combined.truncate(2);
            combined.extend(gen_results.result.into_iter().take(1));
            combined
        },
        None => {
            state.qdrant.search_points(
                SearchPointsBuilder::new(COLLECTION, query_vec.clone(), 5)
                    .with_payload(true)
            ).await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?.result
        }
    };

    // 4. Log results
    info!("--- QDRANT RETRIEVED PATTERNS ---");
    info!("Found {} matching points.", results.len());

    for point in &results {
        if let Some(pattern_id) = point.payload.get("pattern_id") {
            info!("Matched Pattern: {}", pattern_id);
        } else {
            warn!("Point missing pattern_id in payload.");
        }
    }
    info!("---------------------------------");

    // 5. Fetch full patterns from Turso
    let mut pattern_contexts: Vec<String> = Vec::new();
    let mut anti_pattern_contexts: Vec<String> = Vec::new();
    let mut found_pattern_ids: Vec<String> = Vec::new();

    for hit in results {
        
        let pattern_id = match hit.payload.get("pattern_id") {
            Some(v) => {
                let raw = v.to_string();
                let cleaned = raw.trim().replace('"', "").to_string();
                info!("Cleaned pattern_id: '{}'", cleaned);
                cleaned
            },
            None => continue,
        };

        found_pattern_ids.push(pattern_id.clone());
        found_pattern_ids.dedup();

        let rows = state
            .turso_query(
                "SELECT title, explanation, yul_optimized, risk_level, when_not_to_apply \
                 FROM optimization_patterns WHERE id = ?",
                vec![TursoArg::Text(pattern_id.clone())],
            )
            .await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;

        info!("Turso rows for '{}': {}", pattern_id, rows.len());
        if rows.is_empty() {
            warn!("No Turso data found for pattern: {}", pattern_id);
            continue; // skip this pattern
        }

        if let Some(row) = rows.first() {
            let get = |key: &str| {
                row.get(key)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            };

            pattern_contexts.push(format!(
                "PATTERN: {}\nExplanation: {}\nOptimized YUL:\n{}\nRisk: {}\nDo NOT apply when: {}",
                get("title"),
                get("explanation"),
                get("yul_optimized"),
                get("risk_level"),
                get("when_not_to_apply"),
            ));
        }
    }

    let anti_results = state.qdrant.search_points(
    SearchPointsBuilder::new(COLLECTION, query_vec.clone(), 2)
        .with_payload(true)
        .filter(Filter::must([
            Condition::matches("type", "antipattern".to_string())
        ]))
    ).await
    .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?.result;

    for hit in anti_results {
        
        let pattern_id = match hit.payload.get("pattern_id") {
            Some(v) => {
                let raw = v.to_string();
                let cleaned = raw.trim().replace('"', "").to_string();
                info!("Cleaned pattern_id: '{}'", cleaned);
                cleaned
            },
            None => continue,
        };

        found_pattern_ids.push(pattern_id.clone());
        found_pattern_ids.dedup();

        let rows = state
            .turso_query(
                "SELECT title, explanation, wrong_code, correct_code, correct_usage_table \
                 FROM optimization_patterns WHERE id = ?",
                vec![TursoArg::Text(pattern_id.clone())],
            )
            .await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;

        info!("Turso rows for '{}': {}", pattern_id, rows.len());
        if rows.is_empty() {
            warn!("No Turso data found for pattern: {}", pattern_id);
            continue; // skip this pattern
        }

        if let Some(row) = rows.first() {
            let get = |key: &str| {
                row.get(key)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            };

            anti_pattern_contexts.push(format!(
                "ANTI_PATTERN: {}\nExplanation: {}\nWrong Code:\n{}\nCorrect Code {}\n Usage Table: {}",
                get("title"),
                get("explanation"),
                get("wrong_code"),
                get("correct_code"),
                get("correct_usage_table"),
            ));
        }
    }

    let context = format!(
        "follow {} and dont follow {}",
        pattern_contexts.join("\n\n---\n\n"),
        anti_pattern_contexts.join("\n\n---\n\n")
    );
    
    
    // 6. Call DeepSeek
    let optimized_code =
        call_deepseek(&state.http, &payload.contract_source, &context, &state.deepseek_api_key)
            .await
            .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e))?;

    Ok(Json(OptimizeResponse {
        analysis: format!(
            "Gaslite found {} relevant patterns.",
            found_pattern_ids.len()
        ),
        suggested_patterns: found_pattern_ids,
        optimized_code,
    }))
}


/// Parses raw Solidity to extract the contract category and function signatures
fn analyze_contract_ast(source: &str) -> (Option<&'static str>, Vec<String>) {
    let mut detected_cat: Option<&'static str> = None;
    let mut key_ops: Vec<String> = Vec::new();
 
    let Ok((source_unit, _)) = solang_parser::parse(source, 0) else {
        // fallback: basic string scan if parser fails
        return analyze_contract_fallback(source);
    };
 
    for part in source_unit.0 {
        let SourceUnitPart::ContractDefinition(def) = part else { continue };
 
        // ── 1. Inheritance detection ──────────────────────────────────────────
        for base in &def.base {
            let base_name = base.name.identifiers.iter()
                .map(|id| id.name.to_lowercase())
                .collect::<Vec<_>>()
                .join(".");
 
            if detected_cat.is_none() {
                detected_cat = match base_name.as_str() {
                    s if s.contains("erc721")  => Some("erc721"),
                    s if s.contains("erc1155") => Some("erc1155"),
                    s if s.contains("erc20")   => Some("erc20"),
                    s if s.contains("erc2981") => Some("erc2981"),
                    _                           => None,
                };
            }
        }
 
        // ── 2. Function-level analysis ────────────────────────────────────────
        for contract_part in &def.parts {
            let ContractPart::FunctionDefinition(func) = contract_part else { continue };
 
            // 2a. Function name heuristics
            if let Some(name_ident) = &func.name {
                let func_name = name_ident.name.as_str();
                key_ops.push(func_name.to_string());
 
                if detected_cat.is_none() {
                    detected_cat = match func_name {
                        // ERC721
                        "ownerOf" | "tokenURI" | "setApprovalForAll"
                        | "getApproved" | "safeTransferFrom" => Some("erc721"),
 
                        // ERC1155
                        "safeBatchTransferFrom" | "balanceOfBatch"
                        | "onERC1155Received" => Some("erc1155"),
 
                        // ERC20
                        "totalSupply" | "allowance" | "permit" => Some("erc20"),
 
                        // DeFi staking
                        "stake" | "unstake" | "rewardPerToken"
                        | "earned" | "getReward" | "notifyRewardAmount" => Some("defi_staking"),
 
                        // DeFi AMM
                        "swap" | "addLiquidity" | "removeLiquidity"
                        | "getAmountOut" | "getAmountIn" => Some("defi_amm"),
 
                        // DeFi lending
                        "borrow" | "repay" | "liquidate"
                        | "supply" | "withdraw" | "getHealthFactor" => Some("defi_lending"),
 
                        _ => None,
                    };
                }
            }
 
            // 2b. Function body analysis
            if let Some(body) = &func.body {
                let body_str = format!("{body:?}");
 
                // EVM operation signals
                if body_str.contains("Assembly") || body_str.contains("assembly") {
                    key_ops.push("inline_assembly".to_string());
                }
                if body_str.contains("For(") || body_str.contains("While(") {
                    key_ops.push("loop_iteration".to_string());
                }
                if body_str.contains("Emit(") {
                    key_ops.push("event_emission".to_string());
                }
 
                // External call detection
                if body_str.contains("transferFrom") {
                    key_ops.push("erc20_transfer_from".to_string());
                    if detected_cat.is_none() { detected_cat = Some("erc20"); }
                }
                if body_str.contains("transfer") && !body_str.contains("transferFrom") {
                    key_ops.push("erc20_transfer".to_string());
                }
                if body_str.contains("IERC20") || body_str.contains("ERC20(") {
                    key_ops.push("erc20_external_call".to_string());
                    if detected_cat.is_none() { detected_cat = Some("erc20"); }
                }
                if body_str.contains("IERC721") || body_str.contains("ERC721(") {
                    key_ops.push("erc721_external_call".to_string());
                    if detected_cat.is_none() { detected_cat = Some("erc721"); }
                }
 
                // ETH handling
                if body_str.contains("selfbalance")
                    || body_str.contains("address(this).balance")
                    || body_str.contains("msg.value") {
                    key_ops.push("eth_handling".to_string());
                }
 
                // Storage patterns
                if body_str.contains("mapping") {
                    key_ops.push("mapping_access".to_string());
                }
 
                // Reward/staking body signals
                if body_str.contains("rewardDebt") || body_str.contains("rewardPerToken") {
                    key_ops.push("reward_calculation".to_string());
                    if detected_cat.is_none() { detected_cat = Some("defi_staking"); }
                }
            }
        }
    }
 
    key_ops.sort();
    key_ops.dedup();
 
    (detected_cat, key_ops)
}
 
// ── Fallback: plain string scan when Solang fails to parse ────────────────────
fn analyze_contract_fallback(source: &str) -> (Option<&'static str>, Vec<String>) {
    let lower = source.to_lowercase();
    let mut key_ops = Vec::new();
 
    let detected_cat = if lower.contains("ownerof") || lower.contains("tokenuri") {
        Some("erc721")
    } else if lower.contains("safebatchtransferfrom") || lower.contains("balanceofbatch") {
        Some("erc1155")
    } else if lower.contains("totalsupply") || lower.contains("allowance") {
        Some("erc20")
    } else if lower.contains("stake") || lower.contains("rewarddebt") {
        Some("defi_staking")
    } else if lower.contains("swap") || lower.contains("amountout") {
        Some("defi_amm")
    } else if lower.contains("borrow") || lower.contains("collateral") {
        Some("defi_lending")
    } else {
        None
    };
 
    if lower.contains("transferfrom") { key_ops.push("erc20_transfer_from".to_string()); }
    if lower.contains("selfbalance") || lower.contains("msg.value") {
        key_ops.push("eth_handling".to_string());
    }
    if lower.contains("assembly") { key_ops.push("inline_assembly".to_string()); }
    if lower.contains("for ") || lower.contains("while ") {
        key_ops.push("loop_iteration".to_string());
    }
    if lower.contains("emit ") { key_ops.push("event_emission".to_string()); }
 
    key_ops.sort();
    key_ops.dedup();
 
    (detected_cat, key_ops)
}
 
// ── Category → Qdrant filter mapping ─────────────────────────────────────────
// Only token standards get a category filter.
// DeFi categories fall through to unfiltered search since those
// knowledge base entries don't have specific category values yet.
fn filter_category(cat: Option<&'static str>) -> Option<&'static str> {
    match cat {
        Some("erc20") => Some("erc20"),
        Some("erc721") => Some("erc721"),
        Some("erc1155") => Some("erc1155"),
        Some("erc2981") => Some("erc2981"),
        _ => None, // defi_staking, defi_amm, defi_lending, general → no filter
    }
}

async fn reset_collection(
    State(state): State<Arc<AppState>>,
) -> Result<&'static str, (axum::http::StatusCode, String)> {

    state.qdrant
        .delete_collection(COLLECTION)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    state.qdrant
        .create_collection(
            CreateCollectionBuilder::new(COLLECTION)
                .vectors_config(VectorParamsBuilder::new(VECTOR_DIM, Distance::Cosine)),
        )
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let existing = state.qdrant
    .list_collections()
    .await
    .expect("Failed to list Qdrant collections");

    if !existing.collections.iter().any(|c| c.name == COLLECTION) {
        state.qdrant
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION)
                    .vectors_config(VectorParamsBuilder::new(VECTOR_DIM, Distance::Cosine)),
            )
            .await
            .expect("Failed to create Qdrant collection");
    }

    // Create keyword index on category field for filtering
    state.qdrant
        .create_field_index(
            CreateFieldIndexCollectionBuilder::new(
                COLLECTION,
                "category",
                FieldType::Keyword,
            )
        )
        .await
        .expect("Failed to create category index");

    // Also create index on entry_type if you're using that field
    state.qdrant
        .create_field_index(
            CreateFieldIndexCollectionBuilder::new(
                COLLECTION,
                "entry_type", 
                FieldType::Keyword,
            )
        )
        .await
        .expect("Failed to create entry_type index");

        Ok("Collection reset successfully")
}

async fn ingest_local_files(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<IngestLocalRequest>,
) -> Result<Json<IngestLocalResponse>, (axum::http::StatusCode, String)> {
    let mut successful = Vec::new();
    let mut failed = Vec::new();

    for dir_path in payload.directory_paths {
        let dir = Path::new(&dir_path);
        
        if !dir.is_dir() {
            failed.push((dir_path, "Not a valid directory".to_string()));
            continue;
        }

        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
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
                Err(e) => { failed.push((file_name, format!("Read error: {e}"))); continue; }
            };

            let meta: serde_json::Value = match serde_json::from_str(&content) {
                Ok(v) => v,
                Err(e) => { failed.push((file_name, format!("Invalid JSON: {e}"))); continue; }
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
            let sol_before = meta["solidity_before"].as_str().or(meta["pattern_before"].as_str()).unwrap_or("");

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

            let vector = match get_embedding(State(state.clone()), &embed_text).await {
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
                TursoArg::Text(meta["yul_optimized"].as_str().or(meta["pattern_after"].as_str()).unwrap_or("").to_string()),
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
                "category": category
            })
            .try_into()
            .expect("Failed to parse JSON into Qdrant Payload");

            let point = PointStruct::new(Uuid::new_v4().to_string(), vector, qdrant_payload);

            if let Err(e) = state.qdrant.upsert_points(UpsertPointsBuilder::new(COLLECTION, vec![point])).await {
                failed.push((id, format!("Qdrant error: {e}")));
                continue;
            }

            successful.push(id);
        }
    }

    Ok(Json(IngestLocalResponse { successful_patterns: successful, failed_patterns: failed }))
}
// ── helpers ───────────────────────────────────────────────────────────────────

async fn get_embedding(state: State<Arc<AppState>>, text: &str) -> Result<Vec<f32>, String> {
    let mut model = state.embedding_model.lock().unwrap();
    let mut embeddings = model
        .embed(vec![text], None)
        .map_err(|e| format!("Embed error: {e}"))?;
        
    embeddings
        .pop()
        .ok_or_else(|| "Embedding generation returned empty results".to_string())
}

async fn call_deepseek(
    client: &reqwest::Client,
    source_code: &str,
    context: &str,
    api_key: &str,
) -> Result<String, String> {
    let system = format!(
        "You are Gaslite, an elite EVM gas optimizer for the Mantle L2 network. \
        You specialize in YUL assembly optimizations. \
        Return optimized Solidity/YUL code with clear explanations \
        and estimated gas savings on Mantle.\n\
        CRITICAL RULES — never violate these:\n\
        - NEVER use staticcall for state-changing functions (transfer, transferFrom, approve) — use call()\n\
        - ALWAYS use revert(0x1c, 0x04) for 4-byte custom errors, NEVER revert(0x00, 0x04)\n\
        - NEVER use sload(0) or sload(1) for named storage variables — always derive slots properly\n\
        - NEVER use Panic selectors (0x4e487b71) — always use custom 4-byte error selectors\n\
        - NEVER use string reverts like Error(string) (0x08c379a0) — always use custom errors\n\
        - Custom error pattern: mstore(0x00, 0xSELECTOR) then revert(0x1c, 0x04)\n\
        - Use call() for all state-changing external calls\n\
        - Use staticcall() ONLY for pure view functions like balanceOf\n\
        - Scratch space is 0x00-0x3f — safe to use without updating free pointer at 0x40\n\
        - For slot derivation use: mstore(0x0c, SEED) mstore(0x00, key) keccak256(0x0c, 0x20)"
    );

    let user = format!(
        "Optimize this Solidity code using the patterns below.\n\n\
        SOLIDITY:\n```solidity\n{source_code}\n```\n\n\
        RETRIEVED PATTERNS:\n{context}\n\n\
        Rules:\n\
        - Only apply patterns where appropriate for this specific code\n\
        - Never introduce bugs — correctness beats gas savings\n\
        - Use EXACT yul_optimized code from the patterns, do not paraphrase\n\
        - Use EXACT error selectors from the patterns\n\
        - Use scratch space (0x0c offset) for slot derivation\n\
        - Explain each optimization applied and why\n\
        - Note Mantle-specific gas costs where relevant\n\
        - Show optimized code first then explanation\n\
        - If a pattern does NOT apply to this code, say why and skip it"
    );

    let res = client
        .post("https://api.deepseek.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&serde_json::json!({
            "model": "deepseek-chat",
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