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


// ── constants ─────────────────────────────────────────────────────────────────
const COLLECTION: &str = "gaslite_patterns";
const VECTOR_DIM: u64 = 1536;

// ── app state ─────────────────────────────────────────────────────────────────
struct AppState {
    http: reqwest::Client,
    turso_url: String,
    turso_token: String,
    qdrant: Qdrant,
    deepseek_api_key: String,
    kilo_api_key: String,
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
    let kilo_api_key = secrets
        .get("KILO_API_KEY")
        .expect("KILO_API_KEY required");

    let http = reqwest::Client::new();

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
        kilo_api_key,
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
    let query_vec = get_kilo_embedding(&state.http, &query_text, &state.kilo_api_key)
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
    let mut found_pattern_ids: Vec<String> = Vec::new();

    for hit in results {
        let pattern_id = match hit.payload.get("pattern_id") {
            Some(v) => v.to_string().replace('"', ""),
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

    let context = pattern_contexts.join("\n\n---\n\n");

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
    let mut detected_cat = None;
    let mut key_ops = Vec::new();

    // Parse the Solidity source code
    if let Ok((source_unit, _)) = solang_parser::parse(source, 0) {
        for part in source_unit.0 {
            if let SourceUnitPart::ContractDefinition(def) = part {
                
                // 1. Check inherited contracts (e.g., `contract MyToken is ERC20`)
                for base in &def.base {
                    let base_name = base.name.identifiers.iter()
                        .map(|id| id.name.clone())
                        .collect::<Vec<_>>()
                        .join(".");
                    
                    let lower = base_name.to_lowercase();
                    if lower.contains("erc721") { detected_cat = Some("erc721"); }
                    else if lower.contains("erc1155") { detected_cat = Some("erc1155"); }
                    else if lower.contains("erc20") { detected_cat = Some("erc20"); }
                }

                // 2. Extract functions for query enrichment and heuristic matching
                for contract_part in &def.parts {
                    if let ContractPart::FunctionDefinition(func) = contract_part {
                        if let Some(name_ident) = &func.name {
                            let func_name = name_ident.name.as_str();
                            key_ops.push(func_name.to_string());

                            // Signature heuristics if base contract didn't reveal it explicitly
                            match func_name {
                                "ownerOf" | "tokenURI" | "setApprovalForAll" => {
                                    if detected_cat.is_none() { detected_cat = Some("erc721"); }
                                }
                                "safeBatchTransferFrom" | "balanceOfBatch" => {
                                    if detected_cat.is_none() { detected_cat = Some("erc1155"); }
                                }
                                "totalSupply" | "allowance" => {
                                    if detected_cat.is_none() { detected_cat = Some("erc20"); }
                                }
                                "stake" | "rewardPerToken" | "earned" => {
                                    if detected_cat.is_none() { detected_cat = Some("defi_staking"); }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    }
    
    key_ops.sort();
    key_ops.dedup();
    
    (detected_cat, key_ops)
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
            let category = meta["category"].as_str().unwrap_or("general");
            let triggers = meta["trigger_patterns"].to_string();
            let sol_before = meta["solidity_before"].as_str().or(meta["pattern_before"].as_str()).unwrap_or("");

            // Format embedding string
            let embed_text = format!(
                "TOKEN_STANDARD_NAMESPACE: {}\n// Target: {}\n// Triggers: {}\n{}",
                category.to_uppercase(),
                meta["title"].as_str().unwrap_or(""),
                triggers,
                sol_before
            );

            let vector = match get_kilo_embedding(&state.http, &embed_text, &state.kilo_api_key).await {
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
                TursoArg::Text(meta["title"].as_str().unwrap_or("").to_string()),
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

async fn get_kilo_embedding(
    client: &reqwest::Client,
    text: &str,
    api_key: &str,
) -> Result<Vec<f32>, String> {
    let res = client
        .post("https://api.kilo.ai/api/gateway/embeddings")
        .bearer_auth(api_key)
        .json(&serde_json::json!({
            "model": "text-embedding-3-small",
            "input": text
        }))
        .send()
        .await
        .map_err(|e| format!("Kilo request failed: {e}"))?;

    if !res.status().is_success() {
        return Err(format!("Kilo returned status {}", res.status()));
    }

    let json: serde_json::Value = res
        .json()
        .await
        .map_err(|e| format!("Kilo parse error: {e}"))?;

    let embedding = json["data"][0]["embedding"]
        .as_array()
        .ok_or("Missing embedding array")?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect::<Vec<f32>>();

    if embedding.is_empty() {
        return Err("Kilo returned empty embedding".to_string());
    }

    Ok(embedding)
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