use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use libsql::Database;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

// ── constants ─────────────────────────────────────────────────────────────────
const COLLECTION: &str = "gaslite_patterns";
const VECTOR_DIM: u64 = 1536; // text-embedding-3-small

// ── app state ─────────────────────────────────────────────────────────────────
struct AppState {
    turso_db: libsql::Connection,
    qdrant: Qdrant,
    deepseek_api_key: String,
    kilo_api_key: String,
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
    directory_path: String,
}

#[derive(Serialize)]
struct IngestLocalResponse {
    successful_patterns: Vec<String>,
    failed_patterns: Vec<(String, String)>,
}

// ── entry point ───────────────────────────────────────────────────────────────
#[shuttle_runtime::main]
async fn main(
    #[shuttle_runtime::Secrets] secrets: shuttle_runtime::SecretStore,
) -> shuttle_axum::ShuttleAxum {
    let deepseek_api_key = secrets
        .get("DEEPSEEK_API_KEY")
        .expect("DEEPSEEK_API_KEY is required");
    let qdrant_api_key = secrets
        .get("QDRANT_API_KEY")
        .expect("QDRANT_API_KEY is required");
    let qdrant_url = secrets
        .get("QDRANT_CLUSTER_URL")
        .expect("QDRANT_CLUSTER_URL is required");
    let turso_url = secrets
        .get("TURSO_DATABASE_URL")
        .expect("TURSO_DATABASE_URL is required");
    let turso_token = secrets
        .get("TURSO_AUTH_TOKEN")
        .expect("TURSO_AUTH_TOKEN is required");
    let kilo_api_key = secrets
        .get("KILO_API_KEY")
        .expect("KILO_API_KEY is required");

    // turso
    let db = Database::open_remote(turso_url, turso_token)
        .expect("Failed to connect to Turso");
    let turso_db = db.connect().expect("Failed to open Turso connection");

    // run migrations
    turso_db
        .execute(
            "CREATE TABLE IF NOT EXISTS optimization_patterns (
                id               TEXT PRIMARY KEY,
                category         TEXT,
                version          TEXT,
                title            TEXT,
                source           TEXT,
                source_file      TEXT,
                difficulty       TEXT,
                mantle_specific  INTEGER,
                evm_version      TEXT,
                trigger_patterns TEXT,
                solidity_before  TEXT,
                yul_optimized    TEXT,
                patterns_used    TEXT,
                explanation      TEXT,
                risk_level       TEXT,
                when_to_apply    TEXT,
                when_not_to_apply TEXT
            )",
            (),
        )
        .await
        .expect("Failed to run migrations");

    // qdrant
    let qdrant = Qdrant::from_url(&qdrant_url)
        .api_key(qdrant_api_key)
        .build()
        .expect("Failed to connect to Qdrant");

    // create collection if missing
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

    let shared_state = Arc::new(AppState {
        turso_db,
        qdrant,
        deepseek_api_key,
        kilo_api_key,
    });

    let router = Router::new()
        .route("/health", get(health_check))
        .route("/api/optimize", post(optimize_contract))
        .route("/api/admin/ingest-local", post(ingest_local_files))
        .with_state(shared_state);

    Ok(router.into())
}

// ── handlers ──────────────────────────────────────────────────────────────────

async fn health_check() -> &'static str {
    "Gaslite Engine is online and ready to optimize Mantle smart contracts."
}

async fn optimize_contract(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<OptimizeRequest>,
) -> Result<Json<OptimizeResponse>, (axum::http::StatusCode, String)> {
    // 1. embed the incoming code
    let query_vec = get_kilo_embedding(&payload.contract_source, &state.kilo_api_key)
        .await
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e))?;

    // 2. search qdrant for top 3 patterns
    let search_result = state
        .qdrant
        .search_points(
            SearchPointsBuilder::new(COLLECTION, query_vec, 3).with_payload(true),
        )
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // 3. for each hit, fetch full pattern from turso
    let mut pattern_contexts: Vec<String> = Vec::new();
    let mut found_pattern_ids: Vec<String> = Vec::new();

    for hit in search_result.result {
        let pattern_id = match hit.payload.get("pattern_id") {
            Some(v) => v.to_string().replace('"', ""),
            None => continue,
        };

        found_pattern_ids.push(pattern_id.clone());

        let mut rows = state
            .turso_db
            .query(
                "SELECT title, explanation, yul_optimized, risk_level, when_not_to_apply \
                 FROM optimization_patterns WHERE id = ?1",
                libsql::params![pattern_id.clone()],
            )
            .await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        {
            let title: String = row.get(0).unwrap_or_default();
            let explanation: String = row.get(1).unwrap_or_default();
            let yul: String = row.get(2).unwrap_or_default();
            let risk: String = row.get(3).unwrap_or_default();
            let not_apply: String = row.get(4).unwrap_or_default();

            pattern_contexts.push(format!(
                "PATTERN: {title}\n\
                 Explanation: {explanation}\n\
                 Optimized YUL:\n{yul}\n\
                 Risk level: {risk}\n\
                 Do NOT apply when: {not_apply}"
            ));
        }
    }

    let context = pattern_contexts.join("\n\n---\n\n");

    // 4. call deepseek with context
    let optimized_code =
        call_deepseek(&payload.contract_source, &context, &state.deepseek_api_key)
            .await
            .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e))?;

    Ok(Json(OptimizeResponse {
        analysis: format!(
            "Gaslite found {} relevant patterns for your contract.",
            found_pattern_ids.len()
        ),
        suggested_patterns: found_pattern_ids,
        optimized_code,
    }))
}

async fn ingest_local_files(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<IngestLocalRequest>,
) -> Result<Json<IngestLocalResponse>, (axum::http::StatusCode, String)> {
    let mut successful_patterns: Vec<String> = Vec::new();
    let mut failed_patterns: Vec<(String, String)> = Vec::new();

    let dir = Path::new(&payload.directory_path);

    if !dir.is_dir() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            format!("'{}' is not a valid directory", payload.directory_path),
        ));
    }

    let entries = fs::read_dir(dir).map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Cannot read directory: {e}"),
        )
    })?;

    for entry in entries.flatten() {
        let file_path = entry.path();

        if file_path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }

        let file_name = file_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // read + parse JSON
        let content = match fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(e) => {
                failed_patterns.push((file_name, format!("Read error: {e}")));
                continue;
            }
        };

        let meta: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                failed_patterns.push((file_name, format!("Invalid JSON: {e}")));
                continue;
            }
        };

        let id = match meta["id"].as_str() {
            Some(v) => v.to_string(),
            None => {
                failed_patterns.push((file_name, "Missing 'id' field".to_string()));
                continue;
            }
        };

        // build embedding input
        let triggers = meta["trigger_patterns"].to_string();
        let explanation = meta["explanation"]
            .as_str()
            .or(meta["description"].as_str())
            .unwrap_or("");
        let title = meta["title"].as_str().unwrap_or("");
        let when_to_apply = meta["when_to_apply"].as_str().unwrap_or("");

        let embed_text = format!(
            "Title: {title}\nTriggers: {triggers}\nExplanation: {explanation}\nWhen to apply: {when_to_apply}"
        );

        // get embedding
        let vector = match get_kilo_embedding(&embed_text, &state.kilo_api_key).await {
            Ok(v) => v,
            Err(e) => {
                failed_patterns.push((id, format!("Embedding error: {e}")));
                continue;
            }
        };

        // write to turso
        let sql = "INSERT OR REPLACE INTO optimization_patterns \
            (id, category, version, title, source, source_file, difficulty, \
             mantle_specific, evm_version, trigger_patterns, solidity_before, \
             yul_optimized, patterns_used, explanation, risk_level, \
             when_to_apply, when_not_to_apply) \
            VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17)";

        let params = libsql::params![
            id.clone(),
            meta["category"].as_str().unwrap_or("general"),
            meta["version"].as_str().unwrap_or("1.0"),
            title,
            meta["source"].as_str().unwrap_or(""),
            meta["source_file"].as_str().unwrap_or(""),
            meta["difficulty"].as_str().unwrap_or("medium"),
            meta["mantle_specific"].as_bool().unwrap_or(false) as i32,
            meta["evm_version"].as_str().unwrap_or("paris"),
            triggers,
            meta["solidity_before"]
                .as_str()
                .or(meta["pattern_before"].as_str())
                .unwrap_or(""),
            meta["yul_optimized"]
                .as_str()
                .or(meta["pattern_after"].as_str())
                .unwrap_or(""),
            meta["patterns_used"].to_string(),
            explanation,
            meta["risk_level"].as_str().unwrap_or("low"),
            when_to_apply,
            meta["when_not_to_apply"].as_str().unwrap_or("")
        ];

        if let Err(e) = state.turso_db.execute(sql, params).await {
            failed_patterns.push((id, format!("Turso error: {e}")));
            continue;
        }

        // upsert into qdrant
        let mut qdrant_payload: HashMap<String, qdrant_client::qdrant::Value> = HashMap::new();
        qdrant_payload.insert(
            "pattern_id".to_string(),
            qdrant_client::qdrant::Value::from(id.clone()),
        );

        let point = PointStruct::new(Uuid::new_v4().to_string(), vector, qdrant_payload);

        if let Err(e) = state
            .qdrant
            .upsert_points(UpsertPointsBuilder::new(COLLECTION, vec![point]))
            .await
        {
            failed_patterns.push((id, format!("Qdrant error: {e}")));
            continue;
        }

        successful_patterns.push(id);
    }

    Ok(Json(IngestLocalResponse {
        successful_patterns,
        failed_patterns,
    }))
}

// ── helpers ───────────────────────────────────────────────────────────────────

async fn get_kilo_embedding(text: &str, api_key: &str) -> Result<Vec<f32>, String> {
    let client = reqwest::Client::new();

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
        .map_err(|e| format!("Kilo response parse error: {e}"))?;

    let embedding = json["data"][0]["embedding"]
        .as_array()
        .ok_or("Missing embedding array in Kilo response")?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect::<Vec<f32>>();

    if embedding.is_empty() {
        return Err("Kilo returned empty embedding vector".to_string());
    }

    Ok(embedding)
}

async fn call_deepseek(
    source_code: &str,
    context: &str,
    api_key: &str,
) -> Result<String, String> {
    let client = reqwest::Client::new();

    let system = "You are Gaslite, an elite EVM gas optimizer for the Mantle L2 network. \
                  You specialize in YUL assembly optimizations. \
                  Return optimized Solidity/YUL code with clear explanations of each change \
                  and estimated gas savings on Mantle.";

    let user = format!(
        "Optimize this Solidity code using the patterns below.\n\n\
         SOLIDITY TO OPTIMIZE:\n```solidity\n{source_code}\n```\n\n\
         RETRIEVED OPTIMIZATION PATTERNS:\n{context}\n\n\
         Rules:\n\
         - Only apply patterns where appropriate\n\
         - Never introduce bugs — correctness beats gas savings\n\
         - Explain each optimization applied\n\
         - Note Mantle-specific gas costs where relevant\n\
         - Show optimized code first, then explanation"
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
        .map_err(|e| format!("DeepSeek response parse error: {e}"))?;

    json["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or("Missing content in DeepSeek response".to_string())
}