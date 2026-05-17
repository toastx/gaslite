use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use libsql::Database;
use qdrant_client::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// --- App State Configuration ---
struct AppState {
    turso_db: libsql::Connection,
    qdrant_client: QdrantClient,
    deepseek_api_key: String,
    kilo_api_key: String,
}

// --- Request/Response DTOs ---
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
struct SeedPatternPayload {
    metadata: serde_json::Value,
    vector: Vec<f32>, 
}

#[shuttle_runtime::main]
async fn main(
    #[shuttle_runtime::Secrets] secrets: shuttle_runtime::SecretStore,
) -> shuttle_axum::ShuttleAxum {
    
    // 1. Fetch secrets via Shuttle's SecretStore (prevents crashes)
    let deepseek_api_key = secrets.get("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY is required");
    let qdrant_api_key = secrets.get("QDRANT_API_KEY").expect("QDRANT_API_KEY is required");
    let qdrant_url = secrets.get("QDRANT_CLUSTER_URL").expect("QDRANT_CLUSTER_URL is required");
    let turso_url = secrets.get("TURSO_DATABASE_URL").expect("TURSO_DATABASE_URL is required");
    let turso_token = secrets.get("TURSO_AUTH_TOKEN").expect("TURSO_AUTH_TOKEN is required");
    let kilo_api_key = secrets.get("KILO_API_KEY").expect("KILO_API_KEY is required");

    // 2. Initialize Turso DB Client
    let db = Database::open_remote(turso_url, turso_token)
        .expect("Failed to connect to Turso remote database");
    let turso_db = db.connect().expect("Failed to open connection to Turso");

    // 3. Initialize Qdrant Client Configuration
    let config = QdrantClientConfig::from_url(&qdrant_url).with_api_key(qdrant_api_key);
    let qdrant_client = QdrantClient::new(Some(config)).expect("Failed to initialize Qdrant client");

    // 4. Wrap everything in a Thread-Safe Shared State
    let shared_state = Arc::new(AppState {
        turso_db,
        qdrant_client,
        deepseek_api_key,
        kilo_api_key,
    });

    // 5. Build router and register state
    let router = Router::new()
        .route("/health", get(health_check))
        .route("/api/optimize", post(optimize_contract))
        .route("/api/admin/seed", post(seed_gaslite_pattern)) // WIRED UP
        .with_state(shared_state);

    Ok(router.into())
}

// --- Route Handlers ---

async fn health_check() -> &'static str {
    "Gaslite Engine is online and ready to optimize Mantle smart contracts."
}

async fn optimize_contract(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<OptimizeRequest>,
) -> Result<Json<OptimizeResponse>, (axum::http::StatusCode, String)> {
    
    let code_embedding = get_embedding(&payload.contract_source, &state.kilo_api_key).await;

    let search_result = state
        .qdrant_client
        .search_points(&SearchPoints {
            collection_name: "gaslite_patterns".to_string(),
            vector: code_embedding,
            limit: 3,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut pattern_contexts = Vec::new();
    let mut found_pattern_ids = Vec::new();

    for point in search_result.result {
        if let Some(payload) = point.payload.get("pattern_id") {
            if let Some(pattern_id_str) = payload.as_str() {
                found_pattern_ids.push(pattern_id_str.to_string());

                let query = format!("SELECT explanation, yul_optimized FROM optimization_patterns WHERE id = '{}'", pattern_id_str);
                if let Ok(Some(row)) = state.turso_db.query(&query, ()).and_then(|mut r| r.next()) {
                    let expl = row.get::<String>(0).unwrap_or_default();
                    let yul = row.get::<String>(1).unwrap_or_default();
                    pattern_contexts.push(format!("Pattern: {}\nExplanation: {}\nCode:\n{}", pattern_id_str, expl, yul));
                }
            }
        }
    }

    let context_for_llm = pattern_contexts.join("\n---\n");
    
    let deepseek_response = call_deepseek_v3(&payload.contract_source, &context_for_llm, &state.deepseek_api_key)
        .await
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e))?;

    Ok(Json(OptimizeResponse {
        analysis: "Gaslite optimization complete.".to_string(),
        suggested_patterns: found_pattern_ids,
        optimized_code: deepseek_response,
    }))
}

async fn seed_gaslite_pattern(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SeedPatternPayload>,
) -> Result<&'static str, (axum::http::StatusCode, String)> {
    
    let meta = &payload.metadata;
    let id = meta["id"].as_str().ok_or((axum::http::StatusCode::BAD_REQUEST, "Missing ID".to_string()))?;
    let category = meta["category"].as_str().unwrap_or("general");
    let version = meta["version"].as_str().unwrap_or("1.0");
    let title = meta["title"].as_str().unwrap_or("");
    let source = meta["source"].as_str().unwrap_or("");
    let source_file = meta["source_file"].as_str().unwrap_or("");
    let difficulty = meta["difficulty"].as_str().unwrap_or("medium");
    let mantle_specific = meta["mantle_specific"].as_bool().unwrap_or(false) as i32;
    let evm_version = meta["evm_version"].as_str().unwrap_or("paris");
    
    let trigger_patterns = meta["trigger_patterns"].to_string();
    let solidity_before = meta["solidity_before"].as_str().unwrap_or("");
    let yul_optimized = meta["yul_optimized"].as_str().unwrap_or("");
    let patterns_used = meta["patterns_used"].to_string();
    let explanation = meta["explanation"].as_str().unwrap_or("");
    let risk_level = meta["risk_level"].as_str().unwrap_or("low");
    let when_to_apply = meta["when_to_apply"].as_str().unwrap_or("");
    let when_not_to_apply = meta["when_not_to_apply"].as_str().unwrap_or("");

    let sql = "INSERT OR REPLACE INTO optimization_patterns (id, category, version, title, source, source_file, difficulty, mantle_specific, evm_version, trigger_patterns, solidity_before, yul_optimized, patterns_used, explanation, risk_level, when_to_apply, when_not_to_apply) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)";
    
    state.turso_db.execute(sql, libsql::params![
        id, category, version, title, source, source_file, difficulty, mantle_specific, evm_version,
        trigger_patterns, solidity_before, yul_optimized, patterns_used, explanation, risk_level, when_to_apply, when_not_to_apply
    ]).map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Turso Error: {}", e)))?;

    let mut points_payload = std::collections::HashMap::new();
    points_payload.insert("pattern_id".to_string(), id.to_string().into());

    let point_id = uuid::Uuid::new_v4().to_string(); 

    state.qdrant_client.upsert_points(&UpsertPoints {
        collection_name: "gaslite_patterns".to_string(),
        points: vec![PointStruct::new(
            point_id,
            payload.vector,
            points_payload.into()
        )],
        ..Default::default()
    }).await.map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Qdrant Error: {}", e)))?;

    Ok("Gaslite database tracking initialized successfully.")
}

// --- Downstream External API Helpers ---

async fn get_embedding(text: &str, api_key: &str) -> Result<Vec<f32>, String> {
    let client = reqwest::Client::new();
    
    let api_url = "https://api.kilo.ai/api/gateway"; 
    
    let body = serde_json::json!({
        "model": "text-embedding-3-small", 
        "input": text
    });

    let res = client
        .post(api_url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    if res.status().is_success() {
        let json_res: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
        
        // Extract the float array from the standard OpenAI JSON response structure
        let embedding = json_res["data"][0]["embedding"]
            .as_array()
            .ok_or("Failed to parse embedding array from gateway response")?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Vec<f32>>();
            
        Ok(embedding)
    } else {
        Err(format!("Gateway API error: {}", res.status()))
    }
}

async fn call_deepseek_v3(source_code: &str, context: &str, api_key: &str) -> Result<String, String> {
    let client = reqwest::Client::new();
    
    let prompt = format!(
        "You are Gaslite, an elite EVM gas optimizer for Mantle chain.\n\
         Analyze this user code and optimize it using the reference patterns provided.\n\n\
         ### Target User Code:\n{}\n\n\
         ### Reference Optimization Patterns (RAG Context):\n{}\n\n\
         Return ONLY the fully optimized code with relevant inline Yul / Assembly adjustments.",
        source_code, context
    );

    let body = serde_json::json!({
        "model": "deepseek-chat", 
        "messages": [
            {"role": "system", "content": "You are a professional compiler and smart contract optimization architect."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    });

    let res = client
        .post("https://api.deepseek.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if res.status().is_success() {
        let json_res: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
        let optimized_output = json_res["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("Failed to process code response.")
            .to_string();
        Ok(optimized_output)
    } else {
        Err(format!("DeepSeek API returned error status: {}", res.status()))
    }
}