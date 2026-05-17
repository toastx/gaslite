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

// Expected format from your parsed files
#[derive(Serialize, Deserialize, Debug, Clone)]
struct RagPatternJson {
    id: String,
    category: String,
    title: String,
    yul_optimized: String,
    explanation: String,
}

#[tokio::main]
async fn main() -> shuttle_axum::ShuttleAxum {
    // 1. Fetch environment variables safely
    let deepseek_api_key = std::env::var("DEEPSEEK_API_KEY")
        .expect("DEEPSEEK_API_KEY must be set");
    let qdrant_api_key = std::env::var("QDRANT_API_KEY")
        .expect("QDRANT_API_KEY must be set");
    let qdrant_url = std::env::var("QDRANT_CLUSTER_URL")
        .expect("QDRANT_CLUSTER_URL must be set");
    let turso_url = std::env::var("TURSO_DATABASE_URL")
        .expect("TURSO_DATABASE_URL must be set");
    let turso_token = std::env::var("TURSO_AUTH_TOKEN")
        .expect("TURSO_AUTH_TOKEN must be set");

    // 2. Initialize Turso DB Client
    let db = Database::open_remote(turso_url, turso_token)
        .expect("Failed to connect to Turso remote database");
    let turso_db = db.connect().expect("Failed to open connection to Turso");

    // 3. Initialize Qdrant Client Configuration
    let config = QdrantClientConfig::from_url(&qdrant_url)
        .with_api_key(qdrant_api_key);
    let qdrant_client = QdrantClient::new(Some(config))
        .expect("Failed to initialize Qdrant client");

    // 4. Wrap everything in a Thread-Safe Shared State
    let shared_state = Arc::new(AppState {
        turso_db,
        qdrant_client,
        deepseek_api_key,
    });

    // 5. Build router and register state
    let router = Router::new()
        .route("/health", get(health_check))
        .route("/api/optimize", post(optimize_contract))
        .with_state(shared_state);

    Ok(router.into())
}

// --- Route Handlers ---

async fn health_check() -> &'static str {
    "Mantle Gas Optimizer Service is active!"
}

async fn optimize_contract(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<OptimizeRequest>,
) -> Result<Json<OptimizeResponse>, (axum::http::StatusCode, String)> {
    
    // STEP 1: Generate embeddings for user source code via DeepSeek or your embedding service
    // For production, you will call an embedding model (e.g., text-embedding-3-small) here.
    let code_embedding = match get_embedding(&payload.contract_source, &state.deepseek_api_key).await {
        Ok(vec) => vec,
        Err(e) => return Err((axum::http::StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    // STEP 2: Query Qdrant for matching optimization patterns
    let collection_name = "mantle_gas_patterns";
    let search_result = state
        .qdrant_client
        .search_points(&SearchPoints {
            collection_name: collection_name.to_string(),
            vector: code_embedding,
            limit: 3, // Retrieve top 3 matching patterns
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // STEP 3: Extract pattern IDs and fetch structural data/JSON metadata from Turso DB
    let mut pattern_contexts = Vec::new();
    let mut found_pattern_ids = Vec::new();

    for point in search_result.result {
        if let Some(payload) = point.payload.get("pattern_id") {
            if let Some(pattern_id_str) = payload.as_str() {
                found_pattern_ids.push(pattern_id_str.to_string());

                // Query metadata from Turso (assuming you saved your structural JSON strings there indexed by ID)
                let query = format!("SELECT json_data FROM optimization_patterns WHERE id = '{}'", pattern_id_str);
                if let Ok(Some(row)) = state.turso_db.query(&query, ()).and_then(|mut r| r.next()) {
                    if let Ok(raw_json) = row.get::<String>(0) {
                        pattern_contexts.push(raw_json);
                    }
                }
            }
        }
    }

    // STEP 4: Build Context Injection for DeepSeek-V3 Inference
    let context_for_llm = pattern_contexts.join("\n---\n");
    
    let deepseek_response = call_deepseek_v3(
        &payload.contract_source,
        &context_for_llm,
        &state.deepseek_api_key
    ).await
    .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e))?;

    // STEP 5: Return calculated responses back to client
    Ok(Json(OptimizeResponse {
        analysis: "DeepSeek-V3 gas assessment complete using target vector injection.".to_string(),
        suggested_patterns: found_pattern_ids,
        optimized_code: deepseek_response,
    }))
}

// --- Downstream External API Helpers ---

async fn get_embedding(text: &str, _api_key: &str) -> Result<Vec<f32>, String> {
    // Dummy vector placeholder for testing. 
    // Substitute this routine with your real text embedding generation API call.
    Ok(vec![0.023, -0.432, 0.112, 0.984]) 
}

async fn call_deepseek_v3(source_code: &str, context: &str, api_key: &str) -> Result<String, String> {
    let client = reqwest::Client::new();
    
    // Construct the structured prompt payload instructions targeting DeepSeek-V3 execution
    let prompt = format!(
        "You are an elite EVM gas optimizer for Mantle chain.\n\
         Analyze this user code and optimize it using the reference patterns provided.\n\n\
         ### Target User Code:\n{}\n\n\
         ### Reference Optimization Patterns (RAG Context):\n{}\n\n\
         Return ONLY the fully optimized code with relevant inline Yul / Assembly adjustments.",
        source_code, context
    );

    let body = serde_json::json!({
        "model": "deepseek-chat", // DeepSeek-V3 API Identifier
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