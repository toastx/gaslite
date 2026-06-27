//! Admin endpoints: local knowledge-base ingest, Qdrant collection reset, and the
//! Qdrant field-index creation shared by startup and reset.

use std::{fs, path::Path, sync::Arc};

use axum::{Json, extract::State};
use qdrant_client::{
    Payload,
    qdrant::{
        CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance, FieldType,
        PointStruct, UpsertPointsBuilder, VectorParamsBuilder,
    },
};
use tracing::{error, info, warn};
use uuid::Uuid;

use super::dto::{IngestLocalRequest, IngestLocalResponse};
use crate::{
    kb::db::TursoArg,
    state::{AppState, COLLECTION, VECTOR_DIM, load_pattern_matcher},
};

pub(crate) async fn create_qdrant_indexes(state: &AppState) -> Result<(), String> {
    for field in ["category", "type"] {
        state
            .qdrant
            .create_field_index(CreateFieldIndexCollectionBuilder::new(
                COLLECTION,
                field,
                FieldType::Keyword,
            ))
            .await
            .map_err(|e| format!("Failed to create Qdrant index on '{field}': {e}"))?;
        info!("  index created: {}", field);
    }
    Ok(())
}

pub(crate) async fn reset_collection(
    State(state): State<Arc<AppState>>,
) -> Result<&'static str, (axum::http::StatusCode, String)> {
    info!("=== RESET COLLECTION ===");

    state
        .qdrant
        .delete_collection(COLLECTION)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    info!("  deleted : {}", COLLECTION);

    state
        .qdrant
        .create_collection(
            CreateCollectionBuilder::new(COLLECTION)
                .vectors_config(VectorParamsBuilder::new(VECTOR_DIM, Distance::Cosine)),
        )
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    info!("  created : {} ({} dims, cosine)", COLLECTION, VECTOR_DIM);

    create_qdrant_indexes(&state)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;

    // The knowledge base changed — cached optimizations are now stale (L1 + L2).
    state.cache.lock().unwrap().clear();
    if let Err(e) = state.db.execute("DELETE FROM optimize_cache", vec![]).await {
        warn!("  cache   : L2 clear failed: {e}");
    }
    info!("  cache   : cleared (L1 + L2)");

    info!("========================");
    Ok("Collection reset successfully")
}

pub(crate) async fn ingest_local_files(
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
            },
        };

        for entry in entries.flatten() {
            let file_path = entry.path();
            if file_path.extension().and_then(|s| s.to_str()) != Some("json") {
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
                    warn!("    ! read error {}: {}", file_name, e);
                    failed.push((file_name, format!("Read error: {e}")));
                    continue;
                },
            };

            let meta: serde_json::Value = match serde_json::from_str(&content) {
                Ok(v) => v,
                Err(e) => {
                    warn!("    ! invalid JSON {}: {}", file_name, e);
                    failed.push((file_name, format!("Invalid JSON: {e}")));
                    continue;
                },
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
            let sol_before = meta["solidity_before"]
                .as_str()
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

            let vector = match state.embedder.clone().embed(&embed_text).await {
                Ok(v) => v,
                Err(e) => {
                    failed.push((id, format!("Embedding error: {e}")));
                    continue;
                },
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
                TursoArg::Integer(
                    (meta["mantle_specific"].as_bool().unwrap_or(false) as i64).to_string(),
                ),
                TursoArg::Text(meta["evm_version"].as_str().unwrap_or("paris").to_string()),
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

            if let Err(e) = state
                .qdrant
                .upsert_points(UpsertPointsBuilder::new(COLLECTION, vec![point]))
                .await
            {
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

    // Refresh the structural matcher with the newly ingested patterns.
    {
        let matcher = load_pattern_matcher(&state.db).await;
        info!(
            "  structural matcher: {} templates (rebuilt)",
            matcher.len()
        );
        *state.pattern_matcher.write().unwrap() = Arc::new(matcher);
    }

    Ok(Json(IngestLocalResponse {
        successful_patterns: successful,
        failed_patterns: failed,
    }))
}
