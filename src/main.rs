//TODO erc1155

mod admin;
mod ai;
mod analyze;
mod db;
mod dto;
mod embedding;
mod forge;
mod health;
mod logging;
mod normalize;
mod optimize;
mod orchestrator;
mod retrieval;
mod rig_agent;
mod state;
mod tools;
mod utils;
mod verify_agent;

use ai::Embedder;
use db::Turso;
use state::{AppState, COLLECTION, VECTOR_DIM};

use axum::{
    Router,
    routing::{get, post},
};
use qdrant_client::{
    Qdrant,
    qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder},
};
use rig_core::{client::ProviderClient, providers::deepseek};
use std::sync::Arc;
use tracing::{info, warn};

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

    admin::create_qdrant_indexes(&state)
        .await
        .expect("Failed to create Qdrant indexes");

    // Build the structural "Seeker" matcher from the knowledge base.
    {
        let matcher = state::load_pattern_matcher(&state.db).await;
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
        .route("/health", get(health::health_check))
        .route(
            "/api/optimize",
            post(optimize::optimize_contract),
        )
        .route(
            "/api/verify",
            post(forge::verify_contract),
        )
        .route(
            "/api/admin/ingest-local",
            post(admin::ingest_local_files),
        )
        .route(
            "/api/admin/qdrant/reset",
            post(admin::reset_collection),
        )
        .with_state(state)
        // Allow the browser-based web UI (different origin) to call the API.
        .layer(cors_layer());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    info!(
        "Gaslite listening on {}",
        listener.local_addr()?
    );
    axum::serve(listener, router).await?;
    Ok(())
}

/// CORS for the browser web UI. Permissive by default (public optimize API, no
/// cookies/credentials); restrict to a comma-separated allowlist via
/// `CORS_ALLOW_ORIGINS` (e.g. "https://gaslite.example,https://foo.bar").
fn cors_layer() -> tower_http::cors::CorsLayer {
    use tower_http::cors::{Any, CorsLayer};
    let base = CorsLayer::new()
        .allow_methods(Any)
        .allow_headers(Any);
    match std::env::var("CORS_ALLOW_ORIGINS") {
        Ok(list) if !list.trim().is_empty() => {
            let origins: Vec<axum::http::HeaderValue> = list
                .split(',')
                .filter_map(|o| {
                    o.trim()
                        .parse()
                        .ok()
                })
                .collect();
            base.allow_origin(origins)
        }
        _ => base.allow_origin(Any),
    }
}
