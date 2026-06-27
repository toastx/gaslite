//! `GET /health` — liveness of the server plus connectivity probes for the two
//! backing stores (Turso, Qdrant).

use std::sync::Arc;

use axum::{Json, extract::State};
use serde::Serialize;
use tracing::{info, warn};

use crate::state::AppState;

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
pub(crate) struct HealthResponse {
    status: &'static str, // "ok" | "degraded"
    service: &'static str,
    server: &'static str, // "ok" — if this handler runs, the server is up
    checks: HealthChecks,
}

pub(crate) async fn health_check(
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
