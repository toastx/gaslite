//! Shared application state + the cache/knowledge-base helpers that hang off it.

use std::sync::Arc;

use qdrant_client::Qdrant;
use rig_core::providers::deepseek;
use tracing::warn;

use crate::{
    api::dto::OptimizeResponse,
    kb::{
        ai::Embedder,
        db::{Turso, TursoArg},
        normalize,
    },
    logging,
};

// ── constants ─────────────────────────────────────────────────────────────────
pub(crate) const COLLECTION: &str = "gaslite_patterns";
pub(crate) const VECTOR_DIM: u64 = 384;
/// Max functions optimized concurrently (bounds in-flight DeepSeek requests).
pub(crate) const MAX_PARALLEL_FUNCS: usize = 6;
/// Contracts at or below BOTH thresholds skip the router LLM call — they are
/// always routed oneshot, so the round-trip is pure latency.
pub(crate) const ONESHOT_MAX_FUNCS: usize = 4;
pub(crate) const ONESHOT_MAX_BYTES: usize = 4096;

pub(crate) struct AppState {
    pub(crate) db: Arc<Turso>,
    pub(crate) qdrant: Arc<Qdrant>,
    pub(crate) deepseek: deepseek::Client,
    pub(crate) embedder: Arc<Embedder>,
    pub(crate) forge_available: bool,
    /// Result cache keyed on the *normalized* contract source (comments/whitespace
    /// stripped) — see `normalize::lexical_key`. A hit skips the whole agent +
    /// forge pipeline. Only successful/one-shot results are cached so transient
    /// failures can be retried. Cleared on Qdrant reset.
    pub(crate) cache: std::sync::Mutex<std::collections::HashMap<String, OptimizeResponse>>,
    /// Deterministic structural pattern matcher (the "Seeker"), rebuilt from the
    /// knowledge base on startup and after ingest/reset. Read-snapshotted per
    /// request so reads never block writes.
    pub(crate) pattern_matcher: std::sync::RwLock<Arc<normalize::PatternMatcher>>,
    /// Where finished runs are recorded (stubbed: `NoopSink` → tracing). This is the
    /// seam for on-chain Mantle logging — see [`logging`].
    pub(crate) logging: Arc<dyn logging::LoggingSink>,
}

/// L2 cache read — fetch a stored optimization from Turso by normalized key.
///
/// Filters on `verified = 1`. The L2 store is durable and shared across restarts and
/// deployments, so a row written by a deployment without forge (or by a build predating
/// the `verified` column) must never be served as though it were proven.
pub(crate) async fn db_cache_get(db: &Turso, key: &str) -> Option<OptimizeResponse> {
    let rows = db
        .query(
            "SELECT response FROM optimize_cache WHERE cache_key = ? AND verified = 1",
            vec![TursoArg::Text(key.to_string())],
        )
        .await
        .ok()?;
    let json = rows.first()?.get("response")?.as_str()?;
    serde_json::from_str::<OptimizeResponse>(json).ok()
}

/// L2 cache write — persist a forge-verified optimization to Turso (write-through).
///
/// Call this ONLY for a rewrite that forge proved behaviourally equivalent on every
/// target function; rows land with `verified = 1` and are served forever after.
pub(crate) async fn db_cache_put(
    db: &Turso,
    key: &str,
    resp: &OptimizeResponse,
) -> Result<(), String> {
    let json = serde_json::to_string(resp).map_err(|e| e.to_string())?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    db.execute(
        "INSERT OR REPLACE INTO optimize_cache (cache_key, response, created_at, verified) \
         VALUES (?,?,?,1)",
        vec![
            TursoArg::Text(key.to_string()),
            TursoArg::Text(json),
            TursoArg::Integer(now.to_string()),
        ],
    )
    .await
}

/// Load the structural pattern matcher from the knowledge base (Turso).
pub(crate) async fn load_pattern_matcher(db: &Turso) -> normalize::PatternMatcher {
    let rows = match db
        .query(
            "SELECT id, solidity_before FROM optimization_patterns WHERE solidity_before != ''",
            vec![],
        )
        .await
    {
        Ok(rows) => rows,
        Err(e) => {
            warn!("pattern matcher: KB query failed: {e}");
            return normalize::PatternMatcher::default();
        },
    };
    let pairs = rows.into_iter().filter_map(|row| {
        let id = row.get("id")?.as_str()?.to_string();
        let before = row.get("solidity_before")?.as_str()?.to_string();
        Some((id, before))
    });
    normalize::PatternMatcher::build(pairs)
}
