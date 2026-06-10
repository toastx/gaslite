//! Audit-log sink for optimization runs.
//!
//! This is the seam for "log optimization results on the Mantle network". For now
//! only a [`NoopSink`] exists (writes the run to `tracing`). A real on-chain impl —
//! alloy + a funded signer + a deployed logging contract — is a separate, larger
//! piece of work and is intentionally OUT OF SCOPE here. New sinks implement
//! [`LoggingSink`]; `AppState` holds an `Arc<dyn LoggingSink>` so the call site in
//! `optimize_contract` never changes when a chain sink lands.

use std::future::Future;
use std::pin::Pin;

use tracing::info;

/// Boxed future alias so `LoggingSink` stays object-safe (`Arc<dyn LoggingSink>`).
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// One optimization run, the unit we'd anchor on-chain.
#[derive(Debug, Clone)]
pub struct RunLog {
    /// Non-cryptographic hash of the original contract source (run identity).
    pub contract_hash: String,
    /// Which path produced the result: "oneshot" | "decompose" | "fallback".
    pub mode: &'static str,
    pub gas_original: Option<u64>,
    pub gas_optimized: Option<u64>,
    pub gas_saved: Option<i64>,
    pub pattern_ids: Vec<String>,
    /// Unix seconds.
    pub ts: u64,
}

/// Where a finished run is recorded. Today: tracing. Tomorrow: Mantle.
pub trait LoggingSink: Send + Sync {
    fn log_run<'a>(
        &'a self,
        run: &'a RunLog,
    ) -> BoxFuture<'a, Result<(), String>>;
}

/// Default sink — emits the run to `tracing`, performs no I/O.
pub struct NoopSink;

impl LoggingSink for NoopSink {
    fn log_run<'a>(
        &'a self,
        run: &'a RunLog,
    ) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move {
            info!(
                "  run-log (noop): mode={} hash={} gas {}→{} saved={} patterns={} ts={}",
                run.mode,
                run.contract_hash,
                run.gas_original
                    .map_or_else(
                        || "n/a".into(),
                        |v| v.to_string()
                    ),
                run.gas_optimized
                    .map_or_else(
                        || "n/a".into(),
                        |v| v.to_string()
                    ),
                run.gas_saved
                    .map_or_else(
                        || "n/a".into(),
                        |v| v.to_string()
                    ),
                run.pattern_ids
                    .len(),
                run.ts,
            );
            Ok(())
        })
    }
}
