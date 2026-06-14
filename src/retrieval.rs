//! Custom rig `VectorStoreIndex` reproducing Gaslite's composed retrieval:
//! category-filtered + general + antipattern Qdrant search, then full pattern
//! text fetched from Turso. Built per request with the AST-detected category and
//! the contract source baked in.
//!
//! Retrieval is pinned to the contract (NOT rig's per-turn query) and memoised in
//! a shared `OnceCell`, so the composed search runs **once** per request even
//! though `dynamic_context` re-fetches on every agent turn.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use qdrant_client::{
    Qdrant,
    qdrant::{Condition, Filter as QFilter, SearchPointsBuilder},
};
use rig_core::{
    embeddings::embedding::EmbeddingModel,
    vector_store::{
        VectorStoreError, VectorStoreIndex,
        request::{Filter, VectorSearchRequest},
    },
};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::OnceCell;
use tracing::info;

use crate::{
    COLLECTION,
    db::{Turso, TursoArg},
    embedding::FastembedAdapter,
    normalize::PatternMatcher,
};

const TOKEN_CATS: [&str; 5] = ["erc20", "erc721", "erc1155", "erc2981", "accounts"];

/// One retrieved entry: `(score, pattern_id, formatted_document_text)`.
type Hit = (f64, String, String);

#[derive(Clone)]
pub struct GasliteIndex {
    qdrant: Arc<Qdrant>,
    db: Arc<Turso>,
    embedder: FastembedAdapter,
    category: Option<&'static str>,
    /// The function source — the fixed retrieval query for this request.
    query: String,
    /// Deterministic structural matcher (the "Seeker") — the second retrieval
    /// signal alongside embedding search.
    matcher: Arc<PatternMatcher>,
    /// Function name, for attributable logs (agents run concurrently).
    label: Arc<str>,
    /// Memoised composed-search result, shared across clones (rig clones the
    /// index per turn) so the search runs at most once.
    cache: Arc<OnceCell<Vec<Hit>>>,
}

impl GasliteIndex {
    pub fn new(
        qdrant: Arc<Qdrant>,
        db: Arc<Turso>,
        embedder: FastembedAdapter,
        category: Option<&'static str>,
        query: String,
        matcher: Arc<PatternMatcher>,
        label: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            qdrant,
            db,
            embedder,
            category,
            query,
            matcher,
            label: label.into(),
            cache: Arc::new(OnceCell::new()),
        }
    }

    /// Retrieve (memoised). The composed search runs once; later calls clone the
    /// cached result.
    async fn retrieve(&self) -> Result<Vec<Hit>, VectorStoreError> {
        let hits = self
            .cache
            .get_or_try_init(|| self.compute())
            .await?;
        Ok(hits.clone())
    }

    /// Pattern ids for the optimize response. Shares the same cache as the agent.
    pub async fn pattern_ids(&self) -> Result<Vec<String>, VectorStoreError> {
        Ok(self
            .retrieve()
            .await?
            .into_iter()
            .map(|(_, id, _)| id)
            .collect())
    }

    /// The actual composed search: embed the contract, run the
    /// category/general/antipattern Qdrant queries, dedup, fetch text from Turso.
    async fn compute(&self) -> Result<Vec<Hit>, VectorStoreError> {
        // 1. Embed the contract through the rig adapter, cast f64 -> f32 for Qdrant.
        let emb = self
            .embedder
            .embed_text(&self.query)
            .await
            .map_err(VectorStoreError::EmbeddingError)?;
        let qvec: Vec<f32> = emb
            .vec
            .iter()
            .map(|f| *f as f32)
            .collect();

        let is_token = self
            .category
            .map(|c| TOKEN_CATS.contains(&c))
            .unwrap_or(false);

        // 2. All Qdrant searches run CONCURRENTLY — they only depend on the
        //    embedding, so there is no reason to serialize the round-trips.
        //    Token contracts: category-filtered (2) + general (1) + antipattern (2).
        //    Everything else: plain top-3 + antipattern (2).
        let anti_search = SearchPointsBuilder::new(COLLECTION, qvec.clone(), 2)
            .with_payload(true)
            .filter(QFilter::must([Condition::matches(
                "type",
                "antipattern".to_string(),
            )]));
        let (pattern_hits, anti_hits) = if is_token {
            let cat = self
                .category
                .unwrap();
            let (cat_r, gen_r, anti_r) = tokio::join!(
                self.qdrant
                    .search_points(
                        SearchPointsBuilder::new(COLLECTION, qvec.clone(), 2)
                            .with_payload(true)
                            .filter(QFilter::must([Condition::matches(
                                "category",
                                cat.to_string()
                            )])),
                    ),
                self.qdrant
                    .search_points(
                        SearchPointsBuilder::new(COLLECTION, qvec.clone(), 1)
                            .with_payload(true)
                            .filter(QFilter::must_not(
                                TOKEN_CATS
                                    .iter()
                                    .map(|c| Condition::matches("category", c.to_string()))
                                    .collect::<Vec<_>>(),
                            )),
                    ),
                self.qdrant
                    .search_points(anti_search),
            );
            let mut combined = cat_r
                .map_err(qerr)?
                .result;
            combined.extend(
                gen_r
                    .map_err(qerr)?
                    .result,
            );
            (
                combined,
                anti_r
                    .map_err(qerr)?
                    .result,
            )
        } else {
            let (plain_r, anti_r) = tokio::join!(
                self.qdrant
                    .search_points(
                        SearchPointsBuilder::new(COLLECTION, qvec.clone(), 3).with_payload(true),
                    ),
                self.qdrant
                    .search_points(anti_search),
            );
            (
                plain_r
                    .map_err(qerr)?
                    .result,
                anti_r
                    .map_err(qerr)?
                    .result,
            )
        };

        // 3. Collect candidate ids (deduped across sources, in injection order)
        //    together with how each was found — which decides its score + format.
        let mut seen: HashSet<String> = HashSet::new();
        let mut candidates: Vec<(String, HitKind)> = Vec::new();
        for hit in pattern_hits {
            if let Some(id) = hit
                .payload
                .get("pattern_id")
                .map(|v| {
                    v.to_string()
                        .trim()
                        .replace('"', "")
                })
                && seen.insert(id.clone())
            {
                candidates.push((id, HitKind::Pattern(hit.score as f64)));
            }
        }
        for hit in anti_hits {
            if let Some(id) = hit
                .payload
                .get("pattern_id")
                .map(|v| {
                    v.to_string()
                        .trim()
                        .replace('"', "")
                })
                && seen.insert(id.clone())
            {
                candidates.push((id, HitKind::Anti(hit.score as f64)));
            }
        }
        let struct_ids = self
            .matcher
            .match_function(&self.query);
        for id in &struct_ids {
            if seen.insert(id.clone()) {
                candidates.push((id.clone(), HitKind::Structural));
            }
        }

        if candidates.is_empty() {
            info!("  [{}] retrieval: 0 patterns", self.label);
            return Ok(vec![]);
        }

        // 4. ONE batched Turso fetch for every candidate (previously one HTTP
        //    round-trip per id — the dominant retrieval cost).
        let placeholders = vec!["?"; candidates.len()].join(",");
        let sql = format!(
            "SELECT id, title, explanation, yul_optimized, risk_level, when_not_to_apply, \
             solidity_before FROM optimization_patterns WHERE id IN ({placeholders})"
        );
        let args: Vec<TursoArg> = candidates
            .iter()
            .map(|(id, _)| TursoArg::Text(id.clone()))
            .collect();
        let rows = self
            .db
            .query(&sql, args)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(e.into()))?;
        let mut by_id: HashMap<String, &HashMap<String, serde_json::Value>> = HashMap::new();
        for row in &rows {
            if let Some(id) = row
                .get("id")
                .and_then(|v| v.as_str())
            {
                by_id.insert(id.to_string(), row);
            }
        }

        // 5. Format each candidate from its row.
        let mut out: Vec<Hit> = Vec::new();
        let mut struct_added = 0usize;
        for (id, kind) in candidates {
            let Some(row) = by_id.get(&id) else { continue };
            let get = |k: &str| {
                row.get(k)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            };
            match kind {
                HitKind::Pattern(score) => out.push((
                    score,
                    id.clone(),
                    format!(
                        "PATTERN ID: {}\nTitle: {}\nExplanation: {}\nOptimized YUL:\n{}\nRisk: {}\nDo NOT apply when: {}",
                        id,
                        get("title"),
                        get("explanation"),
                        get("yul_optimized"),
                        get("risk_level"),
                        get("when_not_to_apply"),
                    ),
                )),
                HitKind::Anti(score) => out.push((
                    score,
                    id.clone(),
                    format!(
                        "ANTIPATTERN ID: {}\nTitle: {}\nExplanation: {}\nWrong:\n{}\nCorrect:\n{}",
                        id,
                        get("title"),
                        get("explanation"),
                        get("solidity_before"),
                        get("yul_optimized"),
                    ),
                )),
                // Deterministic hits get top score.
                HitKind::Structural => {
                    out.push((
                        1.0,
                        id.clone(),
                        format!(
                            "PATTERN ID: {} (structural match)\nTitle: {}\nExplanation: {}\nOptimized YUL:\n{}\nRisk: {}\nDo NOT apply when: {}",
                            id,
                            get("title"),
                            get("explanation"),
                            get("yul_optimized"),
                            get("risk_level"),
                            get("when_not_to_apply"),
                        ),
                    ));
                    struct_added += 1;
                }
            }
        }

        info!(
            "  [{}] retrieval: {} patterns ({} embedding+antipattern, {} structural{})",
            self.label,
            out.len(),
            out.len() - struct_added,
            struct_added,
            if struct_ids.is_empty() {
                String::new()
            } else {
                format!(" {struct_ids:?}")
            },
        );

        Ok(out)
    }
}

/// Which retrieval signal produced a candidate — decides score and doc format.
enum HitKind {
    Pattern(f64),
    Anti(f64),
    Structural,
}

/// Map a Qdrant error into rig's vector-store error.
fn qerr<E: std::fmt::Display>(e: E) -> VectorStoreError {
    VectorStoreError::DatastoreError(
        e.to_string()
            .into(),
    )
}

impl VectorStoreIndex for GasliteIndex {
    type Filter = Filter<serde_json::Value>;

    // Both methods ignore the request's query/sample count: retrieval is pinned
    // to the contract (`self.query`) and memoised, so every agent turn reuses the
    // same composed result instead of re-embedding and re-searching.
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        // Honor the caller's requested sample count (rig injects ALL docs we return,
        // so we must truncate here). Rank by score descending first, so the highest-
        // signal patterns survive: structural "Seeker" matches (score 1.0) and the
        // best embedding hits stay; weak/antipattern hits drop first.
        let n = req.samples() as usize;
        let mut hits = self
            .retrieve()
            .await?;
        hits.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if n > 0 {
            hits.truncate(n);
        }
        hits.into_iter()
            .map(|(score, id, text)| {
                let doc = json!({ "pattern_id": id.clone(), "context": text });
                let val = serde_json::from_value::<T>(doc)?;
                Ok((score, id, val))
            })
            .collect()
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(self
            .retrieve()
            .await?
            .into_iter()
            .map(|(score, id, _)| (score, id))
            .collect())
    }
}
