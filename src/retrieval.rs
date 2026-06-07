//! Custom rig `VectorStoreIndex` reproducing Gaslite's composed retrieval:
//! category-filtered + general + antipattern Qdrant search, then full pattern
//! text fetched from Turso. Built per request with the AST-detected category
//! baked in, then handed to the agent via `dynamic_context`.

use std::collections::HashSet;
use std::sync::Arc;

use qdrant_client::qdrant::{Condition, Filter as QFilter, SearchPointsBuilder};
use qdrant_client::Qdrant;
use rig_core::embeddings::embedding::EmbeddingModel;
use rig_core::vector_store::request::{Filter, VectorSearchRequest};
use rig_core::vector_store::{VectorStoreError, VectorStoreIndex};
use serde::Deserialize;
use serde_json::json;

use crate::db::{Turso, TursoArg};
use crate::embedding::FastembedAdapter;
use crate::COLLECTION;

const TOKEN_CATS: [&str; 5] = ["erc20", "erc721", "erc1155", "erc2981", "accounts"];

#[derive(Clone)]
pub struct GasliteIndex {
    qdrant: Arc<Qdrant>,
    db: Arc<Turso>,
    embedder: FastembedAdapter,
    category: Option<&'static str>,
}

impl GasliteIndex {
    pub fn new(
        qdrant: Arc<Qdrant>,
        db: Arc<Turso>,
        embedder: FastembedAdapter,
        category: Option<&'static str>,
    ) -> Self {
        Self { qdrant, db, embedder, category }
    }

    /// Embed the query and run the composed search, returning
    /// `(score, pattern_id, formatted_document_text)` per retrieved entry.
    async fn retrieve(&self, query: &str) -> Result<Vec<(f64, String, String)>, VectorStoreError> {
        // 1. Embed the query through the rig adapter, cast f64 -> f32 for Qdrant.
        let emb = self
            .embedder
            .embed_text(query)
            .await
            .map_err(VectorStoreError::EmbeddingError)?;
        let qvec: Vec<f32> = emb.vec.iter().map(|f| *f as f32).collect();

        let is_token = self.category.map(|c| TOKEN_CATS.contains(&c)).unwrap_or(false);

        // 2. Pattern hits: token contracts get category-filtered + 1 general;
        //    everything else gets a plain top-3.
        let pattern_hits = if is_token {
            let cat = self.category.unwrap();
            let cat_r = self
                .qdrant
                .search_points(
                    SearchPointsBuilder::new(COLLECTION, qvec.clone(), 2)
                        .with_payload(true)
                        .filter(QFilter::must([Condition::matches(
                            "category",
                            cat.to_string(),
                        )])),
                )
                .await
                .map_err(|e| VectorStoreError::DatastoreError(e.to_string().into()))?;

            let gen_r = self
                .qdrant
                .search_points(
                    SearchPointsBuilder::new(COLLECTION, qvec.clone(), 1)
                        .with_payload(true)
                        .filter(QFilter::must_not(
                            TOKEN_CATS
                                .iter()
                                .map(|c| Condition::matches("category", c.to_string()))
                                .collect::<Vec<_>>(),
                        )),
                )
                .await
                .map_err(|e| VectorStoreError::DatastoreError(e.to_string().into()))?;

            let mut combined = cat_r.result;
            combined.extend(gen_r.result);
            combined
        } else {
            self.qdrant
                .search_points(
                    SearchPointsBuilder::new(COLLECTION, qvec.clone(), 3).with_payload(true),
                )
                .await
                .map_err(|e| VectorStoreError::DatastoreError(e.to_string().into()))?
                .result
        };

        let mut out: Vec<(f64, String, String)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for hit in pattern_hits {
            let Some(id) = hit.payload.get("pattern_id").map(|v| v.to_string().trim().replace('"', "")) else {
                continue;
            };
            if !seen.insert(id.clone()) {
                continue;
            }
            let rows = self
                .db
                .query(
                    "SELECT title, explanation, yul_optimized, risk_level, when_not_to_apply \
                     FROM optimization_patterns WHERE id = ?",
                    vec![TursoArg::Text(id.clone())],
                )
                .await
                .map_err(|e| VectorStoreError::DatastoreError(e.into()))?;
            if let Some(row) = rows.first() {
                let get = |k: &str| row.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                let text = format!(
                    "PATTERN ID: {}\nTitle: {}\nExplanation: {}\nOptimized YUL:\n{}\nRisk: {}\nDo NOT apply when: {}",
                    id, get("title"), get("explanation"), get("yul_optimized"), get("risk_level"), get("when_not_to_apply"),
                );
                out.push((hit.score as f64, id, text));
            }
        }

        // 3. Antipattern hits — always 2, filtered by type.
        let anti_hits = self
            .qdrant
            .search_points(
                SearchPointsBuilder::new(COLLECTION, qvec, 2)
                    .with_payload(true)
                    .filter(QFilter::must([Condition::matches(
                        "type",
                        "antipattern".to_string(),
                    )])),
            )
            .await
            .map_err(|e| VectorStoreError::DatastoreError(e.to_string().into()))?
            .result;

        for hit in anti_hits {
            let Some(id) = hit.payload.get("pattern_id").map(|v| v.to_string().trim().replace('"', "")) else {
                continue;
            };
            if !seen.insert(id.clone()) {
                continue;
            }
            let rows = self
                .db
                .query(
                    "SELECT title, explanation, solidity_before, yul_optimized \
                     FROM optimization_patterns WHERE id = ?",
                    vec![TursoArg::Text(id.clone())],
                )
                .await
                .map_err(|e| VectorStoreError::DatastoreError(e.into()))?;
            if let Some(row) = rows.first() {
                let get = |k: &str| row.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                let text = format!(
                    "ANTIPATTERN ID: {}\nTitle: {}\nExplanation: {}\nWrong:\n{}\nCorrect:\n{}",
                    id, get("title"), get("explanation"), get("solidity_before"), get("yul_optimized"),
                );
                out.push((hit.score as f64, id, text));
            }
        }

        Ok(out)
    }
}

impl VectorStoreIndex for GasliteIndex {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let hits = self.retrieve(req.query()).await?;
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
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let hits = self.retrieve(req.query()).await?;
        Ok(hits.into_iter().map(|(score, id, _)| (score, id)).collect())
    }
}
