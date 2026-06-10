//! Adapts our FastEmbed-backed `ai::Embedder` (fastembed 5.x) to rig's
//! `EmbeddingModel` trait, so the vector store can run through rig without
//! adopting rig-fastembed (which pins fastembed 4.x). Produces the same
//! 384-dim BGE-Small-EN-v1.5 vectors as the ingested corpus.

use crate::ai::Embedder;
use rig_core::embeddings::embedding::{Embedding, EmbeddingError, EmbeddingModel};
use std::sync::Arc;

const NDIMS: usize = 384;

#[derive(Clone)]
pub struct FastembedAdapter {
    inner: Arc<Embedder>,
}

impl FastembedAdapter {
    pub fn new(inner: Arc<Embedder>) -> Self {
        Self { inner }
    }
}

impl EmbeddingModel for FastembedAdapter {
    const MAX_DOCUMENTS: usize = 256;

    // We construct the adapter via `new()` (wrapping our own loaded model), never
    // through rig's client-based `make`, so there is no provider client type.
    type Client = ();

    fn make(
        _client: &Self::Client,
        _model: impl Into<String>,
        _dims: Option<usize>,
    ) -> Self {
        unimplemented!(
            "FastembedAdapter is built via FastembedAdapter::new(), not EmbeddingModel::make()"
        )
    }

    fn ndims(&self) -> usize {
        NDIMS
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        let docs: Vec<String> = texts
            .into_iter()
            .collect();
        let inner = self
            .inner
            .clone();
        let to_embed = docs.clone();

        let vecs = tokio::task::spawn_blocking(move || inner.embed_blocking(to_embed))
            .await
            .map_err(|e| {
                EmbeddingError::ProviderError(format!(
                    "embed task panicked: {e}"
                ))
            })?
            .map_err(EmbeddingError::ProviderError)?;

        Ok(docs
            .into_iter()
            .zip(vecs)
            .map(|(document, v)| Embedding {
                document,
                vec: v
                    .into_iter()
                    .map(|f| f as f64)
                    .collect(),
            })
            .collect())
    }
}
