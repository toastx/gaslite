//! Text embeddings (FastEmbed). The DeepSeek completion now goes through rig
//! (see `rig_agent.rs`); this module only owns the embedding model.

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::{Arc, Mutex};

/// Wraps the FastEmbed model behind a mutex so embed calls are serialised.
pub struct Embedder(Mutex<TextEmbedding>);

impl Embedder {
    /// Loads the BGE-Small-EN v1.5 model (384-dim vectors).
    pub fn new() -> Result<Arc<Self>, Box<dyn std::error::Error>> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true),
        )?;
        Ok(Arc::new(Self(Mutex::new(
            model,
        ))))
    }

    /// Embeds a single string, offloading the blocking model call to a worker thread.
    pub async fn embed(
        self: Arc<Self>,
        text: &str,
    ) -> Result<Vec<f32>, String> {
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            let mut model = self
                .0
                .lock()
                .unwrap();
            let mut embeddings = model
                .embed(vec![text.as_str()], None)
                .map_err(|e| format!("Embed error: {e}"))?;
            embeddings
                .pop()
                .ok_or_else(|| "Embedding returned empty results".to_string())
        })
        .await
        .map_err(|e| format!("Embedding task panicked: {e}"))?
    }

    /// Synchronous batch embed — locks the model and embeds every text.
    /// Used by the rig `EmbeddingModel` adapter (which wraps this in spawn_blocking).
    pub fn embed_blocking(
        &self,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut model = self
            .0
            .lock()
            .unwrap();
        model
            .embed(
                texts
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                None,
            )
            .map_err(|e| format!("Embed error: {e}"))
    }
}
