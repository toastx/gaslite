<div align="center">

# ⛽ Gaslite

**An AI gas-optimization engine for Solidity, tuned for the Mantle L2.**

Gaslite reads your contract, retrieves battle-tested Yul/assembly optimization
patterns from a curated knowledge base, rewrites each function with a frontier
model, and **verifies the result on a Mantle fork with Foundry** before it ever
reaches you.

<br/>

[![Install the Gaslite Analyzer GitHub App](https://img.shields.io/badge/⚡_Install_to_your_repo-Gaslite_Analyzer-1a73e8?style=for-the-badge&logo=github&logoColor=white)](https://github.com/apps/gaslite-analyzer/installations/new)

<sub>One click → pick a repo → Gaslite reviews gas on every PR.</sub>

</div>

---

## What this repo is

This is the **Gaslite core** — the Rust service that does the actual work. It is
an [Axum](https://github.com/tokio-rs/axum) HTTP server that exposes an
optimization + verification API, backed by a retrieval-augmented (RAG) pipeline
over a hand-curated corpus of Solidity gas patterns.

The web IDE lives in [`webui/`](webui/) and is documented separately — this
README covers only the engine.

## How it works

```
                         POST /api/optimize  { contract_source }
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  1. analyze_contract  (solang-parser)                  │
        │     • split the contract into individual functions     │
        │     • detect a category (erc20 / erc721 / hashing / …) │
        └───────────────────────────────────────────────────────┘
                                    │  per function
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  2. retrieve context                                   │
        │     • embed the function locally (fastembed, 384-d)    │
        │     • Qdrant ANN search: category + general + anti-    │
        │       patterns  →  pattern_ids                         │
        │     • hydrate full pattern rows from Turso (libSQL)    │
        └───────────────────────────────────────────────────────┘
                                    │  parallel, one task / function
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  3. rewrite  (DeepSeek, deepseek-v4-flash, temp 0.1)   │
        │     patterns are fed as templates the model adapts to  │
        │     the contract's real storage layout                 │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
              { analysis, suggested_patterns, optimized_code }

        POST /api/verify  →  Foundry sandbox: forge build + fork-test
                             against Mantle, returns measured gas delta
```

Two data stores work together: **Qdrant** holds the embeddings (fast nearest-
neighbour over pattern vectors), while **Turso** holds the full pattern records
(before/after code, explanation, risk, when-not-to-apply). A search returns
`pattern_id`s from Qdrant which are then hydrated from Turso.

## Tech stack

| Concern            | Choice                                                            |
| ------------------ | ---------------------------------------------------------------- |
| HTTP server        | Axum 0.8 + Tokio                                                  |
| Solidity parsing   | `solang-parser` (function extraction + category detection)        |
| Embeddings         | `fastembed` — `BGESmallENV15`, 384-dim, **runs locally**          |
| Vector search      | Qdrant (`gaslite_patterns`, cosine, 384-dim)                      |
| Pattern store      | Turso / libSQL over the HTTP `/v2/pipeline` API                  |
| Rewrite model      | DeepSeek (`deepseek-v4-flash`) via the OpenAI-compatible API      |
| Verification       | Foundry (`forge`) — build + fork-test against Mantle             |

## API

The server listens on `0.0.0.0:8000`.

| Method & path                 | Body                                   | Returns |
| ----------------------------- | -------------------------------------- | ------- |
| `GET  /health`                | —                                      | `ok` |
| `POST /api/optimize`          | `{ contract_source }`                  | `{ analysis, suggested_patterns[], optimized_code }` |
| `POST /api/verify`            | `{ original_code, optimized_code }`    | `{ compiles, errors[], gas_original, gas_optimized, gas_saved, forge_output }` |
| `POST /api/admin/ingest-local`| `{ directory_paths[] }`                | `{ successful_patterns[], failed_patterns[] }` |
| `POST /api/admin/qdrant/reset`| —                                      | re-creates the empty Qdrant collection |

```bash
curl -s localhost:8000/api/optimize \
  -H 'content-type: application/json' \
  -d '{"contract_source":"contract C { function f(uint[] memory a) public {} }"}' | jq
```

## Knowledge base

The corpus lives in [`rag/`](rag/) and is plain JSON, version-controlled so the
optimizer's behaviour is reviewable:

```
rag/
├── functions/      reference implementations grouped by category
│   ├── erc20/  erc721/  erc1155/  erc2981/
│   ├── accounts/  hashing/  safe_transfer/
└── patterns/       individual Yul optimization patterns + anti-patterns
    ├── CALLDATALOAD_SHR_SELECTOR.json
    ├── BRANCHLESS_CLAMP.json
    ├── ANTIPATTERN_WRONG_NESTED_MAPPING_SLOT.json
    └── …
```

Each pattern record carries `solidity_before`, `yul_optimized`, an
`explanation`, `trigger_patterns` (what it's matched on), `risk_level`,
`when_to_apply` / `when_not_to_apply`, and gas estimates. **Anti-patterns**
(`ANTIPATTERN_*`) are retrieved too, so the model is steered away from
plausible-but-wrong assembly (e.g. mis-derived storage slots).

## Getting started

### Prerequisites

- **Rust** (stable) and **Foundry** (`forge` on `PATH`, for `/api/verify`)
- A **Qdrant** instance and a **Turso** database
- A **DeepSeek** API key

### Configuration

The service reads everything from the environment (a local `.env` is loaded via
`dotenvy`):

| Variable               | Required | Default                       |
| ---------------------- | :------: | ----------------------------- |
| `DEEPSEEK_API_KEY`     |    ✅    | —                             |
| `QDRANT_API_KEY`       |    ✅    | —                             |
| `QDRANT_CLUSTER_URL`   |    ✅    | —                             |
| `TURSO_DATABASE_URL`   |    ✅    | —                             |
| `TURSO_AUTH_TOKEN`     |    ✅    | —                             |
| `DEEPSEEK_BASE_URL`    |          | `https://api.deepseek.com/v1` |
| `MANTLE_RPC_URL`       |          | `https://rpc.mantle.xyz`      |

### Run locally

```bash
cp .env.example .env   # then fill in the values above
cargo run --release    # serves on http://0.0.0.0:8000
```

On boot the service creates the `gaslite_patterns` Qdrant collection if it does
not exist. To populate it, point the ingest endpoint at the knowledge base:

```bash
curl -s localhost:8000/api/admin/ingest-local \
  -H 'content-type: application/json' \
  -d '{"directory_paths":["rag/patterns","rag/functions"]}'
```

### Run with Docker

A multi-stage [`Dockerfile`](Dockerfile) is included (Debian 13 / glibc 2.41 —
required by the prebuilt ONNX Runtime that `fastembed` links):

```bash
docker build -t gaslite .
docker run -p 8000:8000 \
  -e DEEPSEEK_API_KEY=… \
  -e QDRANT_API_KEY=… -e QDRANT_CLUSTER_URL=… \
  -e TURSO_DATABASE_URL=… -e TURSO_AUTH_TOKEN=… \
  -e MANTLE_RPC_URL=… \
  gaslite
```

> On first start `fastembed` downloads its embedding model (~130 MB) into the
> working directory, so the container needs outbound network and a few seconds
> to warm up.

## Delivery surfaces

The core engine is consumed through several front doors:

- 🤖 **[Gaslite Analyzer GitHub App](https://github.com/apps/gaslite-analyzer/installations/new)** — the one-click install above; reviews gas on pull requests.
- 🧩 **VS Code extension** — [`vscode/`](vscode/)
- 🖥️ **Web IDE** — [`webui/`](webui/)

---

<div align="center">
<sub>Built for the Mantle hackathon. Optimizations are model-generated — always
review the <code>forge</code> verification output before shipping.</sub>
</div>
