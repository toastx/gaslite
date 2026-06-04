# syntax=docker/dockerfile:1

###############################################################################
# Stage 1 — build the Rust service
# Use the full Debian-based Rust image so glibc/openssl match the slim runtime.
###############################################################################
FROM rust:1-bookworm AS builder

# Build deps: pkg-config + libssl-dev for reqwest's default (native) TLS,
# ca-certificates so ort/fastembed can download the ONNX Runtime at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- dependency cache layer ----
# Copy only the manifests first and build a dummy bin so the (slow) dependency
# compile + ONNX Runtime download is cached unless Cargo.toml/lock change.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src \
    && echo 'fn main() { println!("stub"); }' > src/main.rs \
    && cargo build --release \
    && rm -rf src

# ---- real build ----
COPY src ./src
# bump mtime so cargo recompiles the real binary over the cached stub
RUN touch src/main.rs && cargo build --release

# Stage the runtime artifacts: the binary plus the ONNX Runtime shared library
# that the `ort` crate downloads next to it (fastembed depends on it).
RUN mkdir -p /out \
    && cp target/release/gaslite /out/ \
    && find target/release -maxdepth 3 -name 'libonnxruntime*.so*' -exec cp -P {} /out/ \; || true

###############################################################################
# Stage 2 — slim runtime
###############################################################################
FROM debian:bookworm-slim AS runtime

# Runtime libs:
#   ca-certificates  -> TLS to DeepSeek / Qdrant / Turso / Mantle RPC
#   libssl3          -> reqwest native-tls
#   libgomp1         -> OpenMP, required by ONNX Runtime
#   libstdc++6       -> C++ runtime for ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Run as an unprivileged user; /app must be writable for the fastembed model
# cache (.fastembed_cache) that is downloaded on first start.
RUN useradd --create-home --uid 10001 gaslite
WORKDIR /app

# Binary + ONNX Runtime .so (kept beside the binary to satisfy its $ORIGIN rpath)
COPY --from=builder /out/ /app/
# Knowledge base files (point the ingest endpoint at /app/rag/functions etc.)
COPY rag ./rag

# Fall back to /app on the library search path in case the rpath is missing.
ENV LD_LIBRARY_PATH=/app
# fastembed writes its model cache here; keep it inside the writable workdir.
ENV HOME=/app

RUN chown -R gaslite:gaslite /app
USER gaslite

EXPOSE 8000

# Required env vars (supply at `docker run -e ...` or via compose / secrets):
#   DEEPSEEK_API_KEY, QDRANT_API_KEY, QDRANT_CLUSTER_URL,
#   TURSO_DATABASE_URL, TURSO_AUTH_TOKEN
# Optional: DEEPSEEK_BASE_URL, MANTLE_RPC_URL
CMD ["/app/gaslite"]
