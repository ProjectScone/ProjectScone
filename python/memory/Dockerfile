# scone-memory as a service: the FastAPI server over whichever stores the
# environment names. The default image carries the Mongo, Qdrant and
# Postgres adapters and the hash embedder; pass EXTRAS to add more, e.g.
#   docker build --build-arg EXTRAS=api,mongo,qdrant,postgres,local-embed .
# for the on-device bge-small model (adds the ONNX runtime, ~200 MB).
FROM python:3.12-slim AS build
ARG EXTRAS=api,mongo,qdrant,postgres
WORKDIR /src
COPY pyproject.toml README.md ./
COPY scone_memory ./scone_memory
RUN python -m venv /opt/scone && /opt/scone/bin/pip install --no-cache-dir -q ".[${EXTRAS}]"

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 scone
COPY --from=build /opt/scone /opt/scone
ENV PATH=/opt/scone/bin:$PATH \
    SCONE_HOST=0.0.0.0 \
    SCONE_PORT=7437 \
    SCONE_SQLITE_PATH=/data/memory.db \
    SCONE_EMBED_CACHE=/data/models
USER scone
VOLUME ["/data"]
EXPOSE 7437
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 CMD curl -fsS http://127.0.0.1:7437/healthz || exit 1
# The server refuses to start without SCONE_API_KEY (or SCONE_API_KEYS).
CMD ["scone-memory", "serve"]
