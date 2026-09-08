# syntax=docker/dockerfile:1
# Build from the repository root; this file's allowlist excludes private infra.
FROM node:24-bookworm-slim AS web
ARG PNPM_VERSION=9.9.0
RUN npm install --global "pnpm@${PNPM_VERSION}"
WORKDIR /build/Webapp
COPY Webapp/package.json Webapp/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY Webapp/ ./
COPY scripts/python-layout.cjs /build/scripts/python-layout.cjs
RUN pnpm exec tsc --noEmit && pnpm exec vite build \
    && node scripts/package.mjs --output /build/console.html

FROM python:3.14-slim-bookworm AS python-build
ARG EXTRAS=api,agents,remote-embed,aws,qdrant
WORKDIR /build
COPY python/memory/pyproject.toml python/memory/README.md python/memory/LICENSE ./
COPY python/memory/src/ ./src/
COPY --from=web /build/console.html ./src/scone_memory/api/console.html
COPY --from=web /build/console.html ./src/scone_memory/api/playground.html
RUN python -m venv /opt/scone \
    && /opt/scone/bin/pip install --no-cache-dir ".[${EXTRAS}]"

FROM python:3.14-slim-bookworm AS runtime
RUN groupadd --gid 10001 scone \
    && useradd --uid 10001 --gid 10001 --no-create-home scone \
    && mkdir /data && chown 10001:10001 /data
COPY --from=python-build /opt/scone /opt/scone
COPY deploy/aws/healthcheck.py /opt/healthcheck.py
ENV PATH=/opt/scone/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SCONE_HOST=0.0.0.0 SCONE_PORT=7437 \
    SCONE_DOCUMENTS=sqlite SCONE_VECTORS=sqlite SCONE_EVENTS=sqlite \
    SCONE_SQLITE_PATH=/data/memory.db SCONE_EMBEDDER=hash
USER 10001:10001
WORKDIR /data
EXPOSE 7437
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "/opt/healthcheck.py"]
CMD ["scone-memory", "serve"]
