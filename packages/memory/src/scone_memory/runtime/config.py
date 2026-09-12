"""Assemble an engine from environment variables.

    SCONE_DOCUMENTS   memory | sqlite | mongo | postgres | elasticsearch   (default memory)
    SCONE_VECTORS     memory | sqlite | qdrant | chroma | lancedb | milvus | postgres | redis | elasticsearch | opensearch | elasticache  (default memory)
    SCONE_EMBEDDER    hash | local | remote     (default hash)

    SCONE_SQLITE_PATH (default ~/.scone-memory/memory.db; both sqlite stores share it)
    SCONE_MONGO_URL, SCONE_MONGO_DB (default scone)
    SCONE_POSTGRES_URL, SCONE_POSTGRES_SCHEMA (default scone); documents, vectors and events share one pool
    SCONE_QDRANT_URL, SCONE_QDRANT_API_KEY, SCONE_QDRANT_COLLECTION (default scone_chunks)
    SCONE_QDRANT_METADATA_INDEXES (optional comma-separated metadata keys to index)
    SCONE_QDRANT_HNSW_EF (optional positive search beam width; unset uses the server default)
    SCONE_CHROMA_PATH (persistent directory) or SCONE_CHROMA_URL (server); neither: in-process, ephemeral
    SCONE_LANCEDB_PATH (database directory, required for lancedb)
    SCONE_REDIS_URL (required for redis; needs the RediSearch module), SCONE_REDIS_PREFIX (default scone_chunks)
    SCONE_ELASTICACHE_URL (explicit rediss endpoint), *_PREFIX, *_USERNAME, *_PASSWORD
    SCONE_ELASTICACHE_CLUSTER_MODE (1), *_ALGORITHM (FLAT), *_BATCH_SIZE (64), *_TIMEOUT (30), *_CA_CERTS
    SCONE_MILVUS_URI (a local .db path runs Milvus Lite; http://host:19530 a server), SCONE_MILVUS_TOKEN, SCONE_MILVUS_COLLECTION
    SCONE_ELASTICSEARCH_URL, SCONE_ELASTICSEARCH_API_KEY, SCONE_ELASTICSEARCH_PREFIX (default scone); one client for all three
    SCONE_OPENSEARCH_URL (explicit self-managed HTTP(S) endpoint), SCONE_OPENSEARCH_INDEX (default scone_vectors)
    SCONE_OPENSEARCH_USERNAME, SCONE_OPENSEARCH_PASSWORD (optional pair; TLS verification always enabled)
    SCONE_OPENSEARCH_AWS_REGION, *_AWS_ACCESS_KEY_ID, *_AWS_SECRET_ACCESS_KEY, *_AWS_SESSION_TOKEN
                                 optional explicit IAM signer for managed domains; no credential discovery
    SCONE_OPENSEARCH_BATCH_SIZE (256), SCONE_OPENSEARCH_MAX_BATCH_BYTES (5242880), SCONE_OPENSEARCH_TIMEOUT (30)
    SCONE_EMBED_MODEL          local: bge-small-en-v1.5 ; remote: model name
    SCONE_EMBED_URL            remote: OpenAI-compatible base, e.g. http://localhost:11434/v1
    SCONE_EMBED_API_KEY        remote: bearer, optional
    SCONE_EMBED_CACHE          local: model cache dir, optional

    SCONE_CONTEXTUAL_EMBEDDINGS=1  embed a date/source/scope prefix with each chunk (experiment 8; off by default)
    SCONE_DEMOTE_RESTATED=1        rank a restated claim ahead of what it replaces (experiment 5; off by default)
    SCONE_MANY_VALUED=knows,owns   predicates whose values hold side by side; any other holds one at a time
    SCONE_RELATION_INVERSE=works_at:employs   which predicates are the other side of which
    SCONE_RELATION_SYMMETRIC=married_to       which read the same both ways
    SCONE_RELATION_TRANSITIVE=part_of         which carry through
    SCONE_ABSTENTION_POLICY        a policy file from `scone calibrate`: the measured floor to abstain by
    SCONE_PROFILE_PREDICATES       only these predicates make a profile (default: all of them)
    SCONE_PROFILE_WITHOUT          predicates a profile never shows
    SCONE_RERANKER_FACTORY        trusted module:factory for an optional reranker
    SCONE_RERANKER_CROSS_ENCODER_DIR, SCONE_RERANKER_CROSS_ENCODER_MODEL
                                 alternatively load preprovisioned CPU model files; both required
    SCONE_RERANKER_CROSS_ENCODER_MAX_PAIR_TOKENS, *_THREADS, *_BATCH_SIZE
                                 optional offline tuning (defaults 512, 2, 8); no model downloads
    SCONE_EVENTS      memory | sqlite | mongo | postgres | elasticsearch | none  (default follows SCONE_DOCUMENTS)
                      (default follows SCONE_DOCUMENTS: sqlite -> sqlite, mongo -> mongo, else memory)
    SCONE_EVENTS_QUERIES  hash | text           (default hash: a sha256 prefix, never the query text)
    SCONE_EVENTS_MAX_AGE_DAYS                    sqlite, mongo and postgres sink retention, optional
    SCONE_EVENTS_MAX     in-memory sink ring size (default 10000)

    SCONE_CHAT_URL, SCONE_CHAT_MODEL   OpenAI-compatible chat model for consolidation; unset = no distiller
    SCONE_CHAT_API_KEY                 optional bearer
    SCONE_CHAT_THINK   true | false    for Ollama reasoning models; unset leaves the field out
    SCONE_CHAT_TIMEOUT                 seconds one chat call may take (default 180)
    SCONE_DISTILL_INTERVAL_S           seconds between consolidation passes (default 30)
    SCONE_DISTILL_BATCH                episodes per pass per space (default 20)
    SCONE_DISTILL_ACCEPT_AT            confidence at or above which extractions enter the ledger
    SCONE_DERIVE       1 | 0          run the derivation pass after extraction (default 0; needs the chat model)
                                       directly; unset = every extraction is proposed for review
    SCONE_MCP_PROPOSE_BELOW            confidence below which a fact submitted over MCP is parked for
                                       review; unset = every submitted fact is a ledger claim, which
                                       is what the Rust server does without --propose-below

    SCONE_API_KEYS    "key:space[:role],..."     bearer keys, the space each one sees, and its role:
                                               read | write | review | full (the default)
    SCONE_API_KEY     one key for the space "default" (used when SCONE_API_KEYS is unset)
    SCONE_HOST, SCONE_PORT                       (default 127.0.0.1:7437)
"""

from __future__ import annotations

import math
import importlib
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, cast

if TYPE_CHECKING:
    from ..providers.aws_auth import AwsSigV4Auth

from ..memory.engine import MemoryEngine
from ..core.errors import InvalidInput
from ..retrieval.reranking import Reranker, validate_candidate_limit, validate_rerank_options


@dataclass(frozen=True)
class Settings:
    documents: str = "memory"
    vectors: str = "memory"
    embedder: str = "hash"
    sqlite_path: str = "~/.scone-memory/memory.db"
    #: Where attachment bytes live. Empty means beside the SQLite file
    #: when there is one, and in memory when there is not.
    blob_dir: str = ""
    blobs: str = "auto"
    s3_bucket: str | None = None
    dynamodb_blob_table: str | None = None
    aws_region: str | None = None
    s3_prefix: str = "attachments/"
    #: Confidence below which a fact an agent submits over MCP is parked
    #: for a person instead of entering the ledger. Unset means none is.
    mcp_propose_below: Optional[float] = None
    mongo_url: Optional[str] = None
    mongo_db: str = "scone"
    postgres_url: Optional[str] = None
    postgres_schema: str = "scone"
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "scone_chunks"
    qdrant_metadata_indexes: tuple[str, ...] = ()
    qdrant_hnsw_ef: int | None = None
    chroma_path: Optional[str] = None
    chroma_url: Optional[str] = None
    lancedb_path: Optional[str] = None
    redis_url: Optional[str] = None
    redis_prefix: str = "scone_chunks"
    elasticache_url: str | None = None
    elasticache_prefix: str = "scone_vectors"
    elasticache_username: str | None = None
    elasticache_password: str | None = field(default=None, repr=False)
    elasticache_cluster_mode: bool = True
    elasticache_algorithm: str = "FLAT"
    elasticache_batch_size: int = 64
    elasticache_timeout: float = 30.0
    elasticache_ca_certs: str | None = None
    milvus_uri: Optional[str] = None
    milvus_token: Optional[str] = None
    milvus_collection: str = "scone_chunks"
    elasticsearch_url: Optional[str] = None
    elasticsearch_api_key: Optional[str] = None
    elasticsearch_prefix: str = "scone"
    opensearch_url: str | None = None
    opensearch_index: str = "scone_vectors"
    opensearch_username: str | None = None
    opensearch_password: str | None = field(default=None, repr=False)
    opensearch_batch_size: int = 256
    opensearch_max_batch_bytes: int = 5 * 1024 * 1024
    opensearch_timeout: float = 30.0
    opensearch_aws_region: str | None = None
    opensearch_aws_access_key_id: str | None = field(default=None, repr=False)
    opensearch_aws_secret_access_key: str | None = field(default=None, repr=False)
    opensearch_aws_session_token: str | None = field(default=None, repr=False)
    embed_model: Optional[str] = None
    embed_url: Optional[str] = None
    embed_api_key: Optional[str] = None
    embed_cache: Optional[str] = None
    chat_url: Optional[str] = None
    chat_model: Optional[str] = None
    chat_api_key: Optional[str] = None
    chat_think: Optional[bool] = None
    #: Seconds to wait on the chat host before giving up on one call. A
    #: local model on a busy machine is the ordinary case, and 180 is not
    #: always enough for it.
    chat_timeout: float = 180.0
    distill_interval_s: float = 30.0
    distill_batch: int = 20
    distill_accept_at: Optional[float] = None
    derive: bool = False
    contextual_embeddings: bool = False
    demote_restated: bool = True
    many_valued: tuple[str, ...] = ()
    relation_inverse: tuple[str, ...] = ()
    relation_symmetric: tuple[str, ...] = ()
    relation_transitive: tuple[str, ...] = ()
    abstention_policy: str | None = None
    profile_predicates: tuple[str, ...] = ()
    profile_without: tuple[str, ...] = ()
    similarity_floor: Optional[float] = None
    candidate_limit: int | None = None
    reranker_factory: str | None = None
    reranker_cross_encoder_dir: str | None = None
    reranker_cross_encoder_model: str | None = None
    reranker_cross_encoder_max_pair_tokens: int | None = None
    reranker_cross_encoder_threads: int | None = None
    reranker_cross_encoder_batch_size: int | None = None
    rerank_limit: int = 32
    rerank_max_bytes: int = 64000
    rerank_timeout: float = 1.0
    events: Optional[str] = None
    events_queries: str = "hash"
    events_max_age_days: Optional[float] = None
    events_max: int = 10_000
    keys: Mapping[str, str] = field(default_factory=dict)
    #: Key -> role; a key not listed here is full.
    roles: Mapping[str, str] = field(default_factory=dict)
    host: str = "127.0.0.1"
    port: int = 7437
    #: Writes embedding at once over HTTP; one more is told to come back.
    ingest_concurrency: int = 4
    #: Episode kinds to the days they are kept; empty means nothing expires.
    retention: Mapping[str, float] = field(default_factory=dict)
    # Naming a journal composes the conversation service onto the memory
    # origin under `serve`; the factory is the same trusted module:callable
    # as `serve-conversations --model-factory`, absent meaning history-only.
    agents_config: Optional[str] = None
    document_jobs_config: Optional[str] = None
    directory_sync_config: Optional[str] = None
    document_media_config: Optional[str] = None
    document_ocr_executable: Optional[str] = None
    document_ocr_language: str = 'eng'
    document_ocr_psm: int = 3
    document_ocr_dpi: int = 150
    conversations_journal: Optional[str] = None
    conversations_model_factory: Optional[str] = None
    # A persona catalog (JSON array of Persona documents) needs a registry
    # (trusted module:callable returning a ProviderRegistry) to bind it.
    conversations_personas: Optional[str] = None
    conversations_registry: Optional[str] = None
    conversations_tool_mode: str = "off"
    conversations_tool_initial_search: bool = True
    conversations_tool_compute: bool = False
    conversations_tool_max_calls: int = 4
    conversations_tool_max_rounds: int = 4
    conversations_tool_timeout: float = 120.0
    answer_review_policy: str = "off"
    answer_review_url: str | None = None
    answer_review_model: str | None = None
    answer_review_api_key: str | None = field(default=None, repr=False)
    answer_review_timeout: float = 20.0
    answer_review_quote_mode: str = "text"
    adaptive_retrieval: bool = False
    adaptive_url: str | None = None
    adaptive_model: str | None = None
    adaptive_api_key: str | None = field(default=None, repr=False)
    adaptive_timeout: float = 15.0
    adaptive_max_rounds: int = 3
    adaptive_max_queries: int = 6
    adaptive_candidate_limit: int = 20
    adaptive_max_evidence_bytes: int = 16000
    adaptive_graph_hops: int = 0
    adaptive_search_history: bool = False
    # Opt-in private local service settings and operational diagnostics.
    model_connections: Optional[str] = None
    log_path: Optional[str] = None

    def __post_init__(self) -> None:
        from .conversation_review import validate_review_settings
        from .conversation_retrieval import validate_adaptive_settings
        from .conversation_tools import validate_tool_settings

        validate_review_settings(self)
        validate_adaptive_settings(self)
        validate_tool_settings(self)
        if self.qdrant_hnsw_ef is not None and (type(self.qdrant_hnsw_ef) is not int or self.qdrant_hnsw_ef < 1):
            raise InvalidInput("SCONE_QDRANT_HNSW_EF must be a positive integer")
        if self.blobs not in ("auto", "memory", "file", "s3"):
            raise InvalidInput("SCONE_BLOBS must be auto, memory, file, or s3")
        if self.blobs in ("memory", "s3") and self.blob_dir:
            raise InvalidInput("SCONE_BLOB_DIR cannot be combined with SCONE_BLOBS=memory or s3")
        if self.blobs == "file" and not self.blob_dir:
            raise InvalidInput("SCONE_BLOBS=file requires SCONE_BLOB_DIR")
        for name, value in (("SCONE_S3_BUCKET", self.s3_bucket),
                            ("SCONE_DYNAMODB_BLOB_TABLE", self.dynamodb_blob_table),
                            ("SCONE_AWS_REGION", self.aws_region)):
            if self.blobs == "s3":
                if type(value) is not str or not value.strip() or "\x00" in value:
                    raise InvalidInput(f"SCONE_BLOBS=s3 requires {name}")
            elif value is not None:
                raise InvalidInput(f"{name} requires SCONE_BLOBS=s3")
        if self.blobs != "s3" and self.s3_prefix != "attachments/":
            raise InvalidInput("SCONE_S3_PREFIX requires SCONE_BLOBS=s3")
        try:
            validate_candidate_limit(self.candidate_limit)
            validate_rerank_options(self.rerank_limit, self.rerank_max_bytes, self.rerank_timeout)
        except InvalidInput as error:
            message = str(error)
            for field_name, env_name in (("candidate_limit", "SCONE_RECALL_CANDIDATES"),
                ("rerank_limit", "SCONE_RERANK_LIMIT"), ("rerank_max_bytes", "SCONE_RERANK_MAX_BYTES"),
                ("rerank_timeout", "SCONE_RERANK_TIMEOUT")):
                message = message.replace(field_name, env_name)
            raise InvalidInput(message) from None
        if self.reranker_factory is not None:
            _reranker_spec(self.reranker_factory)
        for name, value in (("DIR", self.reranker_cross_encoder_dir), ("MODEL", self.reranker_cross_encoder_model)):
            if value is not None and (type(value) is not str or not value.strip() or "\x00" in value):
                raise InvalidInput(f"SCONE_RERANKER_CROSS_ENCODER_{name} must be nonblank text")
        if (self.reranker_cross_encoder_dir is None) != (self.reranker_cross_encoder_model is None):
            raise InvalidInput("SCONE_RERANKER_CROSS_ENCODER_DIR and SCONE_RERANKER_CROSS_ENCODER_MODEL must be configured together")
        if self.reranker_cross_encoder_dir is not None and self.reranker_factory is not None:
            raise InvalidInput("SCONE_RERANKER_CROSS_ENCODER_DIR cannot be combined with SCONE_RERANKER_FACTORY")
        for name, tuning, minimum, maximum in (
            ("MAX_PAIR_TOKENS", self.reranker_cross_encoder_max_pair_tokens, 2, 8192),
            ("THREADS", self.reranker_cross_encoder_threads, 1, 16),
            ("BATCH_SIZE", self.reranker_cross_encoder_batch_size, 1, 128),
        ):
            if tuning is None:
                continue
            if type(tuning) is not int or not minimum <= tuning <= maximum:
                raise InvalidInput(f"SCONE_RERANKER_CROSS_ENCODER_{name} must be an integer in {minimum}..{maximum}")
            if self.reranker_cross_encoder_dir is None:
                raise InvalidInput(f"SCONE_RERANKER_CROSS_ENCODER_{name} requires the cross encoder directory and model")

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        for suffix in ("DIR", "MODEL", "MAX_PAIR_TOKENS", "THREADS", "BATCH_SIZE"):
            name = f"SCONE_RERANKER_CROSS_ENCODER_{suffix}"
            if name in env and type(env[name]) is not str:
                raise InvalidInput(f"{name} must be an environment string")
        for name in ("SCONE_RECALL_CANDIDATES", "SCONE_RERANKER_FACTORY", "SCONE_RERANK_LIMIT",
                     "SCONE_RERANK_MAX_BYTES", "SCONE_RERANK_TIMEOUT"):
            if env.get(name) is not None and not isinstance(env[name], str):
                raise InvalidInput(f"{name} must be an environment string")
        return cls(
            documents=env.get("SCONE_DOCUMENTS", "memory"),
            vectors=env.get("SCONE_VECTORS", "memory"),
            embedder=env.get("SCONE_EMBEDDER", "hash"),
            sqlite_path=env.get("SCONE_SQLITE_PATH", "~/.scone-memory/memory.db"),
            blob_dir=env.get("SCONE_BLOB_DIR", ""),
            blobs=env.get("SCONE_BLOBS", "auto"),
            s3_bucket=env.get("SCONE_S3_BUCKET") or None,
            dynamodb_blob_table=env.get("SCONE_DYNAMODB_BLOB_TABLE") or None,
            aws_region=env.get("SCONE_AWS_REGION") or None,
            s3_prefix=env.get("SCONE_S3_PREFIX", "attachments/"),
            mcp_propose_below=(float(env["SCONE_MCP_PROPOSE_BELOW"])
                               if env.get("SCONE_MCP_PROPOSE_BELOW") else None),
            mongo_url=env.get("SCONE_MONGO_URL"),
            mongo_db=env.get("SCONE_MONGO_DB", "scone"),
            postgres_url=env.get("SCONE_POSTGRES_URL"),
            postgres_schema=env.get("SCONE_POSTGRES_SCHEMA", "scone"),
            qdrant_url=env.get("SCONE_QDRANT_URL"),
            qdrant_api_key=env.get("SCONE_QDRANT_API_KEY"),
            qdrant_collection=env.get("SCONE_QDRANT_COLLECTION", "scone_chunks"),
            qdrant_metadata_indexes=tuple(key.strip() for key in env.get("SCONE_QDRANT_METADATA_INDEXES", "").split(",") if key.strip()),
            qdrant_hnsw_ef=(_environment_integer("SCONE_QDRANT_HNSW_EF", env["SCONE_QDRANT_HNSW_EF"])
                           if env.get("SCONE_QDRANT_HNSW_EF") else None),
            chroma_path=env.get("SCONE_CHROMA_PATH"),
            chroma_url=env.get("SCONE_CHROMA_URL"),
            lancedb_path=env.get("SCONE_LANCEDB_PATH"),
            redis_url=env.get("SCONE_REDIS_URL"),
            redis_prefix=env.get("SCONE_REDIS_PREFIX", "scone_chunks"),
            elasticache_url=env.get("SCONE_ELASTICACHE_URL"),
            elasticache_prefix=env.get("SCONE_ELASTICACHE_PREFIX", "scone_vectors"),
            elasticache_username=env.get("SCONE_ELASTICACHE_USERNAME"),
            elasticache_password=env.get("SCONE_ELASTICACHE_PASSWORD"),
            elasticache_cluster_mode=parse_flag("SCONE_ELASTICACHE_CLUSTER_MODE", env.get("SCONE_ELASTICACHE_CLUSTER_MODE", "1")),
            elasticache_algorithm=env.get("SCONE_ELASTICACHE_ALGORITHM", "FLAT"),
            elasticache_batch_size=_environment_integer("SCONE_ELASTICACHE_BATCH_SIZE", env.get("SCONE_ELASTICACHE_BATCH_SIZE", "64")),
            elasticache_timeout=parse_seconds("SCONE_ELASTICACHE_TIMEOUT", env.get("SCONE_ELASTICACHE_TIMEOUT"), 30.0),
            elasticache_ca_certs=env.get("SCONE_ELASTICACHE_CA_CERTS"),
            milvus_uri=env.get("SCONE_MILVUS_URI"),
            milvus_token=env.get("SCONE_MILVUS_TOKEN"),
            milvus_collection=env.get("SCONE_MILVUS_COLLECTION", "scone_chunks"),
            elasticsearch_url=env.get("SCONE_ELASTICSEARCH_URL"),
            elasticsearch_api_key=env.get("SCONE_ELASTICSEARCH_API_KEY"),
            elasticsearch_prefix=env.get("SCONE_ELASTICSEARCH_PREFIX", "scone"),
            opensearch_url=env.get("SCONE_OPENSEARCH_URL"),
            opensearch_index=env.get("SCONE_OPENSEARCH_INDEX", "scone_vectors"),
            opensearch_username=env.get("SCONE_OPENSEARCH_USERNAME"),
            opensearch_password=env.get("SCONE_OPENSEARCH_PASSWORD"),
            opensearch_batch_size=_environment_integer("SCONE_OPENSEARCH_BATCH_SIZE", env.get("SCONE_OPENSEARCH_BATCH_SIZE", "256")),
            opensearch_max_batch_bytes=_environment_integer("SCONE_OPENSEARCH_MAX_BATCH_BYTES", env.get("SCONE_OPENSEARCH_MAX_BATCH_BYTES", "5242880")),
            opensearch_timeout=parse_seconds("SCONE_OPENSEARCH_TIMEOUT", env.get("SCONE_OPENSEARCH_TIMEOUT"), 30.0),
            opensearch_aws_region=env.get("SCONE_OPENSEARCH_AWS_REGION"),
            opensearch_aws_access_key_id=env.get("SCONE_OPENSEARCH_AWS_ACCESS_KEY_ID"),
            opensearch_aws_secret_access_key=env.get("SCONE_OPENSEARCH_AWS_SECRET_ACCESS_KEY"),
            opensearch_aws_session_token=env.get("SCONE_OPENSEARCH_AWS_SESSION_TOKEN"),
            embed_model=env.get("SCONE_EMBED_MODEL"),
            embed_url=env.get("SCONE_EMBED_URL"),
            embed_api_key=env.get("SCONE_EMBED_API_KEY"),
            embed_cache=env.get("SCONE_EMBED_CACHE"),
            chat_url=env.get("SCONE_CHAT_URL"),
            chat_model=env.get("SCONE_CHAT_MODEL"),
            chat_api_key=env.get("SCONE_CHAT_API_KEY"),
            chat_think={"true": True, "false": False}.get((env.get("SCONE_CHAT_THINK") or "").lower()),
            chat_timeout=parse_seconds("SCONE_CHAT_TIMEOUT", env.get("SCONE_CHAT_TIMEOUT"), 180.0),
            distill_interval_s=float(env.get("SCONE_DISTILL_INTERVAL_S", "30")),
            distill_batch=int(env.get("SCONE_DISTILL_BATCH", "20")),
            derive=parse_flag("SCONE_DERIVE", env.get("SCONE_DERIVE")),
            distill_accept_at=float(env["SCONE_DISTILL_ACCEPT_AT"]) if env.get("SCONE_DISTILL_ACCEPT_AT") else None,
            contextual_embeddings=env.get("SCONE_CONTEXTUAL_EMBEDDINGS") == "1",
            demote_restated=(parse_flag("SCONE_DEMOTE_RESTATED", env["SCONE_DEMOTE_RESTATED"])
                             if env.get("SCONE_DEMOTE_RESTATED") else True),
            many_valued=tuple(item.strip() for item in env.get("SCONE_MANY_VALUED", "").split(",") if item.strip()),
            relation_inverse=tuple(item.strip() for item in env.get("SCONE_RELATION_INVERSE", "").split(",")
                                   if item.strip()),
            relation_symmetric=tuple(item.strip() for item in env.get("SCONE_RELATION_SYMMETRIC", "").split(",")
                                     if item.strip()),
            relation_transitive=tuple(item.strip() for item in env.get("SCONE_RELATION_TRANSITIVE", "").split(",")
                                      if item.strip()),
            abstention_policy=env.get("SCONE_ABSTENTION_POLICY") or None,
            profile_predicates=tuple(item.strip() for item in env.get("SCONE_PROFILE_PREDICATES", "").split(",")
                                     if item.strip()),
            profile_without=tuple(item.strip() for item in env.get("SCONE_PROFILE_WITHOUT", "").split(",")
                                  if item.strip()),
            similarity_floor=float(env["SCONE_SIMILARITY_FLOOR"]) if env.get("SCONE_SIMILARITY_FLOOR") else None,
            candidate_limit=(_environment_integer("SCONE_RECALL_CANDIDATES", env["SCONE_RECALL_CANDIDATES"])
                             if env.get("SCONE_RECALL_CANDIDATES") else None),
            reranker_factory=env.get("SCONE_RERANKER_FACTORY") or None,
            reranker_cross_encoder_dir=env.get("SCONE_RERANKER_CROSS_ENCODER_DIR") or None,
            reranker_cross_encoder_model=env.get("SCONE_RERANKER_CROSS_ENCODER_MODEL") or None,
            reranker_cross_encoder_max_pair_tokens=(_environment_integer("SCONE_RERANKER_CROSS_ENCODER_MAX_PAIR_TOKENS",
                env["SCONE_RERANKER_CROSS_ENCODER_MAX_PAIR_TOKENS"]) if "SCONE_RERANKER_CROSS_ENCODER_MAX_PAIR_TOKENS" in env else None),
            reranker_cross_encoder_threads=(_environment_integer("SCONE_RERANKER_CROSS_ENCODER_THREADS",
                env["SCONE_RERANKER_CROSS_ENCODER_THREADS"]) if "SCONE_RERANKER_CROSS_ENCODER_THREADS" in env else None),
            reranker_cross_encoder_batch_size=(_environment_integer("SCONE_RERANKER_CROSS_ENCODER_BATCH_SIZE",
                env["SCONE_RERANKER_CROSS_ENCODER_BATCH_SIZE"]) if "SCONE_RERANKER_CROSS_ENCODER_BATCH_SIZE" in env else None),
            rerank_limit=_environment_integer("SCONE_RERANK_LIMIT", env.get("SCONE_RERANK_LIMIT", "32")),
            rerank_max_bytes=_environment_integer("SCONE_RERANK_MAX_BYTES", env.get("SCONE_RERANK_MAX_BYTES", "64000")),
            rerank_timeout=parse_seconds("SCONE_RERANK_TIMEOUT", env.get("SCONE_RERANK_TIMEOUT"), 1.0),
            events=env.get("SCONE_EVENTS"),
            events_queries=env.get("SCONE_EVENTS_QUERIES", "hash"),
            events_max_age_days=float(env["SCONE_EVENTS_MAX_AGE_DAYS"]) if env.get("SCONE_EVENTS_MAX_AGE_DAYS") else None,
            events_max=int(env.get("SCONE_EVENTS_MAX", "10000")),
            keys=parse_key_roles(env.get("SCONE_API_KEYS"), env.get("SCONE_API_KEY"))[0],
            roles=parse_key_roles(env.get("SCONE_API_KEYS"), env.get("SCONE_API_KEY"))[1],
            host=env.get("SCONE_HOST", "127.0.0.1"),
            port=int(env.get("SCONE_PORT", "7437")),
            ingest_concurrency=int(env.get("SCONE_INGEST_CONCURRENCY", "4")),
            retention=parse_retention(env.get("SCONE_RETAIN", "")),
            agents_config=env.get("SCONE_AGENTS_CONFIG") or None,
            document_jobs_config=env.get("SCONE_DOCUMENT_JOBS_CONFIG") or None,
            directory_sync_config=env.get("SCONE_DIRECTORY_SYNC_CONFIG") or None,
            document_media_config=env.get("SCONE_DOCUMENT_MEDIA_CONFIG") or None,
            document_ocr_executable=env.get('SCONE_DOCUMENT_OCR_EXECUTABLE') or None,
            document_ocr_language=env.get('SCONE_DOCUMENT_OCR_LANGUAGE', 'eng'),
            document_ocr_psm=int(env.get('SCONE_DOCUMENT_OCR_PSM', '3')),
            document_ocr_dpi=int(env.get('SCONE_DOCUMENT_OCR_DPI', '150')),
            conversations_journal=env.get("SCONE_CONVERSATIONS_JOURNAL") or None,
            conversations_model_factory=env.get("SCONE_CONVERSATIONS_MODEL_FACTORY") or None,
            conversations_personas=env.get("SCONE_CONVERSATIONS_PERSONAS") or None,
            conversations_registry=env.get("SCONE_CONVERSATIONS_REGISTRY") or None,
            answer_review_policy=env.get("SCONE_ANSWER_REVIEW_POLICY", "off"),
            answer_review_url=env.get("SCONE_ANSWER_REVIEW_URL") or None,
            answer_review_model=env.get("SCONE_ANSWER_REVIEW_MODEL") or None,
            answer_review_api_key=env.get("SCONE_ANSWER_REVIEW_API_KEY") or None,
            answer_review_timeout=parse_seconds("SCONE_ANSWER_REVIEW_TIMEOUT", env.get("SCONE_ANSWER_REVIEW_TIMEOUT"), 20.0),
            answer_review_quote_mode=env.get("SCONE_ANSWER_REVIEW_QUOTE_MODE", "text"),
            adaptive_retrieval=parse_flag("SCONE_ADAPTIVE_RETRIEVAL", env.get("SCONE_ADAPTIVE_RETRIEVAL")),
            adaptive_url=env.get("SCONE_ADAPTIVE_URL") or None,
            adaptive_model=env.get("SCONE_ADAPTIVE_MODEL") or None,
            adaptive_api_key=env.get("SCONE_ADAPTIVE_API_KEY") or None,
            adaptive_timeout=parse_seconds("SCONE_ADAPTIVE_TIMEOUT", env.get("SCONE_ADAPTIVE_TIMEOUT"), 15.0),
            adaptive_max_rounds=_environment_integer("SCONE_ADAPTIVE_MAX_ROUNDS", env.get("SCONE_ADAPTIVE_MAX_ROUNDS", "3")),
            adaptive_max_queries=_environment_integer("SCONE_ADAPTIVE_MAX_QUERIES", env.get("SCONE_ADAPTIVE_MAX_QUERIES", "6")),
            adaptive_candidate_limit=_environment_integer("SCONE_ADAPTIVE_CANDIDATE_LIMIT", env.get("SCONE_ADAPTIVE_CANDIDATE_LIMIT", "20")),
            adaptive_max_evidence_bytes=_environment_integer("SCONE_ADAPTIVE_MAX_EVIDENCE_BYTES", env.get("SCONE_ADAPTIVE_MAX_EVIDENCE_BYTES", "16000")),
            adaptive_graph_hops=_environment_integer("SCONE_ADAPTIVE_GRAPH_HOPS", env.get("SCONE_ADAPTIVE_GRAPH_HOPS", "0")),
            adaptive_search_history=parse_flag("SCONE_ADAPTIVE_SEARCH_HISTORY", env.get("SCONE_ADAPTIVE_SEARCH_HISTORY")),
            model_connections=env.get("SCONE_MODEL_CONNECTIONS") or None,
            conversations_tool_mode=env.get("SCONE_CONVERSATIONS_TOOL_MODE", "off"),
            conversations_tool_initial_search=parse_flag("SCONE_CONVERSATIONS_TOOL_INITIAL_SEARCH", env.get("SCONE_CONVERSATIONS_TOOL_INITIAL_SEARCH", "1")),
            conversations_tool_compute=parse_flag("SCONE_CONVERSATIONS_TOOL_COMPUTE", env.get("SCONE_CONVERSATIONS_TOOL_COMPUTE", "0")),
            conversations_tool_max_calls=_environment_integer("SCONE_CONVERSATIONS_TOOL_MAX_CALLS", env.get("SCONE_CONVERSATIONS_TOOL_MAX_CALLS", "4")),
            conversations_tool_max_rounds=_environment_integer("SCONE_CONVERSATIONS_TOOL_MAX_ROUNDS", env.get("SCONE_CONVERSATIONS_TOOL_MAX_ROUNDS", "4")),
            conversations_tool_timeout=parse_seconds("SCONE_CONVERSATIONS_TOOL_TIMEOUT", env.get("SCONE_CONVERSATIONS_TOOL_TIMEOUT"), 120.0),
            log_path=env.get("SCONE_LOG_PATH") or None,
        )


#: What a key may do. read: reads only. write: adds, links and forgets, but
#: never decides. review: decides (approve, decline, exclude, include, batch
#: decisions) but never adds. full: everything.
ROLES = ("read", "write", "review", "full")


def parse_key_roles(many: Optional[str], one: Optional[str]) -> tuple[dict[str, str], dict[str, str]]:
    """``"k1:space-a:read,k2:space-b"`` to ``({"k1": "space-a", "k2": "space-b"},
    {"k1": "read", "k2": "full"})``. A key that appears twice is a
    configuration error, not a last-wins; a role outside ROLES is refused."""
    keys: dict[str, str] = {}
    roles: dict[str, str] = {}
    if many:
        for entry in many.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = [part.strip() for part in entry.split(":")]
            if len(parts) not in (2, 3) or not all(parts):
                raise InvalidInput(f"SCONE_API_KEYS entry must be key:space or key:space:role, got {entry!r}")
            key, space = parts[0], parts[1]
            role = parts[2] if len(parts) == 3 else "full"
            if role not in ROLES:
                raise InvalidInput(f"SCONE_API_KEYS role must be one of {ROLES}, got {role!r}")
            if key in keys:
                raise InvalidInput(f"SCONE_API_KEYS names key {key!r} twice")
            keys[key] = space
            roles[key] = role
    elif one:
        keys[one.strip()] = "default"
        roles[one.strip()] = "full"
    return keys, roles


def parse_keys(many: Optional[str], one: Optional[str]) -> dict[str, str]:
    """The key -> space half of parse_key_roles, for callers that only want spaces."""
    return parse_key_roles(many, one)[0]


def build_profile_policy(settings: Settings):
    """Which claims a profile is made of, as the operator configured it."""
    from ..memory.catalog import ProfilePolicy

    return ProfilePolicy.of(predicates=settings.profile_predicates, without=settings.profile_without)


def build_relation_meanings(settings: Settings):
    """What the space's predicates mean to each other, as the operator
    wrote it, or None when nothing was configured and the graph holds only
    what was said."""
    from ..entities.meanings import RelationMeanings

    opposites: dict[str, str] = {}
    for pair in settings.relation_inverse:
        one, sep, other = pair.partition(":")
        if not sep or not one.strip() or not other.strip():
            raise InvalidInput(
                f"SCONE_RELATION_INVERSE takes pairs written predicate:its opposite, separated by commas; "
                f"{pair!r} is not one")
        opposites[one] = other
    if not (opposites or settings.relation_symmetric or settings.relation_transitive):
        return None
    return RelationMeanings(inverse=opposites, symmetric=settings.relation_symmetric,
                            transitive=settings.relation_transitive)


def build_abstention(settings: Settings):
    """The measured floor to abstain by, when the operator configured one.
    A policy that cannot be read is refused here, not ignored: abstaining
    by a floor nobody measured is what this avoids."""
    from ..retrieval.abstention import AbstentionPolicy

    return AbstentionPolicy.read(settings.abstention_policy) if settings.abstention_policy else None


def build_embedder(settings: Settings):
    if settings.embedder == "hash":
        from ..embedders import HashEmbedder

        return HashEmbedder()
    if settings.embedder == "local":
        from ..embedders import LocalEmbedder

        return LocalEmbedder(settings.embed_model or "bge-small-en-v1.5", settings.embed_cache)
    if settings.embedder == "remote":
        from ..embedders import RemoteEmbedder

        if not settings.embed_url or not settings.embed_model:
            raise InvalidInput("SCONE_EMBEDDER=remote needs SCONE_EMBED_URL and SCONE_EMBED_MODEL")
        return RemoteEmbedder(settings.embed_url, settings.embed_model, settings.embed_api_key)
    raise InvalidInput(f"unknown SCONE_EMBEDDER {settings.embedder!r}")


def build_documents(settings: Settings):
    if settings.documents == "memory":
        from ..backends import InMemoryDocumentStore

        return InMemoryDocumentStore()
    if settings.documents == "sqlite":
        from ..backends import SqliteDocumentStore

        return SqliteDocumentStore(settings.sqlite_path)
    if settings.documents == "mongo":
        from ..backends import MongoDocumentStore

        if not settings.mongo_url:
            raise InvalidInput("SCONE_DOCUMENTS=mongo needs SCONE_MONGO_URL")
        return MongoDocumentStore(settings.mongo_url, settings.mongo_db)
    if settings.documents == "postgres":
        from ..backends import PostgresDocumentStore

        if not settings.postgres_url:
            raise InvalidInput("SCONE_DOCUMENTS=postgres needs SCONE_POSTGRES_URL")
        return PostgresDocumentStore(settings.postgres_url, settings.postgres_schema)
    if settings.documents == "elasticsearch":
        from ..backends import ElasticsearchDocumentStore

        if not settings.elasticsearch_url:
            raise InvalidInput("SCONE_DOCUMENTS=elasticsearch needs SCONE_ELASTICSEARCH_URL")
        return ElasticsearchDocumentStore(settings.elasticsearch_url, settings.elasticsearch_prefix, settings.elasticsearch_api_key)
    raise InvalidInput(f"unknown SCONE_DOCUMENTS {settings.documents!r}")


def _opensearch_auth(settings: Settings) -> AwsSigV4Auth | None:
    region = settings.opensearch_aws_region
    access_key = settings.opensearch_aws_access_key_id
    secret_key = settings.opensearch_aws_secret_access_key
    token = settings.opensearch_aws_session_token
    if all(value is None for value in (region, access_key, secret_key, token)):
        return None
    if region is None or access_key is None or secret_key is None:
        raise InvalidInput("SCONE_OPENSEARCH_AWS_REGION, SCONE_OPENSEARCH_AWS_ACCESS_KEY_ID, and SCONE_OPENSEARCH_AWS_SECRET_ACCESS_KEY are required together")
    from ..providers.aws_auth import AwsSigV4Auth

    return AwsSigV4Auth(access_key, secret_key, region=region, session_token=token)


def build_vectors(settings: Settings, documents=None):
    """``documents`` lets a Postgres vector index share the document
    store's pool when both live in the same database."""
    if settings.vectors == "opensearch":
        from ..backends import OpenSearchVectorIndex

        if not settings.opensearch_url:
            raise InvalidInput("SCONE_VECTORS=opensearch needs SCONE_OPENSEARCH_URL")
        try:
            return OpenSearchVectorIndex(settings.opensearch_url, settings.opensearch_index,
                username=settings.opensearch_username, password=settings.opensearch_password,
                batch_size=settings.opensearch_batch_size, max_batch_bytes=settings.opensearch_max_batch_bytes,
                timeout=settings.opensearch_timeout, auth=_opensearch_auth(settings))
        except ValueError as error:
            raise InvalidInput(f"SCONE_OPENSEARCH configuration: {error}") from error
    if settings.vectors == "elasticache":
        if not settings.elasticache_url:
            raise InvalidInput("SCONE_VECTORS=elasticache needs SCONE_ELASTICACHE_URL")
        from ..backends import ElastiCacheVectorIndex

        try:
            return ElastiCacheVectorIndex(settings.elasticache_url, prefix=settings.elasticache_prefix,
                username=settings.elasticache_username, password=settings.elasticache_password,
                cluster_mode=settings.elasticache_cluster_mode, algorithm=settings.elasticache_algorithm,
                batch_size=settings.elasticache_batch_size, timeout=settings.elasticache_timeout,
                ca_certs=settings.elasticache_ca_certs)
        except ValueError as error:
            raise InvalidInput(f"SCONE_ELASTICACHE configuration: {error}") from error
    if settings.vectors == "elasticsearch":
        from ..backends import ElasticsearchDocumentStore, ElasticsearchVectorIndex

        if isinstance(documents, ElasticsearchDocumentStore):
            return documents.vectors()
        if not settings.elasticsearch_url:
            raise InvalidInput("SCONE_VECTORS=elasticsearch needs SCONE_ELASTICSEARCH_URL")
        return ElasticsearchVectorIndex(settings.elasticsearch_url, settings.elasticsearch_prefix, settings.elasticsearch_api_key)
    if settings.vectors == "postgres":
        from ..backends import PostgresDocumentStore, PostgresVectorIndex

        if isinstance(documents, PostgresDocumentStore):
            return documents.vectors()
        if not settings.postgres_url:
            raise InvalidInput("SCONE_VECTORS=postgres needs SCONE_POSTGRES_URL")
        return PostgresVectorIndex(settings.postgres_url, settings.postgres_schema)
    if settings.vectors == "memory":
        from ..backends import InMemoryVectorIndex

        return InMemoryVectorIndex()
    if settings.vectors == "sqlite":
        from ..backends import SqliteVectorIndex

        return SqliteVectorIndex(settings.sqlite_path)
    if settings.vectors == "qdrant":
        from ..backends import QdrantVectorIndex

        if not settings.qdrant_url:
            raise InvalidInput("SCONE_VECTORS=qdrant needs SCONE_QDRANT_URL")
        return QdrantVectorIndex(settings.qdrant_url, settings.qdrant_collection, settings.qdrant_api_key,
                                 metadata_indexes=settings.qdrant_metadata_indexes, hnsw_ef=settings.qdrant_hnsw_ef)
    if settings.vectors == "chroma":
        from ..backends import ChromaVectorIndex

        return ChromaVectorIndex(path=settings.chroma_path, url=settings.chroma_url)
    if settings.vectors == "lancedb":
        from ..backends import LanceDBVectorIndex

        if not settings.lancedb_path:
            raise InvalidInput("SCONE_VECTORS=lancedb needs SCONE_LANCEDB_PATH")
        return LanceDBVectorIndex(settings.lancedb_path)
    if settings.vectors == "milvus":
        from ..backends import MilvusVectorIndex

        if not settings.milvus_uri:
            raise InvalidInput("SCONE_VECTORS=milvus needs SCONE_MILVUS_URI")
        return MilvusVectorIndex(settings.milvus_uri, settings.milvus_collection, settings.milvus_token)
    if settings.vectors == "redis":
        from ..backends import RedisVectorIndex

        if not settings.redis_url:
            raise InvalidInput("SCONE_VECTORS=redis needs SCONE_REDIS_URL")
        return RedisVectorIndex(settings.redis_url, settings.redis_prefix)
    raise InvalidInput(f"unknown SCONE_VECTORS {settings.vectors!r}")


#: Settings that change what an engine does, so every one of them must
#: reach a bench's per-item engines (see build_in_process_engine).
ENGINE_SETTINGS = ("contextual_embeddings", "similarity_floor", "demote_restated", "candidate_limit",
                   "rerank_limit", "rerank_max_bytes", "rerank_timeout", "many_valued")
#: Settings carried into an engine that are read from a file, not a value.
FILE_SETTINGS = ("abstention_policy",)
#: Settings carried into an engine through a policy they build.
POLICY_SETTINGS = ("profile_predicates", "profile_without")


def _environment_integer(name: str, value: str) -> int:
    if not isinstance(value, str):
        raise InvalidInput(f"{name} must be an integer")
    try:
        return int(value)
    except ValueError:
        raise InvalidInput(f"{name} must be an integer") from None


def _reranker_spec(spec: str) -> tuple[str, str]:
    if not isinstance(spec, str):
        raise InvalidInput("SCONE_RERANKER_FACTORY must name a trusted module:factory")
    module, separator, name = spec.partition(":")
    if not separator or not all(part.isidentifier() for part in module.split(".")) or not name.isidentifier():
        raise InvalidInput("SCONE_RERANKER_FACTORY must name a trusted module:factory")
    return module, name


def build_reranker(settings: Settings) -> Reranker | None:
    """Load an explicit offline model or trusted code; no default is selected.

    The factory is synchronous and takes no required arguments. It returns an
    object with an async rerank method. The operator owns adapter resources and
    shutdown; the engine does not close an injected reranker.
    The built-in cross encoder loads preprovisioned CPU files only. Its optional
    provider module is imported only when the directory/model pair is selected.
    """
    if settings.reranker_cross_encoder_dir is not None and settings.reranker_cross_encoder_model is not None:
        try:
            from ..providers.offline_reranker import OfflineCrossEncoderReranker

            return OfflineCrossEncoderReranker(settings.reranker_cross_encoder_dir,
                model_name=settings.reranker_cross_encoder_model,
                max_pair_tokens=settings.reranker_cross_encoder_max_pair_tokens or 512,
                threads=settings.reranker_cross_encoder_threads or 2,
                batch_size=settings.reranker_cross_encoder_batch_size or 8)
        except Exception:
            raise InvalidInput("SCONE_RERANKER_CROSS_ENCODER requires valid preprovisioned model files and scone-memory[offline-rerank]") from None
    if settings.reranker_factory is None:
        return None
    module, name = _reranker_spec(settings.reranker_factory)
    try:
        factory: object = getattr(importlib.import_module(module), name)
        if not callable(factory) or inspect.iscoroutinefunction(factory) or inspect.isasyncgenfunction(factory):
            raise TypeError("factory must be synchronous")
        inspect.signature(factory).bind()
        adapter: object = factory()
        if inspect.iscoroutine(adapter):
            adapter.close()
            raise TypeError("factory returned a coroutine")
        method = getattr(adapter, "rerank", None)
        if not callable(method) or not inspect.iscoroutinefunction(method):
            raise TypeError("rerank must be async")
        inspect.signature(method).bind("query", ())
    except Exception:
        raise InvalidInput("SCONE_RERANKER_FACTORY must return an adapter with async rerank(query, candidates)") from None
    return cast(Reranker, adapter)


async def build_in_process_engine(settings: Settings, embedder):
    """A fresh engine on in-process stores, carrying every engine setting
    the environment holds. The benches build one per item, and every one
    of them must run under the configuration being measured: twice now a
    new setting has been added and a bench has gone on measuring the
    default under the new name (contextual embeddings, then restatement
    demotion), so both benches build their engines here."""
    from ..backends import InMemoryDocumentStore, InMemoryVectorIndex

    reranker = build_reranker(settings)
    return await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), embedder,
        contextual_embeddings=settings.contextual_embeddings,
        similarity_floor=settings.similarity_floor,
        demote_restated=settings.demote_restated,
        candidate_limit=settings.candidate_limit,
        reranker=reranker,
        rerank_limit=settings.rerank_limit,
        rerank_max_bytes=settings.rerank_max_bytes,
        rerank_timeout=settings.rerank_timeout,
        many_valued=settings.many_valued,
        relation_meanings=build_relation_meanings(settings),
        abstention=build_abstention(settings),
        profile_policy=build_profile_policy(settings),
    ).open()


def parse_seconds(name: str, value, fallback: float) -> float:
    """A length of time from the environment, or a refusal naming the
    setting that was wrong. Falling back to the default without a word
    would leave a run timing out for the one reason it was configured
    not to."""
    if value is None:
        return fallback
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise InvalidInput(f"{name} must be a positive number of seconds, got {value!r}") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise InvalidInput(f"{name} must be a positive number of seconds, got {value!r}")
    return seconds


def build_chat(settings: Settings):
    """The consolidation model, or None when none is configured."""
    if not settings.chat_url and not settings.chat_model:
        return None
    if not (settings.chat_url and settings.chat_model):
        raise InvalidInput("consolidation needs both SCONE_CHAT_URL and SCONE_CHAT_MODEL")
    from ..providers.llm import OpenAICompatibleChat

    return OpenAICompatibleChat(settings.chat_url, settings.chat_model, api_key=settings.chat_api_key,
                                think=settings.chat_think, timeout=settings.chat_timeout)


def build_worker(engine: MemoryEngine, settings: Settings, spaces):
    """A ConsolidationWorker over the configured spaces, or None when there
    is neither a model to distil with nor a retention policy to apply."""
    chat = build_chat(settings)
    if chat is None and not settings.retention:
        return None
    from ..ingestion.worker import ConsolidationWorker

    distiller = None
    if chat is not None:
        from ..ingestion.distill import Distiller

        if settings.distill_accept_at is not None and not 0.0 <= settings.distill_accept_at <= 1.0:
            raise InvalidInput("SCONE_DISTILL_ACCEPT_AT must be within 0..=1")
        distiller = Distiller(engine, chat, accept_at=settings.distill_accept_at)
    deriver = None
    if chat is not None and settings.derive:
        from ..ingestion.derive import Deriver

        deriver = Deriver(engine, chat)
    return ConsolidationWorker(engine, distiller, sorted(set(spaces)), interval_s=settings.distill_interval_s,
                               batch=settings.distill_batch, retention=settings.retention, deriver=deriver)


def parse_flag(name: str, raw: Optional[str]) -> bool:
    """An on/off setting: 1, true, yes, on; 0, false, no, off, or unset."""
    value = (raw or "").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    raise InvalidInput(f"{name} must be 1 or 0, got {raw!r}")


def parse_retention(raw: str) -> dict[str, float]:
    """SCONE_RETAIN: "kind=days,kind=days"; kinds and days are checked the
    way the engine checks them, so a wrong policy stops the server."""
    from ..memory.engine import retention_policy

    policy: dict[str, float] = {}
    for entry in raw.split(","):
        if not entry.strip():
            continue
        kind, sep, days = entry.partition("=")
        try:
            value = float(days) if sep else float("nan")
        except ValueError:
            value = float("nan")
        policy[kind.strip()] = value
    return retention_policy(policy)


def build_events(settings: Settings, documents=None):
    choice = settings.events or {"sqlite": "sqlite", "mongo": "mongo", "postgres": "postgres", "elasticsearch": "elasticsearch"}.get(settings.documents, "memory")
    if choice == "none":
        return None
    if choice == "elasticsearch":
        from ..backends import ElasticsearchDocumentStore, ElasticsearchEventLog

        if isinstance(documents, ElasticsearchDocumentStore):
            return documents.events(settings.events_max_age_days)
        if not settings.elasticsearch_url:
            raise InvalidInput("SCONE_EVENTS=elasticsearch needs SCONE_ELASTICSEARCH_URL")
        return ElasticsearchEventLog(settings.elasticsearch_url, settings.elasticsearch_prefix, settings.elasticsearch_api_key,
                                     max_age_days=settings.events_max_age_days)
    if choice == "postgres":
        from ..backends import PostgresDocumentStore, PostgresEventLog

        if isinstance(documents, PostgresDocumentStore):
            return documents.events(settings.events_max_age_days)
        if not settings.postgres_url:
            raise InvalidInput("SCONE_EVENTS=postgres needs SCONE_POSTGRES_URL")
        return PostgresEventLog(settings.postgres_url, settings.postgres_schema, settings.events_max_age_days)
    if choice == "mongo":
        from ..observability.events import MongoEventLog

        if not settings.mongo_url:
            raise InvalidInput("SCONE_EVENTS=mongo needs SCONE_MONGO_URL")
        return MongoEventLog(settings.mongo_url, settings.mongo_db, settings.events_max_age_days)
    if choice == "memory":
        from ..observability.events import InMemoryEventLog

        return InMemoryEventLog(settings.events_max)
    if choice == "sqlite":
        from ..observability.events import SqliteEventLog

        return SqliteEventLog(settings.sqlite_path, settings.events_max_age_days)
    raise InvalidInput(f"unknown SCONE_EVENTS {settings.events!r}")


def build_blobs(settings: Settings):
    """Select explicit attachment storage or preserve the existing default:
    SCONE_BLOB_DIR, then beside SQLite, otherwise in memory. AWS construction
    is lazy; no SDK, credential resolution, or service call until first use."""
    from ..backends.blobs import FileBlobStore, InMemoryBlobStore

    if settings.blobs == "s3":
        from ..backends.aws_blobs import S3BlobStore

        assert settings.s3_bucket is not None and settings.dynamodb_blob_table is not None and settings.aws_region is not None
        try:
            return S3BlobStore(settings.s3_bucket, settings.dynamodb_blob_table,
                               region_name=settings.aws_region, prefix=settings.s3_prefix)
        except ValueError as error:
            raise InvalidInput(str(error)) from None
    if settings.blobs == "memory":
        return InMemoryBlobStore()
    if settings.blob_dir:
        return FileBlobStore(Path(settings.blob_dir).expanduser())
    if settings.documents == "sqlite" and settings.sqlite_path not in ("", ":memory:"):
        return FileBlobStore(Path(settings.sqlite_path).expanduser().parent / "attachments")
    return InMemoryBlobStore()


async def build_engine(settings: Settings) -> MemoryEngine:
    reranker = build_reranker(settings)
    blobs = build_blobs(settings)
    documents = build_documents(settings)
    if hasattr(documents, "open"):
        await documents.open()
    if settings.events_queries not in ("text", "hash"):
        raise InvalidInput("SCONE_EVENTS_QUERIES must be text or hash")
    events = build_events(settings, documents)
    if hasattr(events, "open"):
        await events.open()
    engine = MemoryEngine(
        documents,
        build_vectors(settings, documents),
        build_embedder(settings),
        events=events,
        record_queries=settings.events_queries == "text",
        contextual_embeddings=settings.contextual_embeddings,
        demote_restated=settings.demote_restated,
        similarity_floor=settings.similarity_floor,
        candidate_limit=settings.candidate_limit,
        reranker=reranker,
        rerank_limit=settings.rerank_limit,
        rerank_max_bytes=settings.rerank_max_bytes,
        rerank_timeout=settings.rerank_timeout,
        many_valued=settings.many_valued,
        relation_meanings=build_relation_meanings(settings),
        abstention=build_abstention(settings),
        profile_policy=build_profile_policy(settings),
        blobs=blobs,
    )
    if settings.embedder == "remote" and engine.embedder.dim == 0:
        await engine.embedder.embed(["warm up"])
    return await engine.open()
