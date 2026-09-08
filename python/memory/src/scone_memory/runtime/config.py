"""Assemble an engine from environment variables.

    SCONE_DOCUMENTS   memory | sqlite | mongo | postgres | elasticsearch   (default memory)
    SCONE_VECTORS     memory | sqlite | qdrant | chroma | lancedb | milvus | postgres | redis | elasticsearch  (default memory)
    SCONE_EMBEDDER    hash | local | remote     (default hash)

    SCONE_SQLITE_PATH (default ~/.scone-memory/memory.db; both sqlite stores share it)
    SCONE_MONGO_URL, SCONE_MONGO_DB (default scone)
    SCONE_POSTGRES_URL, SCONE_POSTGRES_SCHEMA (default scone); documents, vectors and events share one pool
    SCONE_QDRANT_URL, SCONE_QDRANT_API_KEY, SCONE_QDRANT_COLLECTION (default scone_chunks)
    SCONE_CHROMA_PATH (persistent directory) or SCONE_CHROMA_URL (server); neither: in-process, ephemeral
    SCONE_LANCEDB_PATH (database directory, required for lancedb)
    SCONE_REDIS_URL (required for redis; needs the RediSearch module), SCONE_REDIS_PREFIX (default scone_chunks)
    SCONE_MILVUS_URI (a local .db path runs Milvus Lite; http://host:19530 a server), SCONE_MILVUS_TOKEN, SCONE_MILVUS_COLLECTION
    SCONE_ELASTICSEARCH_URL, SCONE_ELASTICSEARCH_API_KEY, SCONE_ELASTICSEARCH_PREFIX (default scone); one client for all three
    SCONE_EMBED_MODEL          local: bge-small-en-v1.5 ; remote: model name
    SCONE_EMBED_URL            remote: OpenAI-compatible base, e.g. http://localhost:11434/v1
    SCONE_EMBED_API_KEY        remote: bearer, optional
    SCONE_EMBED_CACHE          local: model cache dir, optional

    SCONE_CONTEXTUAL_EMBEDDINGS=1  embed a date/source/scope prefix with each chunk (experiment 8; off by default)
    SCONE_DEMOTE_RESTATED=1        rank a restated claim ahead of what it replaces (experiment 5; off by default)
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
    SCONE_RELOAD_PAGES=1 (alias SCONE_UI_DEV=1)  re-read console.html and playground.html per request, ETag from
                                                 file mtime and size, Cache-Control no-store (development)
"""

from __future__ import annotations

import math
import importlib
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, cast

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
    chroma_path: Optional[str] = None
    chroma_url: Optional[str] = None
    lancedb_path: Optional[str] = None
    redis_url: Optional[str] = None
    redis_prefix: str = "scone_chunks"
    milvus_uri: Optional[str] = None
    milvus_token: Optional[str] = None
    milvus_collection: str = "scone_chunks"
    elasticsearch_url: Optional[str] = None
    elasticsearch_api_key: Optional[str] = None
    elasticsearch_prefix: str = "scone"
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
    similarity_floor: Optional[float] = None
    candidate_limit: int | None = None
    reranker_factory: str | None = None
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
    reload_pages: bool = False
    #: Writes embedding at once over HTTP; one more is told to come back.
    ingest_concurrency: int = 4
    #: Episode kinds to the days they are kept; empty means nothing expires.
    retention: Mapping[str, float] = field(default_factory=dict)
    # Naming a journal composes the conversation service onto the memory
    # origin under `serve`; the factory is the same trusted module:callable
    # as `serve-conversations --model-factory`, absent meaning history-only.
    conversations_journal: Optional[str] = None
    conversations_model_factory: Optional[str] = None
    # A persona catalog (JSON array of Persona documents) needs a registry
    # (trusted module:callable returning a ProviderRegistry) to bind it.
    conversations_personas: Optional[str] = None
    conversations_registry: Optional[str] = None
    # Opt-in private local service settings and operational diagnostics.
    model_connections: Optional[str] = None
    log_path: Optional[str] = None

    def __post_init__(self) -> None:
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

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
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
            mcp_propose_below=(float(env["SCONE_MCP_PROPOSE_BELOW"])
                               if env.get("SCONE_MCP_PROPOSE_BELOW") else None),
            mongo_url=env.get("SCONE_MONGO_URL"),
            mongo_db=env.get("SCONE_MONGO_DB", "scone"),
            postgres_url=env.get("SCONE_POSTGRES_URL"),
            postgres_schema=env.get("SCONE_POSTGRES_SCHEMA", "scone"),
            qdrant_url=env.get("SCONE_QDRANT_URL"),
            qdrant_api_key=env.get("SCONE_QDRANT_API_KEY"),
            qdrant_collection=env.get("SCONE_QDRANT_COLLECTION", "scone_chunks"),
            chroma_path=env.get("SCONE_CHROMA_PATH"),
            chroma_url=env.get("SCONE_CHROMA_URL"),
            lancedb_path=env.get("SCONE_LANCEDB_PATH"),
            redis_url=env.get("SCONE_REDIS_URL"),
            redis_prefix=env.get("SCONE_REDIS_PREFIX", "scone_chunks"),
            milvus_uri=env.get("SCONE_MILVUS_URI"),
            milvus_token=env.get("SCONE_MILVUS_TOKEN"),
            milvus_collection=env.get("SCONE_MILVUS_COLLECTION", "scone_chunks"),
            elasticsearch_url=env.get("SCONE_ELASTICSEARCH_URL"),
            elasticsearch_api_key=env.get("SCONE_ELASTICSEARCH_API_KEY"),
            elasticsearch_prefix=env.get("SCONE_ELASTICSEARCH_PREFIX", "scone"),
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
            similarity_floor=float(env["SCONE_SIMILARITY_FLOOR"]) if env.get("SCONE_SIMILARITY_FLOOR") else None,
            candidate_limit=(_environment_integer("SCONE_RECALL_CANDIDATES", env["SCONE_RECALL_CANDIDATES"])
                             if env.get("SCONE_RECALL_CANDIDATES") else None),
            reranker_factory=env.get("SCONE_RERANKER_FACTORY") or None,
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
            reload_pages=env.get("SCONE_RELOAD_PAGES") == "1" or env.get("SCONE_UI_DEV") == "1",
            ingest_concurrency=int(env.get("SCONE_INGEST_CONCURRENCY", "4")),
            retention=parse_retention(env.get("SCONE_RETAIN", "")),
            conversations_journal=env.get("SCONE_CONVERSATIONS_JOURNAL") or None,
            conversations_model_factory=env.get("SCONE_CONVERSATIONS_MODEL_FACTORY") or None,
            conversations_personas=env.get("SCONE_CONVERSATIONS_PERSONAS") or None,
            conversations_registry=env.get("SCONE_CONVERSATIONS_REGISTRY") or None,
            model_connections=env.get("SCONE_MODEL_CONNECTIONS") or None,
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


def build_vectors(settings: Settings, documents=None):
    """``documents`` lets a Postgres vector index share the document
    store's pool when both live in the same database."""
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
        return QdrantVectorIndex(settings.qdrant_url, settings.qdrant_collection, settings.qdrant_api_key)
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
                   "rerank_limit", "rerank_max_bytes", "rerank_timeout")


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
    """Load explicit trusted operator code; no default provider is selected.

    The factory is synchronous and takes no required arguments. It returns an
    object with an async rerank method. The operator owns adapter resources and
    shutdown; the engine does not close an injected reranker.
    """
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
    """Where attachments are kept. SCONE_BLOB_DIR wins; otherwise a
    directory beside the SQLite file, because a server whose database is
    on disk should not lose its evidence on a restart. With no database on
    disk there is nowhere obvious to write, so bytes stay in memory."""
    from ..backends.blobs import FileBlobStore, InMemoryBlobStore

    if settings.blob_dir:
        return FileBlobStore(Path(settings.blob_dir).expanduser())
    if settings.documents == "sqlite" and settings.sqlite_path not in ("", ":memory:"):
        return FileBlobStore(Path(settings.sqlite_path).expanduser().parent / "attachments")
    return InMemoryBlobStore()


async def build_engine(settings: Settings) -> MemoryEngine:
    reranker = build_reranker(settings)
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
        blobs=build_blobs(settings),
    )
    if settings.embedder == "remote" and engine.embedder.dim == 0:
        await engine.embedder.embed(["warm up"])
    return await engine.open()
