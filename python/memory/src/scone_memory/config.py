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
    SCONE_DISTILL_INTERVAL_S           seconds between consolidation passes (default 30)
    SCONE_DISTILL_BATCH                episodes per pass per space (default 20)
    SCONE_DISTILL_ACCEPT_AT            confidence at or above which extractions enter the ledger
                                       directly; unset = every extraction is proposed for review
    SCONE_MCP_PROPOSE_BELOW            confidence below which a fact submitted over MCP is parked for
                                       review; unset = every submitted fact is a ledger claim, which
                                       is what the Rust server does without --propose-below

    SCONE_API_KEYS    "key:space,key2:space2"   bearer keys and the space each one sees
    SCONE_API_KEY     one key for the space "default" (used when SCONE_API_KEYS is unset)
    SCONE_HOST, SCONE_PORT                       (default 127.0.0.1:7437)
    SCONE_RELOAD_PAGES=1 (alias SCONE_UI_DEV=1)  re-read console.html and playground.html per request, ETag from
                                                 file mtime and size, Cache-Control no-store (development)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from .engine import MemoryEngine
from .errors import InvalidInput


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
    distill_interval_s: float = 30.0
    distill_batch: int = 20
    distill_accept_at: Optional[float] = None
    contextual_embeddings: bool = False
    demote_restated: bool = False
    similarity_floor: Optional[float] = None
    events: Optional[str] = None
    events_queries: str = "hash"
    events_max_age_days: Optional[float] = None
    events_max: int = 10_000
    keys: Mapping[str, str] = field(default_factory=dict)
    host: str = "127.0.0.1"
    port: int = 7437
    reload_pages: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
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
            distill_interval_s=float(env.get("SCONE_DISTILL_INTERVAL_S", "30")),
            distill_batch=int(env.get("SCONE_DISTILL_BATCH", "20")),
            distill_accept_at=float(env["SCONE_DISTILL_ACCEPT_AT"]) if env.get("SCONE_DISTILL_ACCEPT_AT") else None,
            contextual_embeddings=env.get("SCONE_CONTEXTUAL_EMBEDDINGS") == "1",
            demote_restated=env.get("SCONE_DEMOTE_RESTATED") == "1",
            similarity_floor=float(env["SCONE_SIMILARITY_FLOOR"]) if env.get("SCONE_SIMILARITY_FLOOR") else None,
            events=env.get("SCONE_EVENTS"),
            events_queries=env.get("SCONE_EVENTS_QUERIES", "hash"),
            events_max_age_days=float(env["SCONE_EVENTS_MAX_AGE_DAYS"]) if env.get("SCONE_EVENTS_MAX_AGE_DAYS") else None,
            events_max=int(env.get("SCONE_EVENTS_MAX", "10000")),
            keys=parse_keys(env.get("SCONE_API_KEYS"), env.get("SCONE_API_KEY")),
            host=env.get("SCONE_HOST", "127.0.0.1"),
            port=int(env.get("SCONE_PORT", "7437")),
            reload_pages=env.get("SCONE_RELOAD_PAGES") == "1" or env.get("SCONE_UI_DEV") == "1",
        )


def parse_keys(many: Optional[str], one: Optional[str]) -> dict[str, str]:
    """``"k1:space-a,k2:space-b"`` to ``{"k1": "space-a", "k2": "space-b"}``.
    A key that appears twice is a configuration error, not a last-wins."""
    keys: dict[str, str] = {}
    if many:
        for entry in many.split(","):
            entry = entry.strip()
            if not entry:
                continue
            key, sep, space = entry.partition(":")
            if not sep or not key.strip() or not space.strip():
                raise InvalidInput(f"SCONE_API_KEYS entry must be key:space, got {entry!r}")
            if key.strip() in keys:
                raise InvalidInput(f"SCONE_API_KEYS names key {key.strip()!r} twice")
            keys[key.strip()] = space.strip()
    elif one:
        keys[one.strip()] = "default"
    return keys


def build_embedder(settings: Settings):
    if settings.embedder == "hash":
        from .embedders import HashEmbedder

        return HashEmbedder()
    if settings.embedder == "local":
        from .embedders import LocalEmbedder

        return LocalEmbedder(settings.embed_model or "bge-small-en-v1.5", settings.embed_cache)
    if settings.embedder == "remote":
        from .embedders import RemoteEmbedder

        if not settings.embed_url or not settings.embed_model:
            raise InvalidInput("SCONE_EMBEDDER=remote needs SCONE_EMBED_URL and SCONE_EMBED_MODEL")
        return RemoteEmbedder(settings.embed_url, settings.embed_model, settings.embed_api_key)
    raise InvalidInput(f"unknown SCONE_EMBEDDER {settings.embedder!r}")


def build_documents(settings: Settings):
    if settings.documents == "memory":
        from .backends import InMemoryDocumentStore

        return InMemoryDocumentStore()
    if settings.documents == "sqlite":
        from .backends import SqliteDocumentStore

        return SqliteDocumentStore(settings.sqlite_path)
    if settings.documents == "mongo":
        from .backends import MongoDocumentStore

        if not settings.mongo_url:
            raise InvalidInput("SCONE_DOCUMENTS=mongo needs SCONE_MONGO_URL")
        return MongoDocumentStore(settings.mongo_url, settings.mongo_db)
    if settings.documents == "postgres":
        from .backends import PostgresDocumentStore

        if not settings.postgres_url:
            raise InvalidInput("SCONE_DOCUMENTS=postgres needs SCONE_POSTGRES_URL")
        return PostgresDocumentStore(settings.postgres_url, settings.postgres_schema)
    if settings.documents == "elasticsearch":
        from .backends import ElasticsearchDocumentStore

        if not settings.elasticsearch_url:
            raise InvalidInput("SCONE_DOCUMENTS=elasticsearch needs SCONE_ELASTICSEARCH_URL")
        return ElasticsearchDocumentStore(settings.elasticsearch_url, settings.elasticsearch_prefix, settings.elasticsearch_api_key)
    raise InvalidInput(f"unknown SCONE_DOCUMENTS {settings.documents!r}")


def build_vectors(settings: Settings, documents=None):
    """``documents`` lets a Postgres vector index share the document
    store's pool when both live in the same database."""
    if settings.vectors == "elasticsearch":
        from .backends import ElasticsearchDocumentStore, ElasticsearchVectorIndex

        if isinstance(documents, ElasticsearchDocumentStore):
            return documents.vectors()
        if not settings.elasticsearch_url:
            raise InvalidInput("SCONE_VECTORS=elasticsearch needs SCONE_ELASTICSEARCH_URL")
        return ElasticsearchVectorIndex(settings.elasticsearch_url, settings.elasticsearch_prefix, settings.elasticsearch_api_key)
    if settings.vectors == "postgres":
        from .backends import PostgresDocumentStore, PostgresVectorIndex

        if isinstance(documents, PostgresDocumentStore):
            return documents.vectors()
        if not settings.postgres_url:
            raise InvalidInput("SCONE_VECTORS=postgres needs SCONE_POSTGRES_URL")
        return PostgresVectorIndex(settings.postgres_url, settings.postgres_schema)
    if settings.vectors == "memory":
        from .backends import InMemoryVectorIndex

        return InMemoryVectorIndex()
    if settings.vectors == "sqlite":
        from .backends import SqliteVectorIndex

        return SqliteVectorIndex(settings.sqlite_path)
    if settings.vectors == "qdrant":
        from .backends import QdrantVectorIndex

        if not settings.qdrant_url:
            raise InvalidInput("SCONE_VECTORS=qdrant needs SCONE_QDRANT_URL")
        return QdrantVectorIndex(settings.qdrant_url, settings.qdrant_collection, settings.qdrant_api_key)
    if settings.vectors == "chroma":
        from .backends import ChromaVectorIndex

        return ChromaVectorIndex(path=settings.chroma_path, url=settings.chroma_url)
    if settings.vectors == "lancedb":
        from .backends import LanceDBVectorIndex

        if not settings.lancedb_path:
            raise InvalidInput("SCONE_VECTORS=lancedb needs SCONE_LANCEDB_PATH")
        return LanceDBVectorIndex(settings.lancedb_path)
    if settings.vectors == "milvus":
        from .backends import MilvusVectorIndex

        if not settings.milvus_uri:
            raise InvalidInput("SCONE_VECTORS=milvus needs SCONE_MILVUS_URI")
        return MilvusVectorIndex(settings.milvus_uri, settings.milvus_collection, settings.milvus_token)
    if settings.vectors == "redis":
        from .backends import RedisVectorIndex

        if not settings.redis_url:
            raise InvalidInput("SCONE_VECTORS=redis needs SCONE_REDIS_URL")
        return RedisVectorIndex(settings.redis_url, settings.redis_prefix)
    raise InvalidInput(f"unknown SCONE_VECTORS {settings.vectors!r}")


#: Settings that change what an engine does, so every one of them must
#: reach a bench's per-item engines (see build_in_process_engine).
ENGINE_SETTINGS = ("contextual_embeddings", "similarity_floor", "demote_restated")


async def build_in_process_engine(settings: Settings, embedder):
    """A fresh engine on in-process stores, carrying every engine setting
    the environment holds. The benches build one per item, and every one
    of them must run under the configuration being measured: twice now a
    new setting has been added and a bench has gone on measuring the
    default under the new name (contextual embeddings, then restatement
    demotion), so both benches build their engines here."""
    from .backends import InMemoryDocumentStore, InMemoryVectorIndex

    return await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), embedder,
        contextual_embeddings=settings.contextual_embeddings,
        similarity_floor=settings.similarity_floor,
        demote_restated=settings.demote_restated,
    ).open()


def build_chat(settings: Settings):
    """The consolidation model, or None when none is configured."""
    if not settings.chat_url and not settings.chat_model:
        return None
    if not (settings.chat_url and settings.chat_model):
        raise InvalidInput("consolidation needs both SCONE_CHAT_URL and SCONE_CHAT_MODEL")
    from .llm import OpenAICompatibleChat

    return OpenAICompatibleChat(settings.chat_url, settings.chat_model, api_key=settings.chat_api_key, think=settings.chat_think)


def build_worker(engine: MemoryEngine, settings: Settings, spaces):
    """A ConsolidationWorker over the configured spaces, or None."""
    chat = build_chat(settings)
    if chat is None:
        return None
    from .distill import Distiller
    from .worker import ConsolidationWorker

    if settings.distill_accept_at is not None and not 0.0 <= settings.distill_accept_at <= 1.0:
        raise InvalidInput("SCONE_DISTILL_ACCEPT_AT must be within 0..=1")
    distiller = Distiller(engine, chat, accept_at=settings.distill_accept_at)
    return ConsolidationWorker(engine, distiller, sorted(set(spaces)), interval_s=settings.distill_interval_s, batch=settings.distill_batch)


def build_events(settings: Settings, documents=None):
    choice = settings.events or {"sqlite": "sqlite", "mongo": "mongo", "postgres": "postgres", "elasticsearch": "elasticsearch"}.get(settings.documents, "memory")
    if choice == "none":
        return None
    if choice == "elasticsearch":
        from .backends import ElasticsearchDocumentStore, ElasticsearchEventLog

        if isinstance(documents, ElasticsearchDocumentStore):
            return documents.events(settings.events_max_age_days)
        if not settings.elasticsearch_url:
            raise InvalidInput("SCONE_EVENTS=elasticsearch needs SCONE_ELASTICSEARCH_URL")
        return ElasticsearchEventLog(settings.elasticsearch_url, settings.elasticsearch_prefix, settings.elasticsearch_api_key,
                                     max_age_days=settings.events_max_age_days)
    if choice == "postgres":
        from .backends import PostgresDocumentStore, PostgresEventLog

        if isinstance(documents, PostgresDocumentStore):
            return documents.events(settings.events_max_age_days)
        if not settings.postgres_url:
            raise InvalidInput("SCONE_EVENTS=postgres needs SCONE_POSTGRES_URL")
        return PostgresEventLog(settings.postgres_url, settings.postgres_schema, settings.events_max_age_days)
    if choice == "mongo":
        from .events import MongoEventLog

        if not settings.mongo_url:
            raise InvalidInput("SCONE_EVENTS=mongo needs SCONE_MONGO_URL")
        return MongoEventLog(settings.mongo_url, settings.mongo_db, settings.events_max_age_days)
    if choice == "memory":
        from .events import InMemoryEventLog

        return InMemoryEventLog(settings.events_max)
    if choice == "sqlite":
        from .events import SqliteEventLog

        return SqliteEventLog(settings.sqlite_path, settings.events_max_age_days)
    raise InvalidInput(f"unknown SCONE_EVENTS {settings.events!r}")


def build_blobs(settings: Settings):
    """Where attachments are kept. SCONE_BLOB_DIR wins; otherwise a
    directory beside the SQLite file, because a server whose database is
    on disk should not lose its evidence on a restart. With no database on
    disk there is nowhere obvious to write, so bytes stay in memory."""
    from .blobs import FileBlobStore, InMemoryBlobStore

    if settings.blob_dir:
        return FileBlobStore(Path(settings.blob_dir).expanduser())
    if settings.documents == "sqlite" and settings.sqlite_path not in ("", ":memory:"):
        return FileBlobStore(Path(settings.sqlite_path).expanduser().parent / "attachments")
    return InMemoryBlobStore()


async def build_engine(settings: Settings) -> MemoryEngine:
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
        blobs=build_blobs(settings),
    )
    if settings.embedder == "remote" and engine.embedder.dim == 0:
        await engine.embedder.embed(["warm up"])
    return await engine.open()
