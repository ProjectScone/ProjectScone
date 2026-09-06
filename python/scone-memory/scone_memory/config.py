"""Assemble an engine from environment variables.

    SCONE_DOCUMENTS   memory | sqlite | mongo   (default memory)
    SCONE_VECTORS     memory | sqlite | qdrant  (default memory)
    SCONE_EMBEDDER    hash | local | remote     (default hash)

    SCONE_SQLITE_PATH (default ~/.scone-memory/memory.db; both sqlite stores share it)
    SCONE_MONGO_URL, SCONE_MONGO_DB (default scone)
    SCONE_QDRANT_URL, SCONE_QDRANT_API_KEY, SCONE_QDRANT_COLLECTION (default scone_chunks)
    SCONE_EMBED_MODEL          local: bge-small-en-v1.5 ; remote: model name
    SCONE_EMBED_URL            remote: OpenAI-compatible base, e.g. http://localhost:11434/v1
    SCONE_EMBED_API_KEY        remote: bearer, optional
    SCONE_EMBED_CACHE          local: model cache dir, optional

    SCONE_EVENTS      memory | sqlite | mongo | none
                      (default follows SCONE_DOCUMENTS: sqlite -> sqlite, mongo -> mongo, else memory)
    SCONE_EVENTS_QUERIES  hash | text           (default hash: a sha256 prefix, never the query text)
    SCONE_EVENTS_MAX_AGE_DAYS                    sqlite sink retention, optional
    SCONE_EVENTS_MAX     in-memory sink ring size (default 10000)

    SCONE_CHAT_URL, SCONE_CHAT_MODEL   OpenAI-compatible chat model for consolidation; unset = no distiller
    SCONE_CHAT_API_KEY                 optional bearer
    SCONE_CHAT_THINK   true | false    for Ollama reasoning models; unset leaves the field out
    SCONE_DISTILL_INTERVAL_S           seconds between consolidation passes (default 30)
    SCONE_DISTILL_BATCH                episodes per pass per space (default 20)
    SCONE_DISTILL_ACCEPT_AT            confidence at or above which extractions enter the ledger
                                       directly; unset = every extraction is proposed for review

    SCONE_API_KEYS    "key:space,key2:space2"   bearer keys and the space each one sees
    SCONE_API_KEY     one key for the space "default" (used when SCONE_API_KEYS is unset)
    SCONE_HOST, SCONE_PORT                       (default 127.0.0.1:7437)
    SCONE_RELOAD_PAGES=1                         re-read console.html and playground.html per request (development)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

from .engine import MemoryEngine
from .errors import InvalidInput


@dataclass(frozen=True)
class Settings:
    documents: str = "memory"
    vectors: str = "memory"
    embedder: str = "hash"
    sqlite_path: str = "~/.scone-memory/memory.db"
    mongo_url: Optional[str] = None
    mongo_db: str = "scone"
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "scone_chunks"
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
            mongo_url=env.get("SCONE_MONGO_URL"),
            mongo_db=env.get("SCONE_MONGO_DB", "scone"),
            qdrant_url=env.get("SCONE_QDRANT_URL"),
            qdrant_api_key=env.get("SCONE_QDRANT_API_KEY"),
            qdrant_collection=env.get("SCONE_QDRANT_COLLECTION", "scone_chunks"),
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
            events=env.get("SCONE_EVENTS"),
            events_queries=env.get("SCONE_EVENTS_QUERIES", "hash"),
            events_max_age_days=float(env["SCONE_EVENTS_MAX_AGE_DAYS"]) if env.get("SCONE_EVENTS_MAX_AGE_DAYS") else None,
            events_max=int(env.get("SCONE_EVENTS_MAX", "10000")),
            keys=parse_keys(env.get("SCONE_API_KEYS"), env.get("SCONE_API_KEY")),
            host=env.get("SCONE_HOST", "127.0.0.1"),
            port=int(env.get("SCONE_PORT", "7437")),
            reload_pages=env.get("SCONE_RELOAD_PAGES") == "1",
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
    raise InvalidInput(f"unknown SCONE_DOCUMENTS {settings.documents!r}")


def build_vectors(settings: Settings):
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
    raise InvalidInput(f"unknown SCONE_VECTORS {settings.vectors!r}")


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


def build_events(settings: Settings):
    choice = settings.events or {"sqlite": "sqlite", "mongo": "mongo"}.get(settings.documents, "memory")
    if choice == "none":
        return None
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


async def build_engine(settings: Settings) -> MemoryEngine:
    documents = build_documents(settings)
    if hasattr(documents, "open"):
        await documents.open()
    if settings.events_queries not in ("text", "hash"):
        raise InvalidInput("SCONE_EVENTS_QUERIES must be text or hash")
    events = build_events(settings)
    if hasattr(events, "open"):
        await events.open()
    engine = MemoryEngine(
        documents,
        build_vectors(settings),
        build_embedder(settings),
        events=events,
        record_queries=settings.events_queries == "text",
    )
    if settings.embedder == "remote" and engine.embedder.dim == 0:
        await engine.embedder.embed(["warm up"])
    return await engine.open()
