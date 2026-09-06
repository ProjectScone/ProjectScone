"""Assemble an engine from environment variables.

    SCONE_DOCUMENTS   memory | mongo            (default memory)
    SCONE_VECTORS     memory | qdrant           (default memory)
    SCONE_EMBEDDER    hash | local | remote     (default hash)

    SCONE_MONGO_URL, SCONE_MONGO_DB (default scone)
    SCONE_QDRANT_URL, SCONE_QDRANT_API_KEY, SCONE_QDRANT_COLLECTION (default scone_chunks)
    SCONE_EMBED_MODEL          local: bge-small-en-v1.5 ; remote: model name
    SCONE_EMBED_URL            remote: OpenAI-compatible base, e.g. http://localhost:11434/v1
    SCONE_EMBED_API_KEY        remote: bearer, optional
    SCONE_EMBED_CACHE          local: model cache dir, optional

    SCONE_API_KEYS    "key:space,key2:space2"   bearer keys and the space each one sees
    SCONE_API_KEY     one key for the space "default" (used when SCONE_API_KEYS is unset)
    SCONE_HOST, SCONE_PORT                       (default 127.0.0.1:7437)
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
    mongo_url: Optional[str] = None
    mongo_db: str = "scone"
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "scone_chunks"
    embed_model: Optional[str] = None
    embed_url: Optional[str] = None
    embed_api_key: Optional[str] = None
    embed_cache: Optional[str] = None
    keys: Mapping[str, str] = field(default_factory=dict)
    host: str = "127.0.0.1"
    port: int = 7437

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        return cls(
            documents=env.get("SCONE_DOCUMENTS", "memory"),
            vectors=env.get("SCONE_VECTORS", "memory"),
            embedder=env.get("SCONE_EMBEDDER", "hash"),
            mongo_url=env.get("SCONE_MONGO_URL"),
            mongo_db=env.get("SCONE_MONGO_DB", "scone"),
            qdrant_url=env.get("SCONE_QDRANT_URL"),
            qdrant_api_key=env.get("SCONE_QDRANT_API_KEY"),
            qdrant_collection=env.get("SCONE_QDRANT_COLLECTION", "scone_chunks"),
            embed_model=env.get("SCONE_EMBED_MODEL"),
            embed_url=env.get("SCONE_EMBED_URL"),
            embed_api_key=env.get("SCONE_EMBED_API_KEY"),
            embed_cache=env.get("SCONE_EMBED_CACHE"),
            keys=parse_keys(env.get("SCONE_API_KEYS"), env.get("SCONE_API_KEY")),
            host=env.get("SCONE_HOST", "127.0.0.1"),
            port=int(env.get("SCONE_PORT", "7437")),
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
    if settings.vectors == "qdrant":
        from .backends import QdrantVectorIndex

        if not settings.qdrant_url:
            raise InvalidInput("SCONE_VECTORS=qdrant needs SCONE_QDRANT_URL")
        return QdrantVectorIndex(settings.qdrant_url, settings.qdrant_collection, settings.qdrant_api_key)
    raise InvalidInput(f"unknown SCONE_VECTORS {settings.vectors!r}")


async def build_engine(settings: Settings) -> MemoryEngine:
    documents = build_documents(settings)
    if hasattr(documents, "open"):
        await documents.open()
    engine = MemoryEngine(documents, build_vectors(settings), build_embedder(settings))
    if settings.embedder == "remote" and engine.embedder.dim == 0:
        await engine.embedder.embed(["warm up"])
    return await engine.open()
