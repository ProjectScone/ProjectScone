"""Storage adapters. ``memory`` is always available; the others import
their driver lazily so the core package installs with no database
client at all."""

from .memory import InMemoryDocumentStore, InMemoryVectorIndex
from .sqlite import SqliteDocumentStore, SqliteVectorIndex

__all__ = [
    "InMemoryDocumentStore",
    "InMemoryVectorIndex",
    "SqliteDocumentStore",
    "SqliteVectorIndex",
    "MongoDocumentStore",
    "QdrantVectorIndex",
    "ChromaVectorIndex",
    "LanceDBVectorIndex",
    "PostgresDocumentStore",
    "PostgresVectorIndex",
    "PostgresEventLog",
]


def __getattr__(name: str):
    if name == "MongoDocumentStore":
        from .mongo import MongoDocumentStore

        return MongoDocumentStore
    if name == "QdrantVectorIndex":
        from .qdrant import QdrantVectorIndex

        return QdrantVectorIndex
    if name == "ChromaVectorIndex":
        from .chroma import ChromaVectorIndex

        return ChromaVectorIndex
    if name == "LanceDBVectorIndex":
        from .lancedb import LanceDBVectorIndex

        return LanceDBVectorIndex
    if name in ("PostgresDocumentStore", "PostgresVectorIndex", "PostgresEventLog"):
        from . import postgres

        return getattr(postgres, name)
    raise AttributeError(name)
