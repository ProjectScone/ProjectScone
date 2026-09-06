"""scone-memory: a temporal memory engine for agents.

    from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("default", "Moved to Lisbon in March", created_at="2024-03-02")
    result = await engine.recall("default", "where do I live")

Swap the two stores for ``MongoDocumentStore`` and ``QdrantVectorIndex``
and nothing else changes.
"""

from .backends import InMemoryDocumentStore, InMemoryVectorIndex
from .embedders import HashEmbedder
from .engine import MemoryEngine, Profile
from .errors import InvalidInput, NotFound, SconeError
from .models import Added, Chunk, Episode, Fact, RecallItem, RecallResult, Status

__all__ = [
    "MemoryEngine",
    "Profile",
    "InMemoryDocumentStore",
    "InMemoryVectorIndex",
    "HashEmbedder",
    "SconeError",
    "InvalidInput",
    "NotFound",
    "Added",
    "Chunk",
    "Episode",
    "Fact",
    "RecallItem",
    "RecallResult",
    "Status",
]

__version__ = "0.1.0"
