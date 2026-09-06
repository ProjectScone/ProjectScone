"""scone-memory: a temporal memory engine for agents.

    from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("default", "Moved to Lisbon in March", created_at="2024-03-02")
    result = await engine.recall("default", "where do I live")

Swap the two stores for ``MongoDocumentStore`` and ``QdrantVectorIndex``
and nothing else changes.
"""

# Attributes resolve lazily (PEP 562): `from scone_memory import MemoryEngine`
# works as before, but importing one submodule, such as the agent hook that
# runs on every prompt, no longer loads the engine, pydantic and every
# backend. The hook's import cost is what the host waits on.
_LAZY = {
    "InMemoryDocumentStore": ".backends",
    "InMemoryVectorIndex": ".backends",
    "DistillError": ".distill",
    "Distiller": ".distill",
    "DistillOutcome": ".distill",
    "InMemoryEventLog": ".events",
    "MongoEventLog": ".events",
    "SqliteEventLog": ".events",
    "HashEmbedder": ".embedders",
    "ImportSummary": ".engine",
    "MemoryEngine": ".engine",
    "Profile": ".engine",
    "Record": ".engine",
    "SyncMemoryEngine": ".sync",
    "InvalidInput": ".errors",
    "NotFound": ".errors",
    "SconeError": ".errors",
    "ChatError": ".llm",
    "FakeChat": ".llm",
    "OpenAICompatibleChat": ".llm",
    "Added": ".models",
    "Chunk": ".models",
    "Episode": ".models",
    "Fact": ".models",
    "RecallItem": ".models",
    "RecallResult": ".models",
    "Status": ".models",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'scone_memory' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "MemoryEngine",
    "SyncMemoryEngine",
    "Profile",
    "Record",
    "ImportSummary",
    "InMemoryDocumentStore",
    "InMemoryVectorIndex",
    "InMemoryEventLog",
    "SqliteEventLog",
    "MongoEventLog",
    "HashEmbedder",
    "SconeError",
    "Distiller",
    "DistillError",
    "DistillOutcome",
    "ChatError",
    "FakeChat",
    "OpenAICompatibleChat",
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
