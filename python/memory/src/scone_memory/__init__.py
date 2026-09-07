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
    "FileBlobStore": ".backends.blobs",
    "InMemoryBlobStore": ".backends.blobs",
    "InMemoryDocumentStore": ".backends",
    "InMemoryVectorIndex": ".backends",
    "DistillError": ".ingestion.distill",
    "Distiller": ".ingestion.distill",
    "DistillOutcome": ".ingestion.distill",
    "InMemoryEventLog": ".observability.events",
    "MongoEventLog": ".observability.events",
    "SqliteEventLog": ".observability.events",
    "HashEmbedder": ".embedders",
    "ImportSummary": ".memory.engine",
    "MemoryEngine": ".memory.engine",
    "Profile": ".memory.engine",
    "Record": ".memory.engine",
    "SyncMemoryEngine": ".memory.sync",
    "Conflict": ".core.errors",
    "InvalidInput": ".core.errors",
    "NotFound": ".core.errors",
    "SconeError": ".core.errors",
    "ChatError": ".providers.llm",
    "FakeChat": ".providers.llm",
    "OpenAICompatibleChat": ".providers.llm",
    "Added": ".core.models",
    "Attachment": ".core.models",
    "BatchDecision": ".core.models",
    "Chunk": ".core.models",
    "DecisionOutcome": ".core.models",
    "Episode": ".core.models",
    "Fact": ".core.models",
    "RecallItem": ".core.models",
    "RecallResult": ".core.models",
    "Status": ".core.models",
    "SourcePage": ".core.ports",
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
    "SourcePage",
    "SyncMemoryEngine",
    "Profile",
    "Record",
    "ImportSummary",
    "InMemoryDocumentStore",
    "InMemoryVectorIndex",
    "InMemoryBlobStore",
    "FileBlobStore",
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
    "Conflict",
    "InvalidInput",
    "NotFound",
    "Added",
    "Attachment",
    "BatchDecision",
    "Chunk",
    "DecisionOutcome",
    "Episode",
    "Fact",
    "RecallItem",
    "RecallResult",
    "Status",
]

__version__ = "0.1.0"
