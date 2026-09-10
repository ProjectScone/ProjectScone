"""Source hydration retains real capture paths, never guessed sessions."""

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.core.ports import NewEvent
from scone_memory.observability.events import SqliteEventLog
from scone_memory.testing import Clock


@pytest.fixture(params=["memory", "sqlite"])
async def graph_engine(request, tmp_path):
    clock = Clock()
    if request.param == "sqlite":
        path = tmp_path / "graph.db"
        documents, vectors, events = SqliteDocumentStore(path), SqliteVectorIndex(path), SqliteEventLog(path)
    else:
        documents, vectors, events = InMemoryDocumentStore(), InMemoryVectorIndex(), InMemoryEventLog()
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), events=events, clock=clock).open()
    yield engine, clock
    await engine.close()


@pytest.mark.parametrize("retrieved", [False, True], ids=["claim", "recall"])
async def test_old_source_keeps_its_recorded_session_path(graph_engine, retrieved):
    engine, _ = graph_engine
    source = await engine.remember("alpha", "Juniper observes Polaris")
    capture = await engine.record("alpha", "agent", {
        "agent": "codex", "session_id": "original", "event": "prompt", "episode_id": source.episode_id,
    })
    await engine.record("alpha", "agent", {"agent": "codex", "session_id": "unrelated", "event": "prompt"})
    if retrieved:
        result = await engine.recall("alpha", "Juniper Polaris")
        assert result.items
    else:
        result = await engine.assert_fact("alpha", "Juniper", "observes", "Polaris", source_episode_id=source.episode_id)

    graph = await engine.graph("alpha", limit=1)
    links = {(edge.source, edge.target, edge.kind) for edge in graph.edges}
    assert ("session:codex:original", f"turn:{capture.event_id}", "has") in links
    assert (f"turn:{capture.event_id}", f"episode:{source.episode_id}", "captured_as") in links
    assert "session:codex:unrelated" not in graph.nodes
    if retrieved:
        assert (f"recall:{result.event_id}", f"chunk:{result.items[0].chunk_id}", "returned") in links
    else:
        assert (f"episode:{source.episode_id}", f"claim:{result.fact_id}", "source_of") in links


async def test_hydration_respects_since_and_space_and_does_not_parse_source_uris(graph_engine):
    engine, clock = graph_engine
    source = await engine.remember("alpha", "Retained source", source="codex://not-a-capture")
    await engine.record("alpha", "agent", {
        "agent": "codex", "session_id": "older", "event": "prompt", "episode_id": source.episode_id,
    })
    clock.now = "2026-01-02T00:00:00.000Z"
    await engine.events.append(NewEvent(space="beta", kind="agent", ts=clock.now, payload={
        "agent": "codex", "session_id": "foreign", "event": "prompt", "episode_id": source.episode_id,
    }))
    await engine.assert_fact("alpha", "source", "is", "retained", source_episode_id=source.episode_id)
    graph = await engine.graph("alpha", since=clock.now, limit=1)
    assert f"episode:{source.episode_id}" in graph.nodes
    assert not any(node.kind == "session" for node in graph.nodes.values())


async def test_source_capture_hydration_is_bounded_and_does_not_duplicate_edges(graph_engine):
    engine, _ = graph_engine
    source = await engine.remember("alpha", "Shared source")
    for session in ["first", "second", "third"]:
        await engine.record("alpha", "agent", {
            "agent": "codex", "session_id": session, "event": "prompt", "episode_id": source.episode_id,
        })
    await engine.assert_fact("alpha", "source", "is", "shared", source_episode_id=source.episode_id)
    small = await engine.graph("alpha", limit=1)
    assert sum(node.kind == "session" for node in small.nodes.values()) == 1
    assert small.truncated
    whole = await engine.graph("alpha", limit=20)
    links = [(edge.source, edge.target, edge.kind) for edge in whole.edges]
    assert len(links) == len(set(links))
    assert sum(node.kind == "session" for node in whole.nodes.values()) == 3


async def test_source_capture_scan_does_not_walk_unbounded_history(graph_engine):
    engine, clock = graph_engine
    source = await engine.remember("alpha", "Old source")
    await engine.record("alpha", "agent", {
        "agent": "codex", "session_id": "beyond-bound", "event": "prompt", "episode_id": source.episode_id,
    })
    for _ in range(2001):
        await engine.events.append(NewEvent(space="alpha", kind="agent", ts=clock.now, payload={
            "agent": "codex", "session_id": "recent", "event": "prompt",
        }))
    await engine.assert_fact("alpha", "source", "is", "old", source_episode_id=source.episode_id)
    graph = await engine.graph("alpha", limit=1)
    assert f"episode:{source.episode_id}" in graph.nodes
    assert "session:codex:beyond-bound" not in graph.nodes
    assert graph.truncated


async def test_focused_session_does_not_expand_into_another_capture_of_the_same_source(graph_engine):
    engine, _ = graph_engine
    source = await engine.remember("alpha", "Shared source")
    for session in ["selected", "other"]:
        await engine.record("alpha", "agent", {
            "agent": "codex", "session_id": session, "event": "prompt", "episode_id": source.episode_id,
        })
    await engine.assert_fact("alpha", "source", "is", "shared", source_episode_id=source.episode_id)
    graph = await engine.graph("alpha", session_id="selected", limit=1)
    assert "session:codex:selected" in graph.nodes
    assert "session:codex:other" not in graph.nodes
