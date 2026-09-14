"""Narrowing in the vector lane, and the window it works in, disclosed.

A recall with a metadata condition, a kind, a source prefix or a date
bound narrows its candidates. The text lane narrows in the store; the
vector lane could not, so it returned its best few and the filter then
removed the ones that did not fit -- and when a selective filter removed
all of them, the answer came back empty while the memory that fit sat
just outside the window, and nothing in the result said so.

Two things change here. The in-process vector indexes evaluate metadata
conditions themselves, so a condition reaches every stored vector; and
a lane that still has to be post-filtered is given a wider window and the
result carries what happened: which lane narrowed where, how deep each
looked, how many candidates the filter removed, and whether a lane's
window was full when it did -- the case where a deeper memory may fit.
"""
from __future__ import annotations

import math

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.core.ports import VectorPoint
from scone_memory.retrieval.filters import parse_filter
from scone_memory.retrieval.recall import LANE_DEPTH, UNFILTERED_DEPTH

QUERY = "what went out"
PUBLISHED = "manifest shipped"
DRAFTS = 60


class Planted:
    """Sixty drafts sit exactly on the question; the one published memory
    sits close but behind all of them, and shares no token with the
    question so the text lane cannot find it. The vector lane is the only
    road to it, and it is sixty-first on that road."""

    id = "planted-v1"
    dim = 4

    async def embed(self, texts):
        out = []
        for text in texts:
            if PUBLISHED in text:
                out.append([0.9, math.sqrt(1 - 0.81), 0.0, 0.0])
            elif QUERY in text or "draft note" in text:
                out.append([1.0, 0.0, 0.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0, 0.0])
        return out


@pytest.fixture(params=["memory", "sqlite"])
async def engine(request, tmp_path):
    if request.param == "sqlite":
        documents, vectors = SqliteDocumentStore(tmp_path / "s.db"), SqliteVectorIndex(tmp_path / "s.db")
    else:
        documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    memory = await MemoryEngine(documents, vectors, Planted()).open()
    for n in range(DRAFTS):
        await memory.remember("s", f"draft note {n} about weather", metadata={"status": "draft"}, source=f"drafts/{n}")
    await memory.remember("s", PUBLISHED, metadata={"status": "published"}, source="release/plan")
    yield memory
    await memory.close()


PUBLISHED_ONLY = {"field": "status", "is": "published"}


async def test_a_selective_condition_reaches_past_the_old_window(engine):
    found = await engine.recall("s", QUERY, limit=5, conditions=PUBLISHED_ONLY)
    assert [item.source for item in found.items] == ["release/plan"]
    narrowing = found.narrowing
    assert narrowing is not None and narrowing.vector_lane == "in_store" and narrowing.postfiltered_out == 0
    assert narrowing.window_exhausted is False


async def test_a_lane_that_cannot_narrow_is_given_a_wider_window(engine, monkeypatch):
    monkeypatch.setattr(type(engine.vectors.inner), "narrows_conditions", False)
    found = await engine.recall("s", QUERY, limit=5, conditions=PUBLISHED_ONLY)
    assert [item.source for item in found.items] == ["release/plan"]
    narrowing = found.narrowing
    assert narrowing.vector_lane == "postfiltered" and narrowing.vector_window == 5 * LANE_DEPTH * UNFILTERED_DEPTH
    assert narrowing.postfiltered_out == DRAFTS and narrowing.window_exhausted is False


async def test_a_full_window_after_postfiltering_is_disclosed(engine, monkeypatch):
    """The caller's candidate limit caps the window at twenty; the filter
    removes all twenty. The answer is empty and the result says the
    window was full when that happened, so empty is not read as none."""
    monkeypatch.setattr(type(engine.vectors.inner), "narrows_conditions", False)
    found = await engine.recall("s", QUERY, limit=5, conditions=PUBLISHED_ONLY, candidate_limit=20)
    assert found.items == []
    narrowing = found.narrowing
    assert narrowing.vector_window == 20 and narrowing.postfiltered_out == 20 and narrowing.window_exhausted is True


async def test_a_source_prefix_widens_the_vector_window_as_well(engine):
    """Kind, source and date bounds live on the episode, which no vector
    row carries; the vector lane is post-filtered for them, and so it
    looks deeper."""
    found = await engine.recall("s", QUERY, limit=5, source_prefix="release/")
    assert [item.source for item in found.items] == ["release/plan"]
    narrowing = found.narrowing
    assert narrowing.vector_lane == "postfiltered" and narrowing.vector_window == 5 * LANE_DEPTH * UNFILTERED_DEPTH


async def test_no_narrowing_means_no_narrowing_field(engine):
    found = await engine.recall("s", QUERY, limit=5)
    assert found.narrowing is None and len(found.items) == 5


async def test_the_evidence_event_carries_the_same_window(tmp_path):
    from scone_memory.observability.events import InMemoryEventLog

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), Planted(), events=InMemoryEventLog()).open()
    try:
        for n in range(DRAFTS):
            await memory.remember("s", f"draft note {n} about weather", metadata={"status": "draft"}, source=f"drafts/{n}")
        await memory.remember("s", PUBLISHED, metadata={"status": "published"}, source="release/plan")
        found = await memory.recall("s", QUERY, limit=5, conditions=PUBLISHED_ONLY)
        events = await memory.events.query("s", kind="recall")
        payload = events[0].payload
        assert payload["narrowing"] == found.narrowing.model_dump() and payload["narrowing"]["window_exhausted"] is False
        assert payload["narrow"]["conditions"] == PUBLISHED_ONLY, "the request's own record is untouched"
    finally:
        await memory.close()


CORPUS = [
    {"status": "draft", "priority": "3", "owner": "ann"},
    {"status": "published", "priority": "5"},
    {"status": "published", "owner": "bob"},
    {"priority": "soon"},
    {"status": "archived", "priority": "1", "owner": "ann"},
    {},
]
FILTERS = [
    {"field": "status", "is": "published"},
    {"field": "status", "is": "published", "not": True},
    {"field": "priority", "at_most": 3},
    {"field": "owner", "present": True},
    {"field": "status", "in": ["draft", "archived"]},
    {"all": [{"field": "status", "is": "published"}, {"field": "owner", "present": True}]},
]


@pytest.mark.parametrize("spec", FILTERS, ids=range(len(FILTERS)))
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_the_vector_lane_returns_exactly_what_the_filter_keeps(tmp_path, backend, spec):
    """The same invariant the text lane holds: never lose a row the
    filter would keep. In the vector lane the filter is applied to the
    same metadata the row carries, so it is exact, not merely generous."""
    index = SqliteVectorIndex(tmp_path / "v.db") if backend == "sqlite" else InMemoryVectorIndex()
    try:
        await index.ensure(2)
        await index.upsert([VectorPoint(chunk_id=i, space="s", episode_id=i, vector=[1.0, 0.0], created_at="2026-01-01T00:00:00Z",
                                        tags=(), metadata=meta) for i, meta in enumerate(CORPUS)])
        condition = parse_filter(spec)
        kept = {i for i, meta in enumerate(CORPUS) if condition.matches(meta)}
        got = {chunk_id for chunk_id, _ in await index.search("s", [1.0, 0.0], 100, conditions=condition)}
        assert got == kept, spec
    finally:
        if hasattr(index, "close"):
            await index.close()
