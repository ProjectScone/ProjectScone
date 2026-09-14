"""Asking recall for one lane, and being told which ran.

Recall fuses a vector lane and a text lane. A caller comparing them, or
one whose question is an exact code or a name no embedding knows, may
want one alone; the reference offers a dense-only or sparse-only mode
per query. A lane not asked for is not run -- a text-only recall makes
no embedding call -- and is not reported as degraded, because nothing
failed. The result and the evidence event name the lanes that ran, and
a vector lane that did not run judges no confidence.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.cli import build_parser, run

pytestmark = pytest.mark.asyncio

NOTES = ["Invoice INV-20931 was paid by the harbour office on the third of May.",
         "The crane survey found rust on the jib and grease missing from the slew ring.",
         "Priya moved the launch to March because the audit ran late."]


class Counting(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return await super().embed(texts)


async def stored(**options):
    embedder = Counting()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder, events=InMemoryEventLog(),
                                **options).open()
    for note in NOTES:
        await engine.remember("default", note)
    embedder.calls = 0
    return engine, embedder


async def test_a_text_only_recall_makes_no_embedding_call_and_says_which_lane_ran():
    engine, embedder = await stored(similarity_floor=0.5)
    try:
        result = await engine.recall("default", "INV-20931", lanes=("text",))
        [event] = await engine.events.query("default", kind="recall")
    finally:
        await engine.close()
    assert embedder.calls == 0
    assert result.lanes == ["text"] and result.degraded == []
    assert result.items and all(set(item.lanes) == {"text"} for item in result.items)
    assert result.top_similarity is None and result.low_confidence is None, "a lane that did not run judges nothing"
    assert event.payload["lanes"] == ["text"]


async def test_a_vector_only_recall_does_not_search_text():
    engine, embedder = await stored()
    searched = 0
    original = engine.documents.search_text

    async def counted(*args, **kwargs):
        nonlocal searched
        searched += 1
        return await original(*args, **kwargs)

    engine.documents.search_text = counted
    try:
        result = await engine.recall("default", "rust on the crane jib", lanes=("vector",))
    finally:
        await engine.close()
    assert searched == 0 and embedder.calls == 1
    assert result.lanes == ["vector"] and result.items and all(set(item.lanes) == {"vector"} for item in result.items)


async def test_by_default_both_lanes_run_and_are_named_in_one_order():
    engine, _ = await stored()
    try:
        default = await engine.recall("default", "crane jib")
        reordered = await engine.recall("default", "crane jib", lanes=("text", "vector", "text"))
    finally:
        await engine.close()
    assert default.lanes == ["vector", "text"] and reordered.lanes == ["vector", "text"]
    assert [item.chunk_id for item in default.items] == [item.chunk_id for item in reordered.items]


async def test_the_only_lane_asked_for_failing_is_an_error_not_an_empty_answer():
    engine, _ = await stored()

    async def explode(*args, **kwargs):
        raise RuntimeError("the index is unwell")

    engine.documents.search_text = explode
    try:
        with pytest.raises(RuntimeError, match="text"):
            await engine.recall("default", "INV-20931", lanes=("text",))
        both = await engine.recall("default", "crane jib")
    finally:
        await engine.close()
    assert both.lanes == ["vector"] and any(note.startswith("text:") for note in both.degraded), \
        "with both asked, a failed lane is degraded and the result names the lane that ran"


@pytest.mark.parametrize("lanes", [(), ("entity",), ("vectors",), "text"])
async def test_lanes_that_do_not_exist_are_refused(lanes):
    engine, _ = await stored()
    try:
        with pytest.raises(InvalidInput):
            await engine.recall("default", "crane", lanes=lanes)
    finally:
        await engine.close()


async def test_lanes_are_asked_for_over_http_and_the_command_line():
    engine, embedder = await stored()
    with TestClient(create_app(engine, {"key-a": "default"})) as client:
        headers = {"authorization": "Bearer key-a"}
        text = client.get("/v1/recall", params={"q": "INV-20931", "lanes": "text"}, headers=headers)
        bogus = client.get("/v1/recall", params={"q": "INV-20931", "lanes": "text,telepathy"}, headers=headers)
        features = client.get("/v1/capabilities", headers=headers).json()["features"]
    calls_over_http = embedder.calls
    out = io.StringIO()
    code = await run(build_parser().parse_args(["recall", "INV-20931", "--lanes", "text", "--json"]), engine,
                     io.StringIO(""), out)
    await engine.close()
    assert text.status_code == 200 and text.json()["lanes"] == ["text"] and calls_over_http == 0
    assert bogus.status_code == 422 and "telepathy" in bogus.text
    assert features["recall.lanes"] is True
    assert code == 0 and json.loads(out.getvalue())["lanes"] == ["text"] and embedder.calls == 0
