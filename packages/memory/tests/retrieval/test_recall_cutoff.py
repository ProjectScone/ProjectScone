"""A recall that stops where the vector scores fall away.

Three notes answer the question and the rest share none of its words, so
their cosines sit far below. With ``cutoff`` the answer holds the three
and says where it cut; without it, the limit is filled with the rest.
A list with no drop is left alone, and settings the cut was not measured
in are refused.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.cli import build_parser, run

pytestmark = pytest.mark.asyncio

MATCHING = ["The crane survey found rust on the jib.",
            "The crane survey found grease missing on the jib.",
            "The crane survey found the jib bolts loose."]
UNRELATED = ["Lunch was soup and bread.", "The harbour board meets on Tuesday.",
             "Invoices go to the second floor.", "Parking moved to the east lot."]
QUESTION = "crane survey jib"


async def engine_with(notes, events=None):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=events).open()
    for note in notes:
        await engine.remember("default", note)
    return engine


async def test_the_answer_stops_at_the_drop_and_says_where():
    engine = await engine_with(MATCHING + UNRELATED)
    try:
        whole = await engine.recall("default", QUESTION, limit=5, lanes=["vector"])
        cut = await engine.recall("default", QUESTION, limit=5, lanes=["vector"], cutoff=True)
    finally:
        await engine.close()
    assert len(whole.items) == 5 and whole.cutoff is None
    assert {item.text for item in cut.items} == set(MATCHING)
    assert cut.cutoff is not None and cut.cutoff.rule == "gap"
    assert cut.cutoff.keep == 3 and cut.cutoff.dropped == 2
    assert all(item.similarity >= cut.cutoff.threshold for item in cut.items)


async def test_a_list_with_no_drop_is_left_as_it_was():
    """Each note shares the same three words with the question, so every score is equal."""
    engine = await engine_with(["crane survey jib alpha.", "crane survey jib bravo.", "crane survey jib charlie.",
                                "crane survey jib delta."])
    try:
        whole = await engine.recall("default", QUESTION, limit=3, lanes=["vector"])
        cut = await engine.recall("default", QUESTION, limit=3, lanes=["vector"], cutoff=True)
    finally:
        await engine.close()
    assert [item.chunk_id for item in cut.items] == [item.chunk_id for item in whole.items]
    assert cut.cutoff.rule == "no-boundary" and cut.cutoff.dropped == 0 and len(cut.items) == 3


async def test_the_cut_is_recorded_with_the_recall():
    from scone_memory import InMemoryEventLog

    events = InMemoryEventLog()
    engine = await engine_with(MATCHING + UNRELATED, events=events)
    try:
        cut = await engine.recall("default", QUESTION, limit=5, lanes=["vector"], cutoff=True)
        [recorded] = await events.query("default", kind="recall")
    finally:
        await engine.close()
    assert recorded.payload["cutoff"]["dropped"] == cut.cutoff.dropped == 2


@pytest.mark.parametrize("options, says", [
    ({"lanes": ["vector", "text"]}, "lanes"),
    ({"lanes": ["text"]}, "lanes"),
    ({"lanes": ["vector"], "cutoff": "yes"}, "boolean"),
])
async def test_settings_it_was_not_measured_in_are_refused(options, says):
    engine = await engine_with(MATCHING)
    try:
        with pytest.raises(InvalidInput, match=says):
            await engine.recall("default", QUESTION, **{"cutoff": True, **options})
    finally:
        await engine.close()


async def test_a_reranker_is_refused_beside_it():
    from scone_memory.retrieval.reranking import RerankScore

    class Ranker:
        async def rerank(self, query, candidates):
            return [RerankScore(candidate.chunk_id, float(index)) for index, candidate in enumerate(candidates)]

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), reranker=Ranker()).open()
    try:
        await engine.remember("default", MATCHING[0])
        with pytest.raises(InvalidInput, match="rerank"):
            await engine.recall("default", QUESTION, lanes=["vector"], cutoff=True)
        allowed = await engine.recall("default", QUESTION, lanes=["vector"], cutoff=True, rerank=False)
    finally:
        await engine.close()
    assert allowed.cutoff is not None


async def test_the_cut_is_asked_for_over_http_and_the_command_line():
    engine = await engine_with(MATCHING + UNRELATED)
    with TestClient(create_app(engine, {"key-a": "default"})) as client:
        headers = {"authorization": "Bearer key-a"}
        made = client.get("/v1/recall", params={"q": QUESTION, "limit": 5, "lanes": "vector", "cutoff": True},
                          headers=headers)
        plain = client.get("/v1/recall", params={"q": QUESTION, "limit": 5, "lanes": "vector"}, headers=headers)
        both_lanes = client.get("/v1/recall", params={"q": QUESTION, "cutoff": True}, headers=headers)
        features = client.get("/v1/capabilities", headers=headers).json()["features"]
    out = io.StringIO()
    code = await run(build_parser().parse_args(["recall", QUESTION, "--limit", "5", "--lanes", "vector",
                                                "--cutoff", "--json"]), engine, io.StringIO(""), out)
    await engine.close()
    assert made.status_code == 200 and made.json()["cutoff"]["dropped"] == 2
    assert {item["text"] for item in made.json()["items"]} == set(MATCHING)
    assert plain.status_code == 200 and "cutoff" not in plain.json() and len(plain.json()["items"]) == 5
    assert both_lanes.status_code in (400, 422) and features["recall.cutoff"] is True
    assert code == 0 and json.loads(out.getvalue())["cutoff"]["keep"] == 3


class Unreadable(InMemoryVectorIndex):
    """An index that hands back one score that is not a number."""

    async def search(self, *args, **kwargs):
        found = await super().search(*args, **kwargs)
        return found[:-1] + [(found[-1][0], float("nan"))]


async def test_a_score_that_is_not_a_number_cuts_nothing_and_says_so():
    engine = await MemoryEngine(InMemoryDocumentStore(), Unreadable(), HashEmbedder()).open()
    try:
        for note in MATCHING + UNRELATED:
            await engine.remember("default", note)
        cut = await engine.recall("default", QUESTION, limit=5, lanes=["vector"], cutoff=True)
    finally:
        await engine.close()
    assert cut.cutoff.rule == "unreadable" and cut.cutoff.dropped == 0 and len(cut.items) == 5
