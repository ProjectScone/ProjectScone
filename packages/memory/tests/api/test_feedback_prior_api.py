"""The feedback prior over HTTP and on the command line: what moved the order, and where its bounds bit."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.retrieval import feedback_prior, lessons
from scone_memory.testing import Clock

QUERY = "harbour crane survey"


async def weighted(weight: float) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2026-05-01T00:00:00.000Z"), events=InMemoryEventLog(),
                                feedback_weight=weight).open()
    for passage in ("The harbour crane survey is booked for May.", "The harbour crane was repainted in May."):
        await engine.remember("alpha", passage)
    return engine


async def judge_second(engine: MemoryEngine) -> int:
    second = (await engine.recall("alpha", QUERY)).items[1].chunk_id
    for asked in (QUERY, QUERY.upper()):
        shown = await engine.recall("alpha", asked)
        assert shown.event_id is not None
        await engine.feedback("alpha", shown.event_id, second, True)
    return second


@pytest.mark.parametrize("weight", [0.0, 0.0002], ids=["off", "on"])
async def test_an_http_recall_carries_what_feedback_added_when_the_weight_is_set(weight):
    engine = await weighted(weight)
    try:
        second = await judge_second(engine)
        with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
            answered = client.get("/v1/recall", params={"q": QUERY}, headers={"authorization": "Bearer key-a"}).json()
    finally:
        await engine.close()
    if weight == 0.0:
        assert "feedback_prior" not in answered
        return
    record = answered["feedback_prior"]
    assert record["boosted"] == 1 and record["events_cut"] is False and record["capped"] == 0
    assert record["returned_terms"] == {str(second): pytest.approx(0.0004)}


async def test_the_command_line_says_when_the_prior_read_was_cut_or_its_term_was(monkeypatch):
    from scone_memory.runtime.cli import build_parser, run

    engine = await weighted(0.0002)
    try:
        await judge_second(engine)

        async def said() -> str:
            out = io.StringIO()
            assert await run(build_parser().parse_args(["--space", "alpha", "recall", QUERY]), engine,
                             io.StringIO(""), out) == 0
            return out.getvalue()

        whole = await said()
        monkeypatch.setattr(feedback_prior, "MAX_FEEDBACK_BOOST", 0.0001)
        capped = await said()
        monkeypatch.setattr(lessons, "MAX_FEEDBACK_EVENTS", 1)
        cut = await said()
    finally:
        await engine.close()
    assert "feedback" not in whole
    assert "cut at its bound of 0.0001 for 1 candidate(s)" in capped and "newest" not in capped
    assert "feedback was read from the newest 1 judgement(s) only" in cut
