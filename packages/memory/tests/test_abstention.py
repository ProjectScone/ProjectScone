"""Research experiment 9: an abstention gate from retrieval confidence.

A recall carries the best cosine its vector lane saw. With a floor
configured, a recall whose best hit sits below it (or that found nothing)
is flagged low_confidence so the reader can say "I don't know" instead of
answering from weak evidence. The floor has no default: it comes from a
measured sweep (the bench prints one), never from a guess in the code.

The hash embedder's cosine tracks token overlap, so an identical query
scores 1.0 and a query with no shared token scores about 0: enough to
exercise both sides of a 0.5 floor deterministically on every backend.
"""

from __future__ import annotations

import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, InvalidInput, MemoryEngine
from scone_memory.runtime import cli
from scone_memory.runtime.config import Settings, build_engine

TEXT = "the deploy runbook lives in the ops wiki under release checklist"


async def test_without_a_floor_the_similarity_is_reported_and_nothing_is_judged(engine):
    await engine.remember("default", TEXT)
    exact = await engine.recall("default", TEXT)
    assert exact.top_similarity == pytest.approx(1.0, abs=1e-5)
    assert exact.low_confidence is None, "no floor, no judgement"
    unrelated = await engine.recall("default", "zebra quartz umbrella")
    assert unrelated.top_similarity is not None and unrelated.top_similarity < 0.2
    assert unrelated.low_confidence is None


async def test_a_floor_flags_weak_evidence_and_clears_strong_evidence(engine):
    engine.similarity_floor = 0.5
    await engine.remember("default", TEXT)
    strong = await engine.recall("default", TEXT)
    assert strong.low_confidence is False and strong.items, "clears the floor: the reader may answer"
    weak = await engine.recall("default", "zebra quartz umbrella")
    assert weak.low_confidence is True, "below the floor: the reader is told the evidence is weak"
    nothing = await engine.recall("other", "anything at all")
    assert nothing.low_confidence is True and nothing.top_similarity is None, "an empty space is the weakest evidence of all"
    events = await engine.events.query("default", kind="recall")
    judged = {e.payload["low_confidence"] for e in events}
    assert judged == {True, False} and all(e.payload["similarity_floor"] == 0.5 for e in events)
    assert all(isinstance(e.payload["top_similarity"], float) for e in events), "the signal is on the evidence log"


async def test_a_degraded_vector_lane_cannot_judge():
    class Broken(InMemoryVectorIndex):
        async def search(self, *a, **kw):
            raise RuntimeError("index offline")

    engine = await MemoryEngine(InMemoryDocumentStore(), Broken(), HashEmbedder(), events=InMemoryEventLog(), similarity_floor=0.5).open()
    await engine.remember("default", TEXT)
    result = await engine.recall("default", "runbook")
    assert result.items, "the text lane still answers"
    assert result.degraded and result.degraded[0].startswith("vectors:")
    assert (result.top_similarity, result.low_confidence) == (None, None), "no similarity was seen, so no verdict is invented"


def test_the_floor_must_be_a_cosine():
    with pytest.raises(InvalidInput, match="similarity_floor"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), similarity_floor=1.5)


async def test_the_floor_comes_from_the_environment(tmp_path):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "m.db"), "SCONE_SIMILARITY_FLOOR": "0.42"}
    settings = Settings.from_env(env)
    assert settings.similarity_floor == 0.42
    engine = await build_engine(settings)
    assert engine.similarity_floor == 0.42
    assert Settings.from_env({}).similarity_floor is None, "off unless configured"


def test_cli_surfaces_the_verdict(tmp_path):
    env = {"SCONE_DOCUMENTS": "sqlite", "SCONE_VECTORS": "sqlite", "SCONE_SQLITE_PATH": str(tmp_path / "cli.db"), "SCONE_SIMILARITY_FLOOR": "0.5"}

    def run(*argv, stdin=""):
        out = io.StringIO()
        code = cli.main(list(argv), env=env, stdin=io.StringIO(stdin), out=out)
        return code, out.getvalue()

    assert run("remember", stdin=TEXT)[0] == 0
    code, text = run("recall", "zebra quartz umbrella", "--json")
    payload = json.loads(text)
    assert code == 0 and payload["low_confidence"] is True and payload["top_similarity"] < 0.5
    code, text = run("recall", "zebra quartz umbrella")
    assert "low confidence: top similarity 0." in text and "floor 0.50" in text, text
    code, text = run("recall", TEXT)
    assert "low confidence" not in text



async def test_http_surfaces_the_verdict():
    from fastapi.testclient import TestClient

    from scone_memory.api import create_app

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), similarity_floor=0.5).open()
    await engine.remember("default", TEXT)
    with TestClient(create_app(engine, {"k": "default"})) as client:
        h = {"authorization": "Bearer k"}
        weak = client.get("/v1/recall", params={"q": "zebra quartz umbrella"}, headers=h).json()
        assert weak["low_confidence"] is True and isinstance(weak["top_similarity"], float) and weak["top_similarity"] < 0.5
        strong = client.get("/v1/recall", params={"q": TEXT}, headers=h).json()
        assert strong["low_confidence"] is False and strong["top_similarity"] == pytest.approx(1.0, abs=1e-5)
