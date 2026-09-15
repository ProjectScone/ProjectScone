"""A recall that does not spend its limit on one passage said three ways.

Fusion ranks by relevance alone, so three near-copies of one note can
take three of five places. With ``diversity`` the places are filled one
at a time, each by the candidate whose relevance, less its likeness to
what was already chosen, is highest: the reference's maximal marginal
relevance, over the fused candidates rather than one vector lane. The
likeness is cosine between the candidates' own vectors, read from the
index where it can give them and embedded once where it cannot, and the
answer says which, and how many places changed hands.
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

COPIES = ["The crane survey found rust on the jib and grease missing from the slew ring.",
          "The crane survey found rust on the jib and grease missing from the slew ring today.",
          "The crane survey found rust on the jib and the grease missing from the slew ring."]
OTHERS = ["The crane survey was booked with the harbour board for the third of May.",
          "Nobody signed the crane survey handover, which caused an argument in July."]
QUESTION = "crane survey rust jib grease slew ring"


class Counting(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return await super().embed(texts)


class Blind(InMemoryVectorIndex):
    """An index that searches but cannot hand back the vectors it holds."""

    vectors_of = None


async def stored(index=None):
    embedder = Counting()
    engine = await MemoryEngine(InMemoryDocumentStore(), index or InMemoryVectorIndex(), embedder).open()
    for note in COPIES + OTHERS:
        await engine.remember("default", note)
    embedder.calls = 0
    return engine, embedder


def copies_in(result) -> int:
    return sum(item.text in COPIES for item in result.items)


async def test_near_copies_give_places_to_other_relevant_passages():
    engine, _ = await stored()
    try:
        plain = await engine.recall("default", QUESTION, limit=3)
        varied = await engine.recall("default", QUESTION, limit=3, diversity=0.7)
    finally:
        await engine.close()
    assert copies_in(plain) == 3, "the fixture must let the copies take every place by relevance alone"
    assert copies_in(varied) == 1 and len(varied.items) == 3
    # Relevance still leads: one of the copies, the most relevant group, is first. Which one can differ,
    # because restatement demotion reorders copies among themselves after the places are filled.
    assert varied.items[0].text in COPIES
    trace = varied.diversity
    assert trace.weight == 0.7 and trace.vectors == "index" and trace.replaced == 2 and trace.candidates >= 5


async def test_no_diversity_is_the_ranking_as_it_was():
    engine, _ = await stored()
    try:
        plain = await engine.recall("default", QUESTION, limit=4)
        zero = await engine.recall("default", QUESTION, limit=4, diversity=0.0)
    finally:
        await engine.close()
    assert [item.chunk_id for item in zero.items] == [item.chunk_id for item in plain.items]
    assert zero.diversity.replaced == 0 and plain.diversity is None


async def test_an_index_that_cannot_give_vectors_back_has_the_candidates_embedded_once_and_says_so():
    engine, embedder = await stored(Blind())
    try:
        varied = await engine.recall("default", QUESTION, limit=3, diversity=0.7)
    finally:
        await engine.close()
    assert copies_in(varied) == 1
    assert varied.diversity.vectors == "embedded" and embedder.calls == 2, "one for the query, one for the candidates"


async def test_candidates_past_the_budget_are_not_diversified_and_are_counted():
    import scone_memory.retrieval.recall as module

    engine, _ = await stored()
    original = module.MAX_DIVERSIFIED
    module.MAX_DIVERSIFIED = 3
    try:
        varied = await engine.recall("default", QUESTION, limit=5, diversity=0.7)
    finally:
        module.MAX_DIVERSIFIED = original
        await engine.close()
    assert varied.diversity.candidates == 3 and varied.diversity.not_diversified >= 2
    assert "budget" in varied.diversity.why


@pytest.mark.parametrize("weight", [-0.1, 1.1, True, "half"])
async def test_a_weight_outside_zero_to_one_is_refused(weight):
    engine, _ = await stored()
    try:
        with pytest.raises(InvalidInput):
            await engine.recall("default", QUESTION, diversity=weight)
    finally:
        await engine.close()


async def test_diversity_is_asked_for_over_http_and_the_command_line():
    engine, _ = await stored()
    with TestClient(create_app(engine, {"key-a": "default"})) as client:
        headers = {"authorization": "Bearer key-a"}
        made = client.get("/v1/recall", params={"q": QUESTION, "limit": 3, "diversity": 0.7}, headers=headers)
        bad = client.get("/v1/recall", params={"q": QUESTION, "diversity": 2}, headers=headers)
        features = client.get("/v1/capabilities", headers=headers).json()["features"]
    out = io.StringIO()
    code = await run(build_parser().parse_args(["recall", QUESTION, "--limit", "3", "--diversity", "0.7", "--json"]),
                     engine, io.StringIO(""), out)
    await engine.close()
    assert made.status_code == 200 and made.json()["diversity"]["replaced"] == 2
    assert sum(item["text"] in COPIES for item in made.json()["items"]) == 1
    assert bad.status_code == 422 and features["recall.diversity"] is True
    assert code == 0 and json.loads(out.getvalue())["diversity"]["vectors"] == "index"


class Partial(InMemoryVectorIndex):
    """An index that gives back some vectors and not others."""

    async def vectors_of(self, space, chunk_ids):
        found = await super().vectors_of(space, chunk_ids)
        return dict(list(found.items())[1:])


class Shrunk(InMemoryVectorIndex):
    """Vectors of the copies handed back a tenth as long: a cosine must not notice."""

    async def vectors_of(self, space, chunk_ids):
        found = await super().vectors_of(space, chunk_ids)
        return {cid: [x * (0.1 if cid <= len(COPIES) else 1.0) for x in vector] for cid, vector in found.items()}


async def test_an_index_that_gives_back_only_some_vectors_has_every_candidate_embedded():
    engine, embedder = await stored(Partial())
    try:
        varied = await engine.recall("default", QUESTION, limit=3, diversity=0.7)
    finally:
        await engine.close()
    assert varied.diversity.vectors == "embedded" and embedder.calls == 2 and copies_in(varied) == 1


async def test_likeness_is_a_cosine_whatever_the_length_of_the_vectors():
    engine, _ = await stored(Shrunk())
    try:
        varied = await engine.recall("default", QUESTION, limit=3, diversity=0.7)
    finally:
        await engine.close()
    assert copies_in(varied) == 1


async def test_vectors_are_read_only_from_the_space_asked_about():
    index = InMemoryVectorIndex()
    engine = await MemoryEngine(InMemoryDocumentStore(), index, HashEmbedder()).open()
    try:
        added = await engine.remember("alpha", COPIES[0])
        [chunk] = await engine.documents.chunks_of("alpha", added.episode_id)
        assert set(await index.vectors_of("alpha", [chunk.chunk_id])) == {chunk.chunk_id}
        assert await index.vectors_of("beta", [chunk.chunk_id]) == {}
    finally:
        await engine.close()


async def test_diversity_beside_a_reranker_is_refused_because_the_reranker_would_undo_it():
    from scone_memory.retrieval.reranking import RerankScore

    class Ranker:
        async def rerank(self, query, candidates):
            return [RerankScore(candidate.chunk_id, float(index)) for index, candidate in enumerate(candidates)]

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), reranker=Ranker()).open()
    try:
        await engine.remember("default", COPIES[0])
        with pytest.raises(InvalidInput, match="rerank"):
            await engine.recall("default", QUESTION, diversity=0.5)
        allowed = await engine.recall("default", QUESTION, diversity=0.5, rerank=False)
    finally:
        await engine.close()
    assert allowed.diversity is not None


async def test_the_sqlite_index_gives_back_its_vectors_by_space(tmp_path):
    from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex

    path = tmp_path / "memory.db"
    index = SqliteVectorIndex(path)
    engine = await MemoryEngine(SqliteDocumentStore(path), index, HashEmbedder()).open()
    try:
        for note in COPIES + OTHERS:
            await engine.remember("default", note)
        varied = await engine.recall("default", QUESTION, limit=3, diversity=0.7)
        ids = [item.chunk_id for item in varied.items]
        read = await index.vectors_of("default", ids)
        elsewhere = await index.vectors_of("other", ids)
        [query] = await HashEmbedder().embed([COPIES[0]])
    finally:
        await engine.close()
    assert varied.diversity.vectors == "index" and copies_in(varied) == 1
    assert set(read) == set(ids) and elsewhere == {}
    assert all(len(vector) == len(query) for vector in read.values())
