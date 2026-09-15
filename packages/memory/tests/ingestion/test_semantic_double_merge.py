"""A second pass over a semantic cut: chunks on either side of a cut that
turn out to be about the same thing are joined again, within the target.

`llama_index`'s double-merging splitter groups sentences, then merges
adjacent groups whose similarity passes a merge threshold. Ours already
has the first half -- sentences packed up to the target and cut at a
valley in similarity -- and the valley test is *relative*: it cuts where
a dip stands out from the rest of that text, so a text that is all about
one subject can still be cut at its deepest dip. The merge threshold is
the absolute check the valley test deliberately does not make.

The texts are constructed so the hash embedder, whose cosine tracks word
overlap, gives each case its shape: `FIRES` is one subject with a cut
the first pass invents, `ONE + OTHER` is a real change of subject.
"""

from __future__ import annotations

import math

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.semantic_chunks import Boundary, semantic_cut, semantic_spans

pytestmark = pytest.mark.asyncio

ONE = ("The cat sat on the mat. The cat was grey and the mat was red. "
       "The cat had sat on that mat every morning for a year. ")
OTHER = ("Quarterly revenue rose eleven percent. Revenue in the northern region "
         "rose fastest of all. Revenue guidance for the next quarter was raised. ")
SAME = ("The cat sat on the mat. The cat was grey. The cat liked the mat. The mat was red. "
        "The cat slept on the mat. ")
AGAIN = "The cat sat on the mat again. The cat was still grey. The grey cat liked the red mat. "
#: One subject throughout, with one out-of-place sentence: the first pass
#: cuts it in two (at 165 of 221 characters), and the halves score 0.699.
FIRES = SAME + "A dog barked once outside. " + AGAIN


def texts_of(text, spans):
    return [text[span.start:span.end] for span in spans]


async def test_a_cut_between_chunks_about_the_same_thing_is_merged():
    unmerged = await semantic_spans(FIRES, HashEmbedder(), target=2000)
    assert [(s.start, s.end) for s in unmerged] == [(0, 165), (165, 221)], "the fixture lost its false cut"
    cut = await semantic_cut(FIRES, HashEmbedder(), target=2000, merge_threshold=0.5)
    assert cut.spans == [Boundary(0, 221, False)]
    assert cut.record() == {"merge_threshold": 0.5, "groups": 2, "merges": 1,
                            "stopped_by_similarity": 0, "stopped_by_size": 0}


async def test_a_real_change_of_subject_is_not_merged_and_the_receipt_says_why():
    text = ONE + OTHER
    cut = await semantic_cut(text, HashEmbedder(), target=400, merge_threshold=0.5)
    assert cut.spans == await semantic_spans(text, HashEmbedder(), target=400)
    assert len(cut.spans) == 2 and cut.spans[0].subject_changed
    assert cut.record() == {"merge_threshold": 0.5, "groups": 2, "merges": 0,
                            "stopped_by_similarity": 1, "stopped_by_size": 0}


async def test_a_threshold_above_the_similarity_keeps_the_cut():
    cut = await semantic_cut(FIRES, HashEmbedder(), target=2000, merge_threshold=0.7)
    assert cut.spans == await semantic_spans(FIRES, HashEmbedder(), target=2000)
    assert (cut.merges, cut.stopped_by_similarity, cut.stopped_by_size) == (0, 1, 0)


async def test_the_target_bounds_a_merge_and_the_receipt_says_it_bit():
    """Alike enough to merge, too long together: the bound decides, and
    says so, rather than the receipt reading as if the text was unalike."""
    cut = await semantic_cut(FIRES, HashEmbedder(), target=200, merge_threshold=0.5)
    assert cut.spans == await semantic_spans(FIRES, HashEmbedder(), target=200)
    assert cut.record() == {"merge_threshold": 0.5, "groups": 2, "merges": 0,
                            "stopped_by_similarity": 0, "stopped_by_size": 1}
    assert all(span.end - span.start <= 200 for span in cut.spans)


async def test_size_and_similarity_are_counted_apart_in_one_text():
    """At target 120 the first pass makes three chunks: 0.517 alike but
    165 characters together, then 0.299 alike and 113 together."""
    unmerged = await semantic_spans(FIRES, HashEmbedder(), target=120)
    assert [(s.start, s.end) for s in unmerged] == [(0, 108), (108, 165), (165, 221)]
    strict = await semantic_cut(FIRES, HashEmbedder(), target=120, merge_threshold=0.5)
    assert strict.spans == unmerged
    assert (strict.groups, strict.merges, strict.stopped_by_similarity, strict.stopped_by_size) == (3, 0, 1, 1)
    loose = await semantic_cut(FIRES, HashEmbedder(), target=120, merge_threshold=0.25)
    assert [(s.start, s.end) for s in loose.spans] == [(0, 108), (108, 221)]
    assert (loose.groups, loose.merges, loose.stopped_by_similarity, loose.stopped_by_size) == (3, 1, 0, 1)
    assert all(span.end - span.start <= 120 for span in loose.spans)


async def test_a_sentence_cut_only_by_size_is_counted_as_the_bound_biting():
    """One sentence longer than the target is cut by size and nothing
    else; its pieces are the same sentence, so what kept them apart is the
    bound, and the receipt must not say they were unalike."""
    text = "the harbour crane lifted crates " * 60
    unmerged = await semantic_spans(text, HashEmbedder(), target=300)
    assert len(unmerged) > 2
    cut = await semantic_cut(text, HashEmbedder(), target=300, merge_threshold=0.5)
    assert cut.spans == unmerged
    assert cut.record() == {"merge_threshold": 0.5, "groups": len(unmerged), "merges": 0,
                            "stopped_by_similarity": 0, "stopped_by_size": len(unmerged) - 1}


async def test_without_a_threshold_the_cut_is_the_first_pass_and_has_no_receipt():
    cut = await semantic_cut(FIRES, HashEmbedder(), target=2000)
    assert cut.spans == await semantic_spans(FIRES, HashEmbedder(), target=2000)
    assert cut.record() is None


async def test_nothing_to_cut_is_an_empty_cut():
    cut = await semantic_cut("   ", HashEmbedder(), target=400, merge_threshold=0.5)
    assert cut.spans == [] and cut.record() == {"merge_threshold": 0.5, "groups": 0, "merges": 0,
                                                "stopped_by_similarity": 0, "stopped_by_size": 0}


class _Counting:
    def __init__(self):
        self.inner, self.calls = HashEmbedder(), 0
        self.id, self.dim = self.inner.id, self.inner.dim

    async def embed(self, texts):
        self.calls += 1
        return await self.inner.embed(texts)


async def test_merging_asks_the_embedder_nothing_more_and_is_deterministic():
    """The merge compares the sentence vectors the first pass already has;
    re-embedding each joined text, as the reference does, would make the
    cost grow with every merge and the result depend on a second call."""
    embedder = _Counting()
    first = await semantic_cut(FIRES, embedder, target=120, merge_threshold=0.25)
    assert embedder.calls == 1
    second = await semantic_cut(FIRES, embedder, target=120, merge_threshold=0.25)
    assert (first.spans, first.record()) == (second.spans, second.record())


def _unit(degrees):
    return [math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]


def _groups(count):
    return [(Boundary(10 * index, 10 * index + 10, False), index, index) for index in range(count)]


@pytest.mark.parametrize("angles, merged", [
    # 0 and 50 degrees merge (cos 0.64); their centre, at 25, is 35 from
    # the third (cos 0.82). The first chunk alone is 60 away (cos 0.5).
    ((0, 50, 60), [(0, 30)]),
    # The third, at 100 degrees, is 50 from the second (cos 0.64) but 75
    # from the merged centre (cos 0.26): it is compared with the whole.
    ((0, 50, 100), [(0, 20), (20, 30)]),
])
async def test_a_merged_chunk_is_compared_by_all_of_its_sentences(angles, merged):
    from scone_memory.ingestion.semantic_chunks import _merged

    spans, counts = _merged(_groups(3), [_unit(a) for a in angles], 0.6, 100)
    assert [(s.start, s.end) for s in spans] == merged
    assert counts == (3 - len(merged), len(merged) - 1, 0)


async def test_a_similarity_exactly_at_the_threshold_merges():
    from scone_memory.ingestion.semantic_chunks import _merged

    spans, counts = _merged(_groups(2), [[1.0, 0.0], [1.0, 0.0]], 1.0, 100)
    assert [(s.start, s.end) for s in spans] == [(0, 20)] and counts == (1, 0, 0)


async def test_a_merge_that_lands_exactly_on_the_target_is_allowed():
    from scone_memory.ingestion.semantic_chunks import _merged

    spans, counts = _merged(_groups(2), [[1.0, 0.0], [1.0, 0.0]], 0.5, 20)
    assert [(s.start, s.end) for s in spans] == [(0, 20)] and counts == (1, 0, 0)
    spans, counts = _merged(_groups(2), [[1.0, 0.0], [1.0, 0.0]], 0.5, 19)
    assert len(spans) == 2 and counts == (0, 0, 1)


async def test_the_merged_chunk_keeps_whether_its_own_end_was_a_change_of_subject():
    from scone_memory.ingestion.semantic_chunks import _merged

    groups = [(Boundary(0, 10, True), 0, 0), (Boundary(10, 20, True), 1, 1), (Boundary(20, 30, False), 2, 2)]
    spans, _ = _merged(groups, [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], 0.5, 100)
    assert spans == [Boundary(0, 20, True), Boundary(20, 30, False)]


@pytest.mark.parametrize("threshold", [0, 0.0, -0.5, 1.01, float("nan"), float("inf"), True, "0.5"])
async def test_a_threshold_that_is_not_a_similarity_is_refused(threshold):
    from scone_memory.ingestion.semantic_chunks import merge_refused

    assert merge_refused(threshold) is not None
    with pytest.raises(ValueError, match="semantic_merge_threshold"):
        await semantic_cut(FIRES, HashEmbedder(), target=400, merge_threshold=threshold)
    with pytest.raises(InvalidInput, match="semantic_merge_threshold"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), semantic_merge_threshold=threshold)


async def test_a_threshold_in_range_is_accepted():
    from scone_memory.ingestion.semantic_chunks import merge_refused

    assert merge_refused(1) is None and merge_refused(0.01) is None and merge_refused(1.0) is None


@pytest.fixture
async def merging():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                chunk_target=2000, semantic_merge_threshold=0.5).open()
    yield memory
    await memory.close()


async def stored_texts(engine, episode_id):
    return [chunk.text for chunk in await engine.documents.chunks_of("default", episode_id)]


async def test_a_record_that_asks_for_semantic_gets_the_second_pass_and_its_receipt(merging):
    added = await merging.remember("default", FIRES, source="notes.txt", chunking="semantic")
    assert added.chunking == "semantic" and added.chunks == 1
    assert added.structure == {"merge_threshold": 0.5, "groups": 2, "merges": 1,
                               "stopped_by_similarity": 0, "stopped_by_size": 0}
    assert await stored_texts(merging, added.episode_id) == [FIRES]


async def test_a_record_that_does_not_ask_for_semantic_is_not_touched_by_the_threshold(merging):
    added = await merging.remember("default", FIRES, source="notes.txt")
    assert added.chunking == "length" and added.structure is None


async def test_an_engine_without_a_threshold_cuts_semantic_records_as_before():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                chunk_target=2000).open()
    try:
        assert engine.semantic_merge_threshold is None
        added = await engine.remember("default", FIRES, source="notes.txt", chunking="semantic")
        assert added.chunks == 2 and added.structure is None
    finally:
        await engine.close()


async def test_a_semantic_engine_merges_every_record_by_its_rule():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=2000,
                                semantic_aware=True, semantic_merge_threshold=0.5).open()
    try:
        added = await engine.remember("default", FIRES, source="notes.txt")
        assert added.chunking == "semantic" and added.chunks == 1 and added.structure["merges"] == 1
    finally:
        await engine.close()


async def test_recovery_after_a_crash_merges_as_the_uninterrupted_write_did(merging):
    """The crash shape: the episode row and its mark, no chunks. Recovery
    re-cuts from the stored content under the engine's threshold and must
    land on exactly the chunks an uninterrupted write stores."""
    from scone_memory.ingestion.batch import validated_record
    from scone_memory.ingestion.records import Record

    text = FIRES
    expected = texts_of(text, (await semantic_cut(text, HashEmbedder(), target=2000, merge_threshold=0.5)).spans)
    assert expected != texts_of(text, await semantic_spans(text, HashEmbedder(), target=2000)), \
        "the fixture must merge, or recovery without the threshold would pass too"
    new = validated_record("default", Record(text, source="notes.txt", chunking="semantic"), merging.clock())
    await merging.documents.mark_inflight("default", new.content_hash)
    episode = await merging.documents.insert_episode(new)
    report = await merging.recover()
    assert (report.completed, report.rechunked) == (1, 1), report
    assert await stored_texts(merging, episode.episode_id) == expected
    uninterrupted = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                       chunk_target=2000, semantic_merge_threshold=0.5).open()
    try:
        added = await uninterrupted.remember("default", text, source="notes.txt", chunking="semantic")
        assert await stored_texts(uninterrupted, added.episode_id) == expected
    finally:
        await uninterrupted.close()


class _Retuning:
    """An embedder that changes the engine's threshold while it is asked."""

    def __init__(self):
        self.inner = HashEmbedder()
        self.id, self.dim = self.inner.id, self.inner.dim
        self.engine = None

    async def embed(self, texts):
        if self.engine is not None:
            self.engine.semantic_merge_threshold = 0.9
        return await self.inner.embed(texts)


async def test_a_replacement_prepared_under_another_threshold_is_refused():
    embedder = _Retuning()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder, chunk_target=2000,
                                semantic_merge_threshold=0.5).open()
    try:
        from scone_memory.ingestion.records import Record

        await engine.replace("default", Record(ONE, source="notes.txt", dedup_key="k", chunking="semantic"))
        embedder.engine = engine
        with pytest.raises(InvalidInput, match="ingestion configuration changed"):
            await engine.replace("default", Record(FIRES, source="notes.txt", dedup_key="k", chunking="semantic"))
        assert (await engine.status("default")).episodes == 1
        assert (await engine.recall("default", "cat mat")).items[0].text == ONE, \
            "the old source must stand when preparation is refused"
    finally:
        await engine.close()


async def test_over_http_a_semantic_record_on_a_merging_engine_carries_the_counts(merging):
    from httpx import ASGITransport, AsyncClient

    from scone_memory.api import create_app

    async with AsyncClient(transport=ASGITransport(app=create_app(merging, {"k": "default"})),
                           base_url="http://fixture") as client:
        answer = await client.post("/v1/episodes", json={"content": FIRES, "source": "notes.txt", "chunking": "semantic"},
                                   headers={"authorization": "Bearer k"})
        assert answer.status_code == 200, answer.text
        assert answer.json()["chunking"] == "semantic" and answer.json()["chunks"] == 1
        assert answer.json()["structure"] == {"merge_threshold": 0.5, "groups": 2, "merges": 1,
                                              "stopped_by_similarity": 0, "stopped_by_size": 0}
