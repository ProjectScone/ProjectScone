"""The best passages at both ends of a model's context, when asked; the
ranked order otherwise, and always recoverable."""
import json

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.core.models import RecallItem, RecallResult
from scone_memory.realtime.context import MemoryContext
from scone_memory.retrieval.reading_order import arranged, ends_first, ranked, validate_reading_order


@pytest.mark.parametrize("length", range(0, 8))
def test_ends_first_puts_odd_ranks_at_the_front_and_even_ranks_at_the_back_and_keeps_every_item(length):
    best_first = list(range(1, length + 1))
    order = ends_first(best_first)
    assert sorted(order) == best_first and len(order) == length
    if length >= 4:
        assert order[0] == 1 and order[-1] == 2 and order[1] == 3 and order[-2] == 4
    assert arranged(best_first, "ranked") == best_first and arranged(best_first, "ends") == order
    assert ranked(order, "ends") == best_first and ranked(best_first, "ranked") == best_first, "the arrangement is undone from its name"


def test_ends_first_is_a_fixed_arrangement():
    assert ends_first(["a", "b", "c", "d", "e"]) == ["a", "c", "e", "d", "b"]
    assert validate_reading_order("ends") == "ends"
    with pytest.raises(InvalidInput, match="reading_order must be one of ranked, ends"):
        validate_reading_order("shuffled")


async def test_a_context_renders_its_chosen_passages_ends_first_when_asked_and_says_so(engine, monkeypatch):
    added = [await engine.remember("alpha", f"Passage number {n} about the harbour lights", source=f"s{n}") for n in range(1, 6)]
    items = []
    for rank, outcome in enumerate(added, 1):
        chunks = await engine.documents.chunks_of("alpha", outcome.episode_id)
        episode = await engine.episode("alpha", outcome.episode_id)
        items.append(RecallItem(chunk_id=chunks[0].chunk_id, episode_id=outcome.episode_id, text=chunks[0].text,
                                source=episode.source, created_at=episode.created_at, score=1.0 / rank, metadata={}))

    async def recall(*args, **kwargs):
        return RecallResult(items=list(items))
    monkeypatch.setattr(engine, "recall", recall)
    messages = [{"role": "user", "content": "harbour lights?"}]
    ranked_block, ranked_receipt = await MemoryContext(engine, "alpha", "s", limit=5).prepare(messages)
    ends_block, ends_receipt = await MemoryContext(engine, "alpha", "s", limit=5, reading_order="ends").prepare(messages)
    assert ranked_receipt["reading_order"] == "ranked" and ends_receipt["reading_order"] == "ends"
    ranked_ids = [r["chunk_id"] for r in ranked_receipt["references"]]
    ends_ids = [r["chunk_id"] for r in ends_receipt["references"]]
    assert ranked_ids == [i.chunk_id for i in items]
    assert ends_ids == ends_first(ranked_ids), "the references follow the block"
    assert ranked(ends_ids, ends_receipt["reading_order"]) == ranked_ids, "and the ranked order comes back from the receipt's name"
    rendered = json.loads(ends_block[0]["content"].split("\n", 1)[1])["sources"]
    assert [s["chunk_id"] for s in rendered] == ends_ids and rendered[0]["text"].startswith("Passage number 1")
    assert rendered[-1]["text"].startswith("Passage number 2"), "the second best sits last, where a model still reads"
    assert ends_receipt["context_sha256"] != ranked_receipt["context_sha256"] and ends_receipt["omitted_count"] == 0
    with pytest.raises(ValueError, match="reading_order must be one of"):
        MemoryContext(engine, "alpha", "s", reading_order="middle")
