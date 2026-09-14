"""A widened passage cut back to the sentences that bear on the question.

A window of sentences either side of a hit holds the answer more often
than the hit alone, and holds a good deal besides. The reference's
optimizer embeds every sentence of a node and drops the ones least like
the question, including, when they score low, the sentences the node was
found by. Here the sentences a passage was retrieved for are never
dropped: only what widening added around them is scored, by the weight of
the question's words each one names or by an embedder, and at most a
share of it is kept. Every run of kept sentences is the episode's own
bytes; runs that are not adjacent are joined by a marker that says text
was left out, never spliced into one quote.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.core.models import RecallItem
from scone_memory.retrieval.compress import GAP, compress
from scone_memory.retrieval.window import widen

pytestmark = pytest.mark.asyncio

S1 = "The harbour board met on Monday to plan the summer."
S2 = "The crane survey was booked for the third of May."
S3 = "Dr. Okafor found rust on the jib."
S4 = "It needed grease and a new slew ring."
S5 = "The canteen menu changed in April."
S6 = "The yard repaired the jib before June."
S7 = "Parking passes are renewed each winter."
TEXT = " ".join((S1, S2, S3, S4, S5, S6, S7))
QUESTION = "What was wrong with the crane jib?"


async def _widened(content: str = TEXT, needle: str = S4, before: int = 3, after: int = 3):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    added = await engine.remember("default", content, source="crane.txt")
    start = content.encode().index(needle.encode())
    hit = RecallItem(chunk_id=7, episode_id=added.episode_id, text=needle, score=1.0, created_at="2026-09-14",
                     start=start, end=start + len(needle.encode()))
    opened = await widen(engine, "default", [hit], before=before, after=after, unit="sentences")
    await engine.close()
    return hit, opened.items[0]


async def test_the_sentences_around_a_hit_that_name_no_question_word_are_dropped():
    hit, passage = await _widened()
    made = await compress([passage], QUESTION, hits=[hit], keep=1.0)
    [one] = made.items
    assert one.text == f"{S2} {S3} {S4}{GAP}{S6}"
    assert made.sentences_dropped == 3 and made.compressed == 1


async def test_the_sentence_a_passage_was_found_by_is_kept_though_it_names_no_question_word():
    hit, passage = await _widened()
    made = await compress([passage], QUESTION, hits=[hit], keep=0.0)
    assert made.items[0].text == S4, "the hit scores lowest and is still the passage"
    assert made.sentences_dropped == 6


async def test_every_run_kept_is_the_episodes_own_bytes():
    hit, passage = await _widened()
    # Lines counted for the whole passage no longer describe a text with sentences left out.
    passage = passage.model_copy(update={"first_line": 1, "last_line": 1})
    made = await compress([passage], QUESTION, hits=[hit], keep=1.0)
    assert (made.items[0].first_line, made.items[0].last_line) == (None, None)
    raw = TEXT.encode()
    runs = made.runs[7]
    assert [raw[start:end].decode() for start, end in runs] == [f"{S2} {S3} {S4}", S6]
    assert made.items[0].text == GAP.join(raw[start:end].decode() for start, end in runs)
    assert (made.items[0].start, made.items[0].end) == (runs[0][0], runs[-1][1])


async def test_a_rarer_question_word_outweighs_a_commoner_one_when_the_share_is_short():
    # "crane" is named once in the passage, "jib" twice: with room for one sentence, crane's wins.
    hit, passage = await _widened()
    made = await compress([passage], QUESTION, hits=[hit], keep=1 / 6)
    assert made.items[0].text == f"{S2}{GAP}{S4}"


async def test_between_sentences_of_equal_weight_the_nearer_to_the_hit_is_kept():
    far = "The jib was painted in March."
    near = "The jib was greased in July."
    content = " ".join((far, S1, S5, S4, near, S7))
    hit, passage = await _widened(content)
    made = await compress([passage], "jib", hits=[hit], keep=1 / 5)
    assert made.items[0].text == f"{S4} {near}"


async def test_at_most_the_share_asked_for_is_kept():
    hit, passage = await _widened()
    made = await compress([passage], QUESTION, hits=[hit], keep=0.34)
    # Two of the six sentences around the hit: crane's, then the nearer of the two naming jib.
    assert made.items[0].text == f"{S2} {S3} {S4}"
    # 0.3 of six is 1.8 sentences: one is kept, never rounded up past the share.
    shorter = await compress([passage], QUESTION, hits=[hit], keep=0.3)
    assert shorter.items[0].text == f"{S2}{GAP}{S4}"


async def test_a_hit_ending_in_the_space_after_its_sentence_keeps_all_of_itself():
    hit, passage = await _widened(needle=S4 + " ")
    made = await compress([passage], QUESTION, hits=[hit], keep=0.0)
    assert made.items[0].text == S4 + " "


async def test_a_passage_that_was_not_widened_is_returned_as_it_was():
    hit, _ = await _widened()
    made = await compress([hit], QUESTION, hits=[hit], keep=0.0)
    assert made.items == (hit,) and made.compressed == 0 and made.sentences_dropped == 0


async def test_a_passage_with_no_hit_named_is_left_as_it_was_and_counted():
    hit, passage = await _widened()
    made = await compress([passage], QUESTION, hits=[], keep=0.0)
    assert made.items == (passage,) and made.unpinned == 1 and "no retrieved span" in made.why


async def test_the_record_counts_bytes_and_states_its_rules_as_unmeasured():
    hit, passage = await _widened()
    made = await compress([passage], QUESTION, hits=[hit], keep=1.0)
    record = made.record()
    assert record["bytes_before"] == len(passage.text.encode())
    assert record["bytes_after"] == len(made.items[0].text.encode())
    assert record["rules"] == {"scorer": "terms", "keep": 1.0, "gap": GAP, "measured": False}
    assert record["runs"] == {"7": [[start, end] for start, end in made.runs[7]]}
    assert record["sentences_dropped"] == 3 and record["compressed"] == 1


class Directions:
    """Vectors chosen per sentence, so which sentence is nearest the question is known."""

    id = "directions"
    dim = 2

    def __init__(self, near: set[str]):
        self.near = near
        self.calls = 0
        self.embedded = 0

    async def embed(self, texts):
        self.calls += 1
        self.embedded += len(texts)
        return [[1.0, 0.0] if text == QUESTION or text in self.near else [0.0, 1.0] for text in texts]


async def test_an_embedder_scores_the_sentences_in_one_call():
    hit, passage = await _widened()
    embedder = Directions(near={S5})
    made = await compress([passage], QUESTION, hits=[hit], keep=1 / 6, scorer="embedding", embedder=embedder)
    assert made.items[0].text == f"{S4} {S5}"
    assert embedder.calls == 1 and embedder.embedded == 7 and made.record()["sentences_embedded"] == 6


async def test_past_the_embedding_budget_a_passage_is_left_whole_and_counted():
    import scone_memory.retrieval.compress as module

    hit, passage = await _widened()
    other_hit, other = await _widened(needle=S3, before=1, after=1)
    other_hit = other_hit.model_copy(update={"chunk_id": 8})
    other = other.model_copy(update={"chunk_id": 8})
    original = module.MAX_EMBEDDED_SENTENCES
    module.MAX_EMBEDDED_SENTENCES = 7
    try:
        made = await compress([passage, other], QUESTION, hits=[hit, other_hit], keep=0.0, scorer="embedding",
                              embedder=Directions(near=set()))
    finally:
        module.MAX_EMBEDDED_SENTENCES = original
    assert made.items[0].text == S4 and made.items[1] == other
    assert made.unscored == 1 and "budget" in made.why and made.record()["unscored"] == 1


@pytest.mark.parametrize("arguments", [{"keep": -0.1}, {"keep": 1.5}, {"keep": True}, {"keep": "half"},
                                       {"scorer": "model"}, {"scorer": "embedding"}, {"query": "   "}])
async def test_bad_compressions_are_refused(arguments):
    hit, passage = await _widened()
    options = {"query": QUESTION, "keep": 0.5, **arguments}
    with pytest.raises(InvalidInput):
        await compress([passage], options.pop("query"), hits=[hit], **options)


async def test_a_question_of_only_common_words_compresses_nothing_and_says_why():
    hit, passage = await _widened()
    made = await compress([passage], "what is it?", hits=[hit], keep=0.0)
    assert made.items == (passage,) and made.compressed == 0
    assert "names no word" in made.why


async def test_a_hit_that_is_only_the_space_between_sentences_keeps_the_sentence_after_it():
    content = f"{S1}\n\n{S4} {S5}"
    hit, passage = await _widened(content, needle="\n\n", before=1, after=1)
    assert passage.text == content
    made = await compress([passage], "grease", hits=[hit], keep=0.0)
    assert made.items[0].text == f"\n\n{S4}"


@pytest.mark.parametrize("window", [{"before": 0, "after": 3}, {"before": 3, "after": 0}])
async def test_a_retrieved_span_outside_the_passage_pins_nothing_and_is_counted(window):
    # A span before the passage starts, or after it ends: neither can be placed in its text.
    _, passage = await _widened(**window)
    stray, _ = await _widened(needle=S2 if window["before"] == 0 else S7, before=0, after=0)
    made = await compress([passage], QUESTION, hits=[stray], keep=0.0)
    assert made.items == (passage,) and made.unpinned == 1


async def test_a_passage_whose_every_sentence_is_kept_is_returned_as_it_was_and_not_counted():
    hit, passage = await _widened()
    made = await compress([passage], "harbour crane jib canteen parking", hits=[hit], keep=1.0)
    assert made.items == (passage,) and made.compressed == 0 and made.runs == {}
