"""Long messages still reach memory, through a bounded, traceable query.

Recall accepts at most MAX_QUERY characters. A conversation turn that pastes
a draft and asks about it used to fail recall outright, so the model answered
with no memory at all. The conversation layer now searches with verbatim
excerpts of the message and records exactly which spans it kept.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.validation import MAX_QUERY
from scone_memory.retrieval.query_formulation import formulate_query

FILLER = "Here is my draft itinerary for the team offsite, please read it carefully. "


def excerpts(message: str, kept: tuple[tuple[int, int], ...]) -> str:
    return "\n".join(message[start:end] for start, end in kept)


def test_a_message_within_the_limit_is_searched_as_written() -> None:
    formulated = formulate_query("When does the Zanzibar ferry leave?")
    assert formulated.method == "verbatim"
    assert formulated.text == "When does the Zanzibar ferry leave?"


def test_a_long_message_keeps_its_question_verbatim_within_the_limit() -> None:
    message = FILLER * 40 + "When does the Zanzibar ferry leave?"
    formulated = formulate_query(message)
    assert formulated.method == "extract"
    assert len(formulated.text) <= MAX_QUERY
    assert "When does the Zanzibar ferry leave?" in formulated.text
    assert formulated.text == excerpts(message, formulated.kept)
    assert formulated.source_chars == len(message)


def test_kept_spans_are_ordered_and_never_overlap() -> None:
    message = "Opening line about the Lisbon office. " + FILLER * 30 + "Does the lease end in May? " + FILLER * 30 + "Thanks."
    kept = formulate_query(message).kept
    assert list(kept) == sorted(kept)
    assert all(end <= start for (_, end), (start, _) in zip(kept, kept[1:]))
    assert all(message[start:end].strip() == message[start:end] and end > start for start, end in kept)


def test_a_question_in_the_middle_is_kept() -> None:
    message = FILLER * 30 + "Can you check whether the dates match the Lisbon contract? " + FILLER * 30
    assert "Can you check whether the dates match the Lisbon contract?" in formulate_query(message).text


def test_a_question_is_kept_even_when_its_words_are_common() -> None:
    # Every entry carries a word of its own, so on rarity alone the entries
    # would fill the query and push out a question made of shared words. The
    # question is longer than any entry, so it cannot slip into leftover room.
    entries = [f"Entry {number} records the item{number} shipment for the harbour ledger. " for number in range(40)]
    question = "Which entry records the shipment for the harbour ledger, and which entry records the harbour shipment?"
    message = "".join(entries[:20]) + question + " " + "".join(entries[20:])
    assert question in formulate_query(message).text


def test_distinctive_sentences_outrank_repeated_filler() -> None:
    message = (FILLER * 20 + "The Zanzibar ferry schedule changed after the monsoon. " + FILLER * 20
               + "What did we decide?")
    assert "The Zanzibar ferry schedule changed after the monsoon." in formulate_query(message).text


@pytest.mark.parametrize("pad", range(4))
def test_one_unbroken_run_keeps_its_head_and_tail_cut_at_spaces(pad: int) -> None:
    # A cut through a word leaves a piece that is not in the list: every word
    # starts with "w" and holds its own number before the "x". Uneven widths
    # move where the raw cuts fall, so some land inside a word.
    words = [f"w{number}x" + "y" * ((number * 7 + pad) % 4) for number in range(400)]
    message = " ".join(words)
    formulated = formulate_query(message)
    assert len(formulated.text) <= MAX_QUERY
    assert formulated.text.startswith(words[0] + " ") and formulated.text.endswith(" " + words[-1])
    assert len(formulated.kept) == 2
    assert all(set(message[start:end].split()) <= set(words) for start, end in formulated.kept)


def test_unspaced_text_splits_at_its_own_sentence_marks() -> None:
    message = "今日は会議がありました。" * 60 + "ザンジバルのフェリーは何時に出ますか？" + "今日は会議がありました。" * 60
    formulated = formulate_query(message)
    assert len(formulated.text) <= MAX_QUERY
    assert "ザンジバルのフェリーは何時に出ますか？" in formulated.text


def test_the_same_message_always_gives_the_same_query() -> None:
    message = FILLER * 25 + "Does the lease end in May? " + FILLER * 25
    assert formulate_query(message) == formulate_query(message)


def test_blank_text_over_the_limit_gives_an_empty_query() -> None:
    formulated = formulate_query(" \n" * 1000)
    assert formulated.text == "" and formulated.kept == ()


@pytest.mark.parametrize("limit", [0, 1, 15, True])
def test_a_budget_too_small_for_an_excerpt_is_refused(limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        formulate_query("abcdef " * 10, limit=limit)


@pytest.fixture
async def memory() -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("alpha", "The Zanzibar ferry leaves at 07:30 on weekdays.")
    return engine


async def test_a_conversation_turn_over_the_limit_still_recalls_memory(memory: MemoryEngine) -> None:
    from scone_memory.realtime.context import MemoryContext
    message = FILLER * 14 + "When does the Zanzibar ferry leave?"
    assert len(message) > MAX_QUERY
    _, receipt = await MemoryContext(memory, "alpha", "s1").prepare([{"role": "user", "content": message}])
    assert receipt["status"] == "prepared" and receipt["error_type"] is None
    assert len(receipt["references"]) >= 1
    formulation = receipt["query_formulation"]
    assert formulation["method"] == "extract" and formulation["source_chars"] == len(message)
    assert formulation["query_chars"] <= MAX_QUERY
    assert all(0 <= start < end <= len(message) for start, end in formulation["kept"])


async def test_a_short_turn_records_no_formulation(memory: MemoryEngine) -> None:
    from scone_memory.realtime.context import MemoryContext
    _, receipt = await MemoryContext(memory, "alpha", "s1").prepare(
        [{"role": "user", "content": "When does the Zanzibar ferry leave?"}])
    assert receipt["status"] == "prepared" and "query_formulation" not in receipt


async def test_the_chat_helper_recalls_for_a_long_message(memory: MemoryEngine) -> None:
    from scone_memory.integrations.chat import recall_context
    message = FILLER * 14 + "When does the Zanzibar ferry leave?"
    outgoing, receipt = await recall_context(memory, "alpha", [{"role": "user", "content": message}])
    assert receipt.injected and len(receipt.query) <= MAX_QUERY
    assert "When does the Zanzibar ferry leave?" in receipt.query
    assert any("07:30" in str(item.get("content")) for item in outgoing)
