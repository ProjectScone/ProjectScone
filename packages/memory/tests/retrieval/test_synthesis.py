"""A summary that says where each sentence came from, and what it left out.

A broad question over many passages is answered in bounded rounds: the
model writes notes, each with a verbatim quote from the passage it came
from, and a note survives only if that quote is found in that passage. A
second call may merge the notes into a summary, but a merged sentence
survives only by citing notes that exist. So every sentence the reader
sees carries a quote a machine checked, and every count of what was left
out is on the record.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from pydantic import ValidationError

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.providers.llm import ChatError, FakeChat
from scone_memory.retrieval import synthesis as module
from scone_memory.retrieval.synthesis import Passage, SynthesisLimits, synthesize, synthesize_passages

pytestmark = pytest.mark.asyncio

QUESTION = "What happened with the launch?"
P1 = Passage("chunk:1", "Priya moved the launch to March because the audit ran late.")
P2 = Passage("chunk:2", "Tomas said the audit found two billing errors in February.")
P3 = Passage("chunk:3", "The launch party is booked for the twelfth of March.")
ONE_EACH = SynthesisLimits(max_round_bytes=70)  # every passage above is under 70 bytes and any two are over


def notes(*rows: tuple[str, str, str]) -> str:
    return json.dumps({"notes": [{"sentence": s, "passage": p, "quote": q} for s, p, q in rows]})


def summary(*rows: tuple[str, list[str]]) -> str:
    return json.dumps({"summary": [{"sentence": s, "notes": n} for s, n in rows]})


N1 = ("The launch moved to March.", "chunk:1", "moved the launch to March")
N2 = ("The audit found two billing errors.", "chunk:2", "found two billing errors")
N3 = ("A party is booked in March.", "chunk:3", "booked for the twelfth of March")


class SlowChat:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return notes(N1)


class CitingChat:
    """A model that reads the passages it was given and quotes each one's opening words."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append(user)
        rows = []
        for match in re.finditer(r"^\[(chunk:\d+)\] (.+)$", user, flags=re.M):
            words = match.group(2).split()
            rows.append((f"Note on {match.group(1)}.", match.group(1), " ".join(words[:4])))
        return notes(*rows)


async def test_one_round_is_one_call_and_every_sentence_carries_a_checked_quote():
    model = FakeChat([notes(N1, N2)])
    result = await synthesize_passages(model, QUESTION, [P1, P2, P3])
    assert result.status == "synthesized"
    assert [s.text for s in result.sentences] == [N1[0], N2[0]]
    first = result.sentences[0].citations
    assert len(first) == 1 and first[0].passage_id == "chunk:1" and first[0].quote == "moved the launch to March"
    assert P1.text[first[0].start:first[0].end] == first[0].quote
    assert result.model_calls == 1 and result.folded is False and len(model.calls) == 1
    system, user = model.calls[0]
    assert QUESTION in user and "[chunk:3] " + P3.text in user, "every passage of the round reaches the model, by its id"
    assert result.text() == f"{N1[0]} [chunk:1]\n{N2[0]} [chunk:2]"
    assert result.passages_given == result.passages_read == 3 and result.passages_unread == 0


async def test_a_note_whose_quote_is_not_in_its_passage_is_dropped_and_counted():
    model = FakeChat([notes(N1, ("The launch moved to April.", "chunk:1", "moved the launch to April"))])
    result = await synthesize_passages(model, QUESTION, [P1, P2])
    assert [s.text for s in result.sentences] == [N1[0]]
    assert result.notes_kept == 1 and result.notes_dropped_unquoted == 1


async def test_a_note_citing_a_passage_the_round_did_not_hold_is_dropped_and_counted():
    model = FakeChat([notes(N1, ("Someone said something.", "chunk:9", "moved the launch"))])
    result = await synthesize_passages(model, QUESTION, [P1, P2])
    assert [s.text for s in result.sentences] == [N1[0]]
    assert result.notes_dropped_unknown == 1


async def test_malformed_notes_are_dropped_not_guessed_at():
    reply = json.dumps({"notes": [{"sentence": "No quote here.", "passage": "chunk:1"}, "not a note",
                                  {"sentence": "x" * (module.MAX_SENTENCE_CHARS + 1), "passage": "chunk:1",
                                   "quote": "moved the launch"},
                                  {"sentence": N1[0], "passage": "chunk:1", "quote": N1[2]}]})
    result = await synthesize_passages(FakeChat([reply]), QUESTION, [P1])
    assert [s.text for s in result.sentences] == [N1[0]]
    assert result.notes_dropped_malformed == 3


async def test_a_reply_with_no_note_list_is_unreadable():
    result = await synthesize_passages(FakeChat([json.dumps({"summary": []})]), QUESTION, [P1])
    assert result.rounds[0].status == "unreadable" and result.status == "no_evidence"


async def test_passages_over_the_bound_are_refused_not_cut():
    model = FakeChat([notes(N1)])
    result = await synthesize_passages(model, QUESTION, [P1, P2, P3], limits=SynthesisLimits(max_passages=2))
    assert result.status == "unavailable" and "bound" in result.reasons[0]
    assert model.calls == [], "the model was never asked"
    assert result.passages_given == 3 and result.passages_read == 0 and result.sentences == ()


async def test_a_passage_larger_than_a_round_is_refused_and_the_rest_are_read():
    big = Passage("chunk:8", "x" * 200)
    model = FakeChat([notes(N1)])
    result = await synthesize_passages(model, QUESTION, [big, P1], limits=SynthesisLimits(max_round_bytes=100))
    assert result.passages_oversize == 1 and result.passages_read == 1
    assert "xxx" not in model.calls[0][1]
    assert [s.text for s in result.sentences] == [N1[0]]
    assert result.status == "partial" and any("oversize" in reason for reason in result.reasons)


async def test_rounds_are_packed_by_bytes_and_capped_with_the_unread_counted():
    model = FakeChat([notes(N1), notes(N2), summary(("Launch and audit.", ["n1", "n2"]))])
    limits = SynthesisLimits(max_round_bytes=70, max_rounds=2)
    result = await synthesize_passages(model, QUESTION, [P1, P2, P3], limits=limits)
    assert [r.passage_count for r in result.rounds] == [1, 1]
    assert result.passages_read == 2 and result.passages_unread == 1 and result.truncated
    assert result.status == "partial" and any("unread" in reason for reason in result.reasons)
    assert result.model_calls == 3


async def test_two_rounds_fold_into_sentences_that_cite_by_note():
    merged = "The launch moved to March after an audit that found two billing errors."
    model = FakeChat([notes(N1), notes(N2), summary((merged, ["n1", "n2"]))])
    result = await synthesize_passages(model, QUESTION, [P1, P2], limits=ONE_EACH)
    assert result.folded is True and result.status == "synthesized"
    assert [s.text for s in result.sentences] == [merged]
    assert [c.passage_id for c in result.sentences[0].citations] == ["chunk:1", "chunk:2"]
    assert [c.quote for c in result.sentences[0].citations] == [N1[2], N2[2]]
    assert result.text() == f"{merged} [chunk:1, chunk:2]"
    fold_prompt = model.calls[2][1]
    assert f"[n1] (chunk:1) {N1[0]}" in fold_prompt and f"[n2] (chunk:2) {N2[0]}" in fold_prompt
    assert N1[2] not in fold_prompt, "the fold sees the notes, not the passages"


async def test_a_fold_sentence_citing_no_known_note_is_dropped_and_counted():
    model = FakeChat([notes(N1), notes(N2), summary(("Kept.", ["n2"]), ("Dropped.", ["n7"]), ("Also dropped.", []))])
    result = await synthesize_passages(model, QUESTION, [P1, P2], limits=ONE_EACH)
    assert [s.text for s in result.sentences] == ["Kept."]
    assert result.fold_dropped_uncited == 2


async def test_a_fold_that_cannot_be_read_shows_the_notes_unmerged():
    model = FakeChat([notes(N1), notes(N2), "I would rather not answer in JSON."])
    result = await synthesize_passages(model, QUESTION, [P1, P2], limits=ONE_EACH)
    assert result.folded is False and result.status == "partial"
    assert [s.text for s in result.sentences] == [N1[0], N2[0]]
    assert any("fold" in reason for reason in result.reasons)


async def test_one_note_from_two_rounds_is_not_folded():
    model = FakeChat([notes(N1), notes()])
    result = await synthesize_passages(model, QUESTION, [P1, P2], limits=ONE_EACH)
    assert result.model_calls == 2 and result.folded is False
    assert [s.text for s in result.sentences] == [N1[0]] and result.status == "synthesized"


async def test_an_unreadable_round_is_counted_and_the_next_round_still_runs():
    model = FakeChat(["garbage", notes(N2)])
    result = await synthesize_passages(model, QUESTION, [P1, P2], limits=ONE_EACH)
    assert [r.status for r in result.rounds] == ["unreadable", "noted"]
    assert [s.text for s in result.sentences] == [N2[0]] and result.status == "partial"


async def test_a_failing_model_stops_the_rounds_and_says_so():
    model = FakeChat([ChatError("down")])
    result = await synthesize_passages(model, QUESTION, [P1, P2, P3], limits=ONE_EACH)
    assert result.status == "unavailable" and result.reasons[0].startswith("model failed: ChatError")
    assert [r.status for r in result.rounds] == ["failed"] and result.passages_unread == 3
    assert len(model.calls) == 1


async def test_a_failure_before_the_rounds_bound_is_not_reported_as_the_bound():
    model = FakeChat([ChatError("down")])
    result = await synthesize_passages(model, QUESTION, [P1, P2, P3], limits=SynthesisLimits(max_round_bytes=70, max_rounds=1))
    assert result.passages_unread == 3 and result.truncated is False
    assert result.reasons == ("model failed: ChatError",), "the model stopped the rounds, not the bound of one round"


async def test_the_deadline_stops_the_rounds():
    model = SlowChat(0.5)
    result = await synthesize_passages(model, QUESTION, [P1, P2], limits=SynthesisLimits(max_round_bytes=70, timeout_s=0.2))
    assert result.rounds[0].status == "timeout" and result.status == "unavailable"
    assert model.calls == 1 and result.passages_unread == 2
    assert any("timeout" in reason for reason in result.reasons)


async def test_sentences_are_capped_with_the_offered_count_on_the_record():
    model = FakeChat([notes(N1, N2, N3)])
    result = await synthesize_passages(model, QUESTION, [P1, P2, P3], limits=SynthesisLimits(max_sentences=1))
    assert [s.text for s in result.sentences] == [N1[0]]
    assert result.sentences_offered == 3 and result.truncated and result.status == "partial"


async def test_no_passages_is_no_evidence_and_no_call():
    model = FakeChat()
    result = await synthesize_passages(model, QUESTION, [])
    assert result.status == "no_evidence" and model.calls == [] and result.sentences == ()


async def test_notes_that_all_fail_the_check_leave_no_evidence():
    model = FakeChat([notes(("Wrong.", "chunk:1", "not in the passage"))])
    result = await synthesize_passages(model, QUESTION, [P1])
    assert result.status == "no_evidence" and result.notes_dropped_unquoted == 1


async def test_limits_are_checked():
    with pytest.raises(ValidationError):
        SynthesisLimits(max_rounds=0)
    with pytest.raises(ValidationError):
        SynthesisLimits(max_passages=51)
    with pytest.raises(ValidationError):
        SynthesisLimits(max_round_bytes=10)


async def test_a_question_that_is_not_text_is_refused():
    with pytest.raises(InvalidInput):
        await synthesize_passages(FakeChat(), "   ", [P1])


async def test_the_record_says_what_was_left_out_and_never_claims_accuracy():
    model = FakeChat([notes(N1, ("Bad.", "chunk:1", "nope"))])
    result = await synthesize_passages(model, QUESTION, [P1], limits=SynthesisLimits(max_sentences=1))
    record = result.record()
    assert record["verified_accuracy"] is False and record["status"] == "synthesized"
    assert record["notes"] == {"kept": 1, "dropped_unquoted": 1, "dropped_unknown": 0, "dropped_malformed": 0}
    assert record["passages"] == {"given": 1, "read": 1, "unread": 0, "oversize": 0, "cited": 1}
    assert record["sentences"][0] == {"text": N1[0], "citations": [{"passage": "chunk:1", "quote": N1[2], "start": 6, "end": 31}]}
    assert record["rounds"][0] == {"round": 1, "passages": 1, "bytes": len(P1.text.encode()), "answer_bytes": 0,
                                   "notes_returned": 2, "notes_kept": 1, "status": "noted"}
    assert record["refine_dropped_carried"] == 0
    assert record["model_calls"] == 1 and record["folded"] is False and record["truncated"] is False


async def test_the_engine_path_recalls_and_cites_chunks():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for text in (P1.text, P2.text, P3.text):
        await engine.remember("s", text)
    model = CitingChat()
    result = await synthesize(engine, model, "s", QUESTION, limits=SynthesisLimits(max_passages=3))
    assert result.status == "synthesized" and len(result.sentences) == 3
    recalled = {item.chunk_id: item.text for item in (await engine.recall("s", QUESTION, limit=3)).items}
    for sentence in result.sentences:
        (citation,) = sentence.citations
        chunk_id = int(citation.passage_id.removeprefix("chunk:"))
        assert citation.quote in recalled[chunk_id]
    assert result.passages_given == 3
