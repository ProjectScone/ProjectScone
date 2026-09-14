"""A bounded summary of many passages, where every sentence carries a checked quote.

A broad question -- "what is known about the launch?" -- is not answered
by five passages and a score. It needs many passages read and a few
sentences written. The sentences are the model's; what makes them usable
is that each one names the passage it came from and a quote from that
passage that this code found there. A sentence without that is not
shown, and is counted.

The work is bounded in rounds. Passages are packed in the order given
(the caller's ranking) into rounds of at most ``max_round_bytes``; each
round is one model call that returns notes, one sentence each with a
passage id and a verbatim quote. A note survives only if the id names a
passage of that round and the quote is found in that passage. When more
than one round left notes, one more call may fold them into a summary
whose sentences cite notes by id; a sentence that cites no known note is
not shown. A fold that cannot be read leaves the notes as they are.

The result says how many passages were read, unread, or too large for a
round, how many notes and sentences were dropped and why, and it never
claims accuracy: the quotes are checked, the sentences are not.

This is not a refine chain that threads one growing answer through every
chunk. Rounds are independent, so an early passage cannot shape what a
later one is allowed to say; a failed round costs only its own passages;
and the citation check is mechanical rather than an instruction the model
may or may not follow.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import InvalidInput
from ..core.models import RecallItem
from ..core.validation import MAX_LIMIT
from ..providers.llm import ChatModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

#: Characters a note's or a summary's sentence may run to; longer is dropped as malformed.
MAX_SENTENCE_CHARS = 1_000
#: Characters a note's quote may run to; longer is dropped as malformed.
MAX_QUOTE_CHARS = 2_000

Status = Literal["synthesized", "partial", "unavailable", "no_evidence"]
RoundStatus = Literal["noted", "unreadable", "failed", "timeout"]

_NOTES_SYSTEM = (
    "You write notes for a summary that answers a question from the passages given, and from nothing else. "
    "Each note is one sentence in your own words, the id of the one passage it comes from, and a quote copied "
    "exactly, character for character, from that passage that supports it. Leave out anything the passages do "
    "not say. Reply with JSON only, of the form "
    '{"notes": [{"sentence": "...", "passage": "chunk:1", "quote": "..."}]}, '
    "and with an empty list when nothing in the passages bears on the question."
)
_FOLD_SYSTEM = (
    "You merge notes into a short summary that answers a question. Each summary sentence is written from one "
    "or more of the notes and lists the ids of the notes it draws on. Add nothing the notes do not say, and "
    "cite no note a sentence does not use. Reply with JSON only, of the form "
    '{"summary": [{"sentence": "...", "notes": ["n1", "n2"]}]}.'
)


class SynthesisLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    #: Passages one synthesis may read; more is refused, not cut. Capped by what one recall returns.
    max_passages: int = Field(default=24, ge=1, le=MAX_LIMIT)
    #: Bytes of passage text one round may hold; a passage over it on its own is refused.
    max_round_bytes: int = Field(default=12_000, ge=64, le=64_000)
    #: Rounds one synthesis may run; passages beyond them are left unread and counted.
    max_rounds: int = Field(default=6, ge=1, le=16)
    #: Sentences shown; the rest are cut, ``truncated`` says so and the offered count is on the record.
    max_sentences: int = Field(default=24, ge=1, le=100)
    #: Seconds for the whole synthesis, every model call included.
    timeout_s: float = Field(default=120.0, ge=0.1, le=600, allow_inf_nan=False)
    #: Passages a widened synthesis may read: the hits' whole sessions, cut at the byte budget and this count.
    max_widened: int = Field(default=200, ge=1, le=1000)


@dataclass(frozen=True)
class Passage:
    """One passage as the model will see it: an id it can cite and the text, verbatim."""

    id: str
    text: str
    source: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or not isinstance(self.text, str) or not self.text.strip():
            raise InvalidInput("a passage needs an id and some text")


@dataclass(frozen=True)
class Citation:
    """A quote this code found in the passage: ``text[start:end] == quote``, in characters."""

    passage_id: str
    quote: str
    start: int
    end: int


@dataclass(frozen=True)
class Sentence:
    text: str
    #: Never empty: a sentence without a checked citation is not a Sentence.
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class SynthesisRound:
    """What one round was given and what came back; no passage text."""

    round_number: int
    passage_count: int
    passage_bytes: int
    notes_returned: int
    notes_kept: int
    status: RoundStatus


@dataclass(frozen=True)
class _Note:
    sentence: str
    citation: Citation


@dataclass(frozen=True)
class Synthesis:
    """The sentences, with the citations each one earned, and everything left out."""

    question: str
    status: Status
    sentences: tuple[Sentence, ...]
    rounds: tuple[SynthesisRound, ...]
    passages_given: int
    #: Passages of rounds whose notes were read; the rest are unread or oversize.
    passages_read: int
    passages_unread: int
    passages_oversize: int
    notes_kept: int
    notes_dropped_unquoted: int
    notes_dropped_unknown: int
    notes_dropped_malformed: int
    folded: bool
    fold_dropped_uncited: int
    sentences_offered: int
    truncated: bool
    model_calls: int
    reasons: tuple[str, ...]
    #: The quotes were checked against their passages; the sentences were not checked against anything.
    verified_accuracy: Literal[False] = False
    #: When the hits' sessions were read whole: sessions, hits, bytes left out and whether the budget bit.
    widening: Optional[dict[str, object]] = None

    def text(self) -> str:
        lines = []
        for sentence in self.sentences:
            ids = list(dict.fromkeys(citation.passage_id for citation in sentence.citations))
            lines.append(f"{sentence.text} [{', '.join(ids)}]")
        return "\n".join(lines)

    def record(self) -> dict[str, object]:
        return {
            "schema_version": 1, "question": self.question, "status": self.status,
            "sentences": [{"text": sentence.text,
                           "citations": [{"passage": c.passage_id, "quote": c.quote, "start": c.start, "end": c.end}
                                         for c in sentence.citations]} for sentence in self.sentences],
            "sentences_offered": self.sentences_offered, "truncated": self.truncated,
            "folded": self.folded, "fold_dropped_uncited": self.fold_dropped_uncited,
            "passages": {"given": self.passages_given, "read": self.passages_read,
                         "unread": self.passages_unread, "oversize": self.passages_oversize},
            "notes": {"kept": self.notes_kept, "dropped_unquoted": self.notes_dropped_unquoted,
                      "dropped_unknown": self.notes_dropped_unknown, "dropped_malformed": self.notes_dropped_malformed},
            "rounds": [{"round": r.round_number, "passages": r.passage_count, "bytes": r.passage_bytes,
                        "notes_returned": r.notes_returned, "notes_kept": r.notes_kept, "status": r.status}
                       for r in self.rounds],
            "model_calls": self.model_calls, "reasons": list(self.reasons), "verified_accuracy": False,
            "widening": self.widening,
        }


def _object(reply: object) -> Optional[dict[str, object]]:
    """The first JSON object in the reply, or None; a model may wrap it in prose, not omit it."""
    if not isinstance(reply, str):
        return None
    start = reply.find("{")
    while start >= 0:
        depth = 0
        for index in range(start, len(reply)):
            if reply[index] == "{":
                depth += 1
            elif reply[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(reply[start:index + 1])
                    except ValueError:
                        break
                    return value if isinstance(value, dict) else None
        start = reply.find("{", start + 1)
    return None


def _pack(passages: Sequence[Passage], max_round_bytes: int) -> tuple[list[list[Passage]], int]:
    """Rounds in the order given, each under the byte bound; passages over it alone are refused."""
    rounds: list[list[Passage]] = []
    current: list[Passage] = []
    used = 0
    oversize = 0
    for passage in passages:
        size = len(passage.text.encode("utf-8"))
        if size > max_round_bytes:
            oversize += 1
            continue
        if current and used + size > max_round_bytes:
            rounds.append(current)
            current, used = [], 0
        current.append(passage)
        used += size
    if current:
        rounds.append(current)
    return rounds, oversize


class _Calls:
    """The model calls of one synthesis, under one deadline."""

    def __init__(self, model: ChatModel, timeout_s: float) -> None:
        self._model = model
        self._timeout_s = timeout_s
        self._deadline = time.monotonic() + timeout_s
        self.count = 0

    async def ask(self, system: str, user: str) -> tuple[Optional[str], Optional[RoundStatus], Optional[str]]:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            return None, "timeout", f"timeout: the {self._timeout_s}s deadline passed before the call"
        self.count += 1
        try:
            reply = await asyncio.wait_for(self._model.complete(system, user), remaining)
        except TimeoutError:
            return None, "timeout", f"timeout: the {self._timeout_s}s deadline passed during the call"
        except Exception as error:  # a model that fails is a failed round, not a failed synthesis
            return None, "failed", f"model failed: {type(error).__name__}"
        return reply, None, None


@dataclass
class _Tally:
    unquoted: int = 0
    unknown: int = 0
    malformed: int = 0


def _read_notes(reply: str, held: Mapping[str, Passage], tally: _Tally) -> Optional[tuple[int, list[_Note]]]:
    """The notes of a reply that survive the check, and how many the reply held; None when it is not notes."""
    body = _object(reply)
    if body is None or not isinstance(body.get("notes"), list):
        return None
    kept: list[_Note] = []
    rows = body["notes"]
    assert isinstance(rows, list)
    for row in rows:
        if not isinstance(row, dict):
            tally.malformed += 1
            continue
        sentence, passage_id, quote = row.get("sentence"), row.get("passage"), row.get("quote")
        if (not isinstance(sentence, str) or not sentence.strip() or len(sentence) > MAX_SENTENCE_CHARS
                or not isinstance(passage_id, str) or not isinstance(quote, str) or not quote.strip()
                or len(quote) > MAX_QUOTE_CHARS):
            tally.malformed += 1
            continue
        passage = held.get(passage_id)
        if passage is None:
            tally.unknown += 1
            continue
        start = passage.text.find(quote)
        if start < 0:
            tally.unquoted += 1
            continue
        kept.append(_Note(sentence.strip(), Citation(passage_id, quote, start, start + len(quote))))
    return len(rows), kept


def _read_fold(reply: str, notes: Sequence[_Note]) -> Optional[tuple[list[Sentence], int]]:
    """The summary sentences that cite known notes, and how many cited none; None when it is not a summary."""
    body = _object(reply)
    if body is None or not isinstance(body.get("summary"), list):
        return None
    by_id = {f"n{index}": note for index, note in enumerate(notes, 1)}
    sentences: list[Sentence] = []
    uncited = 0
    rows = body["summary"]
    assert isinstance(rows, list)
    for row in rows:
        text = row.get("sentence") if isinstance(row, dict) else None
        ids = row.get("notes") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_SENTENCE_CHARS or not isinstance(ids, list):
            uncited += 1
            continue
        cited = [by_id[i] for i in ids if isinstance(i, str) and i in by_id]
        if not cited:
            uncited += 1
            continue
        citations = tuple(dict.fromkeys(note.citation for note in cited))
        sentences.append(Sentence(text.strip(), citations))
    return sentences, uncited


def _notes_prompt(question: str, passages: Sequence[Passage]) -> str:
    listed = "\n".join(f"[{passage.id}] {passage.text}" for passage in passages)
    return f"Question: {question}\n\nPassages:\n{listed}"


def _fold_prompt(question: str, notes: Sequence[_Note]) -> str:
    listed = "\n".join(f"[n{index}] ({note.citation.passage_id}) {note.sentence}" for index, note in enumerate(notes, 1))
    return f"Question: {question}\n\nNotes:\n{listed}"


async def synthesize_passages(model: ChatModel, question: str, passages: Sequence[Passage], *,
                              limits: Optional[SynthesisLimits] = None) -> Synthesis:
    """Sentences about ``question`` from ``passages``, each with a quote found in its passage.

    Passages are read in the order given, so give them ranked. Over
    ``limits.max_passages`` is refused, not cut; a passage larger than a
    round is refused on its own and the rest are read."""
    limits = limits or SynthesisLimits()
    if not isinstance(question, str) or not question.strip():
        raise InvalidInput("a question must have something in it")
    given = tuple(passages)
    reasons: list[str] = []
    rounds: list[SynthesisRound] = []
    notes: list[_Note] = []
    tally = _Tally()
    calls = _Calls(model, limits.timeout_s)
    read = 0
    oversize = 0
    if len(given) > limits.max_passages:
        reasons.append(f"{len(given)} passages over the bound of {limits.max_passages}; refused rather than cut")
        return _finish(question, given, rounds, notes, tally, calls, reasons, read, oversize,
                       folded=False, fold_uncited=0, fold_attempted=False, limits=limits, refused=True)
    plan, oversize = _pack(given, limits.max_round_bytes)
    if oversize:
        reasons.append(f"{oversize} passage(s) oversize for a round of {limits.max_round_bytes} bytes; refused rather than cut")
    if not plan:
        reasons.append("no passages to read")
    for number, group in enumerate(plan[:limits.max_rounds], 1):
        size = sum(len(passage.text.encode("utf-8")) for passage in group)
        reply, failure, reason = await calls.ask(_NOTES_SYSTEM, _notes_prompt(question, group))
        if reply is None:
            assert failure is not None and reason is not None
            rounds.append(SynthesisRound(number, len(group), size, 0, 0, failure))
            reasons.append(reason)
            break
        parsed = _read_notes(reply, {passage.id: passage for passage in group}, tally)
        if parsed is None:
            rounds.append(SynthesisRound(number, len(group), size, 0, 0, "unreadable"))
            reasons.append(f"round {number}: the reply could not be read as notes")
            continue
        returned, kept = parsed
        rounds.append(SynthesisRound(number, len(group), size, returned, len(kept), "noted"))
        notes.extend(kept)
        read += len(group)
    capped = len(plan) > limits.max_rounds
    if capped:
        reasons.append(f"{sum(len(group) for group in plan[limits.max_rounds:])} passage(s) unread: "
                       f"the bound of {limits.max_rounds} round(s) was reached")
    folded = False
    fold_uncited = 0
    fold_attempted = False
    sentences: list[Sentence] = [Sentence(note.sentence, (note.citation,)) for note in notes]
    noted_rounds = sum(1 for r in rounds if r.status == "noted" and r.notes_kept)
    if noted_rounds >= 2 and len(notes) >= 2:
        fold_attempted = True
        reply, failure, reason = await calls.ask(_FOLD_SYSTEM, _fold_prompt(question, notes))
        merged = _read_fold(reply, notes) if reply is not None else None
        if reply is None:
            reasons.append(f"fold {reason}; notes shown unmerged")
        elif merged is None:
            reasons.append("fold: the reply could not be read as a summary; notes shown unmerged")
        elif not merged[0]:
            fold_uncited = merged[1]
            reasons.append("fold: no sentence cited a known note; notes shown unmerged")
        else:
            sentences, fold_uncited = merged
            folded = True
    return _finish(question, given, rounds, notes, tally, calls, reasons, read, oversize, folded=folded,
                   fold_uncited=fold_uncited, fold_attempted=fold_attempted, limits=limits, refused=False,
                   sentences=sentences, capped=capped)


def _finish(question: str, given: Sequence[Passage], rounds: Sequence[SynthesisRound], notes: Sequence[_Note],
            tally: _Tally, calls: _Calls, reasons: list[str], read: int, oversize: int, *, folded: bool,
            fold_uncited: int, fold_attempted: bool, limits: SynthesisLimits, refused: bool,
            sentences: Sequence[Sentence] = (), capped: bool = False) -> Synthesis:
    offered = len(sentences)
    cut = offered > limits.max_sentences
    shown = tuple(sentences[:limits.max_sentences])
    if cut:
        reasons.append(f"{offered} sentences offered, {limits.max_sentences} shown: the sentence bound was reached")
    # A bound of the limits cut something: rounds or sentences. A failed
    # round leaves passages unread too, but that is a failure, not a bound.
    truncated = capped or cut
    unread = len(given) - read - oversize
    status: Status
    if not shown:
        broken = refused or any(r.status in ("failed", "timeout") for r in rounds)
        status = "unavailable" if broken else "no_evidence"
    elif unread or oversize or truncated or (fold_attempted and not folded):
        # A round that failed or could not be read leaves its passages unread, so it is covered above.
        status = "partial"
    else:
        status = "synthesized"
    return Synthesis(question, status, shown, tuple(rounds), len(given), read, unread, oversize, len(notes),
                     tally.unquoted, tally.unknown, tally.malformed, folded, fold_uncited, offered, truncated,
                     calls.count, tuple(reasons))


def passages_from_recall(items: Sequence[RecallItem]) -> tuple[Passage, ...]:
    """Recalled passages as the model will see them, cited by chunk id, in recall order."""
    return tuple(Passage(f"chunk:{item.chunk_id}", item.text, item.source, item.created_at)
                 for item in items if item.text.strip())


@dataclass(frozen=True)
class Widened:
    """The hits' sessions read whole, in the hits' order, under a byte budget."""

    passages: tuple[Passage, ...]
    sessions: int
    hits: int
    omitted_bytes: int
    truncated: bool

    def record(self) -> dict[str, object]:
        return {"sessions": self.sessions, "hits": self.hits, "omitted_bytes": self.omitted_bytes,
                "truncated": self.truncated}


async def widened_passages(engine: "MemoryEngine", space: str, items: Sequence[RecallItem], *,
                           max_bytes: int) -> Widened:
    """Every chunk of every episode the hits name, episodes in the hits' order, chunks in theirs.

    Recall credits a session when one slice of it matches; the slice that
    matches the question's words is often not the one that holds the
    fact. Reading the session whole, bounded by bytes, is how a
    synthesizer sees what a reader would. Bytes past the budget are
    counted, not read."""
    if type(max_bytes) is not int or not 1 <= max_bytes <= 1_000_000:
        raise InvalidInput("max_bytes must be from 1 to 1,000,000")
    sources: dict[int, Optional[str]] = {}
    for item in items:
        sources.setdefault(item.episode_id, item.source)
    passages: list[Passage] = []
    used = 0
    omitted = 0
    for episode_id, source in sources.items():
        for chunk in await engine.documents.chunks_of(space, episode_id):
            if not chunk.text.strip():
                continue
            size = len(chunk.text.encode("utf-8"))
            if used + size > max_bytes:
                omitted += size
                continue
            used += size
            passages.append(Passage(f"chunk:{chunk.chunk_id}", chunk.text, source, chunk.created_at))
    return Widened(tuple(passages), len(sources), len(items), omitted, omitted > 0)


async def synthesize(engine: "MemoryEngine", model: ChatModel, space: str, question: str, *,
                     limits: Optional[SynthesisLimits] = None, as_of: Optional[str] = None,
                     tags: Sequence[str] = (), where: Mapping[str, str] | None = None, kind: Optional[str] = None,
                     source_prefix: Optional[str] = None, since: Optional[str] = None,
                     until: Optional[str] = None, widen_bytes: Optional[int] = None) -> Synthesis:
    """Recall up to ``limits.max_passages`` passages for ``question`` and synthesize over them.

    With ``widen_bytes`` the hits' sessions are read whole under that
    budget (and ``limits.max_widened`` passages), and the record says
    what widening did."""
    limits = limits or SynthesisLimits()
    found = await engine.recall(space, question, limit=limits.max_passages, as_of=as_of, tags=tags, where=where,
                                kind=kind, source_prefix=source_prefix, since=since, until=until)
    if widen_bytes is None:
        return await synthesize_passages(model, question, passages_from_recall(found.items), limits=limits)
    widened = await widened_passages(engine, space, found.items, max_bytes=widen_bytes)
    passages = widened.passages
    truncated = widened.truncated
    if len(passages) > limits.max_widened:
        passages, truncated = passages[:limits.max_widened], True
    made = await synthesize_passages(model, question, passages,
                                     limits=limits.model_copy(update={"max_passages": max(1, limits.max_widened)}))
    return replace(made, widening={**widened.record(), "truncated": truncated})
