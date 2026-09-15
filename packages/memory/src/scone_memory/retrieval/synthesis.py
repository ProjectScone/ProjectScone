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

That is the ``evidence`` mode, the default. Two more modes answer the
same way the reference framework's Refine and Accumulate do, under the
same check:

``refine`` threads one answer through the rounds. The first round that
leaves a note is the answer; each later round is one call that sees the
answer so far -- every sentence with its passage id and its quote, not
the passages again -- and the new round's passages, and returns the whole
answer rewritten. A rewritten sentence survives only if its quote is
found in a passage read so far, and a readable rewrite replaces the
answer; a sentence of the answer so far whose citation the rewrite does
not carry is counted in ``refine_dropped_carried``, with a reason, and a
sentence it repeats (same passage, same quote) in ``notes_carried``. A
rewrite that cannot be read, or whose every sentence fails the check,
leaves the answer as it stood, and ``refine_kept_prior`` counts those
rounds. The rounds are packed by bytes as above, except that a round
after the first answer carries that answer inside its bound: its passages
get ``max_round_bytes`` less the answer's bytes, and each round's record
says both. That is the reference's CompactAndRefine (its default), which
repacks each chunk around the existing answer. Passages are never cut, so
when the answer leaves no room for the next passage the rounds stop, the
rest are unread, and ``truncated`` says the bound bit. An early passage
can shape what a later round keeps, which is the trade against
``evidence``.

``accumulate`` gives the model one passage per call and joins what each
call left, in passage order, with no fold: a note must quote the passage
its own call held. It spends the most calls on the passages it reads.

``facts`` extracts first and writes second. Each passage is one call, as
in ``accumulate``, that asks for the atomic facts the passage states that
bear on the question, each a short sentence with a quote from that
passage; the passage's id is the code's, not the model's, so a fact
cannot name the wrong passage, and a fact whose quote is not in the
passage its call held is dropped and counted in ``notes_dropped_unquoted``
(a fact without a quote is malformed). When any fact survives, one more
call writes the answer from the checked facts alone -- each fact with its
quote, never the passages -- and every sentence must name the facts it
uses; a sentence that names none it was given is dropped and counted in
``fold_dropped_uncited``, and a shown sentence carries the quotes of the
facts it names. The call asks for at most ``max_sentences`` sentences,
the bound that cuts what is shown. It holds at most ``max_round_bytes`` of facts, in
the order they were found: the facts past the bound are not sent, are
counted in ``facts_unsent``, and the answer is ``truncated``. An answer
that cannot be read, fails, or cites nothing, or a first fact too large
to send, leaves the facts shown as they are, and the synthesis is
``partial``. ``facts_used`` counts the distinct facts the shown sentences
name (two facts with one quote are two; with no answer written, the facts
shown), and ``folded`` says the answer was written.

In every mode the calls are bounded by ``max_rounds`` (``evidence`` may
add its fold and ``facts`` its answer), every call is counted in
``model_calls``, and the record says how many passages were read and how
many the shown sentences cite.
The citation check is mechanical rather than an instruction the model may
or may not follow.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass
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
Mode = Literal["evidence", "refine", "accumulate", "facts"]
#: The ways a synthesis can read its rounds; ``evidence`` is the default.
MODES: tuple[str, ...] = ("evidence", "refine", "accumulate", "facts")
RoundStatus = Literal["noted", "unreadable", "failed", "timeout"]

_NOTES_SYSTEM = (
    "You write notes for a summary that answers a question from the passages given, and from nothing else. "
    "Each note is one sentence in your own words, the id of the one passage it comes from, and a quote copied "
    "exactly, character for character, from that passage that supports it. Leave out anything the passages do "
    "not say. Reply with JSON only, of the form "
    '{"notes": [{"sentence": "...", "passage": "chunk:1", "quote": "..."}]}, '
    "and with an empty list when nothing in the passages bears on the question."
)
_REFINE_SYSTEM = (
    "You refine an answer to a question with new passages, using the answer so far and the new passages and "
    "nothing else. The answer so far is a list of sentences, each with the id of the passage it came from and "
    "a quote from that passage. Return the whole refined answer: keep each sentence that still holds, with "
    "its passage id and its quote exactly as given; correct or add sentences only from what the new passages "
    "say, each with the id of the one passage it comes from and a quote copied exactly, character for "
    "character, from that passage. When the new passages add nothing, return the answer so far unchanged. "
    "Reply with JSON only, of the form "
    '{"notes": [{"sentence": "...", "passage": "chunk:1", "quote": "..."}]}.'
)
_FACTS_SYSTEM = (
    "You pick out the facts a passage states that bear on a question, from that passage and nothing else. "
    "Each fact is one short sentence that says one thing the passage states, and a quote copied exactly, "
    "character for character, from the passage that states it. Give each fact once. Write no fact about the "
    "passage itself or about what it does not say, and leave out what does not bear on the question. Reply "
    "with JSON only, of the form "
    '{"facts": [{"fact": "...", "quote": "..."}]}, '
    "and with an empty list when nothing in the passage bears on the question."
)
_ANSWER_SYSTEM = (
    "You answer a question from numbered facts, each given with the quote it was taken from, and from nothing "
    "else. Use only the facts that help answer the question and leave the others out; not every fact belongs "
    "in the answer. Say each thing once. Each sentence lists the ids of the facts it uses, and cites no fact it "
    "does not use. Add nothing the facts do not say. Reply with JSON only, of the form "
    '{"answer": [{"sentence": "...", "facts": ["f1", "f2"]}]}.'
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
    #: Bytes of passage text one round may hold, a refine round's answer so far included; a passage over it
    #: on its own is refused.
    max_round_bytes: int = Field(default=12_000, ge=64, le=64_000)
    #: Rounds one synthesis may run; passages beyond them are left unread and counted.
    max_rounds: int = Field(default=6, ge=1, le=16)
    #: Sentences shown; the rest are cut, ``truncated`` says so and the offered count is on the record.
    max_sentences: int = Field(default=24, ge=1, le=100)
    #: Seconds for the whole synthesis, every model call included.
    timeout_s: float = Field(default=120.0, ge=0.1, le=600, allow_inf_nan=False)


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
    #: Bytes of the answer so far a refine round carried beside its passages; 0 in any other round.
    answer_bytes: int = 0
    #: Kept notes of a refine rewrite that repeat a sentence of the answer so far (same passage and quote), so
    #: ``notes_kept - notes_carried`` is what the round added; 0 in any other round.
    notes_carried: int = 0


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
    mode: Mode = "evidence"
    #: Refine rounds whose reply left the answer as it stood: unreadable, or no sentence kept a checked quote.
    refine_kept_prior: int = 0
    #: Checked sentences of the answer so far that a readable refine rewrite left out, over every round.
    refine_dropped_carried: int = 0
    #: Kept notes that repeat the answer so far, over every refine round: ``notes_kept`` counts them again.
    notes_carried: int = 0
    #: Distinct facts the shown sentences name, in ``facts`` mode (with no answer written, the facts shown); 0 in any
    #: other mode.
    facts_used: int = 0
    #: Checked facts left out of the answer call by its byte bound, in ``facts`` mode; 0 in any other mode.
    facts_unsent: int = 0
    #: The quotes were checked against their passages; the sentences were not checked against anything.
    verified_accuracy: Literal[False] = False

    @property
    def passages_cited(self) -> int:
        """Distinct passages the shown sentences cite: the passages the answer used, not the ones read."""
        return len({citation.passage_id for sentence in self.sentences for citation in sentence.citations})

    def text(self) -> str:
        lines = []
        for sentence in self.sentences:
            ids = list(dict.fromkeys(citation.passage_id for citation in sentence.citations))
            lines.append(f"{sentence.text} [{', '.join(ids)}]")
        return "\n".join(lines)

    def record(self) -> dict[str, object]:
        return {
            "schema_version": 1, "question": self.question, "status": self.status, "mode": self.mode,
            "sentences": [{"text": sentence.text,
                           "citations": [{"passage": c.passage_id, "quote": c.quote, "start": c.start, "end": c.end}
                                         for c in sentence.citations]} for sentence in self.sentences],
            "sentences_offered": self.sentences_offered, "truncated": self.truncated,
            "folded": self.folded, "fold_dropped_uncited": self.fold_dropped_uncited,
            "passages": {"given": self.passages_given, "read": self.passages_read,
                         "unread": self.passages_unread, "oversize": self.passages_oversize,
                         "cited": self.passages_cited},
            "notes": {"kept": self.notes_kept, "carried": self.notes_carried, "dropped_unquoted": self.notes_dropped_unquoted,
                      "dropped_unknown": self.notes_dropped_unknown, "dropped_malformed": self.notes_dropped_malformed},
            "rounds": [{"round": r.round_number, "passages": r.passage_count, "bytes": r.passage_bytes,
                        "answer_bytes": r.answer_bytes, "notes_returned": r.notes_returned,
                        "notes_kept": r.notes_kept, "notes_carried": r.notes_carried, "status": r.status}
                       for r in self.rounds],
            "refine_kept_prior": self.refine_kept_prior, "refine_dropped_carried": self.refine_dropped_carried,
            # Only a mode that extracts facts has counts of them; elsewhere zeros would read as none found.
            "facts": ({"extracted": self.notes_kept, "used": self.facts_used, "unsent": self.facts_unsent,
                       "sentences_dropped_uncited": self.fold_dropped_uncited} if self.mode == "facts" else None),
            "model_calls": self.model_calls, "reasons": list(self.reasons), "verified_accuracy": False,
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


def _size(passage: Passage) -> int:
    return len(passage.text.encode("utf-8"))


def _take(passages: Sequence[Passage], start: int, room: int, *, one_each: bool) -> list[Passage]:
    """The next round from ``start``, in the order given: passages while they fit ``room`` bytes, or one."""
    taken: list[Passage] = []
    used = 0
    for passage in passages[start:]:
        if (taken and one_each) or used + _size(passage) > room:
            break
        taken.append(passage)
        used += _size(passage)
    return taken


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


def _texts(sentence: object, quote: object) -> Optional[tuple[str, str]]:
    """A sentence and a quote, both non-empty text within their bounds; None when either is not."""
    if (not isinstance(sentence, str) or not sentence.strip() or len(sentence) > MAX_SENTENCE_CHARS
            or not isinstance(quote, str) or not quote.strip() or len(quote) > MAX_QUOTE_CHARS):
        return None
    return sentence, quote


def _quoted(sentence: str, quote: str, passage: Passage, tally: _Tally) -> Optional[_Note]:
    """The note, when its quote is found in the passage; otherwise counted as unquoted."""
    start = passage.text.find(quote)
    if start < 0:
        tally.unquoted += 1
        return None
    return _Note(sentence.strip(), Citation(passage.id, quote, start, start + len(quote)))


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
        passage_id = row.get("passage")
        texts = _texts(row.get("sentence"), row.get("quote"))
        if texts is None or not isinstance(passage_id, str):
            tally.malformed += 1
            continue
        passage = held.get(passage_id)
        if passage is None:
            tally.unknown += 1
            continue
        note = _quoted(*texts, passage, tally)
        if note is not None:
            kept.append(note)
    return len(rows), kept


def _read_facts(reply: str, passage: Passage, tally: _Tally) -> Optional[tuple[int, list[_Note]]]:
    """The facts of a reply whose quotes are found in ``passage``, and how many it held; None when it is not facts."""
    body = _object(reply)
    if body is None or not isinstance(body.get("facts"), list):
        return None
    kept: list[_Note] = []
    rows = body["facts"]
    assert isinstance(rows, list)
    for row in rows:
        texts = _texts(row.get("fact"), row.get("quote")) if isinstance(row, dict) else None
        if texts is None:
            tally.malformed += 1
            continue
        note = _quoted(*texts, passage, tally)
        if note is not None:
            kept.append(note)
    return len(rows), kept


@dataclass
class _Cited:
    """The sentences of a fold or an answer that cite known notes, the notes each names, and how many cited none."""

    sentences: list[Sentence]
    uses: list[frozenset[int]]
    uncited: int


def _read_fold(reply: str, notes: Sequence[_Note], *, rows_key: str = "summary", ids_key: str = "notes",
               prefix: str = "n") -> Optional[_Cited]:
    """The sentences under ``rows_key`` that cite known notes by ``prefix`` and number; None when there is no such list."""
    body = _object(reply)
    if body is None or not isinstance(body.get(rows_key), list):
        return None
    by_id = {f"{prefix}{index}": index - 1 for index in range(1, len(notes) + 1)}
    cited = _Cited([], [], 0)
    rows = body[rows_key]
    assert isinstance(rows, list)
    for row in rows:
        text = row.get("sentence") if isinstance(row, dict) else None
        ids = row.get(ids_key) if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_SENTENCE_CHARS or not isinstance(ids, list):
            cited.uncited += 1
            continue
        named = [by_id[i] for i in ids if isinstance(i, str) and i in by_id]
        if not named:
            cited.uncited += 1
            continue
        citations = tuple(dict.fromkeys(notes[index].citation for index in named))
        cited.sentences.append(Sentence(text.strip(), citations))
        cited.uses.append(frozenset(named))
    return cited


def _notes_prompt(question: str, passages: Sequence[Passage]) -> str:
    listed = "\n".join(f"[{passage.id}] {passage.text}" for passage in passages)
    return f"Question: {question}\n\nPassages:\n{listed}"


def _answer_so_far(answer: Sequence[_Note]) -> str:
    """The answer as a refine call carries it: each sentence with its passage id and quote."""
    return "\n".join(f'[s{index}] ({note.citation.passage_id}) {note.sentence} Quote: "{note.citation.quote}"'
                     for index, note in enumerate(answer, 1))


def _refine_prompt(question: str, so_far: str, passages: Sequence[Passage]) -> str:
    new_passages = "\n".join(f"[{passage.id}] {passage.text}" for passage in passages)
    return f"Question: {question}\n\nAnswer so far:\n{so_far}\n\nNew passages:\n{new_passages}"


def _left_out(answer: Sequence[_Note], rewrite: Sequence[_Note]) -> int:
    """Sentences of the answer whose citation no sentence of the rewrite carries; a reworded sentence keeps its quote."""
    carried = Counter(note.citation for note in rewrite)
    return sum((Counter(note.citation for note in answer) - carried).values())


def _facts_prompt(question: str, passage: Passage) -> str:
    return f"Question: {question}\n\nPassage:\n{passage.text}"


def _fact_lines(notes: Sequence[_Note]) -> list[str]:
    """The facts as the answer call sees them: an id, the fact, and its quote; no passage."""
    return [f'[f{index}] {note.sentence} Quote: "{note.citation.quote}"' for index, note in enumerate(notes, 1)]


def _lines_within(lines: Sequence[str], room: int) -> int:
    """How many of ``lines``, from the first, fit ``room`` bytes joined by newlines."""
    used = 0
    for count, line in enumerate(lines):
        used += len(line.encode("utf-8")) + (1 if count else 0)
        if used > room:
            return count
    return len(lines)


def _fold_prompt(question: str, notes: Sequence[_Note]) -> str:
    listed = "\n".join(f"[n{index}] ({note.citation.passage_id}) {note.sentence}" for index, note in enumerate(notes, 1))
    return f"Question: {question}\n\nNotes:\n{listed}"


async def synthesize_passages(model: ChatModel, question: str, passages: Sequence[Passage], *,
                              limits: Optional[SynthesisLimits] = None, mode: Mode = "evidence") -> Synthesis:
    """Sentences about ``question`` from ``passages``, each with a quote found in its passage.

    Passages are read in the order given, so give them ranked. Over
    ``limits.max_passages`` is refused, not cut; a passage larger than a
    round is refused on its own and the rest are read. ``mode`` is how the
    rounds are read: ``evidence``, ``refine``, ``accumulate`` or ``facts``
    (see the module's account)."""
    limits = limits or SynthesisLimits()
    if not isinstance(question, str) or not question.strip():
        raise InvalidInput("a question must have something in it")
    if mode not in MODES:
        raise InvalidInput(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
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
        return _finish(question, given, rounds, tally, calls, reasons, read, oversize,
                       folded=False, fold_uncited=0, fold_attempted=False, limits=limits, refused=True, mode=mode)
    fitting = [passage for passage in given if _size(passage) <= limits.max_round_bytes]
    oversize = len(given) - len(fitting)
    # Passages of rounds whose reply was read: what a refined sentence may quote.
    pool: dict[str, Passage] = {}
    kept_prior = 0
    dropped_carried = 0
    capped = crowded = False
    if oversize:
        reasons.append(f"{oversize} passage(s) oversize for a round of {limits.max_round_bytes} bytes; refused rather than cut")
    if not fitting:
        reasons.append("no passages to read")
    start = 0
    while start < len(fitting):
        if len(rounds) == limits.max_rounds:
            capped = True
            reasons.append(f"{len(fitting) - start} passage(s) unread: the bound of {limits.max_rounds} round(s) was reached")
            break
        number = len(rounds) + 1
        refining = mode == "refine" and bool(notes)
        so_far = _answer_so_far(notes) if refining else ""
        carried = len(so_far.encode("utf-8"))
        group = _take(fitting, start, limits.max_round_bytes - carried, one_each=mode in ("accumulate", "facts"))
        if not group:
            # Only a carried answer can leave no room: every passage here fits a round on its own.
            crowded = True
            reasons.append(f"{len(fitting) - start} passage(s) unread: the answer so far ({carried} bytes) left no room "
                           f"for the next passage in a round of {limits.max_round_bytes} bytes")
            break
        start += len(group)
        size = sum(_size(passage) for passage in group)
        if refining:
            reply, failure, reason = await calls.ask(_REFINE_SYSTEM, _refine_prompt(question, so_far, group))
        elif mode == "facts":
            reply, failure, reason = await calls.ask(_FACTS_SYSTEM, _facts_prompt(question, group[0]))
        else:
            reply, failure, reason = await calls.ask(_NOTES_SYSTEM, _notes_prompt(question, group))
        if reply is None:
            assert failure is not None and reason is not None
            rounds.append(SynthesisRound(number, len(group), size, 0, 0, failure, carried))
            reasons.append(reason)
            break
        held = {passage.id: passage for passage in group}
        if mode == "facts":
            parsed = _read_facts(reply, group[0], tally)
        else:
            parsed = _read_notes(reply, {**pool, **held} if refining else held, tally)
        if parsed is None:
            rounds.append(SynthesisRound(number, len(group), size, 0, 0, "unreadable", carried))
            stands = "; the answer so far stands" if refining else ""
            kept_prior += refining
            reasons.append(f"round {number}: the reply could not be read as {'facts' if mode == 'facts' else 'notes'}{stands}")
            continue
        returned, kept = parsed
        pool.update(held)
        read += len(group)
        repeated = 0
        if not refining:
            notes.extend(kept)
        elif kept:
            left_out = _left_out(notes, kept)
            repeated = len(notes) - left_out
            if left_out:
                dropped_carried += left_out
                reasons.append(f"round {number}: {left_out} sentence(s) of the answer so far left out of the rewrite")
            notes = kept
        else:
            kept_prior += 1
            reasons.append(f"round {number}: no refined sentence kept a checked quote; the answer so far stands")
        rounds.append(SynthesisRound(number, len(group), size, returned, len(kept), "noted", carried, repeated))
    folded = False
    fold_uncited = 0
    fold_attempted = False
    sentences: list[Sentence] = [Sentence(note.sentence, (note.citation,)) for note in notes]
    # The notes each sentence stands on; unmerged, a sentence is its own note.
    uses = [frozenset({index}) for index in range(len(notes))]
    unsent = 0
    noted_rounds = sum(1 for r in rounds if r.status == "noted" and r.notes_kept)
    if mode == "evidence" and noted_rounds >= 2 and len(notes) >= 2:
        fold_attempted = True
        reply, failure, reason = await calls.ask(_FOLD_SYSTEM, _fold_prompt(question, notes))
        merged = _read_fold(reply, notes) if reply is not None else None
        if reply is None:
            reasons.append(f"fold {reason}; notes shown unmerged")
        elif merged is None:
            reasons.append("fold: the reply could not be read as a summary; notes shown unmerged")
        elif not merged.sentences:
            fold_uncited = merged.uncited
            reasons.append("fold: no sentence cited a known note; notes shown unmerged")
        else:
            sentences, uses, fold_uncited = merged.sentences, merged.uses, merged.uncited
            folded = True
    if mode == "facts" and notes:
        fold_attempted = True
        lines = _fact_lines(notes)
        sent = _lines_within(lines, limits.max_round_bytes)
        unsent = len(notes) - sent
        if unsent:
            reasons.append(f"answer: {unsent} fact(s) left out: the facts past {limits.max_round_bytes} bytes did not "
                           "fit the answer's round")
        if not sent:
            reasons.append("answer: no fact fit the answer's round; facts shown unmerged")
        else:
            prompt = (f"Question: {question}\n\nFacts:\n" + "\n".join(lines[:sent])
                      + f"\n\nAnswer in at most {limits.max_sentences} sentence(s).")
            reply, failure, reason = await calls.ask(_ANSWER_SYSTEM, prompt)
            written = (_read_fold(reply, notes[:sent], rows_key="answer", ids_key="facts", prefix="f")
                       if reply is not None else None)
            if reply is None:
                reasons.append(f"answer {reason}; facts shown unmerged")
            elif written is None:
                reasons.append("answer: the reply could not be read as an answer; facts shown unmerged")
            elif not written.sentences:
                fold_uncited = written.uncited
                reasons.append("answer: no sentence cited a known fact; facts shown unmerged")
            else:
                sentences, uses, fold_uncited = written.sentences, written.uses, written.uncited
                folded = True
    used = len(frozenset().union(*uses[:limits.max_sentences])) if mode == "facts" else 0
    return _finish(question, given, rounds, tally, calls, reasons, read, oversize, folded=folded,
                   fold_uncited=fold_uncited, fold_attempted=fold_attempted, limits=limits, refused=False,
                   sentences=sentences, capped=capped or crowded or bool(unsent), mode=mode, kept_prior=kept_prior,
                   dropped_carried=dropped_carried, facts_used=used, facts_unsent=unsent)


def _finish(question: str, given: Sequence[Passage], rounds: Sequence[SynthesisRound],
            tally: _Tally, calls: _Calls, reasons: list[str], read: int, oversize: int, *, folded: bool,
            fold_uncited: int, fold_attempted: bool, limits: SynthesisLimits, refused: bool,
            sentences: Sequence[Sentence] = (), capped: bool = False, mode: Mode = "evidence",
            kept_prior: int = 0, dropped_carried: int = 0, facts_used: int = 0, facts_unsent: int = 0) -> Synthesis:
    offered = len(sentences)
    cut = offered > limits.max_sentences
    shown = tuple(sentences[:limits.max_sentences])
    if cut:
        reasons.append(f"{offered} sentences offered, {limits.max_sentences} shown: the sentence bound was reached")
    # A bound of the limits cut something: rounds, a round's room beside a refine answer, the facts an answer call
    # could hold, or sentences. A failed round leaves passages unread too, but that is a failure, not a bound.
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
    # Notes kept over every round: a refined answer's rewrites are counted as the rounds kept them.
    kept = sum(r.notes_kept for r in rounds)
    return Synthesis(question, status, shown, tuple(rounds), len(given), read, unread, oversize, kept,
                     tally.unquoted, tally.unknown, tally.malformed, folded, fold_uncited, offered, truncated,
                     calls.count, tuple(reasons), mode, kept_prior, dropped_carried,
                     sum(r.notes_carried for r in rounds), facts_used, facts_unsent)


def passages_from_recall(items: Sequence[RecallItem]) -> tuple[Passage, ...]:
    """Recalled passages as the model will see them, cited by chunk id, in recall order."""
    return tuple(Passage(f"chunk:{item.chunk_id}", item.text, item.source, item.created_at)
                 for item in items if item.text.strip())


async def synthesize(engine: "MemoryEngine", model: ChatModel, space: str, question: str, *,
                     limits: Optional[SynthesisLimits] = None, mode: Mode = "evidence", as_of: Optional[str] = None,
                     tags: Sequence[str] = (), where: Mapping[str, str] | None = None, kind: Optional[str] = None,
                     source_prefix: Optional[str] = None, since: Optional[str] = None,
                     until: Optional[str] = None) -> Synthesis:
    """Recall up to ``limits.max_passages`` passages for ``question`` and synthesize over them in ``mode``."""
    limits = limits or SynthesisLimits()
    found = await engine.recall(space, question, limit=limits.max_passages, as_of=as_of, tags=tags, where=where,
                                kind=kind, source_prefix=source_prefix, since=since, until=until)
    return await synthesize_passages(model, question, passages_from_recall(found.items), limits=limits, mode=mode)
