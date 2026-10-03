"""Named-entity mentions in stored records, read into the entity graph.

Entities used to exist only as the ends of claims a model or a code reader
made, so a record naming Acme put Acme in the graph only if a distiller
happened to state something about it. A recognizer reads every record
instead, and each named thing it finds becomes a claim that the record
names it -- ``<source> scone:mentions organisation Acme`` -- quoted from
the sentence it was found in and extracted rather than stated
(``entities.mentions`` says why the kind rides on the predicate). The
projection, the entity routes, the MCP tools, duplicate suggestions and
merges then see it like any other entity, with its kind ``inferred`` from
the recognizer's label and a conflict, never an override, when another
hint disagrees.

**Values are left out.** A date, time, amount, percentage, quantity,
ordinal or cardinal is not a thing, and as a mention it would be an
attribute of the record ("notes.md mentions 3 MB"), which says nothing
anyone asks about while filling the record's page and the mention cap.
The classifier already keeps such shapes out of the graph when they are
the objects of claims; a recognizer finding them does not change what
they are. They are counted, not recorded.

**Kinds.** OntoNotes' labels map onto the existing kinds, with one added:
NORP (nationalities, religious and political groups) becomes
``nationality``, since none of the others fit and the benchmark found the
recognizer reliable there (94.3 F1). WORK_OF_ART, LAW and LANGUAGE are
recorded as entities of unknown kind rather than given kinds of their
own: each is rare, a kind is something every view and filter must then
know, and an unknown kind says exactly what is known.

**Bounded, and said so.** At most ``MAX_TEXT_CHARS`` of a record are read,
and at most ``MAX_MENTIONS`` distinct (name, kind) pairs are recorded per
record; a repeated pair is recorded once. Every bound that bit is a count
on the outcome, and on the worker's pass report, never a silence.

**Idempotent.** A mention is placed like any claim: the same name from the
same record at the same moment restates the fact already held and writes
nothing. Which records are pending is read from the ledger (no mention
fact cites them yet), plus, for records that named nothing, this
instance's memory -- as the distiller does, a fresh recorder reads those
once more and writes nothing new.

**Failures are retried, then parked.** A record counts as read only once
every mention planned for it is in the ledger. One whose writes stop
partway stays pending although some mention already cites it, and is
read again; so is one the recognizer could not read. A batch the
recognizer refuses is read again one text at a time, so one bad record
cannot hold back the rest. After ``MAX_ATTEMPTS`` failures a record is
parked: skipped, and counted as parked on every pass, so the queue moves
on and nothing is silent. Failures are kept in memory, as the
distiller's are: a record whose writes stopped partway before a restart
is not known to be incomplete after it.

The recognizer runs in the background consolidation worker, not at
ingest: the best model takes about 9 ms a sentence on a CPU, which a
document of a few hundred sentences would add to every write.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional, Protocol, Sequence, cast

from ..core import forget_after
from ..core.errors import InvalidInput, SconeError
from ..core.models import Episode
from ..core.timeutil import parse_rfc3339
from ..core.validation import check_space, entity_key, normalise_term
from ..entities.mentions import is_mention, mention_predicate

if TYPE_CHECKING:
    from ..entities.kinds import EntityKind
    from ..memory.engine import MemoryEngine

__all__ = ["DEFAULT_MODEL", "MAX_MENTIONS", "MAX_QUOTE_CHARS", "MAX_TEXT_CHARS", "VALUE_LABELS", "EntityRecognizer",
           "Mention", "MentionOutcome", "MentionPlan", "MentionRecorder", "MentionTotals", "SpacyRecognizer", "ontonotes_kind",
           "plan_mentions", "quote_for"]

#: The most accurate English pipeline in the entity-recognition benchmark
#: (89.5 F1 on held-out OntoNotes 5, ~9 ms a sentence on a CPU).
#: ``en_core_web_lg`` needs no PyTorch and is about four times faster at
#: 84.8 F1.
DEFAULT_MODEL = "en_core_web_trf"
#: Distinct (name, kind) pairs recorded from one record.
MAX_MENTIONS = 64
#: Characters of one record the recognizer reads.
MAX_TEXT_CHARS = 100_000
#: The longest sentence quoted as a mention's evidence; past it the
#: mention's own words are quoted instead.
MAX_QUOTE_CHARS = 300
#: Failures before a record is parked and no longer read by this instance.
MAX_ATTEMPTS = 3

#: OntoNotes labels for values rather than things.
VALUE_LABELS = frozenset({"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"})
_ONTONOTES: dict[str, "EntityKind"] = {
    "PERSON": "person", "ORG": "organisation", "GPE": "place", "LOC": "place", "FAC": "place",
    "PRODUCT": "product", "EVENT": "event", "NORP": "nationality",
}
_SENTENCE_ENDS = ".!?\n"


@dataclass(frozen=True)
class Mention:
    """A span of a text a recognizer read as a name: character offsets,
    half-open, into the text it was given."""

    start: int
    end: int
    text: str
    #: The recognizer's own label, such as ``ORG``.
    label: str
    #: The kind that label maps to, or None when none does.
    kind: Optional["EntityKind"]


class EntityRecognizer(Protocol):
    """Reads texts for named entities. Synchronous and CPU-bound: callers
    run it off the event loop."""

    #: Names the recognizer and its model on every pass report.
    name: str

    def recognize(self, texts: Sequence[str]) -> list[list[Mention]]: ...


def ontonotes_kind(label: str) -> Optional["EntityKind"]:
    """The kind an OntoNotes label gives, or None for a label with no kind
    here (WORK_OF_ART, LAW, LANGUAGE, a value, or one this map does not
    know)."""
    return _ONTONOTES.get(label)


class _Span(Protocol):
    start_char: int
    end_char: int
    text: str
    label_: str


class _Doc(Protocol):
    @property
    def ents(self) -> Sequence[_Span]: ...


class _Pipeline(Protocol):
    pipe_names: list[str]

    def pipe(self, texts: Iterable[str], *, batch_size: int) -> Iterable[_Doc]: ...


class SpacyRecognizer:
    """A spaCy pipeline's entity recognizer. spaCy is imported here, not
    when Scone is, so nothing loads it unless recognition is configured;
    a missing package or model is refused by name with the command that
    installs it."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, batch_size: int = 8) -> None:
        try:
            import spacy
        except ImportError:
            raise InvalidInput("SCONE_ENTITY_RECOGNIZER=spacy needs spaCy: "
                               "pip install 'scone-memory[entities]'") from None
        try:
            nlp = cast(_Pipeline, spacy.load(model_name))
        except OSError:
            raise InvalidInput(f"the spaCy model {model_name!r} is not installed: "
                               f"python -m spacy download {model_name}") from None
        if "ner" not in nlp.pipe_names:
            raise InvalidInput(f"the spaCy model {model_name!r} has no entity recognizer (ner)")
        self._nlp = nlp
        self._batch_size = batch_size
        self.name = f"spacy/{model_name}"

    def recognize(self, texts: Sequence[str]) -> list[list[Mention]]:
        return [[Mention(span.start_char, span.end_char, span.text, span.label_, ontonotes_kind(span.label_))
                 for span in doc.ents]
                for doc in self._nlp.pipe(texts, batch_size=self._batch_size)]


# -- from mentions to claims ---------------------------------------------------


@dataclass(frozen=True)
class Planned:
    """One mention to record: the name as written, its kind, and the
    quote that grounds it in the record."""

    name: str
    kind: Optional["EntityKind"]
    quote: str


@dataclass(frozen=True)
class MentionPlan:
    planned: tuple[Planned, ...]
    #: Value mentions (dates, amounts, ...) left out.
    values: int = 0
    #: Mentions of a (name, kind) pair already planned from this record.
    repeated: int = 0
    #: Distinct pairs past ``MAX_MENTIONS``, not recorded.
    cut: int = 0
    #: Mentions whose span did not hold their text and whose text is not
    #: in the record either, so nothing could quote them.
    unplaced: int = 0
    #: Mentions whose sentence and own words were both longer than
    #: ``MAX_QUOTE_CHARS``, so no bounded quote could ground them.
    too_long: int = 0
    #: Mentions quoted by their own words because their sentence was
    #: longer than ``MAX_QUOTE_CHARS``.
    narrowed: int = 0


def quote_for(text: str, start: int, end: int, *, limit: int = MAX_QUOTE_CHARS) -> Optional[str]:
    """The sentence around ``text[start:end]``, trimmed, when it is at most
    ``limit`` characters; None otherwise. Always a substring of ``text``,
    since the ledger refuses a quote its source does not hold."""
    left = max(text.rfind(mark, 0, start) for mark in _SENTENCE_ENDS) + 1
    ends = [found for mark in _SENTENCE_ENDS if (found := text.find(mark, end)) >= 0]
    right = min(ends) + 1 if ends else len(text)
    sentence = text[left:right].strip()
    return sentence if sentence and len(sentence) <= limit else None


def plan_mentions(text: str, mentions: Iterable[Mention], *, limit: int = MAX_MENTIONS,
                  quote_limit: int = MAX_QUOTE_CHARS) -> MentionPlan:
    """What one record's mentions come to, in the order they are written:
    values left out, each (name, kind) pair once, at most ``limit`` pairs,
    and every one of those counted."""
    planned: list[Planned] = []
    seen: set[tuple[str, Optional[str]]] = set()
    over: set[tuple[str, Optional[str]]] = set()
    values = repeated = unplaced = narrowed = too_long = 0
    for mention in sorted(mentions, key=lambda item: (item.start, item.end)):
        name = " ".join(mention.text.split())
        if mention.label in VALUE_LABELS:
            values += 1
            continue
        if not name:
            continue
        pair = (entity_key(name), mention.kind)
        if pair in seen or pair in over:
            repeated += 1
            continue
        if len(seen) >= limit:
            over.add(pair)
            continue
        placed = 0 <= mention.start < mention.end <= len(text) and text[mention.start:mention.end] == mention.text
        quote = quote_for(text, mention.start, mention.end, limit=quote_limit) if placed else None
        if quote is None:
            if not mention.text.strip() or mention.text not in text:
                unplaced += 1
                continue
            quote = mention.text.strip()
            if len(quote) > quote_limit:
                too_long += 1
                continue
            narrowed += int(placed)
        seen.add(pair)
        planned.append(Planned(name, mention.kind, quote))
    return MentionPlan(tuple(planned), values=values, repeated=repeated, cut=len(over), unplaced=unplaced,
                       narrowed=narrowed, too_long=too_long)


@dataclass
class MentionOutcome:
    episode_id: int
    #: Mention facts newly written.
    added: int = 0
    #: Mentions that restated a fact already held.
    restated: int = 0
    values: int = 0
    repeated: int = 0
    cut: int = 0
    unplaced: int = 0
    too_long: int = 0
    narrowed: int = 0
    #: Characters of the record past ``MAX_TEXT_CHARS``, not read.
    text_cut: int = 0
    #: Why the record was not read at all, when it was not: ``code`` for
    #: a source file, whose names the code readers already record, or
    #: ``parked`` for one that failed ``MAX_ATTEMPTS`` times.
    skipped: Optional[str] = None
    error: Optional[str] = None


def record_name(episode: Episode) -> str:
    """What a record is called as the subject of its mentions: its source,
    or the episode when it has none."""
    source = (episode.source or "").strip()
    return source or f"episode {episode.episode_id}"


def _is_code(episode: Episode) -> bool:
    from .code import code_language

    source = (episode.source or "").strip()
    return bool(source) and code_language(source.rsplit("/", 1)[-1]) is not None


class MentionRecorder:
    """Reads pending records through a recognizer and records what they
    name. ``max_mentions`` and ``max_chars`` exist so a test can make the
    bounds bite; deployments leave them alone."""

    def __init__(self, engine: "MemoryEngine", recognizer: EntityRecognizer, *, max_mentions: int = MAX_MENTIONS,
                 max_chars: int = MAX_TEXT_CHARS, max_attempts: int = MAX_ATTEMPTS) -> None:
        if max_mentions < 1 or max_chars < 1 or max_attempts < 1:
            raise ValueError("mention bounds must be positive")
        self.engine = engine
        self.recognizer = recognizer
        self.max_mentions = max_mentions
        self.max_chars = max_chars
        self.max_attempts = max_attempts
        # Both keyed by (space, episode id) and kept in memory, as the
        # distiller keeps its own.
        self._done: set[tuple[str, int]] = set()
        self._failures: dict[tuple[str, int], int] = {}

    def parked(self, space: str) -> list[int]:
        return sorted(episode for (owner, episode), count in self._failures.items()
                      if owner == space and count >= self.max_attempts)

    async def record_episode(self, space: str, episode_id: int) -> MentionOutcome:
        """Read one record now, pending or not."""
        episode = await self.engine.episode(space, episode_id)
        return (await self._record(space, [episode]))[0]

    async def record_pending(self, space: str, limit: int = 20) -> list[MentionOutcome]:
        """Read up to ``limit`` pending records, oldest first, in one call
        to the recognizer, and report every parked record as skipped. A
        record that fails is reported on its outcome and counted against
        it; it never stops the others."""
        check_space(space)
        pending = await self._pending(space)
        parked = [MentionOutcome(episode.episode_id, skipped="parked") for episode in pending
                  if self._failures.get((space, episode.episode_id), 0) >= self.max_attempts]
        waiting = [episode for episode in pending
                   if self._failures.get((space, episode.episode_id), 0) < self.max_attempts]
        return [*parked, *await self._record(space, waiting[:max(limit, 0)])]

    async def _pending(self, space: str) -> list[Episode]:
        documents = self.engine.documents
        counts = await documents.counts(space)
        if counts.episodes == 0:
            return []
        episodes = await documents.recent_episodes(space, counts.episodes)
        cited = {fact.source_episode_id for fact in await documents.list_facts(space, include_closed=True)
                 if is_mention(fact)}
        moment = parse_rfc3339(self.engine.clock())
        # A record some mention cites is read, unless its writes stopped
        # partway: then a failure is held against it and it is read again.
        fresh = [episode for episode in episodes
                 if episode.content.strip()
                 and (episode.episode_id not in cited or (space, episode.episode_id) in self._failures)
                 and (space, episode.episode_id) not in self._done
                 and not forget_after.is_due(episode.metadata, moment)]
        return sorted(fresh, key=lambda episode: (parse_rfc3339(episode.created_at), episode.episode_id))

    async def _record(self, space: str, episodes: Sequence[Episode]) -> list[MentionOutcome]:
        outcomes = {episode.episode_id: MentionOutcome(episode.episode_id) for episode in episodes}
        readable = []
        for episode in episodes:
            if _is_code(episode):
                outcomes[episode.episode_id].skipped = "code"
                self._done.add((space, episode.episode_id))
            else:
                readable.append(episode)
        if readable:
            await self.engine._living(space)
            texts = [episode.content[:self.max_chars] for episode in readable]
            found = await self._recognized(texts)
            for episode, text, mentions in zip(readable, texts, found):
                outcome = outcomes[episode.episode_id]
                outcome.text_cut = len(episode.content) - len(text)
                try:
                    if isinstance(mentions, Exception):
                        raise mentions
                    await self._write(space, episode, plan_mentions(text, mentions, limit=self.max_mentions), outcome)
                except Exception as error:  # noqa: BLE001 - one record's fault is that record's, counted
                    self._fail(space, episode.episode_id, outcome, error)
                    continue
                self._failures.pop((space, episode.episode_id), None)
                self._done.add((space, episode.episode_id))
        return [outcomes[episode.episode_id] for episode in episodes]

    def _fail(self, space: str, episode_id: int, outcome: MentionOutcome, error: Exception) -> None:
        key = (space, episode_id)
        self._failures[key] = self._failures.get(key, 0) + 1
        detail = str(error) if isinstance(error, SconeError) else ""
        outcome.error = f"{type(error).__name__}: {detail}" if detail else type(error).__name__

    async def _recognized(self, texts: Sequence[str]) -> list[list[Mention] | Exception]:
        """The recognizer's mentions for each text, or the error it raised
        on that text. A batch it refuses, or answers with the wrong count,
        is read again a text at a time, so one bad record fails alone."""
        try:
            found = await asyncio.to_thread(self.recognizer.recognize, texts)
            if len(found) == len(texts):
                return list(found)
        except Exception:  # noqa: BLE001 - retried below, one text at a time
            pass
        alone: list[list[Mention] | Exception] = []
        for text in texts:
            try:
                single = await asyncio.to_thread(self.recognizer.recognize, [text])
                if len(single) != 1:
                    raise InvalidInput(f"recognizer {self.recognizer.name} returned {len(single)} results for 1 text")
                alone.append(single[0])
            except Exception as error:  # noqa: BLE001 - this text's failure, counted against its record
                alone.append(error)
        return alone

    async def _write(self, space: str, episode: Episode, plan: MentionPlan, outcome: MentionOutcome) -> None:
        outcome.values, outcome.repeated, outcome.cut = plan.values, plan.repeated, plan.cut
        outcome.unplaced, outcome.narrowed, outcome.too_long = plan.unplaced, plan.narrowed, plan.too_long
        subject = record_name(episode)
        held: dict[str, set[int]] = {}
        for item in plan.planned:
            predicate = mention_predicate(item.kind)
            if predicate not in held:
                held[predicate] = {fact.fact_id for fact in await self.engine.documents.facts_for(
                    space, normalise_term(subject, "subject"), predicate)}
            fact = await self.engine._assert_placed(
                space, subject, predicate, item.name, valid_from=episode.created_at,
                source_episode_id=episode.episode_id, origin="extracted", quote=item.quote)
            if fact.fact_id in held[predicate]:
                outcome.restated += 1
            else:
                held[predicate].add(fact.fact_id)
                outcome.added += 1


@dataclass
class MentionTotals:
    """A pass's outcomes summed, as the worker reports them."""

    read: int = 0
    added: int = 0
    restated: int = 0
    #: Distinct pairs the per-record cap left out.
    cut: int = 0
    #: Records read only up to ``MAX_TEXT_CHARS``.
    text_cut: int = 0
    #: Mentions quoted by their own words rather than their sentence.
    narrowed: int = 0
    #: Mentions or records not recorded, by reason.
    left_out: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def _leave_out(self, reason: str, count: int) -> None:
        if count:
            self.left_out[reason] = self.left_out.get(reason, 0) + count

    @classmethod
    def of(cls, outcomes: Iterable[MentionOutcome]) -> "MentionTotals":
        totals = cls()
        for outcome in outcomes:
            if outcome.error is not None:
                totals.errors[str(outcome.episode_id)] = outcome.error[:500]
            elif outcome.skipped is not None:
                totals._leave_out(f"skipped_{outcome.skipped}", 1)
            else:
                totals.read += 1
                totals.added += outcome.added
                totals.restated += outcome.restated
                totals.cut += outcome.cut
                totals.text_cut += outcome.text_cut > 0
                totals.narrowed += outcome.narrowed
                for reason in ("values", "repeated", "unplaced", "too_long"):
                    totals._leave_out(reason, getattr(outcome, reason))
        return totals
