"""BEAM: memory over conversations of 100K to 10M tokens, scored by rubric.

BEAM ("Beyond a Million Tokens", Tavakoli et al., arXiv 2510.27246; data
at Hugging Face ``Mohammadta/BEAM``, CC BY-SA 4.0, fetched separately and
never shipped here) gives each long user-assistant conversation twenty
probing questions, two for each of ten memory abilities: abstention,
contradiction resolution, event ordering, information extraction,
instruction following, knowledge update, multi-session reasoning,
preference following, summarization, temporal reasoning.

The data is read as written:

- ``chat`` is a list of sessions, each a list of messages with ``id``,
  ``role`` and ``content``; a session's first message carries a
  ``time_anchor`` like ``March-15-2024`` that the rest inherit. The
  ``->-> s,t`` index marker at the end of some messages is not text and is
  removed, whatever follows it.
- ``probing_questions`` is a Python literal (the dataset card says to read
  it with ``ast.literal_eval``, which evaluates no code), or already a
  mapping. Each question carries its ``rubric`` (the nuggets an answer is
  scored on) and, for every ability but abstention, ``source_chat_ids``:
  the messages holding the answer, as a list, a nested list, or a mapping
  from a role (``original_info``, ``updated_info``, ...) to ids. An id that
  names no message is counted, never silently dropped.

What Scone reports:

- **Evidence recall at k** over those messages, from retrieval alone.
- **Stale-only**, for knowledge-update questions: the original fact was
  recalled and its update was not. That is the state in which a memory
  answers confidently with what is no longer true.
- With a reader, the **nugget score** the paper uses: a judge marks each
  rubric item 0, 0.5 or 1 and the marks are averaged. For event ordering
  the paper's own measure, **Kendall tau-b** between the rubric's order
  and the order the answer gives the events it mentions, is reported in
  its own column with the share of events mentioned at all.
- **The judge's credit for known-wrong answers**: each nugget-scored
  question with a factual reference also has the reference of a
  same-ability question from another conversation, about another user's
  life, scored against its rubric. A judge that gives that credit makes
  every score above it worth less, so the two are reported together.
- With ``directives``, each user message is read for standing directives
  (``memory.directives``) once the conversation is remembered, and the
  reader sees their block above the recalled messages, as a conversation
  would on every turn. **In context** is then the share of a question's
  evidence messages that reached the reader by either path, beside
  evidence recall, which stays retrieval's alone.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Coroutine, Iterator, Optional, Sequence

from ..ingestion.records import Record
from ..memory.directives import (DEFAULT_BLOCK_BYTES, MAX_BLOCK_BYTES, MAX_BLOCK_LIMIT, MIN_BLOCK_BYTES,
                                 record_directives, standing_directives)
from ..providers.llm import ChatModel
from .evaluators import _verdict, dated_line

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

ABILITIES = ("abstention", "contradiction_resolution", "event_ordering", "information_extraction",
             "instruction_following", "knowledge_update", "multi_session_reasoning", "preference_following",
             "summarization", "temporal_reasoning")
#: Where each ability keeps its reference: the first of these a question has.
REFERENCE_FIELDS = ("answer", "ideal_answer", "ideal_response", "ideal_summary", "expected_compliance")
#: References that state facts, and so make a meaningful wrong answer for
#: another question; instruction and preference questions describe a
#: behaviour instead, and abstention references all say "no information".
FACTUAL_FIELDS = ("answer", "ideal_answer", "ideal_summary")
TIME_ANCHOR = "%B-%d-%Y"
#: The index annotation some messages end with: ``->-> 1,2``, and in the
#: data also ``->-> 2,N/A``, ``->-> 5,14)`` and ``->-> 2,22, 24``.
MARKER = re.compile(r"\s*->->[^\n]*\s*\Z")
NUGGET_MARKS = (0, 0.5, 1)


@dataclass(frozen=True)
class Message:
    id: int
    role: str
    text: str
    when: datetime


@dataclass(frozen=True)
class Probe:
    ability: str
    question: str
    rubric: tuple[str, ...]
    reference: Optional[str]
    #: True when the reference states facts (see FACTUAL_FIELDS).
    factual: bool
    evidence: tuple[int, ...]
    #: The roles a mapping of source ids named, each with its ids.
    roles: tuple[tuple[str, tuple[int, ...]], ...] = ()
    unreadable_evidence: int = 0


@dataclass(frozen=True)
class BeamConversation:
    conversation_id: str
    messages: tuple[Message, ...]
    probes: tuple[Probe, ...]


def parse_time_anchor(text: str) -> datetime:
    """``March-15-2024`` as midnight UTC; any other form raises."""
    return datetime.strptime(text.strip(), TIME_ANCHOR).replace(tzinfo=timezone.utc)


def _leaves(value: object) -> Iterator[object]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    elif value is not None:
        yield value


def source_ids(raw: object, known: set[int]) -> tuple[tuple[int, ...], tuple[tuple[str, tuple[int, ...]], ...], int]:
    """The message ids ``source_chat_ids`` names, in order and once each;
    the roles of a mapping with their ids; and how many entries name no
    message."""
    ids: list[int] = []
    unreadable = 0
    for leaf in _leaves(raw):
        if type(leaf) is int and leaf in known:
            if leaf not in ids:
                ids.append(leaf)
        else:
            unreadable += 1
    roles: list[tuple[str, tuple[int, ...]]] = []
    if isinstance(raw, dict):
        for role, value in raw.items():
            roles.append((str(role), tuple(leaf for leaf in _leaves(value) if type(leaf) is int and leaf in known)))
    return tuple(ids), tuple(roles), unreadable


def _rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("reading BEAM's parquet files needs pyarrow; or pass the rows as JSON") from error
        return list(pq.read_table(path).to_pylist())
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError("BEAM JSON is a list of conversation rows")
    return rows


def _messages(chat: object) -> list[Message]:
    if not isinstance(chat, list):
        raise ValueError("a BEAM chat is a list of sessions")
    messages: list[Message] = []
    anchor: Optional[datetime] = None
    since_anchor = 0
    seen: set[int] = set()
    for session in chat:
        if not isinstance(session, list):
            raise ValueError("a BEAM session is a list of messages")
        for item in session:
            if not isinstance(item, dict) or type(item.get("id")) is not int or not isinstance(item.get("content"), str):
                raise ValueError("a BEAM message has an integer id and text content")
            if item.get("time_anchor"):
                stated = parse_time_anchor(str(item["time_anchor"]))
                # A restated anchor keeps the count, so its messages stay in order.
                if stated != anchor:
                    anchor, since_anchor = stated, 0
            if anchor is None:
                raise ValueError(f"message {item['id']} comes before any time anchor")
            if item["id"] in seen:
                raise ValueError(f"message id {item['id']} appears twice")
            seen.add(item["id"])
            role = str(item.get("role") or "unknown")
            # Seconds since the anchor keep its messages in order within its day.
            messages.append(Message(item["id"], role, f"{role}: {MARKER.sub('', item['content'])}",
                                    anchor + timedelta(seconds=since_anchor)))
            since_anchor += 1
    return messages


def _probes(raw: object, known: set[int]) -> list[Probe]:
    questions = ast.literal_eval(raw) if isinstance(raw, str) else raw
    if not isinstance(questions, dict):
        raise ValueError("BEAM probing questions are a mapping from ability to questions")
    probes: list[Probe] = []
    for ability in questions:
        if ability not in ABILITIES:
            raise ValueError(f"unknown BEAM ability {ability!r}")
        for item in questions[ability]:
            if not isinstance(item, dict) or not isinstance(item.get("question"), str):
                raise ValueError(f"a {ability} question has no question text")
            rubric = item.get("rubric")
            if not isinstance(rubric, list) or not rubric or not all(isinstance(nugget, str) for nugget in rubric):
                raise ValueError(f"a {ability} question has no rubric")
            found = next((name for name in REFERENCE_FIELDS if isinstance(item.get(name), str)), None)
            ids, roles, unreadable = source_ids(item.get("source_chat_ids"), known)
            probes.append(Probe(ability, item["question"], tuple(rubric), item[found] if found else None,
                                found in FACTUAL_FIELDS, ids, roles, unreadable))
    return probes


def load_beam(path: str | Path) -> list[BeamConversation]:
    conversations: list[BeamConversation] = []
    for row in _rows(path):
        if any(c.conversation_id == str(row.get("conversation_id")) for c in conversations):
            # Each conversation is its own memory space, named by its id.
            raise ValueError(f"conversation id {row.get('conversation_id')!r} appears twice")
        messages = _messages(row.get("chat"))
        known = {message.id for message in messages}
        conversations.append(BeamConversation(str(row.get("conversation_id")), tuple(messages),
                                              tuple(_probes(row.get("probing_questions"), known))))
    return conversations


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Kendall's tau-b between two rankings of the same items; None when
    either ranking has no untied pair to compare."""
    if len(x) != len(y):
        raise ValueError("both rankings rank the same items")
    concordant = discordant = tied_x = tied_y = pairs = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            pairs += 1
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0:
                tied_x += 1
            if dy == 0:
                tied_y += 1
            if dx and dy:
                if (dx > 0) == (dy > 0):
                    concordant += 1
                else:
                    discordant += 1
    denominator = math.sqrt((pairs - tied_x) * (pairs - tied_y))
    return (concordant - discordant) / denominator if denominator else None


READER_SYSTEM = (
    "You are an assistant with memory of a long conversation with this user. Below are messages recalled from it, "
    "oldest first, each dated. Answer the user's new message from them. If they do not contain what is asked, say "
    "there is no information about it in our conversation. If they contradict each other on what is asked, point "
    "out the contradiction and ask which is correct. Follow any instruction or preference the user stated in them.")

NUGGET_SYSTEM = (
    "You grade a response to a user's question against numbered rubric items. For each item, score 1 if the "
    "response fully satisfies it, 0.5 if it partly does, 0 if it does not. Grade every item. Reply with exactly one "
    'JSON object: {"scores": [{"index": <integer>, "score": 0 | 0.5 | 1}]}.')

ORDER_SYSTEM = (
    "You align a response with numbered reference events. For each event, give the position (1 for the first) at "
    "which the response mentions it, or null if it does not mention it. Reply with exactly one JSON object: "
    '{"positions": [{"index": <integer>, "position": <integer or null>}]}.')


def _numbered(items: Sequence[str]) -> str:
    return "\n".join(f"[{index}] {item}" for index, item in enumerate(items))


async def nugget_marks(judge: ChatModel, question: str, rubric: Sequence[str], response: str) -> Optional[list[float]]:
    """The judge's 0, 0.5 or 1 for each rubric item, or None when its reply
    does not give exactly one such mark per item."""
    verdict = _verdict(await judge.complete(NUGGET_SYSTEM, f"Question:\n{question}\n\nRubric:\n{_numbered(rubric)}\n\n"
                                                           f"Response:\n{response}"))
    rows = verdict.get("scores") if verdict else None
    if not isinstance(rows, list):
        return None
    marks: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("index")) is not int or not 0 <= row["index"] < len(rubric):
            return None
        mark = row.get("score")
        # A second mark for one item leaves which one counts unsaid.
        if isinstance(mark, bool) or mark not in NUGGET_MARKS or row["index"] in marks:
            return None
        marks[row["index"]] = float(mark)
    return [marks[i] for i in range(len(rubric))] if len(marks) == len(rubric) else None


async def event_positions(judge: ChatModel, events: Sequence[str], response: str) -> Optional[list[Optional[int]]]:
    """Where the response mentions each reference event (None if it does
    not), or None when the judge's reply does not say for every event."""
    verdict = _verdict(await judge.complete(ORDER_SYSTEM, f"Reference events:\n{_numbered(events)}\n\nResponse:\n{response}"))
    rows = verdict.get("positions") if verdict else None
    if not isinstance(rows, list):
        return None
    positions: dict[int, Optional[int]] = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("index")) is not int or not 0 <= row["index"] < len(events):
            return None
        position = row.get("position")
        if (position is not None and (type(position) is not int or position < 1)) or row["index"] in positions:
            return None
        positions[row["index"]] = position
    return [positions[i] for i in range(len(events))] if len(positions) == len(events) else None


def decoy_for(pool: Sequence[tuple[str, Probe]], index: int) -> Optional[str]:
    """The reference of the next factual question of the same ability in
    another conversation: about another user's life, so wrong for this one
    by construction. None for a question without a factual reference, or
    when no other conversation has one."""
    owner, probe = pool[index]
    if not probe.factual or probe.reference is None:
        return None
    for step in range(1, len(pool)):
        other_owner, other = pool[(index + step) % len(pool)]
        if other_owner != owner and other.ability == probe.ability and other.factual and other.reference:
            return other.reference
    return None


@dataclass
class ProbeResult:
    conversation_id: str
    ability: str
    question: str
    evidence: list[int]
    top_sources: list[int]
    evidence_recall: Optional[float]
    all_evidence: Optional[bool]
    #: Knowledge update only: the original fact recalled and its update not.
    stale_only: Optional[bool]
    recall_ms: float
    prediction: Optional[str] = None
    marks: Optional[list[float]] = None
    score: Optional[float] = None
    #: Event ordering only.
    tau_b: Optional[float] = None
    events_mentioned: Optional[float] = None
    decoy: Optional[str] = None
    decoy_score: Optional[float] = None
    error: Optional[str] = None
    #: With directives: the share of evidence messages the reader saw,
    #: recalled or in the directive block.
    in_context: Optional[float] = None


@dataclass
class BeamReport:
    conversations: int
    questions: int
    k: int
    reader: Optional[str]
    judge: Optional[str]
    embedder: str
    started_at: str
    finished_at: str
    #: Per ability, and "all".
    scores: dict[str, dict[str, Optional[float]]]
    evidence_unreadable: int
    #: Mean nugget score the judge gave known-wrong answers; None with no judge.
    judge_decoy_score: Optional[float] = None
    judge_audited: int = 0
    judge_unreadable: int = 0
    errors: int = 0
    #: With directives: the block's byte bound, and per conversation what
    #: reading its user messages recorded and what the block showed.
    directive_bytes: Optional[int] = None
    directives: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Whether each message shown to the reader carried its relative dates resolved.
    date_notes: bool = False
    results: list[ProbeResult] = field(default_factory=list)

    def as_dict(self, with_items: bool = True) -> dict:
        d = asdict(self)
        if not with_items:
            d.pop("results")
        return d


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def summarize(results: Sequence[ProbeResult]) -> dict[str, dict[str, Optional[float]]]:
    groups = {ability: [r for r in results if r.ability == ability] for ability in ABILITIES}
    groups["all"] = list(results)
    table: dict[str, dict[str, Optional[float]]] = {}
    for name, rows in groups.items():
        retrieved = [r for r in rows if r.evidence_recall is not None]
        updates = [r for r in rows if r.stale_only is not None]
        table[name] = {
            "n": float(len(rows)),
            # The questions the two evidence columns are over: abstention and
            # some others name no source message.
            "evidenced": float(len(retrieved)),
            "evidence_recall": _mean([r.evidence_recall for r in retrieved if r.evidence_recall is not None]),
            "all_evidence": _mean([1.0 if r.all_evidence else 0.0 for r in retrieved]),
            "in_context": _mean([r.in_context for r in retrieved if r.in_context is not None]),
            "stale_only": _mean([1.0 if r.stale_only else 0.0 for r in updates]),
            "score": _mean([r.score for r in rows if r.score is not None]),
            "scored": float(sum(1 for r in rows if r.score is not None)),
            "tau_b": _mean([r.tau_b for r in rows if r.tau_b is not None]),
            "events_mentioned": _mean([r.events_mentioned for r in rows if r.events_mentioned is not None]),
        }
    return table


def _stale_only(probe: Probe, top: Sequence[int]) -> Optional[bool]:
    roles = dict(probe.roles)
    if probe.ability != "knowledge_update" or not roles.get("original_info") or not roles.get("updated_info"):
        return None
    return any(i in top for i in roles["original_info"]) and not any(i in top for i in roles["updated_info"])


async def run_beam(
    make_engine: Callable[[], "asyncio.Future[MemoryEngine] | Coroutine[object, object, MemoryEngine] | MemoryEngine"],
    conversations: Sequence[BeamConversation],
    reader: Optional[ChatModel] = None,
    reader_name: Optional[str] = None,
    judge: Optional[ChatModel] = None,
    judge_name: Optional[str] = None,
    k: int = 10,
    questions: Optional[int] = None,
    abilities: Optional[set[str]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    concurrency: int = 1,
    decoys_from: Sequence[BeamConversation] = (),
    directives: bool = False,
    directive_bytes: int = DEFAULT_BLOCK_BYTES,
    date_notes: bool = False,
) -> BeamReport:
    """Every conversation in its own engine: all messages remembered, then
    each question recalled at limit k, answered from the top k in time
    order when a reader is given, and scored on its rubric, with the
    judge's credit for a known-wrong answer beside it, when a judge is.

    Up to `concurrency` questions of a conversation are in flight at once;
    results keep the questions' order. Above 1, a question's recall_ms
    includes waiting on the others.

    Decoys come from the conversations run and from `decoys_from`, so a run
    of one conversation can still audit its judge against the others.

    With `directives`, every question of a conversation is asked after all
    of it, so one directive block, bounded by `directive_bytes`, serves
    them all.

    With `date_notes`, each message shown to the reader is followed by the
    days its relative dates mean, resolved against its own date
    (`retrieval.dates.date_note`); the message's own text is unchanged."""
    if judge is not None and reader is None:
        raise ValueError("a judge scores a reader's answers; give a reader too")
    if date_notes and reader is None:
        raise ValueError("date notes are for a reader; give a reader too")
    if concurrency < 1:
        raise ValueError("concurrency is a number of questions in flight, at least 1")
    if directives and (type(directive_bytes) is not int or not MIN_BLOCK_BYTES <= directive_bytes <= MAX_BLOCK_BYTES):
        raise ValueError(f"directive_bytes is from {MIN_BLOCK_BYTES} to {MAX_BLOCK_BYTES}")
    if abilities and not abilities <= set(ABILITIES):
        raise ValueError(f"abilities are among {ABILITIES}")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    ran = {c.conversation_id for c in conversations}
    pool = [(c.conversation_id, probe)
            for c in [*conversations, *(c for c in decoys_from if c.conversation_id not in ran)] for probe in c.probes]
    where = {id(probe): index for index, (_, probe) in enumerate(pool)}
    chosen = {c.conversation_id: [p for p in (c.probes[:questions] if questions else c.probes)
                                  if not abilities or p.ability in abilities] for c in conversations}
    unreadable = sum(p.unreadable_evidence for asked in chosen.values() for p in asked)
    total = sum(len(asked) for asked in chosen.values())
    results: list[ProbeResult] = []
    unread = 0
    directive_report: dict[str, dict[str, int]] = {}
    done = 0
    embedder_id = ""
    for conversation in conversations:
        engine = make_engine()
        if asyncio.iscoroutine(engine) or isinstance(engine, asyncio.Future):
            engine = await engine
        embedder_id = engine.embedder.id
        space = conversation.conversation_id
        messages = {message.id: message for message in conversation.messages}
        added = await engine.remember_many(space, [
            Record(content=m.text, kind="conversation", source=str(m.id), dedup_key=str(m.id),
                   metadata={"role": m.role}, created_at=m.when.isoformat().replace("+00:00", "Z"))
            for m in conversation.messages
        ])
        block_text, shown_ids = "", frozenset[int]()
        if directives:
            if len(added) != len(conversation.messages):
                raise RuntimeError(f"{len(conversation.messages)} messages remembered as {len(added)} records")
            message_of = {receipt.episode_id: m.id for receipt, m in zip(added, conversation.messages)}
            tally = dict.fromkeys(("read", "added", "restated", "cut", "clipped", "text_cut"), 0)
            # Every message is offered; only those stored as the user's are read.
            for receipt in added:
                outcome = await record_directives(engine, space, receipt.episode_id)
                tally["read"] += outcome.read
                tally["added"] += outcome.added
                tally["restated"] += outcome.restated
                tally["cut"] += outcome.cut
                tally["clipped"] += outcome.clipped
                tally["text_cut"] += outcome.text_cut
            block = await standing_directives(engine, space, limit=MAX_BLOCK_LIMIT, max_bytes=directive_bytes)
            block_text = block.text
            shown_ids = frozenset(message_of[e] for e in block.episode_ids if e in message_of)
            directive_report[space] = {**tally, "held": block.held, "shown": len(block.fact_ids),
                                       "omitted": block.omitted, "bytes": len(block.text.encode())}

        # Every question of this conversation finishes before the loop moves
        # on, so the closure sees this conversation's engine and messages.
        async def ask(probe: Probe) -> ProbeResult:
            nonlocal unread
            t0 = time.perf_counter()
            pack = await engine.recall(space, probe.question, limit=k)
            ms = round((time.perf_counter() - t0) * 1000, 3)
            top: list[int] = []
            for found in pack.items:
                if found.source and found.source.isdigit() and int(found.source) not in top:
                    top.append(int(found.source))
            hits = sum(1 for i in probe.evidence if i in top)
            result = ProbeResult(conversation.conversation_id, probe.ability, probe.question, list(probe.evidence), top,
                                 round(hits / len(probe.evidence), 4) if probe.evidence else None,
                                 hits == len(probe.evidence) if probe.evidence else None, _stale_only(probe, top), ms)
            if directives and probe.evidence:
                seen = shown_ids | set(top)
                result.in_context = round(sum(1 for i in probe.evidence if i in seen) / len(probe.evidence), 4)
            if reader is None:
                return result
            ordered = sorted((messages[i] for i in top if i in messages), key=lambda m: (m.when, m.id))
            listing = "\n".join(dated_line(m.when, m.text, date_notes) for m in ordered) or "(nothing remembered matches)"
            standing = f"{block_text}\n\n" if block_text else ""
            try:
                result.prediction = (await reader.complete(READER_SYSTEM, f"{standing}Recalled messages:\n{listing}\n\n"
                                                                          f"User: {probe.question}")).strip()
                if judge is None:
                    return result
                if probe.ability == "event_ordering":
                    # The paper scores ordering by tau-b alone; the rubric is the order.
                    positions = await event_positions(judge, probe.rubric, result.prediction)
                    if positions is None:
                        unread += 1
                        return result
                    aligned = [(i, p) for i, p in enumerate(positions) if p is not None]
                    result.events_mentioned = round(len(aligned) / len(probe.rubric), 4)
                    tau = kendall_tau_b([i for i, _ in aligned], [p for _, p in aligned]) if len(aligned) > 1 else None
                    result.tau_b = None if tau is None else round(tau, 4)
                    return result
                result.marks = await nugget_marks(judge, probe.question, probe.rubric, result.prediction)
                if result.marks is None:
                    unread += 1
                else:
                    result.score = round(sum(result.marks) / len(result.marks), 4)
                result.decoy = decoy_for(pool, where[id(probe)])
                if result.decoy is not None:
                    decoy_marks = await nugget_marks(judge, probe.question, probe.rubric, result.decoy)
                    if decoy_marks is None:
                        unread += 1
                    else:
                        result.decoy_score = round(sum(decoy_marks) / len(decoy_marks), 4)
            except Exception as error:  # noqa: BLE001 - one failed call is counted, not fatal
                result.error = f"{type(error).__name__}: {error}"
            return result

        gate = asyncio.Semaphore(concurrency)

        async def bounded(probe: Probe) -> ProbeResult:
            nonlocal done
            async with gate:
                result = await ask(probe)
            done += 1
            if progress:
                progress(done, total)
            return result

        results.extend(await asyncio.gather(*(bounded(probe) for probe in chosen[conversation.conversation_id])))
        for store in (engine.documents, engine.vectors, engine.events):
            if store is not None and hasattr(store, "close"):
                try:
                    await store.close()
                except Exception:  # noqa: BLE001
                    pass
    audited = [r.decoy_score for r in results if r.decoy_score is not None]
    return BeamReport(
        conversations=len(conversations), questions=len(results), k=k,
        reader=reader_name if reader is not None else None, judge=judge_name if judge is not None else None,
        embedder=embedder_id, started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        scores=summarize(results), evidence_unreadable=unreadable,
        judge_decoy_score=_mean(audited) if judge is not None else None, judge_audited=len(audited),
        judge_unreadable=unread, errors=sum(1 for r in results if r.error),
        directive_bytes=directive_bytes if directives else None, directives=directive_report,
        date_notes=date_notes, results=results,
    )
