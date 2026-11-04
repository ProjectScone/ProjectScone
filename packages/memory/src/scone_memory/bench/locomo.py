"""LoCoMo: very long conversations between two people, and questions about them.

The data (Maharana et al., ACL 2024; snap-research/locomo, ``data/locomo10.json``)
is ten conversations of 19 to 35 dated sessions and 1,986 questions. It is
licensed CC BY-NC 4.0: it is fetched by whoever runs the bench and never
shipped with Scone.

Each question carries a category number. The original scorer
(``task_eval/evaluation.py``) treats them as follows, and the names are the
ones later papers use, checked against the questions themselves:

- 1 multi-hop: the answer is comma-separated parts, each scored by the best
  matching part of the prediction;
- 2 temporal: 246 of the 321 ask "When";
- 3 open-domain: inference beyond the text; the answer is cut at its first
  ``;`` before scoring;
- 4 single-hop;
- 5 adversarial: the question pins one speaker's event on the other, or
  asks about something that never happened. 444 of the 446 have no answer,
  only the trap (``adversarial_answer``). A reply is right only when it says
  the information is not there, which the original scorer tests as
  "no information available" or "not mentioned" in the reply. This is the
  category that measures hallucination.

Scone's reading: each turn is one episode, "<speaker>: <text>" (with the
caption of a shared photo), whose source is its dialogue id ("D3:12") and
whose time is its session's time plus a minute per turn. Per question:

- retrieval, with no model: the share of the evidence turns in the top k,
  and whether all of them are;
- the answer, when a reader is given: the original token F1 with Porter
  stems, per category;
- a judgment, when a judge is given: CORRECT or WRONG against the answer.
  A judge is never reported alone. Each judged question also hands the
  judge a known-wrong answer (another question's answer in the same
  conversation and category, sharing no word with the right one), and the
  share the judge accepts is reported beside its score. One audit found the
  usual LoCoMo judge accepting most wrong answers; a judged number without
  its false-accept rate cannot be read.

Evidence ids are read as written where they can be: "D8:6; D9:17" is two
ids, "D30:05" is "D30:5". An id that names no turn is counted, not guessed,
and a question left with no readable evidence is out of the retrieval
numbers and counted there too.
"""

from __future__ import annotations

import asyncio
import json
import re
import string
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Coroutine, Optional, Sequence

from ..memory.engine import MemoryEngine, Record
from ..providers.llm import ChatModel
from .evaluators import _verdict

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
#: What the original scorer looks for in a reply to an adversarial question.
NOT_AVAILABLE = ("no information available", "not mentioned")
SESSION_TIME = "%I:%M %p on %d %B, %Y"
EVIDENCE_ID = re.compile(r"D:?(\d+):(\d+)")


@dataclass(frozen=True)
class Turn:
    dia_id: str
    speaker: str
    text: str
    when: datetime


@dataclass(frozen=True)
class Question:
    question: str
    category: int
    #: None for the adversarial questions that only carry a trap.
    answer: Optional[str]
    evidence: tuple[str, ...]
    #: Evidence entries that name no turn of this conversation.
    unreadable_evidence: int = 0
    trap: Optional[str] = None


@dataclass(frozen=True)
class Conversation:
    sample_id: str
    speakers: tuple[str, str]
    turns: tuple[Turn, ...]
    questions: tuple[Question, ...]


def parse_session_time(text: str) -> datetime:
    """"1:56 pm on 8 May, 2023" as a UTC time. Anything else is an error:
    the session time is every turn's time."""
    return datetime.strptime(text.strip(), SESSION_TIME).replace(tzinfo=timezone.utc)


def evidence_ids(raw: Sequence[str], known: set[str]) -> tuple[tuple[str, ...], int]:
    """The turn ids a question's evidence names, in order and once each, and
    how many entries named no known turn."""
    found: list[str] = []
    unreadable = 0
    for entry in raw:
        parts = [part for part in re.split(r"[;\s]+", str(entry)) if part]
        for part in parts or [""]:
            match = EVIDENCE_ID.fullmatch(part)
            dia_id = f"D{int(match.group(1))}:{int(match.group(2))}" if match else None
            if dia_id is None or dia_id not in known:
                unreadable += 1
            elif dia_id not in found:
                found.append(dia_id)
    return tuple(found), unreadable


def turn_text(turn: dict[str, object]) -> str:
    text = f"{turn['speaker']}: {turn.get('text', '')}".rstrip()
    caption = turn.get("blip_caption")
    return f"{text} [shares a photo of {caption}]" if isinstance(caption, str) and caption.strip() else text


def load_locomo(path: str | Path) -> list[Conversation]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("LoCoMo data is a list of conversations")
    conversations: list[Conversation] = []
    for row in rows:
        conversation = row["conversation"]
        sessions = sorted((int(key.split("_")[1]) for key in conversation
                           if re.fullmatch(r"session_\d+", key) and isinstance(conversation[key], list)))
        turns: list[Turn] = []
        for number in sessions:
            start = parse_session_time(conversation[f"session_{number}_date_time"])
            for index, turn in enumerate(conversation[f"session_{number}"]):
                turns.append(Turn(str(turn["dia_id"]), str(turn["speaker"]), turn_text(turn), start + timedelta(minutes=index)))
        if len({turn.dia_id for turn in turns}) != len(turns):
            raise ValueError(f"{row['sample_id']}: a dialogue id appears twice")
        known = {turn.dia_id for turn in turns}
        questions: list[Question] = []
        for qa in row["qa"]:
            category = int(qa["category"])
            if category not in CATEGORIES:
                raise ValueError(f"{row['sample_id']}: unknown category {category}")
            evidence, unreadable = evidence_ids(qa.get("evidence") or [], known)
            answer = qa.get("answer")
            trap = qa.get("adversarial_answer")
            questions.append(Question(str(qa["question"]), category, None if answer is None else str(answer),
                                      evidence, unreadable, None if trap is None else str(trap)))
        conversations.append(Conversation(str(row["sample_id"]), (str(conversation["speaker_a"]), str(conversation["speaker_b"])),
                                          tuple(turns), tuple(questions)))
    return conversations


# The original scorer, ported line for line (task_eval/evaluation.py).

def normalize_answer(text: str) -> str:
    text = text.replace(",", "").lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def _stemmer() -> Callable[[str], str]:
    try:
        from nltk.stem import PorterStemmer
    except ImportError as error:  # the answer score needs it; retrieval does not
        raise RuntimeError("LoCoMo's answer F1 uses NLTK's Porter stemmer; install nltk") from error
    return PorterStemmer().stem


def token_f1(prediction: str, truth: str, stem: Optional[Callable[[str], str]] = None) -> float:
    stem = stem or _stemmer()
    predicted = [stem(word) for word in normalize_answer(prediction).split()]
    expected = [stem(word) for word in normalize_answer(truth).split()]
    same = sum((Counter(predicted) & Counter(expected)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(predicted), same / len(expected)
    return 2 * precision * recall / (precision + recall)


def multi_f1(prediction: str, truth: str, stem: Optional[Callable[[str], str]] = None) -> float:
    """Each comma-separated part of the answer, scored by its best match
    among the parts of the prediction, averaged."""
    stem = stem or _stemmer()
    predictions = [part.strip() for part in prediction.split(",")]
    truths = [part.strip() for part in truth.split(",")]
    return sum(max(token_f1(p, t, stem) for p in predictions) for t in truths) / len(truths)


def official_score(category: int, prediction: str, answer: Optional[str], stem: Optional[Callable[[str], str]] = None) -> float:
    if category == 5:
        low = prediction.lower()
        return 1.0 if any(phrase in low for phrase in NOT_AVAILABLE) else 0.0
    if answer is None:
        raise ValueError(f"a category {category} question has no answer")
    if category == 3:
        answer = answer.split(";")[0].strip()
    return multi_f1(prediction, answer, stem) if category == 1 else token_f1(prediction, answer, stem)


READER_SYSTEM = (
    "You answer a question about a long conversation between two people from the turns of it you "
    "remember. Each turn is dated by its session. Use only these turns. Resolve relative times "
    "(\"yesterday\", \"last week\") against the session date. Check who said or did what: if the "
    "question attributes something to the wrong person, or the turns do not say, reply exactly: "
    "Not mentioned. Otherwise reply with a short answer only, no explanation."
)
JUDGE_SYSTEM = (
    "You grade an answer to a question about a conversation against the gold answer. It is CORRECT "
    "when it states the same fact as the gold answer, in any wording; a date may be written another "
    "way if it is the same date. It is WRONG when it states a different fact, adds a claim the gold "
    "answer contradicts, or does not answer. Reply with exactly one JSON object: "
    '{"verdict": "CORRECT"} or {"verdict": "WRONG"}.'
)


def decoy_for(questions: Sequence[Question], index: int, stem: Optional[Callable[[str], str]] = None) -> Optional[str]:
    """A known-wrong answer for question ``index``: the next answer, cycling,
    in the same category that shares no stemmed word with the right one."""
    target = questions[index]
    if target.answer is None:
        return None
    for step in range(1, len(questions)):
        other = questions[(index + step) % len(questions)]
        if other.category == target.category and other.answer and token_f1(other.answer, target.answer, stem) == 0.0:
            return other.answer
    return None


async def judge_answer(judge: ChatModel, question: str, gold: str, answer: str) -> Optional[bool]:
    """True for CORRECT, False for WRONG, None when the reply cannot be read."""
    reply = await judge.complete(JUDGE_SYSTEM, f"Question: {question}\nGold answer: {gold}\nAnswer: {answer}")
    verdict = _verdict(reply)
    value = verdict.get("verdict") if verdict else None
    return {"CORRECT": True, "WRONG": False}.get(value.strip().upper()) if isinstance(value, str) else None


@dataclass
class QuestionResult:
    sample_id: str
    question: str
    category: int
    answer: Optional[str]
    evidence: list[str]
    top_sources: list[str]
    #: None when the question has no readable evidence.
    evidence_recall: Optional[float]
    all_evidence: Optional[bool]
    recall_ms: float
    prediction: Optional[str] = None
    score: Optional[float] = None
    judged_correct: Optional[bool] = None
    decoy: Optional[str] = None
    decoy_accepted: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class LocomoReport:
    conversations: int
    questions: int
    k: int
    reader: Optional[str]
    judge: Optional[str]
    embedder: str
    started_at: str
    finished_at: str
    #: Per category name, and "all", "all but adversarial".
    scores: dict[str, dict[str, Optional[float]]]
    evidence_unreadable: int
    no_readable_evidence: int
    #: Share of known-wrong answers the judge called CORRECT; None with no judge.
    judge_false_accept: Optional[float] = None
    judge_audited: int = 0
    judge_unreadable: int = 0
    errors: int = 0
    results: list[QuestionResult] = field(default_factory=list)

    def as_dict(self, with_items: bool = True) -> dict:
        d = asdict(self)
        if not with_items:
            d.pop("results")
        return d


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def summarize(results: Sequence[QuestionResult]) -> dict[str, dict[str, Optional[float]]]:
    groups: dict[str, list[QuestionResult]] = {name: [r for r in results if r.category == number] for number, name in CATEGORIES.items()}
    groups["all"] = list(results)
    groups["all but adversarial"] = [r for r in results if r.category != 5]
    table: dict[str, dict[str, Optional[float]]] = {}
    for name, rows in groups.items():
        retrieved = [r for r in rows if r.evidence_recall is not None]
        scored = [r for r in rows if r.score is not None]
        judged = [r for r in rows if r.judged_correct is not None]
        table[name] = {
            "n": float(len(rows)),
            "evidence_recall": _mean([r.evidence_recall for r in retrieved if r.evidence_recall is not None]),
            "all_evidence": _mean([1.0 if r.all_evidence else 0.0 for r in retrieved]),
            "score": _mean([r.score for r in scored if r.score is not None]),
            "answered": float(len(scored)),
            "judged_correct": _mean([1.0 if r.judged_correct else 0.0 for r in judged]),
            "judged": float(len(judged)),
        }
    return table


async def run_locomo(
    make_engine: Callable[[], "asyncio.Future[MemoryEngine] | Coroutine[object, object, MemoryEngine] | MemoryEngine"],
    conversations: Sequence[Conversation],
    reader: Optional[ChatModel] = None,
    reader_name: Optional[str] = None,
    judge: Optional[ChatModel] = None,
    judge_name: Optional[str] = None,
    k: int = 10,
    questions: Optional[int] = None,
    categories: Optional[set[int]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    concurrency: int = 1,
) -> LocomoReport:
    """Every conversation in its own engine: all turns remembered, then each
    question recalled at limit k, answered from the top k in time order when
    a reader is given, and judged with an audit when a judge is given.

    Up to `concurrency` questions of a conversation are in flight at once;
    results keep the questions' order whatever order they finish in. Above 1,
    a question's recall_ms includes waiting on the others."""
    if judge is not None and reader is None:
        raise ValueError("a judge grades a reader's answers; give a reader too")
    if concurrency < 1:
        raise ValueError("concurrency is a number of questions in flight, at least 1")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    stem = _stemmer() if reader is not None else None
    results: list[QuestionResult] = []
    embedder_id = ""
    # The first N of each conversation, then the categories asked for; every
    # count in the report is over these questions alone.
    chosen = {c.sample_id: [q for q in (c.questions[:questions] if questions else c.questions)
                            if not categories or q.category in categories] for c in conversations}
    unreadable = sum(q.unreadable_evidence for asked in chosen.values() for q in asked)
    total = sum(len(asked) for asked in chosen.values())
    done = 0
    for conversation in conversations:
        engine = make_engine()
        if asyncio.iscoroutine(engine) or isinstance(engine, asyncio.Future):
            engine = await engine
        embedder_id = engine.embedder.id
        space = conversation.sample_id
        turns = {turn.dia_id: turn for turn in conversation.turns}
        await engine.remember_many(space, [
            Record(content=turn.text, kind="conversation", source=turn.dia_id, dedup_key=turn.dia_id,
                   created_at=turn.when.isoformat().replace("+00:00", "Z"))
            for turn in conversation.turns
        ])
        # Every question of this conversation finishes before the loop moves
        # on, so the closure sees this conversation's engine and turns.
        async def ask(item: Question) -> QuestionResult:
            t0 = time.perf_counter()
            pack = await engine.recall(space, item.question, limit=k)
            ms = round((time.perf_counter() - t0) * 1000, 3)
            top: list[str] = []
            for found in pack.items:
                if found.source and found.source not in top:
                    top.append(found.source)
            hits = sum(1 for dia_id in item.evidence if dia_id in top)
            result = QuestionResult(
                conversation.sample_id, item.question, item.category, item.answer, list(item.evidence), top,
                round(hits / len(item.evidence), 4) if item.evidence else None,
                hits == len(item.evidence) if item.evidence else None, ms)
            if reader is not None:
                ordered = sorted((turns[s] for s in top if s in turns), key=lambda turn: turn.when)
                listing = "\n".join(f"[{turn.when.day} {turn.when:%B %Y}] {turn.text}" for turn in ordered) \
                    or "(nothing remembered matches)"
                try:
                    result.prediction = (await reader.complete(READER_SYSTEM, f"Turns:\n{listing}\n\nQuestion: {item.question}")).strip()
                    result.score = round(official_score(item.category, result.prediction, item.answer, stem), 4)
                    # Adversarial questions are scored by the phrase test alone, as
                    # LoCoMo's scorer does, even the two whose rows carry an answer.
                    if judge is not None and item.answer is not None and item.category != 5:
                        result.judged_correct = await judge_answer(judge, item.question, item.answer, result.prediction)
                        result.decoy = decoy_for(conversation.questions, conversation.questions.index(item), stem)
                        if result.decoy is not None:
                            result.decoy_accepted = await judge_answer(judge, item.question, item.answer, result.decoy)
                except Exception as error:  # noqa: BLE001 - one failed call is counted, not fatal
                    result.error = f"{type(error).__name__}: {error}"
            return result

        gate = asyncio.Semaphore(concurrency)

        async def bounded(item: Question) -> QuestionResult:
            nonlocal done
            async with gate:
                result = await ask(item)
            done += 1
            if progress:
                progress(done, total)
            return result

        results.extend(await asyncio.gather(*(bounded(item) for item in chosen[conversation.sample_id])))
        for store in (engine.documents, engine.vectors, engine.events):
            if store is not None and hasattr(store, "close"):
                try:
                    await store.close()
                except Exception:  # noqa: BLE001
                    pass
    audited = [r for r in results if r.decoy is not None and r.decoy_accepted is not None]
    judged_rows = [r for r in results
                   if judge is not None and r.answer is not None and r.category != 5 and r.prediction is not None]
    return LocomoReport(
        conversations=len(conversations), questions=len(results), k=k,
        reader=reader_name if reader is not None else None, judge=judge_name if judge is not None else None,
        embedder=embedder_id, started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        scores=summarize(results), evidence_unreadable=unreadable,
        no_readable_evidence=sum(1 for r in results if r.evidence_recall is None),
        judge_false_accept=_mean([1.0 if r.decoy_accepted else 0.0 for r in audited]) if judge is not None else None,
        judge_audited=len(audited),
        judge_unreadable=sum(1 for r in judged_rows if r.judged_correct is None)
        + sum(1 for r in judged_rows if r.decoy is not None and r.decoy_accepted is None),
        errors=sum(1 for r in results if r.error), results=results,
    )
