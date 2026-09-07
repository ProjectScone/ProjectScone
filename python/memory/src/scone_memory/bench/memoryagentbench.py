"""MemoryAgentBench, Conflict Resolution split (research experiment 5).

The split (ai-hyz/MemoryAgentBench, Conflict_Resolution parquet) is eight
"FactConsolidation" items: a numbered list of facts in which later facts
replace earlier ones about the same subject and relation, then 100
questions each, single-hop (sh) or multi-hop (mh), at four context sizes
(6k, 32k, 64k, 262k tokens). The competency is forgetting the superseded
fact: answering "pesäpallo was created in the country of Finland" after
fact 261 said Philippines is the failure every published system makes.

Scone's reading: each fact is one episode, stored in list order with a
created_at one minute apart, so recency is real time and the engine's
chronological ordering applies. Two things are measured per question:

- retrieval: is the gold fact (the last fact carrying a gold answer) in
  the top k, and does a stale fact (an earlier fact with the same words
  before the answer) rank above it; both need no model;
- accuracy: a reader answers from the top k in chronological order and is
  scored by case-insensitive substring against the gold answers, which is
  the paper's scoring for this split as far as its code shows; the reader
  is optional and the report says which one ran, or that none did.

Nothing here is comparable to the paper's numbers unless the same reader
and the same k are stated beside them.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..memory.engine import MemoryEngine, Record
from ..providers.llm import ChatModel

FACT_LINE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")


@dataclass(frozen=True)
class ConflictItem:
    source: str  # e.g. factconsolidation_sh_6k
    facts: tuple[str, ...]  # fact bodies in list order, numbers stripped
    questions: tuple[tuple[str, tuple[str, ...]], ...]  # (question, gold answers)

    @property
    def hops(self) -> str:
        return "multi" if "_mh_" in self.source else "single"


def parse_facts(context: str) -> list[str]:
    """The numbered lines of a FactConsolidation context, in order. Lines
    that are not "N. fact" (the "Here is a list of facts:" header, blanks)
    are skipped; a gap or reorder in the numbering is an error because the
    number is the fact's time."""
    facts: list[str] = []
    for line in context.splitlines():
        m = FACT_LINE.match(line)
        if not m:
            continue
        if int(m.group(1)) != len(facts):
            raise ValueError(f"fact numbering breaks at line {line!r}: expected {len(facts)}")
        facts.append(m.group(2))
    return facts


def load_conflict_resolution(path: str | Path) -> list[ConflictItem]:
    """Rows of the split from its parquet file (needs pyarrow) or from a
    JSON array of {context, questions, answers, metadata: {source}} rows,
    the same shape, for tests and exports."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq  # optional; the parquet is the published form

        rows = pq.read_table(path).to_pylist()
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for row in rows:
        source = str((row.get("metadata") or {}).get("source") or "")
        questions = tuple(
            (str(q), tuple(str(a) for a in answers))
            for q, answers in zip(row["questions"], row["answers"])
        )
        items.append(ConflictItem(source=source, facts=tuple(parse_facts(row["context"])), questions=questions))
    return items


def content_words(text: str) -> set[str]:
    return {w for w in re.split(r"[^0-9a-zA-Z\u00c0-\u024f']+", text.lower()) if len(w) > 3}


def gold_fact(facts: Sequence[str], answers: Sequence[str], question: str = "") -> Optional[int]:
    """Index of the gold fact: among the facts carrying a gold answer, the
    ones sharing the most of the question's words (an answer string such
    as "India" or "Shahnameh" recurs in unrelated facts, so the answer
    alone does not name the fact), and of those the last, since a later
    fact about the same thing supersedes an earlier one."""
    wanted = [a.lower() for a in answers if a.strip()]
    asked = content_words(question) - {w for a in wanted for w in content_words(a)}
    best: Optional[int] = None
    best_score = -1
    for i, fact in enumerate(facts):
        low = fact.lower()
        if not any(a in low for a in wanted):
            continue
        score = len(asked & content_words(fact))
        if score >= best_score:
            best, best_score = i, score
    return best


def stale_facts(facts: Sequence[str], gold: int, answers: Sequence[str]) -> list[int]:
    """Earlier facts that say the same thing about the same subject with a
    different object: they share the gold fact's words up to the answer."""
    low = facts[gold].lower()
    cut = min((low.find(a.lower()) for a in answers if a.strip() and a.lower() in low), default=-1)
    if cut <= 0:
        return []
    prefix = low[:cut].rstrip()
    return [i for i in range(gold) if facts[i].lower().startswith(prefix) and facts[i].lower() != low]


def judge_ranking(top_sources: Sequence[str], gold: Optional[int], stale: Sequence[int]) -> tuple[bool, bool]:
    """(gold in the top k, a stale fact ranked above the gold or present
    without it) from the sources of the ranked recall items."""
    ranks = {s: r for r, s in enumerate(top_sources)}
    gold_rank = ranks.get(source_of(gold)) if gold is not None else None
    stale_ranks = [ranks[source_of(i)] for i in stale if source_of(i) in ranks]
    if not stale_ranks:
        return gold_rank is not None, False
    return gold_rank is not None, gold_rank is None or min(stale_ranks) < gold_rank


def source_of(index: Optional[int]) -> str:
    return f"fact-{index}"


READER_SYSTEM = (
    "You answer a question from a list of remembered facts. The facts are in the order they were "
    "learned, oldest first; when two facts disagree, the later one is true and the earlier one is "
    "forgotten. Reply with the answer only, no explanation. If the facts do not say, reply: unknown."
)


@dataclass
class QuestionResult:
    question: str
    answers: list[str]
    gold: Optional[int]
    stale: list[int]
    top_sources: list[str]
    gold_at_k: bool
    stale_above_gold: bool
    recall_ms: float
    answer: Optional[str] = None
    correct: Optional[bool] = None
    error: Optional[str] = None
    #: E36: the ledger claims recall returned, and whether one carries the
    #: answer (claim_gold) or an inferred one does (derived_gold). None
    #: when nothing was distilled, so there were no claims to score.
    claims: list[int] = field(default_factory=list)
    claim_gold_at_k: Optional[bool] = None
    derived_gold_at_k: Optional[bool] = None
    derived_hits: list[int] = field(default_factory=list)


@dataclass
class ConflictReport:
    source: str
    hops: str
    facts: int
    questions: int
    k: int
    reader: Optional[str]
    gold_at_k: float
    stale_above_gold: float
    answered: int
    correct: int
    accuracy: Optional[float]
    errors: int
    embedder: str
    started_at: str
    finished_at: str
    #: E36 stages: the model, extractions approved, bridges proposed by
    #: the derivation pass, whether they were approved for the run, and
    #: the claim scores as fractions (None when nothing was distilled).
    model: Optional[str] = None
    distilled: Optional[int] = None
    derived: Optional[int] = None
    derive_approved: bool = False
    claim_gold_at_k: Optional[float] = None
    derived_gold_at_k: Optional[float] = None
    results: list[QuestionResult] = field(default_factory=list)

    def as_dict(self, with_items: bool = True) -> dict:
        d = asdict(self)
        if not with_items:
            d.pop("results")
        return d


def correct_answer(answer: str, answers: Sequence[str]) -> bool:
    low = answer.lower()
    return any(a.strip() and a.lower() in low for a in answers)


async def run_conflict_resolution(
    make_engine: Callable[[], "MemoryEngine"],
    item: ConflictItem,
    reader: Optional[ChatModel] = None,
    reader_name: Optional[str] = None,
    k: int = 10,
    questions: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    model: Optional[ChatModel] = None,
    model_name: Optional[str] = None,
    distill: bool = False,
    derive: bool = False,
    derive_approve: bool = False,
) -> ConflictReport:
    """One item: every fact remembered in order, then each question
    recalled at limit k and, when a reader is given, answered from the
    top k in chronological order.

    E36 stages, each off by default: ``distill`` extracts claims from
    every statement with ``model`` and approves them for the run (the
    split has no reviewer); ``derive`` then runs one derivation pass over
    them, leaving its bridges proposed as they would arrive in production,
    or approved with ``derive_approve`` to measure the ceiling. Recall's
    claims are scored beside its episodes either way."""
    import asyncio

    started = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    engine = make_engine()
    if asyncio.iscoroutine(engine) or isinstance(engine, asyncio.Future):
        engine = await engine
    space = "facts"
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    await engine.remember_many(space, [
        Record(
            content=fact, kind="note", source=source_of(i),
            created_at=(base + timedelta(minutes=i)).isoformat().replace("+00:00", "Z"),
        )
        for i, fact in enumerate(item.facts)
    ])
    distilled: Optional[int] = None
    derived: Optional[int] = None
    if model is not None and distill:
        from ..ingestion.distill import Distiller

        distilled = 0
        for outcome in await Distiller(engine, model).distill_pending(space, limit=len(item.facts)):
            for fact in outcome.added:
                if fact.status == "proposed":
                    await engine.approve(space, fact.fact_id)
                distilled += 1
        if derive:
            from ..ingestion.derive import Deriver

            bridges = await Deriver(engine, model).derive(space, limit_groups=len(item.facts))
            derived = len(bridges.proposed)
            if derive_approve:
                for fact in bridges.proposed:
                    await engine.approve(space, fact.fact_id)
    asked = list(item.questions)[:questions] if questions else list(item.questions)
    results: list[QuestionResult] = []
    for n, (question, answers) in enumerate(asked, 1):
        gold = gold_fact(item.facts, answers, question)
        stale = stale_facts(item.facts, gold, answers) if gold is not None else []
        t0 = time.perf_counter()
        pack = await engine.recall(space, question, limit=k)
        ms = round((time.perf_counter() - t0) * 1000, 3)
        top = [i.source or "" for i in pack.items]
        at_k, above = judge_ranking(top, gold, stale)
        result = QuestionResult(question, list(answers), gold, stale, top, at_k, above, ms)
        if distilled is not None:
            claims = list(pack.facts)
            carrying = [c for c in claims if correct_answer(f"{c.subject} {c.predicate} {c.object}", answers)]
            result.claims = [c.fact_id for c in claims]
            result.claim_gold_at_k = bool(carrying)
            result.derived_hits = [c.fact_id for c in carrying if c.origin == "inferred"]
            result.derived_gold_at_k = bool(result.derived_hits)
        if reader is not None:
            # Oldest first, as the system prompt promises; the rank is not shown.
            ordered = sorted(pack.items, key=lambda i: i.created_at)
            listing = "\n".join(f"- {i.text}" for i in ordered) or "- (nothing remembered matches)"
            if distilled is not None and pack.facts:
                listing += "\n" + "\n".join(f"- claim: {c.subject} {c.predicate} {c.object}" for c in pack.facts)
            try:
                result.answer = await reader.complete(READER_SYSTEM, f"Facts:\n{listing}\n\nQuestion: {question}")
                result.correct = correct_answer(result.answer, answers)
            except Exception as e:  # noqa: BLE001 - one failed call is counted, not fatal
                result.error = f"{type(e).__name__}: {e}"
        results.append(result)
        if progress:
            progress(n, len(asked))
    for store in (engine.documents, engine.vectors, engine.events):
        if store is not None and hasattr(store, "close"):
            try:
                await store.close()
            except Exception:  # noqa: BLE001
                pass
    n = len(results)
    answered = [r for r in results if r.correct is not None]
    return ConflictReport(
        source=item.source, hops=item.hops, facts=len(item.facts), questions=n, k=k,
        reader=reader_name if reader is not None else None,
        gold_at_k=round(sum(r.gold_at_k for r in results) / n, 4) if n else 0.0,
        stale_above_gold=round(sum(r.stale_above_gold for r in results) / n, 4) if n else 0.0,
        answered=len(answered), correct=sum(1 for r in answered if r.correct),
        accuracy=round(sum(1 for r in answered if r.correct) / len(answered), 4) if answered else None,
        errors=sum(1 for r in results if r.error), embedder=engine.embedder.id,
        started_at=started, finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        model=model_name if model is not None and distill else None,
        distilled=distilled, derived=derived, derive_approved=bool(derive and derive_approve),
        claim_gold_at_k=round(sum(bool(r.claim_gold_at_k) for r in results) / n, 4) if n and distilled is not None else None,
        derived_gold_at_k=round(sum(bool(r.derived_gold_at_k) for r in results) / n, 4) if n and distilled is not None else None,
        results=results,
    )
