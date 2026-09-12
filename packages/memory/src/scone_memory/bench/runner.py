from __future__ import annotations

import asyncio
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Coroutine, Callable, Iterable, Optional, Sequence

from ..memory.engine import MemoryEngine, Record
from ..core.models import RecallItem


@dataclass(frozen=True)
class BenchItem:
    question_id: str
    question_type: str
    question: str
    question_date: str
    sessions: tuple[tuple[str, ...], ...]  # each session: "role: content" lines
    session_ids: tuple[str, ...]
    session_dates: tuple[str, ...]
    answer_session_ids: tuple[str, ...]

    @property
    def has_evidence(self) -> bool:
        return bool(self.answer_session_ids)


def iso_date(raw: str) -> str:
    """'2023/05/20 (Sat) 02:21' -> '2023-05-20T02:21:00Z', as the Rust harness does."""
    raw = raw.strip()
    if len(raw) < 10:
        return ""
    date = raw[:10].replace("/", "-")
    tail = raw.rsplit(" ", 1)[-1]
    time_part = tail if ":" in tail else "00:00"
    return f"{date}T{time_part}:00Z"


def load_items(path: str | Path) -> list[BenchItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset must be a JSON array")
    items = []
    for raw in data:
        sessions = tuple(
            tuple(f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in session)
            for session in raw.get("haystack_sessions", [])
        )
        items.append(BenchItem(
            question_id=str(raw.get("question_id", "")),
            question_type=str(raw.get("question_type", "")),
            question=str(raw.get("question", "")),
            question_date=str(raw.get("question_date", "")),
            sessions=sessions,
            session_ids=tuple(str(x) for x in raw.get("haystack_session_ids", [])),
            session_dates=tuple(iso_date(str(d)) for d in raw.get("haystack_dates", [])),
            answer_session_ids=tuple(str(x) for x in raw.get("answer_session_ids", [])),
        ))
    return items


def stratified_sample(items: Sequence[BenchItem], n: int, seed: int = 42) -> list[BenchItem]:
    """The Rust harness's proportional stratified sample, item for item.

    ``n`` items spread across question types by their population share,
    drawn with the same xorshift64 generator in the same order (types
    sorted bytewise, a swap-remove draw per pick, then a deterministic
    top-up when rounding leaves the sample short), so the Python engine
    is measured on exactly the items E21 and E22 ran on: seed 42, n=60
    or n=200. Needed because longmemeval_s is ordered by type, so head
    sampling measures one class (found 2026-08-28)."""
    mask = (1 << 64) - 1
    state = max(int(seed), 1) & mask

    def next_draw() -> int:
        nonlocal state
        state ^= (state << 13) & mask
        state ^= state >> 7
        state ^= (state << 17) & mask
        return state

    def swap_remove(pool: list[BenchItem], idx: int) -> BenchItem:
        picked = pool[idx]
        pool[idx] = pool[-1]
        pool.pop()
        return picked

    by_type: dict[str, list[BenchItem]] = {}
    for it in items:
        by_type.setdefault(it.question_type, []).append(it)
    total = max(len(items), 1)
    sample: list[BenchItem] = []
    for question_type in sorted(by_type, key=lambda t: t.encode("utf-8")):
        pool = list(by_type[question_type])
        take = (n * len(pool) + total // 2) // total
        for _ in range(min(take, len(pool))):
            sample.append(swap_remove(pool, next_draw() % len(pool)))
    chosen = {it.question_id for it in sample}
    pool = [it for it in items if it.question_id not in chosen]
    while len(sample) < n and pool:
        sample.append(swap_remove(pool, next_draw() % len(pool)))
    return sample[:n]


@dataclass
class ItemResult:
    question_id: str
    question_type: str
    has_evidence: bool
    retrieved_sessions: list[str]  # source of each returned item, in rank order
    stored_bytes: int
    returned_bytes: int
    recall_ms: float
    degraded: list[str] = field(default_factory=list)
    error: Optional[str] = None
    answer_sessions: list[str] = field(default_factory=list)
    #: Experiment 9: the best cosine the vector lane saw, and the engine's
    #: verdict when it had a floor (None when it had none or could not judge).
    top_similarity: Optional[float] = None
    low_confidence: Optional[bool] = None
    #: Experiment 3: facts that held at the question date, and the closed
    #: chain behind them when the run asked for history. Both are zero
    #: unless something distilled facts into the item's space first.
    facts: int = 0
    history_facts: int = 0

    def any_at(self, k: int) -> bool:
        seen = set(self.retrieved_sessions[:k])
        return any(a in seen for a in self.answer_sessions)

    def all_at(self, k: int) -> bool:
        seen = set(self.retrieved_sessions[:k])
        return bool(self.answer_sessions) and all(a in seen for a in self.answer_sessions)


def cross_partner(items: Sequence[BenchItem], idx: int) -> Optional[BenchItem]:
    """Another item whose question has no evidence in this item's haystack:
    the next item in run order (wrapping) none of whose answer sessions
    appear among this item's session ids. Its question, asked of this
    item's store, is a no-evidence query for the abstention sweep
    (experiment 9). LongMemEval's own abstention items cannot serve: their
    answer sessions are in the haystack (the topic is there, the detail is
    not), so has_evidence is true for every one of the 500."""
    here = set(items[idx].session_ids)
    for step in range(1, len(items)):
        other = items[(idx + step) % len(items)]
        if other.answer_session_ids and here.isdisjoint(other.answer_session_ids) and other.question != items[idx].question:
            return other
    return None


#: Candidate floors for the abstention sweep, 0.30 to 0.90 by 0.05.
FLOOR_GRID: tuple[float, ...] = tuple(round(0.30 + 0.05 * i, 2) for i in range(13))
#: question_type of a cross-item query result (never scored for recall).
CROSS_ITEM = "cross-item"


def flagged(top_similarity: Optional[float], floor: float) -> bool:
    """The engine's rule, restated for the sweep: nothing found, or the best
    hit below the floor, is weak evidence."""
    return top_similarity is None or top_similarity < floor


def abstention_sweep(results: Sequence["ItemResult"], floors: Sequence[float] = FLOOR_GRID) -> Optional[dict]:
    """For each candidate floor: the share of no-evidence items that would be
    flagged (abstention caught) and the share of evidence items that would be
    flagged (answers wrongly withheld). Items whose recall errored or whose
    vector lane was degraded are outside both denominators, and the counts
    say so. None when no no-evidence item ran, because then the first
    number cannot be measured at all."""
    judged = [r for r in results if r.error is None and not any(d.startswith("vectors:") for d in r.degraded)]
    without = [r for r in judged if not r.has_evidence]
    with_ev = [r for r in judged if r.has_evidence]
    if not without:
        return None
    return {
        "floors": list(floors),
        "no_evidence_n": len(without),
        "cross_item_n": sum(r.question_type == CROSS_ITEM for r in without),
        "evidence_n": len(with_ev),
        "abstain_rate": {f: round(sum(flagged(r.top_similarity, f) for r in without) / len(without), 4) for f in floors},
        "false_abstain_rate": {f: (round(sum(flagged(r.top_similarity, f) for r in with_ev) / len(with_ev), 4) if with_ev else None) for f in floors},
        # The rates are rounded for reading; the counts are what a floor is
        # chosen by, since one withheld answer of 21 rounds to 0.0476 and a
        # target of 0.0476 must not take a floor that cost more than that.
        "false_abstain_n": {f: sum(flagged(r.top_similarity, f) for r in with_ev) for f in floors},
        "abstain_n": {f: sum(flagged(r.top_similarity, f) for r in without) for f in floors},
    }


@dataclass
class RunReport:
    dataset: str
    items: int
    scored: int  # items in the denominator
    include_abstention: bool
    #: Whether passage merging was on. Recorded because a saved run whose
    #: metadata cannot say which configuration produced it is not evidence
    #: of anything.
    merge: bool
    #: Bytes of window either side, 0 when off. Recorded for the same
    #: reason.
    window: int
    ks: list[int]
    recall_any: dict[int, float]
    recall_all: dict[int, float]
    by_type: dict[str, dict[str, float]]
    context_reduction_median: Optional[float]
    recall_ms_p50: Optional[float]
    recall_ms_p95: Optional[float]
    errors: int
    embedder: str
    document_store: str
    vector_index: str
    python: str
    platform: str
    started_at: str
    finished_at: str
    results: list[ItemResult] = field(default_factory=list)
    #: Experiment 9 sweep (see abstention_sweep); None when it cannot be measured.
    abstention: Optional[dict] = None
    #: The floor the engines ran with, if any, and how many items each verdict got.
    similarity_floor: Optional[float] = None
    #: Whether the engines embedded the date/source prefix (experiment 8), read
    #: from the engine itself so the report says what ran, not what was asked.
    contextual_embeddings: bool = False
    low_confidence_counts: dict[str, int] = field(default_factory=dict)
    #: Experiment 9: one no-evidence query per item (another item's question
    #: whose evidence is absent here), kept apart from the scored results.
    cross_results: list[ItemResult] = field(default_factory=list)
    #: Experiment 3: whether history was asked for, and how many items got any facts / any history back.
    history: bool = False
    items_with_facts: int = 0
    items_with_history: int = 0

    def as_dict(self, with_items: bool = True) -> dict:
        d = asdict(self)
        if not with_items:
            d.pop("results")
            d.pop("cross_results")
        return d


def nearest_rank(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    import math
    return s[min(len(s), max(1, math.ceil(q * len(s)))) - 1]


async def run(
    make_engine: Callable[[], "asyncio.Future[MemoryEngine] | Coroutine[object, object, MemoryEngine] | MemoryEngine"],
    items: Iterable[BenchItem],
    ks: Sequence[int] = (5, 10, 15),
    limit: Optional[int] = None,
    include_abstention: bool = False,
    dataset: str = "",
    progress: Optional[Callable[[int, int], None]] = None,
    history: bool = False,
    cross_queries: bool = False,
    merge: bool = False,
    window: int = 0,
) -> RunReport:
    """``make_engine`` returns a fresh engine (or an awaitable of one) per
    item, so an item's memory never leaks into the next. ``limit`` is the
    recall limit and defaults to max(ks). ``history`` passes
    ``history=True`` to every recall (experiment 3) and counts what came
    back; it changes nothing unless facts exist in the item's space.
    ``cross_queries`` asks each item's store one other item's question
    whose evidence is absent (see cross_partner) after the real recall;
    those results feed the abstention sweep only.

    ``merge`` joins neighbouring chunks of one episode into the passage
    holding them before anything is scored. It cannot change *which*
    episodes came back, so it cannot change session recall at the recall
    limit -- but it compacts the list, so an episode below the cut can
    move above it, and recall at a k smaller than the limit can change in
    either direction. That is the thing worth measuring."""
    import asyncio
    from datetime import datetime, timezone

    items = list(items)
    k_max = limit or max(ks)
    results: list[ItemResult] = []
    cross: list[ItemResult] = []
    started = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    engine_meta = {"embedder": "", "document_store": "", "vector_index": ""}
    floor: Optional[float] = None
    contextual = False
    for n, item in enumerate(items, 1):
        engine = make_engine()
        if asyncio.iscoroutine(engine) or isinstance(engine, asyncio.Future):
            engine = await engine
        engine_meta = {"embedder": engine.embedder.id, "document_store": engine.documents.name, "vector_index": engine.vectors.name}
        floor = getattr(engine, "similarity_floor", None)
        contextual = bool(getattr(engine, "contextual_embeddings", False))
        space = "item"
        result = ItemResult(item.question_id, item.question_type, item.has_evidence, [], 0, 0, 0.0, answer_sessions=list(item.answer_session_ids))
        try:
            records = []
            for s_idx, session in enumerate(item.sessions):
                text = "\n".join(session)
                if not text.strip():
                    continue  # real datasets contain empty sessions
                result.stored_bytes += len(text.encode())
                records.append(Record(
                    content=text, kind="conversation",
                    source=item.session_ids[s_idx] if s_idx < len(item.session_ids) else None,
                    created_at=item.session_dates[s_idx] if s_idx < len(item.session_dates) and item.session_dates[s_idx] else None,
                ))
            await engine.remember_many(space, records)
            t0 = time.perf_counter()
            pack = await engine.recall(space, item.question, limit=k_max, history=history)
            if window:
                from ..retrieval.window import widen

                opened = await widen(engine, space, pack.items, before=window, after=window)
                pack = pack.model_copy(update={
                    "items": list(opened.items),
                    "returned_bytes": sum(len(i.text.encode()) for i in opened.items)})
            if merge:
                from ..retrieval.merging import merge_neighbours

                joined = await merge_neighbours(engine, space, pack.items)
                # The byte count has to be recomputed, not carried over.
                # Merging replaces fragments with the span containing them,
                # which across a gap is *longer* than the fragments were --
                # keeping recall's number would report a context reduction
                # that never happened, and a measurement published from it
                # would be wrong in the flattering direction. The
                # definition here is recall's own: the bytes of the text
                # actually handed back.
                pack = pack.model_copy(update={
                    "items": list(joined.items),
                    "returned_bytes": sum(len(i.text.encode()) for i in joined.items)})
            result.recall_ms = round((time.perf_counter() - t0) * 1000, 3)
            result.retrieved_sessions = [i.source or "" for i in pack.items]
            result.returned_bytes = pack.returned_bytes
            result.degraded = list(pack.degraded)
            result.top_similarity = pack.top_similarity
            result.low_confidence = pack.low_confidence
            result.facts = len(pack.facts)
            result.history_facts = len(pack.history)
            partner = cross_partner(items, n - 1) if cross_queries else None
            if partner is not None:
                t0 = time.perf_counter()
                stray = await engine.recall(space, partner.question, limit=k_max)
                cross.append(ItemResult(
                    f"{partner.question_id}@{item.question_id}", CROSS_ITEM, False,
                    [i.source or "" for i in stray.items], result.stored_bytes, stray.returned_bytes,
                    round((time.perf_counter() - t0) * 1000, 3), degraded=list(stray.degraded),
                    top_similarity=stray.top_similarity, low_confidence=stray.low_confidence,
                ))
        except Exception as e:  # noqa: BLE001 - one bad item must not lose the run; it is counted
            result.error = f"{type(e).__name__}: {e}"
        results.append(result)
        for store in (engine.documents, engine.vectors, engine.events):
            if store is not None and hasattr(store, "close"):
                try:
                    await store.close()
                except Exception:  # noqa: BLE001
                    pass
        if progress:
            progress(n, len(items))

    scored = [r for r in results if r.error is None and (r.has_evidence or include_abstention)]
    denom = len(scored)
    recall_any = {k: round(sum(r.any_at(k) for r in scored) / denom, 4) if denom else 0.0 for k in ks}
    recall_all = {k: round(sum(r.all_at(k) for r in scored) / denom, 4) if denom else 0.0 for k in ks}
    by_type: dict[str, dict[str, float]] = {}
    for qt in sorted({r.question_type for r in scored}):
        rows = [r for r in scored if r.question_type == qt]
        by_type[qt] = {"n": len(rows), **{f"any@{k}": round(sum(r.any_at(k) for r in rows) / len(rows), 4) for k in ks},
                       **{f"all@{k}": round(sum(r.all_at(k) for r in rows) / len(rows), 4) for k in ks}}
    reductions = [1 - r.returned_bytes / r.stored_bytes for r in results if r.error is None and r.stored_bytes]
    verdicts: dict[str, int] = {}
    for r in results + cross:
        if r.error is None and r.low_confidence is not None:
            key = ("no_evidence" if not r.has_evidence else "evidence") + ("_flagged" if r.low_confidence else "_cleared")
            verdicts[key] = verdicts.get(key, 0) + 1
    latencies = [r.recall_ms for r in results if r.error is None]
    return RunReport(
        dataset=dataset, items=len(results), scored=denom, include_abstention=include_abstention,
        merge=merge, window=window, ks=list(ks),
        recall_any=recall_any, recall_all=recall_all, by_type=by_type,
        context_reduction_median=nearest_rank(reductions, 0.5), recall_ms_p50=nearest_rank(latencies, 0.5), recall_ms_p95=nearest_rank(latencies, 0.95),
        errors=sum(1 for r in results if r.error), python=platform.python_version(), platform=platform.platform(),
        started_at=started, finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        results=results, abstention=abstention_sweep(results + cross), similarity_floor=floor, low_confidence_counts=verdicts,
        contextual_embeddings=contextual, cross_results=cross,
        history=history, items_with_facts=sum(1 for r in results if r.facts), items_with_history=sum(1 for r in results if r.history_facts),
        **engine_meta,
    )
