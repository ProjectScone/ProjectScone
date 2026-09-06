from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from ..engine import MemoryEngine, Record
from ..models import RecallItem


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

    def any_at(self, k: int) -> bool:
        seen = set(self.retrieved_sessions[:k])
        return any(a in seen for a in self.answer_sessions)

    def all_at(self, k: int) -> bool:
        seen = set(self.retrieved_sessions[:k])
        return bool(self.answer_sessions) and all(a in seen for a in self.answer_sessions)


@dataclass
class RunReport:
    dataset: str
    items: int
    scored: int  # items in the denominator
    include_abstention: bool
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

    def as_dict(self, with_items: bool = True) -> dict:
        d = asdict(self)
        if not with_items:
            d.pop("results")
        return d


def nearest_rank(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    import math
    return s[min(len(s), max(1, math.ceil(q * len(s)))) - 1]


async def run(
    make_engine: Callable[[], "asyncio.Future[MemoryEngine] | MemoryEngine"],
    items: Iterable[BenchItem],
    ks: Sequence[int] = (5, 10, 15),
    limit: Optional[int] = None,
    include_abstention: bool = False,
    dataset: str = "",
    progress: Optional[Callable[[int, int], None]] = None,
) -> RunReport:
    """``make_engine`` returns a fresh engine (or an awaitable of one) per
    item, so an item's memory never leaks into the next. ``limit`` is the
    recall limit and defaults to max(ks)."""
    import asyncio
    from datetime import datetime, timezone

    items = list(items)
    k_max = limit or max(ks)
    results: list[ItemResult] = []
    started = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    engine_meta = {"embedder": "", "document_store": "", "vector_index": ""}
    for n, item in enumerate(items, 1):
        engine = make_engine()
        if asyncio.iscoroutine(engine) or isinstance(engine, asyncio.Future):
            engine = await engine
        engine_meta = {"embedder": engine.embedder.id, "document_store": engine.documents.name, "vector_index": engine.vectors.name}
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
            pack = await engine.recall(space, item.question, limit=k_max)
            result.recall_ms = round((time.perf_counter() - t0) * 1000, 3)
            result.retrieved_sessions = [i.source or "" for i in pack.items]
            result.returned_bytes = pack.returned_bytes
            result.degraded = list(pack.degraded)
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
    latencies = [r.recall_ms for r in results if r.error is None]
    return RunReport(
        dataset=dataset, items=len(results), scored=denom, include_abstention=include_abstention, ks=list(ks),
        recall_any=recall_any, recall_all=recall_all, by_type=by_type,
        context_reduction_median=nearest_rank(reductions, 0.5), recall_ms_p50=nearest_rank(latencies, 0.5), recall_ms_p95=nearest_rank(latencies, 0.95),
        errors=sum(1 for r in results if r.error), python=platform.python_version(), platform=platform.platform(),
        started_at=started, finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        results=results, **engine_meta,
    )
