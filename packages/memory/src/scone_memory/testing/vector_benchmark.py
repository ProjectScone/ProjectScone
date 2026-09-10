"""Search-only sample collection, not a full database benchmark or QA score.

Supply a preloaded, disposable VectorIndex and independently established nearest
neighbors. This utility never writes/deletes vectors or deploys services. Run
the adapter's behavioral contract separately before using timings to compare it.
Progress observers can feed the console; they are outside the search timing.
"""

from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Mapping, Sequence, TypedDict

from ..core.ports import VectorIndex
from ..core.timeutil import now_rfc3339


class SearchSample(TypedDict):
    case: str
    repeat: int
    limit: int
    expected_count: int
    returned_ids: list[int]
    recall_at_k: float | None
    latency_ms: float
    error_category: str | None
    empty_result_correct: bool | None


@dataclass(frozen=True)
class SearchCase:
    name: str
    space: str
    vector: Sequence[float]
    expected_ids: tuple[int, ...]
    limit: int = 5
    as_of: str | None = None
    tags: tuple[str, ...] = ()
    where: Mapping[str, str] = field(default_factory=dict)


def _latencies(samples: Sequence[SearchSample]) -> dict:
    """Nearest-rank quantiles: sorted samples[ceil(p*n)-1]; ms, no estimates."""
    values = sorted(sample["latency_ms"] for sample in samples)
    return {"n": len(values), "p50": values[math.ceil(.50 * len(values)) - 1] if values else None,
            "p95": values[math.ceil(.95 * len(values)) - 1] if values else None,
            "method": "nearest-rank", "includes_failures": True}


async def measure_search(
    index: VectorIndex,
    cases: Sequence[SearchCase],
    *,
    dataset_id: str,
    backend_version: str,
    deployment: str,
    repeats: int = 3,
    on_progress: Callable[[dict], Awaitable[None]] | None = None,
) -> dict:
    """Collect serial, client-observed search latency and macro neighbor recall.

    Recall per nonempty case is |returned IDs intersect expected IDs| / number
    of expected IDs. Mean includes failed searches as zero; empty ground truth
    is excluded and gets an explicit empty_result_correct check instead.
    Completion means execution finished, not semantic accuracy or conformance.
    No warmup is hidden: every call is recorded, and repeats reuse the same cases.
    Observer exceptions are counted but do not change search outcomes; async
    cancellation propagates instead of reporting a completed run.
    """
    cases = tuple(cases)
    if not cases:
        raise ValueError("at least one search case is required")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not all((dataset_id, backend_version, deployment)):
        raise ValueError("dataset_id, backend_version and deployment must be explicit")
    for case in cases:
        if not case.name or not case.space or not 1 <= case.limit <= 50:
            raise ValueError("cases need name, space and limit in 1..50")
        if len(set(case.expected_ids)) != len(case.expected_ids) or len(case.expected_ids) > case.limit:
            raise ValueError("expected neighbor IDs must be unique and fit the requested limit")

    started_at = now_rfc3339()
    total = len(cases) * repeats
    observer_failures = 0

    async def notify(status: str, done: int) -> None:
        nonlocal observer_failures
        if on_progress is not None:
            try:
                await on_progress({"status": status, "done": done, "total": total})
            except Exception:
                observer_failures += 1

    await notify("running", 0)
    samples: list[SearchSample] = []
    for repeat in range(repeats):
        for case in cases:
            returned_ids = []
            error = None
            started = time.perf_counter()
            try:
                hits = await index.search(case.space, case.vector, case.limit, as_of=case.as_of,
                                          tags=case.tags, where=case.where)
                latency_ms = (time.perf_counter() - started) * 1000
                returned_ids = [hit[0] for hit in hits]
                if (len(returned_ids) > case.limit or len(set(returned_ids)) != len(returned_ids)
                        or not all(math.isfinite(hit[1]) for hit in hits)):
                    error = "invalid_result"
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                error = type(exc).__name__  # no raw exception text / endpoint secrets
            recall = None
            if case.expected_ids:
                recall = 0.0 if error else len(set(returned_ids).intersection(case.expected_ids)) / len(case.expected_ids)
            samples.append({"case": case.name, "repeat": repeat, "limit": case.limit,
                            "expected_count": len(case.expected_ids), "returned_ids": returned_ids,
                            "recall_at_k": recall, "latency_ms": latency_ms, "error_category": error,
                            "empty_result_correct": (not returned_ids and error is None) if not case.expected_ids else None})
            await notify("running", len(samples))

    failed = sum(sample["error_category"] is not None for sample in samples)
    status = "failed" if failed else "completed"
    await notify(status, total)
    recalls = [s["recall_at_k"] for s in samples if s["recall_at_k"] is not None]
    return {"schema_version": 1, "product": "python", "backend": index.name,
            "backend_version": backend_version, "deployment": deployment, "dataset_id": dataset_id,
            "runtime": platform.python_version(), "platform": platform.platform(),
            "started_at": started_at, "finished_at": now_rfc3339(),
            "status": status, "concurrency": 1, "repeats": repeats, "hidden_warmups": 0,
            "sample_count": len(samples), "failed_samples": failed, "observer_failures": observer_failures,
            "recall_at_k": {"value": sum(recalls) / len(recalls) if recalls else None, "n": len(recalls),
                            "denominator": "nonempty ground-truth samples, including failures"},
            "latency_ms": _latencies(samples), "samples": samples,
            "not_measured": ["server-only latency", "throughput", "ingestion", "recovery", "RSS", "storage bytes",
                             "answer quality", "semantic evidence recall", "context tokens", "model cost"]}
