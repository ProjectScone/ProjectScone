"""Metrics computed from recorded evidence, and from nothing else.

Every figure carries its ``n``, the name of its denominator, the window
it was computed over, and a definition a reader can check against the
events. A figure with no evidence is ``None`` with ``n = 0``; it is
never a default. Coverage is reported alongside: how many events were
considered, the earliest one retained, and whether the read was
truncated, because lost events mean partial coverage, not zero
activity. Operational figures (latency, lane overlap, bytes) are not
answer quality, and the definitions say so.

Quantiles use the nearest-rank method (no interpolation): the value at
position ``ceil(q * n)`` of the sorted sample.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.ports import Event
from ..core.timeutil import parse_rfc3339
from .payload import mapping, sequence, items, number, integer


@dataclass
class Window:
    since: Optional[str]
    until: Optional[str]


@dataclass
class Coverage:
    events_considered: int
    earliest_retained: Optional[str]
    latest: Optional[str]
    truncated: bool
    schema_versions: list[int]


@dataclass
class Metric:
    name: str
    value: object
    n: int
    denominator: str
    unit: str
    definition: str
    caveat: Optional[str] = None


@dataclass
class Report:
    window: Window
    coverage: Coverage
    failures: dict[str, int]
    embedders: list[str]
    metrics: list[Metric] = field(default_factory=list)
    daily: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def nearest_rank(sorted_values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank quantile: the value at ceil(q * n), 1-based."""
    if not sorted_values:
        return None
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def compute(events: Iterable[Event], since: Optional[str] = None, until: Optional[str] = None, truncated: bool = False) -> Report:
    """Summarise the events that fall in ``[since, until)``.

    ``truncated`` says whether the caller's read hit its limit, so the
    report can say its coverage is partial.
    """
    lo = parse_rfc3339(since) if since else None
    hi = parse_rfc3339(until) if until else None
    kept: list[Event] = []
    for e in events:
        t = parse_rfc3339(e.ts)
        if (lo is None or t >= lo) and (hi is None or t < hi):
            kept.append(e)
    kept.sort(key=lambda e: (e.ts, e.event_id))

    failures = Counter(e.kind for e in kept if "error" in e.payload)
    ok = [e for e in kept if "error" not in e.payload]
    by_kind: dict[str, list[Event]] = defaultdict(list)
    for e in ok:
        by_kind[e.kind].append(e)

    report = Report(
        window=Window(since, until),
        coverage=Coverage(
            events_considered=len(kept),
            earliest_retained=kept[0].ts if kept else None,
            latest=kept[-1].ts if kept else None,
            truncated=truncated,
            schema_versions=sorted({e.schema_version for e in kept}),
        ),
        failures=dict(failures),
        embedders=sorted({str(e.payload.get("embedder")) for e in ok if e.payload.get("embedder")}),
    )
    report.metrics += recall_metrics(by_kind["recall"])
    report.metrics += ingest_metrics(by_kind["remember"])
    report.metrics += ledger_metrics(by_kind["fact_assert"], by_kind["fact_close"])
    report.metrics += feedback_metrics(by_kind["recall"], by_kind["feedback"])
    report.metrics += scope_metrics(by_kind["recall"])
    report.daily = daily_counts(kept)
    return report


def recall_metrics(recalls: list[Event]) -> list[Metric]:
    out: list[Metric] = []
    out.append(Metric("recall.count", len(recalls), len(recalls), "successful recalls", "recalls",
                      "Recall events without an error field in the window."))
    totals = sorted(number(mapping(e.payload["latency_ms"])["total"]) for e in recalls
                    if isinstance(e.payload.get("latency_ms"), dict) and "total" in mapping(e.payload["latency_ms"]))
    for q, name in ((0.5, "p50"), (0.95, "p95")):
        out.append(Metric(f"recall.latency_ms.{name}", nearest_rank(totals, q), len(totals), "successful recalls with a measured total",
                          "ms", f"Nearest-rank {name} of latency_ms.total measured by perf_counter around the whole recall.",
                          "Operational timing on this machine; says nothing about answer quality."))
    for lane in ("embed", "vector", "text"):
        vals = sorted(number(mapping(e.payload["latency_ms"])[lane]) for e in recalls
                      if isinstance(e.payload.get("latency_ms"), dict) and lane in mapping(e.payload["latency_ms"]))
        out.append(Metric(f"recall.latency_ms.{lane}.p50", nearest_rank(vals, 0.5), len(vals),
                          f"successful recalls where the {lane} step ran", "ms",
                          f"Nearest-rank median of latency_ms.{lane}."))

    nonempty = [e for e in recalls if e.payload.get("items")]
    tops = [items(e.payload["items"])[0] for e in nonempty]
    both = sum(1 for t in tops if set(mapping(t.get("lanes") or {}).keys()) >= {"vector", "text"})
    vonly = sum(1 for t in tops if set(mapping(t.get("lanes") or {}).keys()) == {"vector"})
    tonly = sum(1 for t in tops if set(mapping(t.get("lanes") or {}).keys()) == {"text"})
    n = len(nonempty)
    for name, count in (("both_lanes", both), ("vector_only", vonly), ("text_only", tonly)):
        out.append(Metric(f"recall.top_item.{name}_share", _share(count, n), n, "successful recalls that returned at least one item",
                          "share", f"Recalls whose top item was returned by {name.replace('_', ' ')}, over recalls with items.",
                          "Lane overlap is an observation about the retrievers agreeing, not relevance."))

    by_embedder: dict[str, list[float]] = defaultdict(list)
    for e in nonempty:
        sim = items(e.payload["items"])[0].get("similarity")
        if sim is not None and e.payload.get("embedder"):
            by_embedder[str(e.payload["embedder"])].append(number(sim))
    for embedder, sims in sorted(by_embedder.items()):
        sims.sort()
        out.append(Metric(f"recall.top_similarity.{embedder}", {"p10": nearest_rank(sims, 0.1), "median": nearest_rank(sims, 0.5),
                                                              "p90": nearest_rank(sims, 0.9)}, len(sims),
                          f"successful recalls with a top item that the vector lane scored, embedder {embedder}", "cosine",
                          "Nearest-rank p10/median/p90 of the top item's cosine similarity, grouped by full embedder id.",
                          "Uncalibrated: no confidence or quality class is derived from it. Hash embedders measure word overlap, not meaning."))

    reductions = sorted(1.0 - number(e.payload["returned_bytes"]) / number(e.payload["space_bytes"])
                        for e in recalls if number(e.payload.get("space_bytes") or 0) > 0)
    out.append(Metric("recall.byte_context_reduction.median", nearest_rank(reductions, 0.5), len(reductions),
                      "successful recalls in a non-empty space", "share of bytes",
                      "Median of 1 - returned_bytes / space_bytes, where space_bytes is every stored byte in the space.",
                      "Bytes, not tokens, and not the whole prompt; undefined on an empty space."))
    degraded = sum(1 for e in recalls if e.payload.get("degraded"))
    out.append(Metric("recall.degraded_count", degraded, len(recalls), "successful recalls", "recalls",
                      "Recalls that answered with one lane after the other failed."))
    judged = [e for e in recalls if e.payload.get("low_confidence") is not None]
    flagged = sum(1 for e in judged if e.payload["low_confidence"])
    floors = sorted({number(e.payload["similarity_floor"]) for e in judged if e.payload.get("similarity_floor") is not None})
    out.append(Metric("recall.low_confidence_share", _share(flagged, len(judged)), len(judged),
                      "successful recalls judged against a similarity floor", "share",
                      "Recalls flagged low_confidence (best vector hit below the engine's floor, or nothing found) over recalls that had "
                      "a floor and a working vector lane." + (f" Floors seen: {floors}." if floors else ""),
                      "Says how often the reader was told the evidence is weak, not whether that was right; abstention accuracy "
                      "needs labelled questions (the bench sweep)."))
    return out


def ingest_metrics(remembers: list[Event]) -> list[Metric]:
    fresh = sum(integer(e.payload.get("fresh", 0)) for e in remembers)
    dedup = sum(integer(e.payload.get("deduplicated", 0)) for e in remembers)
    records = fresh + dedup
    return [
        Metric("ingest.calls", len(remembers), len(remembers), "successful remember calls", "calls",
               "remember/remember_many calls without an error field."),
        Metric("ingest.fresh_episodes", fresh, records, "records offered", "episodes", "Records stored as new episodes."),
        Metric("ingest.deduplicated", dedup, records, "records offered", "records",
               "Records identical to an episode already in the space (or earlier in the batch); nothing stored."),
        Metric("ingest.dedup_fraction", _share(dedup, records), records, "records offered", "share",
               "deduplicated / (fresh + deduplicated)."),
        Metric("ingest.stored_bytes", sum(integer(e.payload.get("bytes", 0)) for e in remembers), records, "records offered", "bytes",
               "Bytes of newly stored episode content (deduplicated records contribute none).",
               "Input bytes offered are not recorded separately yet."),
        Metric("ingest.chunks", sum(integer(e.payload.get("chunks", 0)) for e in remembers), fresh, "fresh episodes", "chunks",
               "Chunks created for fresh episodes."),
    ]


def ledger_metrics(asserts: list[Event], closes: list[Event]) -> list[Metric]:
    outcomes = Counter(str(e.payload.get("outcome")) for e in asserts)
    superseded = sum(len(sequence(e.payload.get("superseded") or [])) for e in asserts)
    n = len(asserts)
    return [
        Metric("ledger.asserted", outcomes["new_active"] + outcomes["new_closed"], n, "assert_fact calls", "facts",
               "Assertions that stored a new fact (active, or already closed because a later fact bounded it)."),
        Metric("ledger.asserted_already_closed", outcomes["new_closed"], n, "assert_fact calls", "facts",
               "New facts stored closed because a fact starting later already existed (late-arriving history)."),
        Metric("ledger.restated", outcomes["restated"], n, "assert_fact calls", "facts",
               "Assertions that matched a fact already holding at that time; nothing changed."),
        Metric("ledger.superseded", superseded, n, "assert_fact calls", "facts",
               "Existing facts whose interval was truncated by a newer assertion.",
               "Counts ledger events, not factual accuracy or the number of distinct beliefs."),
        Metric("ledger.manual_closures", sum(1 for e in closes if e.payload.get("reason_kind") == "manual"), len(closes),
               "fact_close calls", "facts", "Facts closed by a person with a written reason."),
    ]


def feedback_metrics(recalls: list[Event], feedback: list[Event]) -> list[Metric]:
    latest: dict[tuple[int, int], bool] = {}
    for e in feedback:  # events arrive oldest first, so the last write wins
        latest[(integer(e.payload["recall_event_id"]), integer(e.payload["chunk_id"]))] = bool(e.payload["useful"])
    judged = len(latest)
    useful = sum(1 for v in latest.values() if v)
    returned = sum(len(sequence(e.payload.get("items") or [])) for e in recalls)
    return [
        Metric("feedback.judged_items", judged, returned, "items returned by recalls in the window", "items",
               "Distinct (recall, chunk) pairs with at least one judgement; the latest judgement per pair is kept.",
               "Actor is the space's key holder; no per-person identity is recorded yet."),
        Metric("feedback.useful_share", _share(useful, judged), judged, "judged items", "share",
               "useful / judged over the latest judgement per item.",
               "Usefulness among judged items only. People judge what they notice, so this is not precision or recall."),
        Metric("feedback.coverage", _share(judged, returned), returned, "items returned by recalls in the window", "share",
               "judged items / returned items."),
    ]


def scope_metrics(recalls: list[Event]) -> list[Metric]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for e in recalls:
        for key, value in mapping(e.payload.get("where") or {}).items():
            counts[key][str(value)] += 1
    top = {key: dict(c.most_common(5)) for key, c in sorted(counts.items())}
    return [
        Metric("recall.where_values.top5", top, len(recalls), "successful recalls", "recalls",
               "Explicit where-filter values recorded on recall events, top five per key.",
               "Counts of filters people asked for, not users served and not a permission boundary."),
    ]


def daily_counts(events: list[Event]) -> dict[str, dict[str, int]]:
    """Events per UTC day and kind, for the sparkline. Coverage caveat: a
    day with no retained events shows nothing, which is not zero activity."""
    days: dict[str, Counter] = defaultdict(Counter)
    for e in events:
        days[e.ts[:10]][e.kind] += 1
    return {day: dict(c) for day, c in sorted(days.items())}


def _share(count: int, n: int) -> Optional[float]:
    return None if n == 0 else round(count / n, 4)
