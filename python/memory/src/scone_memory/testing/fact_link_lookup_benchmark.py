"""Measure bounded in-memory relationship lookup as unrelated links grow.

Run before and after the implementation change with the same sizes and repeats.
Fixtures live only in an isolated in-memory document store; no engine, model,
network, or production data is involved. Setup time is excluded from latency.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

from ..backends.memory import InMemoryDocumentStore
from ..core.ports import NewFactLink

STAMP = "2026-09-07T00:00:00Z"


def new_link(space: str, left: int, right: int, kind: str = "supports") -> NewFactLink:
    return NewFactLink(space=space, from_fact=left, to_fact=right, kind=kind, created_at=STAMP)


async def measure(distractors: int, repeats: int, other_space: bool) -> dict[str, object]:
    store = InMemoryDocumentStore()
    selected = [new_link("selected", 101, 102), new_link("selected", 101, 102, "contradicts"),
                new_link("selected", 102, 103)]
    for record in selected:
        await store.insert_fact_link(record)
    duplicate = selected[-1]
    for offset in range(distractors):
        duplicate = new_link("unrelated" if other_space else "selected", 1_000_000 + offset * 2, 1_000_001 + offset * 2)
        await store.insert_fact_link(duplicate)
    for _ in range(5):
        await store.fact_links_between("selected", [103, 101, 102], 49)
        await store.insert_fact_link(duplicate)
    lookup_times: list[float] = []
    duplicate_times: list[float] = []
    expected = [1, 2, 3]
    for _ in range(repeats):
        started = time.perf_counter_ns()
        found = await store.fact_links_between("selected", [103, 101, 102], 49)
        lookup_times.append((time.perf_counter_ns() - started) / 1_000_000)
        assert [link.link_id for link in found] == expected
        started = time.perf_counter_ns()
        same = await store.insert_fact_link(duplicate)
        duplicate_times.append((time.perf_counter_ns() - started) / 1_000_000)
        assert same.link_id == 3 + distractors
    assert len(store._links) == 3 + distractors
    return {"distractors": distractors, "unrelated_space": other_space, "links_total": 3 + distractors,
            "repeats": repeats, "returned_ids": expected, "returned_ids_verified_every_trial": True,
            "lookup_median_ms": round(statistics.median(lookup_times), 6),
            "lookup_max_ms": round(max(lookup_times), 6),
            "duplicate_insert_median_ms": round(statistics.median(duplicate_times), 6),
            "duplicate_insert_max_ms": round(max(duplicate_times), 6)}


async def run(sizes: list[int], repeats: int, label: str) -> dict[str, object]:
    results = []
    for other_space in (False, True):
        for size in sizes:
            result = await measure(size, repeats, other_space)
            print(json.dumps(result), flush=True)
            results.append(result)
    backend = Path(__file__).resolve().parents[1] / "backends" / "memory.py"
    return {"schema_version": 1, "label": label, "workload": "bounded induced fact links and idempotent insert",
            "python": platform.python_version(), "platform": platform.platform(),
            "backend_sha256": hashlib.sha256(backend.read_bytes()).hexdigest(),
            "setup_excluded": True, "warmup_calls": 5, "response_limit": 49, "selected_fact_ids": [103, 101, 102],
            "limitations": ["Warm local microbenchmark with three returned links; not end-to-end recall latency.",
                           "One process per before/after run; timings vary with machine load.",
                           "Direct document-store fixture writes omit fact validation and graph source hydration."],
            "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[0, 1000, 10000])
    parser.add_argument("--repeats", type=int, default=201)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10000 or any(not 0 <= size <= 100000 for size in args.sizes):
        parser.error("repeats must be1..10000 and sizes0..100000")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(run(args.sizes, args.repeats, args.label))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
