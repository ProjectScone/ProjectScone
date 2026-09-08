"""Measure scoped recall as unrelated records grow, without a chat model.

Run: python -m scone_memory.testing.scoped_recall_benchmark --output results.json
Uses temporary local stores and deterministic embeddings; never live memory.
This measures one scope-starvation workload, not general semantic recall quality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import tempfile
import time

from ..backends import InMemoryDocumentStore, InMemoryVectorIndex, SqliteDocumentStore, SqliteVectorIndex
from ..embedders import HashEmbedder
from ..memory.engine import MemoryEngine, Record


async def measure(backend: str, distractors: int, repeats: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="scone-recall-scale-") as directory:
        path = str(Path(directory) / "memory.db")
        documents: SqliteDocumentStore | InMemoryDocumentStore = SqliteDocumentStore(path) if backend == "sqlite" else InMemoryDocumentStore()
        vectors: SqliteVectorIndex | InMemoryVectorIndex = SqliteVectorIndex(path) if backend == "sqlite" else InMemoryVectorIndex()
        engine = await MemoryEngine(documents, vectors, HashEmbedder(),
                                    clock=lambda: "2026-09-07T00:00:00Z").open()
        try:
            for offset in range(0, distractors, 100):
                await engine.remember_many("fixture", [Record(
                    f"Juniper calibration planning note {number}", kind="conversation", source="chatter",
                    created_at="2026-09-06T00:00:00Z")
                    for number in range(offset, min(offset + 100, distractors))])
            target = await engine.remember("fixture",
                "The retained Juniper calibration manual specifies Polaris as its reference star.",
                kind="file", source="knowledge/core/manual", created_at="2025-01-01T00:00:00Z")
            elapsed: list[float] = []
            found, returned_bytes = 0, []
            for _ in range(repeats):
                started = time.perf_counter()
                result = await engine.recall("fixture", "Juniper calibration", limit=1,
                                             kind="file", source_prefix="knowledge/core/")
                elapsed.append((time.perf_counter() - started) * 1000)
                found += int(any(item.episode_id == target.episode_id for item in result.items))
                returned_bytes.append(result.returned_bytes)
            return {"backend": backend, "distractors": distractors, "records": distractors + 1,
                    "trials": repeats, "target_found": found, "coverage": found / repeats,
                    "latency_ms_median": round(statistics.median(elapsed), 3),
                    "latency_ms_max": round(max(elapsed), 3), "returned_bytes": returned_bytes,
                    "response_limit": 1, "chat_model_calls": 0}
        finally:
            await engine.close()


async def run(sizes: list[int], repeats: int) -> dict[str, object]:
    results = []
    for backend in ("memory", "sqlite"):
        for size in sizes:
            result = await measure(backend, size, repeats)
            results.append(result)
            print(json.dumps(result), flush=True)
    return {"schema_version": 1, "workload": "scoped lexical recall under same-topic distractors",
            "embedder": "deterministic hash", "limitations": [
                "One relevant record per query; not a semantic recall benchmark.",
                "SQLite vector search remains a linear scan; this does not establish million-record latency.",
                "Cold/warm query costs are combined; no cross-machine performance claim."], "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[0, 100, 1000, 10000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100 or any(not 0 <= size <= 100000 for size in args.sizes):
        parser.error("repeats must be 1..100 and sizes 0..100000")
    report = asyncio.run(run(args.sizes, args.repeats))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
