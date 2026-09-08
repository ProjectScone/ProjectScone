"""Controlled candidate-window coverage fixture; not a semantic model benchmark.

A deterministic index returns ten retained records in a fixed order. Only the
last carries ANSWER. A scripted ranker assigns its known fixture score; this
measures pipeline reach, byte budgets and overhead, not learned relevance.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import time
from typing import Mapping, Sequence

from ..backends.memory import InMemoryDocumentStore, InMemoryVectorIndex
from ..core.ports import TextFilter
from ..embedders import HashEmbedder
from ..memory.engine import MemoryEngine
from ..retrieval.reranking import RerankCandidate, RerankScore

STAMP = "2026-09-07T00:00:00.000Z"


class OrderedVectors(InMemoryVectorIndex):
    def __init__(self) -> None:
        super().__init__()
        self.fixture_ids: list[int] = []

    async def search(self, space: str, query: Sequence[float], limit: int,
                     as_of: str | None = None, tags: Sequence[str] = (),
                     where: Mapping[str, str] | None = None) -> list[tuple[int, float]]:
        return [(chunk_id, .95 - rank / 100) for rank, chunk_id in enumerate(self.fixture_ids[:limit])]


class NoTextLane(InMemoryDocumentStore):
    async def search_text(self, space: str, query: str, limit: int, filter: TextFilter) -> list[tuple[int, float]]:
        return []


class FixtureRanker:
    async def rerank(self, query: str, candidates: tuple[RerankCandidate, ...]) -> list[RerankScore]:
        return [RerankScore(candidate.chunk_id, 1 if candidate.text.startswith("ANSWER") else 0) for candidate in candidates]


async def run(repeats: int) -> dict[str, object]:
    vectors = OrderedVectors()
    engine = await MemoryEngine(NoTextLane(), vectors, HashEmbedder(), clock=lambda: STAMP).open()
    try:
        for number in range(10):
            added = await engine.remember("fixture", f"{'ANSWER' if number == 9 else 'NOISE!'} record {number:02d}", created_at=STAMP)
            vectors.fixture_ids.append((await engine.documents.chunks_of("fixture", added.episode_id))[0].chunk_id)
        target = vectors.fixture_ids[-1]
        rows: list[dict[str, object]] = []
        for label, depth, configured in (("legacy_depth_no_ranker", None, False), ("independent_depth_no_ranker", 16, False),
                                         ("legacy_depth_scripted_ranker", None, True), ("independent_depth_scripted_ranker", 16, True)):
            engine.reranker = FixtureRanker() if configured else None
            elapsed: list[float] = []
            found = 0
            returned_bytes: list[int] = []
            payload_bytes: list[int] = []
            candidates_sent: list[int] = []
            returned_ids: list[list[int]] = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                result = await engine.recall("fixture", "Which retained record is the answer?", limit=1, candidate_limit=depth)
                elapsed.append((time.perf_counter_ns() - start) / 1_000_000)
                ids = [item.chunk_id for item in result.items]
                found += int(target in ids)
                returned_bytes.append(result.returned_bytes)
                payload_bytes.append(result.rerank.payload_bytes if result.rerank else 0)
                candidates_sent.append(result.rerank.candidates_sent if result.rerank else 0)
                returned_ids.append(ids)
                assert len(ids) <= 1 and result.returned_bytes == 16
            row = {"mode": label, "trials": repeats, "candidate_limit": depth, "output_limit": 1,
                   "target_chunk_id": target, "target_found": found, "coverage": found / repeats,
                   "returned_ids": returned_ids, "returned_bytes": returned_bytes,
                   "ranker_payload_bytes": payload_bytes, "candidates_sent": candidates_sent,
                   "latency_ms_median": round(statistics.median(elapsed), 6)}
            rows.append(row)
            print(json.dumps(row), flush=True)
        return {"schema_version": 1, "workload": "fixed candidate-order diagnostic", "records": 10,
                "chat_model_calls": 0, "reranker": "scripted exact fixture marker, not a semantic model",
                "limitations": ["Measures candidate-window reach and pipeline overhead only.",
                               "Does not estimate real reranker model quality, token usage or model latency.",
                               "Warm tiny local stores; not a corpus-scale or end-to-end answer benchmark."],
                "results": rows}
    finally:
        await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000:
        parser.error("repeats must be between 1 and 1000")
    report = asyncio.run(run(args.repeats))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
