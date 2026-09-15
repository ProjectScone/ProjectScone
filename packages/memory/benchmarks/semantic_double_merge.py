"""Semantic chunking with and without its second, merging pass, on the comparative runner.

LongMemEval-S, the Rust harness's stratified sample (``--n``, ``--seed``),
the hashed-token embedder on both sides, no model and no reranker. Every
row is ``bench.comparative.compare()`` itself -- a fresh in-memory engine
per item, sessions folded from passages, scored by its rule against
LlamaIndex's BM25 + vector fusion (``hybrid=True``) -- so the reference
column is the same in every row and the rows differ only in how our
sessions were cut:

- ``length``: the engine's default cut, for scale;
- ``semantic``: ``semantic_aware=True``, the first pass alone;
- ``semantic+merge@T``: the same, with ``semantic_merge_threshold=T``.

Each record's receipt is kept, so a row also says what the second pass
did: chunks stored, first-pass chunks, merges, and how many joins
similarity and the size target each stopped.

``--pairs`` instead reads the scale a threshold has to be chosen on: the
first pass's neighbouring chunks that would fit the target together, and
the cosine between their mean sentence vectors, as percentiles.

Run from packages/memory:

    PYTHONPATH=$PWD/src python benchmarks/semantic_double_merge.py \\
        --data ../../bench-data/longmemeval_s.json --n 50 --seed 42 --thresholds 0.2,0.4 --out merge.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench.comparative import compare
from scone_memory.bench.runner import BenchItem, load_items, stratified_sample
from scone_memory.ingestion import semantic_chunks
from scone_memory.ingestion.chunker import DEFAULT_TARGET

KS = (5, 10, 15)
COUNTS = ("groups", "merges", "stopped_by_similarity", "stopped_by_size")


async def row(items: list[BenchItem], name: str, options: dict[str, Any], log: Any) -> dict[str, Any]:
    tally: dict[str, int] = {"records": 0, "chunks": 0, **{key: 0 for key in COUNTS}}

    async def make_engine() -> MemoryEngine:
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), **options).open()
        remember_many = engine.remember_many

        async def counted(space: str, records: Any, **kwargs: Any) -> Any:
            added = await remember_many(space, records, **kwargs)
            for receipt in added:
                tally["records"] += 1
                tally["chunks"] += receipt.chunks
                for key in COUNTS:
                    tally[key] += int((receipt.structure or {}).get(key, 0))  # type: ignore[call-overload]
            return added

        engine.remember_many = counted  # type: ignore[method-assign]
        return engine

    started = time.perf_counter()
    report = await compare(items, make_engine, HashEmbedder(), ks=KS, hybrid=True)
    record = report.record()
    print(f"{name} done in {time.perf_counter() - started:.0f}s", file=log, flush=True)

    def side(label: str) -> dict[str, float]:
        scores = record["sides"][label]
        return {"R@5": scores["recall_any"]["5"], "all@5": scores["recall_all"]["5"], "R@10": scores["recall_any"]["10"],
                "R@15": scores["recall_any"]["15"], "MRR": scores["mrr"], "NDCG@5": scores["ndcg"]["5"]}

    return {"options": options, "scone": side("scone"), "llamaindex": side("llamaindex"), "receipts": tally,
            "per_item": record["per_item"], "sessions": {item["question_id"]: item["scone"] for item in record["items"]},
            "wall_seconds": round(time.perf_counter() - started)}


async def pairs(items: list[BenchItem], target: int = DEFAULT_TARGET) -> dict[str, Any]:
    """Cosines between neighbouring first-pass chunks, split by whether the
    two would fit ``target`` together -- only those can ever be merged."""
    embedder = HashEmbedder()
    fitting: list[float] = []
    too_long: list[float] = []
    sessions = groups = 0
    for item in items:
        for session in item.sessions:
            text = "\n".join(session)
            if not text.strip():
                continue
            sessions += 1
            sentences = semantic_chunks.sentence_spans(text)
            vectors = await embedder.embed([text[s.start:s.end].strip() for s in sentences])
            cuts = semantic_chunks._valleys(semantic_chunks._block_similarity(vectors, semantic_chunks.BLOCK),
                                            semantic_chunks.SENSITIVITY)
            made = semantic_chunks._assemble(text, sentences, cuts, target)
            groups += len(made)
            for (left, first, last), (right, start, end) in zip(made, made[1:]):
                similar = semantic_chunks._cosine(semantic_chunks._centre(vectors[first:last + 1]),
                                                  semantic_chunks._centre(vectors[start:end + 1]))
                (fitting if right.end - left.start <= target else too_long).append(similar)

    def percentiles(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {str(p): round(ordered[int(p / 100 * (len(ordered) - 1))], 3) for p in (10, 25, 50, 75, 90, 95)}

    return {"sessions": sessions, "groups": groups, "fitting_pairs": len(fitting), "too_long_pairs": len(too_long),
            "fitting_cosine_percentiles": percentiles(fitting), "too_long_cosine_percentiles": percentiles(too_long),
            "fitting_at_or_above": {str(t): sum(value >= t for value in fitting) for t in (0.1, 0.2, 0.3, 0.4, 0.5)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", default="0.2,0.4")
    parser.add_argument("--no-length", action="store_true", help="skip the length row")
    parser.add_argument("--pairs", action="store_true", help="print the neighbouring-chunk cosine distribution only")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    items = [item for item in stratified_sample(load_items(args.data), args.n, seed=args.seed) if item.has_evidence]
    if args.pairs:
        found = asyncio.run(pairs(items))
        Path(args.out).write_text(json.dumps(found, indent=1))
        print(json.dumps(found))
        return
    configs: list[tuple[str, dict[str, Any]]] = [] if args.no_length else [("length", {})]
    configs.append(("semantic", {"semantic_aware": True}))
    thresholds: list[Optional[float]] = [float(part) for part in args.thresholds.split(",") if part.strip()]
    configs += [(f"semantic+merge@{t}", {"semantic_aware": True, "semantic_merge_threshold": t}) for t in thresholds]
    started = time.perf_counter()
    rows = {name: asyncio.run(row(items, name, options, sys.stderr)) for name, options in configs}
    result = {"rows": rows,
              "sample": {"dataset": Path(args.data).name, "n": args.n, "seed": args.seed, "scored": len(items),
                         "stratified_by": "question_type"},
              "run": {"argv": sys.argv, "python": platform.python_version(),
                      "wall_seconds": round(time.perf_counter() - started), "ks": list(KS),
                      "embedder": HashEmbedder().id, "reference": "compare(hybrid=True)"}}
    Path(args.out).write_text(json.dumps(result, indent=1))
    width = max(len(name) for name in rows)
    for name, data in rows.items():
        print(f"{name:<{width}}  " + "  ".join(f"{key} {value:.3f}" for key, value in data["scone"].items())
              + "  " + "  ".join(f"{key} {value}" for key, value in data["receipts"].items()))
    first = next(iter(rows.values()))
    print(f"{'llamaindex hybrid':<{width}}  " + "  ".join(f"{key} {value:.3f}" for key, value in first["llamaindex"].items()))


if __name__ == "__main__":
    main()
