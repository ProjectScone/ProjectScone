"""The scoreboard's first row, swept: which retrieval defaults beat LlamaIndex's best.

LongMemEval-S, the Rust harness's stratified sample (``--n``, ``--seed``),
the hashed-token embedder on both sides, no model and no reranker. The
reference is LlamaIndex's own BM25 retriever fused with its vector
retriever by reciprocal rank (``compare(hybrid=True)``), plus its default
vector retriever for scale. Ours is the in-memory engine, every item in
a fresh engine per chunk size, asked once per setting: fusion mode x
vector weight x diversity, and each lane alone. Both sides are folded to
distinct sessions and scored by ``bench.comparative``'s own rule, so a
row here and a ``compare()`` run on the same engine agree number for
number (the ``--check`` flag runs ``compare()`` once and says whether
they do).

Ingestion is the costly part, so settings read at recall time share one
engine per item and chunk size: fusion and diversity are recall
arguments, and the vector weight is the engine attribute every recall
reads (``MemoryEngine.vector_weight``), set between recalls.

Run from packages/memory:

    PYTHONPATH=$PWD/src python benchmarks/northstar_defaults.py \\
        --data ../../bench-data/longmemeval_s.json --n 50 --seed 42 --out northstar.json

``--holdout-of 42:50`` draws the sample from the items *outside* the
seed-42 n=50 sample instead, to check a winner on items it was not
chosen on.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench.comparative import _scores, compare, distinct_sessions, llamaindex_session_ranking
from scone_memory.bench.runner import BenchItem, load_items, stratified_sample
from scone_memory.memory.engine import Record
from scone_memory.retrieval.fusion import PER_EPISODE_CAP

KS = (5, 10, 15)
RECALL_LIMIT = max(KS) * PER_EPISODE_CAP


def _records(item: BenchItem) -> list[Record]:
    """The sessions as the bench runner stores them, one record each."""
    records = []
    for index, session in enumerate(item.sessions):
        text = "\n".join(session)
        if not text.strip():
            continue
        records.append(Record(content=text, kind="conversation",
                              source=item.session_ids[index] if index < len(item.session_ids) else None,
                              created_at=(item.session_dates[index] if index < len(item.session_dates)
                                          and item.session_dates[index] else None)))
    return records


def _name(chunk: int, fusion: str, weight: Optional[float], diversity: Optional[float], lanes: Sequence[str]) -> str:
    if tuple(lanes) != ("vector", "text"):
        return f"chunk={chunk} lanes={'+'.join(lanes)}"
    return f"chunk={chunk} fusion={fusion} vector_weight={weight} diversity={diversity}"


def _row(rankings: list[tuple[Sequence[str], set[str]]]) -> dict[str, Any]:
    scores = _scores(rankings, KS)
    return {"R@5": scores.recall_any[5], "all@5": scores.recall_all[5], "R@10": scores.recall_any[10],
            "R@15": scores.recall_any[15], "MRR": scores.mrr}


async def sweep(items: list[BenchItem], *, chunks: Sequence[int], fusions: Sequence[str], weights: Sequence[float],
                diversities: Sequence[Optional[float]], log: Any) -> dict[str, Any]:
    embedder = HashEmbedder()
    rankings: dict[str, list[list[str]]] = {}
    relevant: list[set[str]] = []
    default_weight: Optional[float] = None
    started = time.perf_counter()
    for position, item in enumerate(items, 1):
        relevant.append(set(item.answer_session_ids))
        for name, hybrid in (("llamaindex hybrid (BM25+vector, RRF, chunk 512 tokens)", True),
                             ("llamaindex default (vector, chunk 512 tokens)", False)):
            ranked = await llamaindex_session_ranking(item, embedder, k=max(KS), hybrid=hybrid)
            rankings.setdefault(name, []).append(list(ranked))
        records = _records(item)
        for chunk in chunks:
            engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                        chunk_target=chunk).open()
            default_weight = engine.vector_weight
            await engine.remember_many("item", records)
            asks: list[tuple[str, float, Optional[float], tuple[str, ...]]] = [
                (fusion, weight, diversity, ("vector", "text"))
                for fusion in fusions for weight in weights for diversity in diversities]
            asks += [("rank", default_weight, None, ("text",)), ("rank", default_weight, None, ("vector",))]
            for fusion, weight, diversity, lanes in asks:
                engine.vector_weight = weight
                pack = await engine.recall("item", item.question, limit=RECALL_LIMIT, fusion=fusion,
                                           diversity=diversity, lanes=lanes)
                folded = distinct_sessions([hit.source or "" for hit in pack.items], max(KS))
                rankings.setdefault(_name(chunk, fusion, weight, diversity, lanes), []).append(list(folded))
            engine.vector_weight = default_weight
        print(f"{position}/{len(items)} {time.perf_counter() - started:.0f}s", file=log, flush=True)
    rows = {name: _row([(ranked, answers) for ranked, answers in zip(per_item, relevant)])
            for name, per_item in rankings.items()}
    return {"rows": rows, "rankings": rankings, "engine_default_vector_weight": default_weight,
            "question_ids": [item.question_id for item in items]}


async def check(items: list[BenchItem], log: Any) -> dict[str, Any]:
    """``compare()`` itself at engine defaults with the hybrid reference: the
    official path the sweep's rows must agree with."""
    def make_engine() -> Any:
        return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    report = await compare(items, make_engine, HashEmbedder(), ks=KS, hybrid=True)
    record = report.record()
    print("compare() done", file=log, flush=True)
    return {side: {"R@5": record["sides"][side]["recall_any"]["5"], "all@5": record["sides"][side]["recall_all"]["5"],
                   "R@10": record["sides"][side]["recall_any"]["10"], "R@15": record["sides"][side]["recall_any"]["15"],
                   "MRR": record["sides"][side]["mrr"]} for side in ("scone", "llamaindex")}


def _floats(raw: str) -> list[Optional[float]]:
    return [None if part in ("none", "off") else float(part) for part in raw.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-of", help="SEED:N -- draw from the items outside that sample")
    parser.add_argument("--chunks", default="700,2000")
    parser.add_argument("--fusions", default="rank,score,distribution")
    parser.add_argument("--weights", default="0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--diversities", default="none,0.3")
    parser.add_argument("--check", action="store_true", help="also run compare() at engine defaults")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    everything = load_items(args.data)
    pool = everything
    if args.holdout_of:
        seed, n = (int(part) for part in args.holdout_of.split(":"))
        taken = {item.question_id for item in stratified_sample(everything, n, seed=seed)}
        pool = [item for item in everything if item.question_id not in taken]
    items = [item for item in stratified_sample(pool, args.n, seed=args.seed) if item.has_evidence]
    weights = [w for w in _floats(args.weights) if w is not None]
    started = time.perf_counter()
    result = asyncio.run(sweep(items, chunks=[int(c) for c in args.chunks.split(",")], fusions=args.fusions.split(","),
                               weights=weights, diversities=_floats(args.diversities), log=sys.stderr))
    if args.check:
        result["compare_at_defaults"] = asyncio.run(check(items, sys.stderr))
    result["sample"] = {"dataset": Path(args.data).name, "n": args.n, "seed": args.seed, "holdout_of": args.holdout_of,
                        "scored": len(items), "stratified_by": "question_type"}
    result["run"] = {"argv": sys.argv, "python": platform.python_version(), "wall_seconds": round(time.perf_counter() - started),
                     "recall_limit": RECALL_LIMIT, "ks": list(KS), "embedder": HashEmbedder().id}
    Path(args.out).write_text(json.dumps(result, indent=1))
    width = max(len(name) for name in result["rows"])
    for name, row in result["rows"].items():
        print(f"{name:<{width}}  " + "  ".join(f"{key} {value:.3f}" for key, value in row.items()))
    if args.check:
        print("compare() at defaults:", json.dumps(result["compare_at_defaults"]))


if __name__ == "__main__":
    main()
