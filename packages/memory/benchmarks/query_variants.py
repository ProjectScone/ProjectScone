"""Does asking a question several ways find more of the answer?

One question is one wording of a need. ``retrieval.query_variants`` asks
a model for several restatements, searches each beside the question
itself, and fuses the rankings by reciprocal rank with the question at
full voice. This measures whether that finds sessions the question alone
misses, on the same LongMemEval-S sample the scoreboard uses, with the
same hashed-token embedder as the single-rewrite measurement.

Every delta is reported with the items behind it: on fifty items a
difference of 0.02 is one item, and a number without its count says
more than it knows.

Run from packages/memory, with a local model served OpenAI-style:

    PYTHONPATH=$PWD/src python benchmarks/query_variants.py \
        --data ../../bench-data/longmemeval_s.json --n 50 --seed 42 \
        --url http://127.0.0.1:11434/v1 --model llama3.1-ctx8k --out variants.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench.comparative import _scores, distinct_sessions
from scone_memory.bench.runner import BenchItem, load_items, stratified_sample
from scone_memory.memory.engine import Record
from scone_memory.providers.llm import OpenAICompatibleChat
from scone_memory.retrieval.query_variants import variant_recall, variants

KS = (5, 10, 15)
SPACE = "item"
RECALL_LIMIT = 40


def _records(item: BenchItem) -> list[Record]:
    """The sessions as the bench runner stores them, one record each."""
    records: list[Record] = []
    for index, session in enumerate(item.sessions):
        text = "\n".join(session)
        if not text.strip():
            continue
        records.append(Record(content=text, kind="conversation",
                              source=item.session_ids[index] if index < len(item.session_ids) else None,
                              created_at=(item.session_dates[index] if index < len(item.session_dates)
                                          and item.session_dates[index] else None)))
    return records


async def _engine() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              chunk_target=700, vector_weight=0.01).open()


def _row(rankings: list[tuple[Sequence[str], set[str]]]) -> dict[str, float]:
    scores = _scores(rankings, KS)
    return {"R@5": scores.recall_any[5], "all@5": scores.recall_all[5], "R@10": scores.recall_any[10],
            "R@15": scores.recall_any[15], "MRR": scores.mrr}


async def run(items: list[BenchItem], model: Any, count: int, log: Any) -> dict[str, Any]:
    asked: list[list[str]] = []
    fused: list[list[str]] = []
    relevant: list[set[str]] = []
    written: list[dict[str, object]] = []
    started = time.perf_counter()
    for position, item in enumerate(items, 1):
        relevant.append(set(item.answer_session_ids))
        engine = await _engine()
        try:
            await engine.remember_many(SPACE, _records(item))
            plain = await engine.recall(SPACE, item.question, limit=RECALL_LIMIT)
            asked.append(list(distinct_sessions([hit.source or "" for hit in plain.items], max(KS))))
            pack, record = await variant_recall(engine, model, SPACE, item.question,
                                                count=count, limit=RECALL_LIMIT)
            fused.append(list(distinct_sessions([hit.source or "" for hit in pack.items], max(KS))))
            written.append({"question_id": item.question_id, **record.record()})
        finally:
            await engine.close()
        print(f"{position}/{len(items)} {time.perf_counter() - started:.0f}s", file=log, flush=True)

    rows = {"question as asked": _row(list(zip(asked, relevant))),
            f"question + {count} variants": _row(list(zip(fused, relevant)))}
    wins, losses = [], []
    for k in KS:
        won = [i.question_id for i, a, f, gold in zip(items, asked, fused, relevant)
               if (gold & set(f[:k])) and not (gold & set(a[:k]))]
        lost = [i.question_id for i, a, f, gold in zip(items, asked, fused, relevant)
                if (gold & set(a[:k])) and not (gold & set(f[:k]))]
        wins.append((k, won)); losses.append((k, lost))
    kept = sum(len(w["kept"]) for w in written)  # type: ignore[arg-type]
    refused = sum(len(w["refused"]) for w in written)  # type: ignore[arg-type]
    return {"rows": rows,
            "per_item": {f"k={k}": {"won": won, "lost": lost}
                         for (k, won), (_, lost) in zip(wins, losses)},
            "variants": {"kept": kept, "refused": refused, "items": len(items),
                         "no_variant_items": sum(1 for w in written if not w["kept"])},
            "written": written}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-of", help="SEED:N -- draw from the items outside that sample")
    parser.add_argument("--count", type=int, default=3, help="variants asked for per question")
    parser.add_argument("--url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pool = load_items(args.data)
    if args.holdout_of:
        seed, _, size = args.holdout_of.partition(":")
        inside = {item.question_id for item in stratified_sample(pool, int(size), int(seed))}
        pool = [item for item in pool if item.question_id not in inside]
    items = stratified_sample(pool, args.n, args.seed)
    model = OpenAICompatibleChat(args.url, args.model, timeout=args.timeout, trust_env=False)
    report = asyncio.run(run(items, model, args.count, sys.stderr))
    report["settings"] = {"n": args.n, "seed": args.seed, "count": args.count, "model": args.model,
                          "holdout_of": args.holdout_of,
                          "embedder": "hash", "chunk": 700}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for name, row in report["rows"].items():
        print(f"{name:28s} " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))
    for k, counts in report["per_item"].items():
        print(f"{k}: {len(counts['won'])} won by variants, {len(counts['lost'])} lost")
    print("variants: " + json.dumps(report["variants"]))


if __name__ == "__main__":
    main()
