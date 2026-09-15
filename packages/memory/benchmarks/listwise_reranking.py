"""Fused recall beside listwise-reranked recall on LongMemEval-S, interleaved, repeated.

    python benchmarks/listwise_reranking.py run DATA OUT.jsonl [--items 20] [--repeats 3]
        [--model llama3.2-ctx8k] [--url http://127.0.0.1:11434/v1] [--timeout 300]
    python benchmarks/listwise_reranking.py summarize OUT.jsonl

Each repeat rebuilds every item's engine (in-memory stores, hashed-token
embedder, engine defaults) and runs the fused recall (``rerank=False``) and
the listwise recall (``rerank=True``) on the same stored passages back to back,
the order alternating by item and repeat. Rankings are folded to distinct
sessions as the comparative runner folds them. The log keeps each query's
sessions, seconds, degraded notes, trace with receipt, and the first 160
characters of each model reply (the scored path never keeps replies).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from scone_memory import HashEmbedder
from scone_memory.bench.comparative import distinct_sessions
from scone_memory.bench.runner import BenchItem, load_items, stratified_sample
from scone_memory.memory.engine import MemoryEngine, Record
from scone_memory.providers.llm import OpenAICompatibleChat
from scone_memory.retrieval.fusion import PER_EPISODE_CAP
from scone_memory.retrieval.listwise import ListwiseReranker
from scone_memory.runtime.config import Settings, build_in_process_engine

K = 5
DEPTH = 15
RECALL_LIMIT = DEPTH * PER_EPISODE_CAP


class Recording:
    """The chat model, keeping each reply for the log."""

    def __init__(self, chat: OpenAICompatibleChat) -> None:
        self.chat = chat
        self.replies: list[str] = []

    async def complete(self, system: str, user: str) -> str:
        reply = await self.chat.complete(system, user)
        self.replies.append(reply)
        return reply


async def ingest(item: BenchItem) -> MemoryEngine:
    engine = await build_in_process_engine(Settings(), HashEmbedder())
    records = []
    for index, session in enumerate(item.sessions):
        text = "\n".join(session)
        if not text.strip():
            continue
        records.append(Record(content=text, kind="conversation",
                              source=item.session_ids[index] if index < len(item.session_ids) else None,
                              created_at=(item.session_dates[index] if index < len(item.session_dates)
                                          and item.session_dates[index] else None)))
    await engine.remember_many("item", records)
    return engine


async def run(args: argparse.Namespace) -> None:
    items = stratified_sample(load_items(args.data), args.items, seed=42)
    chat = Recording(OpenAICompatibleChat(args.url, args.model, timeout=args.timeout, trust_env=False))
    ranker = ListwiseReranker(chat, timeout=args.timeout)
    with Path(args.out).open("a") as log:
        for repeat in range(args.repeats):
            for index, item in enumerate(items):
                engine = await ingest(item)
                try:
                    engine.reranker = ranker
                    order = [False, True] if (repeat + index) % 2 == 0 else [True, False]
                    rows = {}
                    for rerank in order:
                        before = len(chat.replies)
                        started = time.perf_counter()
                        result = await engine.recall("item", item.question, limit=RECALL_LIMIT, rerank=rerank)
                        seconds = time.perf_counter() - started
                        rows["listwise" if rerank else "fused"] = {
                            "sessions": list(distinct_sessions([i.source or "" for i in result.items], DEPTH)),
                            "seconds": round(seconds, 3), "degraded": list(result.degraded),
                            "rerank": result.rerank.model_dump(mode="json") if result.rerank else None,
                            "reply_heads": [reply[:160] for reply in chat.replies[before:]],
                        }
                    log.write(json.dumps({"repeat": repeat, "question_id": item.question_id,
                                          "question_type": item.question_type,
                                          "answer_sessions": list(item.answer_session_ids),
                                          "order": ["listwise" if r else "fused" for r in order], **rows}) + "\n")
                    log.flush()
                    print(repeat, index, item.question_id, {k: v["seconds"] for k, v in rows.items()}, flush=True)
                finally:
                    await engine.close()


def reciprocal_rank(sessions: list[str], relevant: set[str]) -> float:
    return next((1.0 / place for place, session in enumerate(sessions, 1) if session in relevant), 0.0)


def summarize(path: str) -> None:
    rows = [json.loads(line) for line in open(path)]
    by_repeat: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_repeat[row["repeat"]].append(row)
    table = {}
    for repeat, items in sorted(by_repeat.items()):
        scored = [row for row in items if row["answer_sessions"]]
        out: dict[str, object] = {"n": len(scored)}
        for side in ("fused", "listwise"):
            out[f"{side}_r5"] = sum(any(s in set(row["answer_sessions"]) for s in row[side]["sessions"][:K])
                                    for row in scored) / len(scored)
            out[f"{side}_mrr"] = sum(reciprocal_rank(row[side]["sessions"], set(row["answer_sessions"]))
                                     for row in scored) / len(scored)
            out[f"{side}_seconds_median"] = statistics.median(row[side]["seconds"] for row in items)
        hit = [(any(s in set(r["answer_sessions"]) for s in r["fused"]["sessions"][:K]),
                any(s in set(r["answer_sessions"]) for s in r["listwise"]["sessions"][:K])) for r in scored]
        out["r5_wins"], out["r5_losses"] = sum(b and not a for a, b in hit), sum(a and not b for a, b in hit)
        receipts = [row["listwise"]["rerank"]["listwise"] for row in items]
        out["window_outcomes"] = dict(Counter(call["outcome"] for receipt in receipts for call in receipt["calls"]))
        out["fallbacks"] = sum(receipt["fallback"] is not None for receipt in receipts)
        out["queries_over_60s"] = sum(row["listwise"]["seconds"] > 60 for row in items)
        table[repeat] = out
        print(repeat, json.dumps(out))
    medians = {key: round(statistics.median(float(t[key]) for t in table.values()), 4)  # type: ignore[arg-type]
               for key in ("fused_r5", "listwise_r5", "fused_mrr", "listwise_mrr",
                           "fused_seconds_median", "listwise_seconds_median")}
    print("median over repeats:", json.dumps(medians))
    starts, reasons = Counter(), Counter()
    for row in rows:
        receipt = row["listwise"]["rerank"]["listwise"]
        answered = [call for call in receipt["calls"] if call["outcome"] not in ("timeout", "failed")]
        for head, call in zip(row["listwise"]["reply_heads"], answered):
            first = re.match(r"\s*\[(\d+)\]", head)
            starts["began with the last passage shown" if first and int(first.group(1)) == call["passages"] else "other"] += 1
        reasons.update(call["reason"].split(":")[0] for call in receipt["calls"] if call["outcome"] == "unparseable")
    print("replies:", dict(starts), "unparseable reasons:", dict(reasons))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("run")
    measure.add_argument("data")
    measure.add_argument("out")
    measure.add_argument("--items", type=int, default=20)
    measure.add_argument("--repeats", type=int, default=3)
    measure.add_argument("--model", default="llama3.2-ctx8k")
    measure.add_argument("--url", default="http://127.0.0.1:11434/v1")
    measure.add_argument("--timeout", type=float, default=300.0)
    summary = commands.add_parser("summarize")
    summary.add_argument("out")
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run(args))
    else:
        summarize(args.out)


if __name__ == "__main__":
    main()
