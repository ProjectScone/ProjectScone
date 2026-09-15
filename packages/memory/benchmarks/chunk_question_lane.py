"""The question lane on this repository's own documents, with a local model.

Run with Python 3.14 and PYTHONPATH=packages/memory/src, from packages/memory.

``build-eval --seed N`` (model calls): stores the named documents from
``docs/`` in SQLite stores under ``--data`` and writes one evaluation set
with ``bench.questions.write_questions`` (its own prompt, a seeded sample
of chunks, one question each). ``build-lane`` (model calls): the same
store, then the question lane over every chunk with
``ingestion.chunk_questions`` (a prompt worded apart from the bench's),
saved with its model calls and seconds.

``measure`` (no model): the same stores read by an engine with the lane off
and one with it on, in alternating order, ``--repeats`` times. A question is
found at rank r when the r-th returned passage holds its quote
(``bench.questions.measure``). Beside R@k and MRR it reports how many
questions moved, and how close each evaluation question is to the lane's
questions for the chunk holding its quote -- the same model wrote both
sets, so the prompts differ but leakage is reduced, not removed.

``sweep`` (no model): the lane's fusion weight is the context lane's
(``retrieval.recall.CONTEXT_WEIGHT``); this sets that constant for the run
only, over ``--weights``, and scores the evaluation set of ``--choose-seed``
as the set a weight would be chosen on and the questions of the others not
in it as held out.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from pathlib import Path
import shutil
import statistics
import time

import scone_memory.retrieval.recall as recall_module
from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.bench.questions import QuestionReport, QuestionSet, measure, store_corpus, write_questions
from scone_memory.providers.llm import OpenAICompatibleChat
from scone_memory.retrieval.lexical import tokenize

SPACE = "corpus"
DOCS = ("pdf-ocr.md", "answer-review.md", "agent-event-history.md", "directory-sync.md", "chat-exports.md",
        "followup-queries.md", "getting-started.md", "s3-catalog.md", "attachments.md", "space-merge.md",
        "table-querying.md", "http-and-deployment.md", "workflow-pauses.md")


async def engine_at(data: Path, *, question_lane: bool) -> MemoryEngine:
    return await MemoryEngine(SqliteDocumentStore(str(data / "documents.db")), SqliteVectorIndex(str(data / "vectors.db")),
                              HashEmbedder(), question_lane=question_lane).open()


async def stored(args: argparse.Namespace) -> MemoryEngine:
    data = Path(args.data)
    corpus = data / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for name in args.docs:
        shutil.copyfile(Path(__file__).resolve().parent.parent / "docs" / name, corpus / name)
    engine = await engine_at(data, question_lane=True)
    counts = await store_corpus(engine, SPACE, corpus)
    print(f"stored {counts} as {(await engine.documents.counts(SPACE)).chunks} chunk(s)", flush=True)
    return engine


async def build_eval(args: argparse.Namespace) -> None:
    """One evaluation set per seed, each a sample of ``--eval-chunks`` chunks;
    ``measure`` asks every set's questions, each question once."""
    engine = await stored(args)
    path = Path(args.data) / f"eval-questions-{args.seed}.json"
    started = time.perf_counter()
    written = await write_questions(engine, SPACE, OpenAICompatibleChat(args.url, args.model, timeout=args.timeout),
                                    corpus="packages/memory/docs (" + ", ".join(args.docs) + ")",
                                    model_name=args.model, per_chunk=1, max_chunks=args.eval_chunks, seed=args.seed)
    written.save(path)
    print(f"evaluation set in {time.perf_counter() - started:.0f}s: {written.text()}", flush=True)
    await engine.close()


async def build_lane(args: argparse.Namespace) -> None:
    engine = await stored(args)
    lane = await engine.build_chunk_questions(SPACE, OpenAICompatibleChat(args.url, args.model, timeout=args.timeout),
                                              per_chunk=args.per_chunk, model_name=args.model)
    (Path(args.data) / "lane.json").write_text(json.dumps(lane.record(), indent=1, ensure_ascii=False))
    print(lane.text(), flush=True)
    await engine.close()


def evaluation_questions(data: Path) -> tuple[QuestionSet, list[str]]:
    """Every saved set's questions, each question once, and what each set said of itself."""
    sets = [QuestionSet.load(path) for path in sorted(data.glob("eval-questions-*.json"))]
    seen: set[str] = set()
    merged = []
    for written in sets:
        for question in written.questions:
            key = _normal(question.question).casefold()
            if key not in seen:
                seen.add(key)
                merged.append(question)
    first = sets[0]
    return (QuestionSet(corpus=first.corpus, model=first.model, questions=tuple(merged),
                        chunks_total=first.chunks_total, chunks_asked=sum(s.chunks_asked for s in sets),
                        per_chunk=first.per_chunk, seed=first.seed,
                        dropped_unparsed=sum(s.dropped_unparsed for s in sets),
                        dropped_unquoted=sum(s.dropped_unquoted for s in sets),
                        skipped_long=sum(s.skipped_long for s in sets), calls_failed=sum(s.calls_failed for s in sets),
                        dropped_unasked=sum(s.dropped_unasked for s in sets)),
            [f"seed {s.seed}: {s.text()}" for s in sets])


def _normal(text: str) -> str:
    return " ".join(text.split())


async def ranks(engine: MemoryEngine, questions: QuestionSet, depth: int) -> list[int | None]:
    found: list[int | None] = []
    for question in questions.questions:
        items = (await engine.recall(SPACE, question.question, limit=depth)).items
        found.append(next((rank for rank, item in enumerate(items, 1) if question.quote in _normal(item.text)), None))
    return found


async def overlap(engine: MemoryEngine, questions: QuestionSet, lane: dict) -> list[float]:
    """For each evaluation question, the largest word overlap (Jaccard) with a
    lane question of a chunk holding its quote; 0 where no such chunk has one."""
    kept = {entry["chunk_id"]: entry["questions"] for entry in lane["kept"]}
    chunks = {chunk.chunk_id: chunk for chunk in await engine.documents.get_chunks(SPACE, list(kept))}
    scores: list[float] = []
    for question in questions.questions:
        asked = set(tokenize(question.question))
        best = 0.0
        for chunk_id, lane_questions in kept.items():
            if chunk_id in chunks and question.quote in _normal(chunks[chunk_id].text):
                for other in lane_questions:
                    words = set(tokenize(other))
                    if asked | words:
                        best = max(best, len(asked & words) / len(asked | words))
        scores.append(best)
    return scores


async def run_measure(args: argparse.Namespace) -> None:
    data = Path(args.data)
    questions, sets = evaluation_questions(data)
    lane = json.loads((data / "lane.json").read_text())
    engines = {"off": await engine_at(data, question_lane=False), "on": await engine_at(data, question_lane=True)}
    seconds: dict[str, list[float]] = {"off": [], "on": []}
    reports: dict[str, dict] = {}
    for repeat in range(args.repeats):
        for name in (("off", "on") if repeat % 2 == 0 else ("on", "off")):
            started = time.perf_counter()
            report = await measure(engines[name], SPACE, questions, ks=(1, 5, 10))
            seconds[name].append(time.perf_counter() - started)
            payload = report.as_payload()
            if name in reports and reports[name] != payload:
                raise SystemExit(f"lane {name}: repeat {repeat} scored differently; recall is not deterministic here")
            reports[name] = payload
    off_ranks, on_ranks = await ranks(engines["off"], questions, 10), await ranks(engines["on"], questions, 10)
    depth = 11
    closeness = await overlap(engines["on"], questions, lane)
    up = [c for a, b, c in zip(off_ranks, on_ranks, closeness) if (b or depth) < (a or depth)]
    down = [c for a, b, c in zip(off_ranks, on_ranks, closeness) if (b or depth) > (a or depth)]
    result = {
        "questions": len(questions.questions), "eval_sets": sets, "lane": {k: v for k, v in lane.items() if k != "kept"},
        "off": reports["off"], "on": reports["on"],
        "measure_seconds": {name: [round(s, 3) for s in values] for name, values in seconds.items()},
        "measure_seconds_median": {name: round(statistics.median(values), 3) for name, values in seconds.items()},
        "moved_up_at_10": len(up), "moved_down_at_10": len(down),
        "moved_up_with_overlap_at_least_half": sum(1 for c in up if c >= 0.5),
        "moved_down_with_overlap_at_least_half": sum(1 for c in down if c >= 0.5),
        "eval_to_lane_question_overlap": {"median": round(statistics.median(closeness), 3) if closeness else None,
                                          "at_least_half": sum(1 for c in closeness if c >= 0.5),
                                          "none": sum(1 for c in closeness if c == 0)},
        "seconds_per_100_chunks": round(lane["seconds"] / lane["chunks_asked"] * 100, 1) if lane["chunks_asked"] else None,
        "model_calls_per_100_chunks": round(lane["model_calls"] / lane["chunks_asked"] * 100, 1) if lane["chunks_asked"] else None,
    }
    (data / "results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))
    for engine in engines.values():
        await engine.close()


def _scores(report: QuestionReport) -> dict[str, object]:
    return {"questions": report.questions, "r@1": round(report.quote_at[1], 3), "r@5": round(report.quote_at[5], 3),
            "r@10": round(report.quote_at[10], 3), "mrr": round(report.mrr, 3), "unfound": report.unfound}


async def run_sweep(args: argparse.Namespace) -> None:
    data = Path(args.data)
    sets = [QuestionSet.load(path) for path in sorted(data.glob("eval-questions-*.json"))]
    chosen = [s for s in sets if s.seed == args.choose_seed]
    if len(sets) < 2 or len(chosen) != 1:
        raise SystemExit(f"sweep needs the set of seed {args.choose_seed} to choose on and another to hold out")
    chosen_on = chosen[0]
    seen = {_normal(question.question).casefold() for question in chosen_on.questions}
    held_out = dataclasses.replace(chosen_on, questions=tuple(
        question for other in sets if other is not chosen_on for question in other.questions if _normal(question.question).casefold() not in seen))
    splits = {f"choose (seed {chosen_on.seed})": chosen_on, "held out (" + ", ".join(f"seed {s.seed}" for s in sets if s is not chosen_on)
              + f", less questions in seed {chosen_on.seed})": held_out}
    engines = {"off": await engine_at(data, question_lane=False), "on": await engine_at(data, question_lane=True)}
    result: dict[str, object] = {"off": {name: _scores(await measure(engines["off"], SPACE, questions))
                                         for name, questions in splits.items()}}
    default = recall_module.CONTEXT_WEIGHT
    try:
        for weight in args.weights:
            recall_module.CONTEXT_WEIGHT = weight
            result[f"on at {weight}"] = {name: _scores(await measure(engines["on"], SPACE, questions))
                                         for name, questions in splits.items()}
    finally:
        recall_module.CONTEXT_WEIGHT = default
    (data / f"sweep-{chosen_on.seed}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))
    for engine in engines.values():
        await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=("build-eval", "build-lane", "measure", "sweep"))
    parser.add_argument("--data", required=True, help="a directory for the stores, the sets and the results")
    parser.add_argument("--docs", nargs="+", default=list(DOCS), help="documents from docs/ to store")
    parser.add_argument("--url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="llama3.2-ctx8k")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--per-chunk", type=int, default=3)
    parser.add_argument("--eval-chunks", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--choose-seed", type=int, default=42, help="sweep: the evaluation set a weight is chosen on")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    args = parser.parse_args()
    asyncio.run({"build-eval": build_eval, "build-lane": build_lane, "measure": run_measure, "sweep": run_sweep}[args.phase](args))


if __name__ == "__main__":
    main()
