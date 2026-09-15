"""Broad questions, four synthesis modes and the reference's TreeSummarize, one local model, one local judge.

Usage: PYTHONPATH=src python benchmarks/synthesis_modes.py <longmemeval_s.json> <out_dir> [n] [seed] [model] [k] [runs] [sides]
Needs a local Ollama at 127.0.0.1:11434 and llama-index-core installed.
The scoreboard's protocol (comparative-synthesis-v1, 2026-09-13) with the
modes added: for n LongMemEval-S multi-session items (stratified, seed 42),
recall the top k passages with our engine (HashEmbedder, in memory) and
have each side write from the same passages with the same local model:
our evidence mode (notes, then a fold), refine, accumulate, facts
(quoted facts from each passage, then an answer written from them), and
LlamaIndex's TreeSummarize. ``sides``, a comma-separated list, runs only
those sides (default: all five). Sides run one after another per item, their
order rotated item by item. Each written text is judged by the same model
for faithfulness to the passages and relevancy to the question. Our sides
also report the mechanical quoted share: notes kept with a checked quote
over notes the model returned, leaving out on both sides the sentences a
refine rewrite repeated from the answer so far (``notes.carried``), so the
share is of sentences the model wrote new and compares across modes.
For ``facts`` the notes are the facts, so the share is facts kept with a
checked quote over facts returned; its row also counts the facts the
shown answer used, the facts the answer call's bound left unsent, and
the answer sentences dropped for citing no fact.
Rows are appended to run<r>/rows.jsonl as they finish, so a stopped run
keeps what it measured. With runs > 1 the whole protocol repeats, each
item's side order rotated one further per run, and report.json gives
every run's aggregate and the median of each number across runs.
"""
import asyncio, json, statistics, sys, time
from pathlib import Path

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, Record
from scone_memory.bench.comparative import llamaindex_summary
from scone_memory.bench.evaluators import answer_relevancy, faithfulness
from scone_memory.bench.runner import load_items, stratified_sample
from scone_memory.providers.llm import OpenAICompatibleChat
from scone_memory.retrieval import synthesis

EXPECTED = ['gpt4_2ba83207', 'ba358f49', 'c18a7dc8', 'a4996e51', 'a3332713', '60bf93ed', '2318644b', '00ca467f']
OURS = ("evidence", "refine", "accumulate", "facts")
SIDES = (*OURS, "treesummarize")


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


async def judge_text(judge, question, text, contexts):
    if not text.strip():
        return {"faithfulness": None, "relevancy": None, "judged": False}
    f = await faithfulness(judge, answer=text, contexts=contexts)
    r = await answer_relevancy(judge, question=question, answer=text)
    return {"faithfulness": f.score if f.verified else None, "faithfulness_verified": f.verified,
            "faithfulness_reasons": list(f.reasons)[:6],
            "relevancy": r.score if r.verified else None, "relevancy_verified": r.verified, "judged": True}


async def one(item, model, judge, k, order):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    records = [Record(content="\n".join(s), kind="conversation", source=item.session_ids[i],
                      created_at=item.session_dates[i] if i < len(item.session_dates) and item.session_dates[i] else None)
               for i, s in enumerate(item.sessions) if "\n".join(s).strip() and i < len(item.session_ids)]
    await engine.remember_many("item", records)
    found = await engine.recall("item", item.question, limit=k)
    passages = synthesis.passages_from_recall(found.items)
    evidence = set(item.answer_session_ids)
    retrieved_sessions = {p.source for p in passages}
    texts = [p.text for p in passages]
    limits = synthesis.SynthesisLimits(max_passages=max(k, len(passages)), max_round_bytes=6_000, max_rounds=16,
                                       max_sentences=12, timeout_s=600.0)
    row = {"question_id": item.question_id, "question": item.question,
           "evidence_sessions": len(evidence), "order": list(order),
           "retrieved_evidence_share": round(len(evidence & retrieved_sessions) / len(evidence), 3) if evidence else None,
           "passages": len(passages)}
    for side in order:
        t0 = time.perf_counter()
        if side == "treesummarize":
            theirs = await llamaindex_summary(model, item.question, texts)
            seconds = time.perf_counter() - t0
            row[side] = {"failed": theirs.failed, "model_calls": theirs.model_calls, "cites": theirs.cites,
                         "seconds": round(seconds, 1), "text": theirs.text}
            text = theirs.text
        else:
            made = await synthesis.synthesize_passages(model, item.question, passages, limits=limits, mode=side)
            seconds = time.perf_counter() - t0
            text = " ".join(s.text for s in made.sentences)
            cited = {c.passage_id for s in made.sentences for c in s.citations}
            cited_sessions = {p.source for p in passages if p.id in cited}
            returned = sum(r.notes_returned for r in made.rounds)
            fresh = returned - made.notes_carried
            dropped = made.notes_dropped_unquoted + made.notes_dropped_unknown + made.notes_dropped_malformed
            row[side] = {"status": made.status, "sentences": len(made.sentences), "model_calls": made.model_calls,
                         "rounds": len(made.rounds), "passages_read": made.passages_read,
                         "passages_cited": made.passages_cited,
                         "cited_evidence_share": round(len(evidence & cited_sessions) / len(evidence), 3) if evidence else None,
                         "notes_returned": returned, "notes_kept": made.notes_kept, "notes_dropped": dropped,
                         "notes_carried": made.notes_carried, "refine_dropped_carried": made.refine_dropped_carried,
                         "notes_dropped_unquoted": made.notes_dropped_unquoted,
                         "notes_dropped_unknown": made.notes_dropped_unknown,
                         "notes_dropped_malformed": made.notes_dropped_malformed,
                         "quoted_share": round((made.notes_kept - made.notes_carried) / fresh, 3) if fresh else None,
                         "folded": made.folded, "fold_dropped_uncited": made.fold_dropped_uncited,
                         "facts_used": made.facts_used, "facts_unsent": made.facts_unsent,
                         "refine_kept_prior": made.refine_kept_prior, "reasons": list(made.reasons),
                         "round_notes": [[r.notes_returned, r.notes_kept, r.notes_carried] for r in made.rounds],
                         "seconds": round(seconds, 1), "text": made.text()}
        row[side].update(await judge_text(judge, item.question, text, texts))
    return row


def aggregate(rows, sides=SIDES):
    agg = {}
    for side in sides:
        present = [r[side] for r in rows if side in r]
        entry = {"items_with_answer": sum(1 for s in present if s["judged"]),
                 "faithfulness": mean(s.get("faithfulness") for s in present),
                 "faithfulness_judged": sum(1 for s in present if s.get("faithfulness") is not None),
                 "relevancy": mean(s.get("relevancy") for s in present),
                 "relevancy_judged": sum(1 for s in present if s.get("relevancy") is not None),
                 "model_calls": sum(s["model_calls"] for s in present),
                 "seconds": round(sum(s["seconds"] for s in present), 1)}
        if side in OURS:
            returned = sum(s["notes_returned"] for s in present)
            carried = sum(s["notes_carried"] for s in present)
            fresh_kept = sum(s["notes_kept"] for s in present) - carried
            entry.update({"quoted_share": round(fresh_kept / (returned - carried), 3) if returned - carried else None,
                          "quoted": fresh_kept, "written": returned - carried, "notes_carried": carried,
                          "refine_dropped_carried": sum(s["refine_dropped_carried"] for s in present),
                          "notes_returned": returned, "notes_kept": sum(s["notes_kept"] for s in present),
                          "notes_dropped": sum(s["notes_dropped"] for s in present),
                          "passages_read": sum(s["passages_read"] for s in present),
                          "passages_cited": sum(s["passages_cited"] for s in present),
                          "cited_evidence_share": mean(s["cited_evidence_share"] for s in present),
                          "refine_kept_prior": sum(s["refine_kept_prior"] for s in present),
                          "folded": sum(1 for s in present if s["folded"]),
                          "fold_dropped_uncited": sum(s["fold_dropped_uncited"] for s in present),
                          "facts_used": sum(s["facts_used"] for s in present),
                          "facts_unsent": sum(s["facts_unsent"] for s in present),
                          "statuses": {k: sum(1 for s in present if s["status"] == k)
                                       for k in ("synthesized", "partial", "unavailable", "no_evidence")}})
        else:
            entry["failed"] = sum(1 for s in present if s["failed"])
        agg[side] = entry
    agg["retrieved_evidence_share"] = mean(r["retrieved_evidence_share"] for r in rows)
    return agg


MEDIAN_KEYS = ("items_with_answer", "faithfulness", "relevancy", "quoted_share", "model_calls", "passages_read",
               "passages_cited", "cited_evidence_share", "refine_kept_prior", "refine_dropped_carried", "notes_carried",
               "notes_kept", "notes_dropped", "folded", "fold_dropped_uncited", "facts_used", "facts_unsent")


def medians(aggregates, sides=SIDES):
    """Each number's median across runs, beside its spread; a number a run lacks is left out of that median."""
    out = {}
    for side in sides:
        entry = {}
        for key in MEDIAN_KEYS:
            values = [a[side][key] for a in aggregates if a[side].get(key) is not None]
            if values:
                entry[key] = {"median": round(statistics.median(values), 3), "min": min(values), "max": max(values),
                              "runs": len(values)}
        out[side] = entry
    return out


async def one_run(items, model, model_name, k, seed, out: Path, run: int, sides=SIDES):
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    for idx, item in enumerate(items):
        turn = (idx + run - 1) % len(sides)
        order = sides[turn:] + sides[:turn]
        row = await one(item, model, model, k, order)
        rows.append(row)
        with (out / "rows.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"run {run} {idx + 1}/{len(items)} {item.question_id} " + " | ".join(
            f"{s}: {row[s].get('status', 'failed' if row[s].get('failed') else 'written')} f={row[s].get('faithfulness')} "
            f"r={row[s].get('relevancy')} calls={row[s]['model_calls']} {row[s]['seconds']}s" for s in sides), flush=True)
    wall = round(time.perf_counter() - started, 1)
    record = {"protocol": "comparative-synthesis-modes-v1", "run": run, "dataset": "LongMemEval-S multi-session", "n": len(rows),
              "seed": seed, "k": k, "sides": list(sides), "model": model_name, "judge": model_name, "judge_is_writer": True, "embedder": "hash",
              "limits": {"max_round_bytes": 6000, "max_rounds": 16, "max_sentences": 12, "timeout_s": 600.0},
              "llamaindex": {"version": __import__("llama_index.core").core.__version__, "synthesizer": "TreeSummarize"},
              "wall_seconds": wall, "aggregate": aggregate(rows, sides), "items": rows}
    (out / "report.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record["aggregate"], indent=1), flush=True)
    return record


async def main(data: Path, out: Path, n: int, seed: int, model_name: str, k: int, runs: int, sides=SIDES):
    items = [i for i in load_items(data) if i.question_type == "multi-session"]
    items = stratified_sample(items, n, seed=seed)
    if n == 8 and seed == 42:
        assert [i.question_id for i in items] == EXPECTED, "not the scoreboard's items"
    model = OpenAICompatibleChat("http://127.0.0.1:11434/v1", model_name, timeout=600.0)
    records = [await one_run(items, model, model_name, k, seed, out / f"run{run}", run, sides) for run in range(1, runs + 1)]
    aggregates = [r["aggregate"] for r in records]
    summary = {"protocol": "comparative-synthesis-modes-v1", "runs": runs, "model": model_name, "k": k, "n": len(items),
               "sides": list(sides), "wall_seconds": [r["wall_seconds"] for r in records], "per_run": aggregates,
               "median": medians(aggregates, sides)}
    (out / "report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["median"], indent=1), flush=True)


if __name__ == "__main__":
    data = Path(sys.argv[1]); out = Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    model = sys.argv[5] if len(sys.argv) > 5 else "llama3.1-ctx8k:latest"; k = int(sys.argv[6]) if len(sys.argv) > 6 else 12
    runs = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    sides = tuple(sys.argv[8].split(",")) if len(sys.argv) > 8 else SIDES
    unknown = [side for side in sides if side not in SIDES]
    if unknown or not sides:
        sys.exit(f"sides must be drawn from {', '.join(SIDES)}; not {', '.join(unknown) or 'none'}")
    asyncio.run(main(data, out, n, seed, model, k, runs, sides))
