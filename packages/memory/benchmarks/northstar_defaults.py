"""The scoreboard's first row, swept: which retrieval defaults beat LlamaIndex's best.

LongMemEval-S, the Rust harness's stratified sample (``--n``, ``--seed``),
one embedder on both sides (``--embedder``: the hashed-token embedder, or a
local model such as ``bge-small-en-v1.5``), no language model and no reranker. The
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
reads (``MemoryEngine.vector_weight``), set between recalls, as is
``--exact-forms`` (``MemoryEngine.lexical_exact_forms``).

Run from packages/memory:

    PYTHONPATH=$PWD/src python benchmarks/northstar_defaults.py \\
        --data ../../bench-data/longmemeval_s.json --n 50 --seed 42 --out northstar.json

``--holdout-of 42:50`` draws the sample from the items *outside* the
seed-42 n=50 sample instead, to check a winner on items it was not
chosen on.

``--chunks`` names character targets (``700``) and token targets (``512t``,
``MemoryEngine(chunk_tokens=512)``, counted by the model's tokenizer when
it has one). With a real model the cost is embedding: ``--embedding-cache
PATH`` keeps every vector in one SQLite file (the engine's own
``SqliteEmbeddingCache``) that both sides read through, so each text is
embedded once for the run and a rerun embeds nothing; the result says what
each side embedded, what it read back, the seconds the model took, and
whether the cache dropped anything.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench.comparative import (CachedEmbedder, OneThreadCache, VectorCache, _scores, bench_embedder, compare,
                                            distinct_sessions, llamaindex_session_ranking, reference_tokenizer)
from scone_memory.bench.runner import BenchItem, load_items, stratified_sample
from scone_memory.core.ports import Embedder
from scone_memory.ingestion.embedding_cache import SqliteEmbeddingCache
from scone_memory.memory.engine import Record
from scone_memory.retrieval.fusion import PER_EPISODE_CAP

KS = (5, 10, 15)
RECALL_LIMIT = max(KS) * PER_EPISODE_CAP
#: Vectors the run's cache holds before it drops the least recently used.
#: Both samples at every chunking are about a quarter of a million; the
#: result reports ``evicted``, and a non-zero count means a text was
#: embedded twice.
CACHE_ENTRIES = 1_000_000


@dataclass(frozen=True)
class Chunking:
    """One point on the chunk axis: a character target, or a token target."""

    characters: Optional[int]
    tokens: Optional[int]

    @property
    def label(self) -> str:
        return f"chunk={self.characters}" if self.tokens is None else f"chunk_tokens={self.tokens}"


def chunkings(raw: str) -> list[Chunking]:
    """``700,2000,512t``: character targets, and token targets ending in ``t``."""
    found: list[Chunking] = []
    for part in raw.split(","):
        tokens = part.endswith("t")
        digits = part[:-1] if tokens else part
        if not digits.isdigit() or int(digits) < 1:
            raise SystemExit(f"--chunks takes positive character targets and token targets like 512t; not {part!r}")
        found.append(Chunking(None, int(digits)) if tokens else Chunking(int(digits), None))
    return found


async def open_engine(chunking: Chunking, embedder: Embedder) -> MemoryEngine:
    """A fresh in-memory engine cutting at the chunking's target."""
    if chunking.tokens is not None:
        return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder,
                                  chunk_tokens=chunking.tokens).open()
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder,
                              chunk_target=chunking.characters).open()


def embedders(name: str, *, model_cache: Optional[str], vector_cache: Optional[str],
              max_entries: int = CACHE_ENTRIES) -> tuple[Embedder, Embedder, Optional[VectorCache]]:
    """Ours and the reference's embedder: the one model, and with a vector
    cache a counting wrapper each over one shared SQLite cache."""
    model = bench_embedder(name, cache_dir=model_cache)
    if vector_cache is None:
        return model, model, None
    cache = OneThreadCache(lambda: SqliteEmbeddingCache(vector_cache, max_entries))
    return CachedEmbedder(model, cache), CachedEmbedder(model, cache), cache


def recount(embedder: Embedder) -> Embedder:
    """The same model over the same cache with its counts at zero, so a
    later phase's cost is its own; an uncounted embedder is itself."""
    if isinstance(embedder, CachedEmbedder):
        return CachedEmbedder(embedder.inner, embedder.cache)
    return embedder


def _costs(ours: Embedder, theirs: Embedder, cache: Optional[VectorCache]) -> dict[str, Any]:
    costs: dict[str, Any] = {"embedder": ours.id}
    for side, embedder in (("scone", ours), ("llamaindex", theirs)):
        if isinstance(embedder, CachedEmbedder):
            costs[side] = embedder.record()
    if cache is not None:
        costs["cache"] = cache.record()
    return costs


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


def _name(chunk: str, fusion: str, weight: Optional[float], diversity: Optional[float], lanes: Sequence[str],
          exact_forms: Optional[bool] = None, trust: Optional[bool] = None) -> str:
    # Rows name exact forms only when the sweep sets them, so older result files keep their names.
    suffix = "" if exact_forms is None else f" exact_forms={'on' if exact_forms else 'off'}"
    suffix += "" if trust is None else f" lane_trust={'on' if trust else 'off'}"
    if tuple(lanes) != ("vector", "text"):
        return f"{chunk} lanes={'+'.join(lanes)}{suffix}"
    return f"{chunk} fusion={fusion} vector_weight={weight} diversity={diversity}{suffix}"


def _row(rankings: list[tuple[Sequence[str], set[str]]]) -> dict[str, Any]:
    scores = _scores(rankings, KS)
    return {"R@5": scores.recall_any[5], "all@5": scores.recall_all[5], "R@10": scores.recall_any[10],
            "R@15": scores.recall_any[15], "MRR": scores.mrr}


async def sweep(items: list[BenchItem], *, chunkings: Sequence[Chunking], fusions: Sequence[str],
                weights: Sequence[float], diversities: Sequence[Optional[float]], log: Any,
                exact_forms: Sequence[Optional[bool]] = (None,),
                trusts: Sequence[Optional[bool]] = (None,),
                embedders: Optional[tuple[Embedder, Embedder]] = None,
                cache: Optional[VectorCache] = None) -> dict[str, Any]:
    """Every row over ``items``. ``embedders`` is ours and the reference's
    (the hashed-token embedder for both when unset)."""
    ours, embedder = embedders if embedders is not None else (HashEmbedder(), HashEmbedder())
    rankings: dict[str, list[list[str]]] = {}
    relevant: list[set[str]] = []
    default_weight: Optional[float] = None
    default_exact: Optional[bool] = None
    started = time.perf_counter()
    for position, item in enumerate(items, 1):
        relevant.append(set(item.answer_session_ids))
        for name, hybrid in (("llamaindex hybrid (BM25+vector, RRF, chunk 512 tokens)", True),
                             ("llamaindex default (vector, chunk 512 tokens)", False)):
            ranked = await llamaindex_session_ranking(item, embedder, k=max(KS), hybrid=hybrid)
            rankings.setdefault(name, []).append(list(ranked))
        records = _records(item)
        for chunking in chunkings:
            engine = await open_engine(chunking, ours)
            default_weight = engine.vector_weight
            default_exact = engine.lexical_exact_forms
            default_trust = engine.lane_trust
            await engine.remember_many("item", records)
            asks: list[tuple[str, float, Optional[float], tuple[str, ...], Optional[bool], Optional[bool]]] = [
                (fusion, weight, diversity, ("vector", "text"), exact, trust)
                for exact in exact_forms for trust in trusts
                for fusion in fusions for weight in weights for diversity in diversities]
            # A lane on its own has nothing to be trusted against, so the
            # single-lane rows are run once, at the engine's own setting.
            asks += [("rank", default_weight, None, lanes, exact, None) for exact in exact_forms
                     for lanes in (("text",), ("vector",))]
            for fusion, weight, diversity, lanes, exact, trust in asks:
                engine.vector_weight = weight
                engine.lexical_exact_forms = default_exact if exact is None else exact
                engine.lane_trust = default_trust if trust is None else trust
                pack = await engine.recall("item", item.question, limit=RECALL_LIMIT, fusion=fusion,
                                           diversity=diversity, lanes=lanes)
                folded = distinct_sessions([hit.source or "" for hit in pack.items], max(KS))
                rankings.setdefault(_name(chunking.label, fusion, weight, diversity, lanes, exact, trust),
                                    []).append(list(folded))
            engine.vector_weight = default_weight
            engine.lexical_exact_forms = default_exact
            engine.lane_trust = default_trust
        print(f"{position}/{len(items)} {time.perf_counter() - started:.0f}s", file=log, flush=True)
    rows = {name: _row([(ranked, answers) for ranked, answers in zip(per_item, relevant)])
            for name, per_item in rankings.items()}
    return {"rows": rows, "rankings": rankings, "engine_default_vector_weight": default_weight,
            "engine_default_exact_forms": default_exact, "embedding": _costs(ours, embedder, cache),
            "llamaindex_tokenizer": reference_tokenizer(embedder),
            "question_ids": [item.question_id for item in items]}


async def check(items: list[BenchItem], log: Any, embedders: Optional[tuple[Embedder, Embedder]] = None,
                cache: Optional[VectorCache] = None) -> dict[str, Any]:
    """``compare()`` itself at engine defaults with the hybrid reference: the
    official path the sweep's rows must agree with."""
    ours, theirs = embedders if embedders is not None else (HashEmbedder(), HashEmbedder())

    def make_engine() -> Any:
        return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), ours).open()

    report = await compare(items, make_engine, theirs, ks=KS, hybrid=True)
    record = report.record()
    print("compare() done", file=log, flush=True)
    scores: dict[str, Any] = {side: {"R@5": record["sides"][side]["recall_any"]["5"], "all@5": record["sides"][side]["recall_all"]["5"],
                   "R@10": record["sides"][side]["recall_any"]["10"], "R@15": record["sides"][side]["recall_any"]["15"],
                   "MRR": record["sides"][side]["mrr"]} for side in ("scone", "llamaindex")}
    scores["embedding"] = _costs(ours, theirs, cache)
    return scores


def _floats(raw: str) -> list[Optional[float]]:
    return [None if part in ("none", "off") else float(part) for part in raw.split(",")]


def _switches(raw: str) -> list[Optional[bool]]:
    known: dict[str, Optional[bool]] = {"default": None, "off": False, "on": True}
    parts = raw.split(",")
    if any(part not in known for part in parts):
        raise SystemExit(f"--exact-forms takes default, off or on, comma-separated; not {raw!r}")
    return [known[part] for part in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-of", help="SEED:N -- draw from the items outside that sample")
    parser.add_argument("--chunks", default="700,2000", help="character targets, and token targets like 512t")
    parser.add_argument("--fusions", default="rank,score,distribution")
    parser.add_argument("--weights", default="0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--diversities", default="none,0.3")
    parser.add_argument("--lane-trust", default="default",
                        help="off,on -- sweep MemoryEngine.lane_trust; 'default' leaves the engine's own")
    parser.add_argument("--exact-forms", default="default",
                        help="off,on -- sweep MemoryEngine.lexical_exact_forms; 'default' leaves the engine's own")
    parser.add_argument("--check", action="store_true", help="also run compare() at engine defaults")
    parser.add_argument("--embedder", default="hash", help="hash, or a local model such as bge-small-en-v1.5")
    parser.add_argument("--model-cache", help="where fastembed keeps a local model's files")
    parser.add_argument("--embedding-cache", help="a SQLite file of vectors both sides read and write")
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
    axis = chunkings(args.chunks)
    ours, theirs, cache = embedders(args.embedder, model_cache=args.model_cache, vector_cache=args.embedding_cache)
    started = time.perf_counter()
    result = asyncio.run(sweep(items, chunkings=axis, fusions=args.fusions.split(","),
                               weights=weights, diversities=_floats(args.diversities), log=sys.stderr,
                               exact_forms=_switches(args.exact_forms), trusts=_switches(args.lane_trust),
                               embedders=(ours, theirs), cache=cache))
    if args.check:
        result["compare_at_defaults"] = asyncio.run(check(items, sys.stderr, (recount(ours), recount(theirs)), cache))
    result["sample"] = {"dataset": Path(args.data).name, "n": args.n, "seed": args.seed, "holdout_of": args.holdout_of,
                        "scored": len(items), "stratified_by": "question_type"}
    result["run"] = {"argv": sys.argv, "python": platform.python_version(), "wall_seconds": round(time.perf_counter() - started),
                     "recall_limit": RECALL_LIMIT, "ks": list(KS), "embedder": ours.id}
    Path(args.out).write_text(json.dumps(result, indent=1))
    width = max(len(name) for name in result["rows"])
    for name, row in result["rows"].items():
        print(f"{name:<{width}}  " + "  ".join(f"{key} {value:.3f}" for key, value in row.items()))
    print("embedding:", json.dumps(result["embedding"]))
    if args.check:
        print("compare() at defaults:", json.dumps(result["compare_at_defaults"]))


if __name__ == "__main__":
    main()
