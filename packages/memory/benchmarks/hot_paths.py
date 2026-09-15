"""Recall latency and ingestion throughput on a fixed corpus, for before/after.

Run from packages/memory with the tree under test on PYTHONPATH:

    PYTHONPATH=src python benchmarks/hot_paths.py --json out.json --dump recalls.json

The corpus is this package's own docs and source read from one pinned git
revision, never the working tree, so a change to the code under test does not
change what it is measured on. Every record gets one fixed timestamp and the
engine one fixed clock, so recency cannot move a score between runs. The 200
queries are drawn from the corpus by a seeded generator.

Wall-clock numbers follow whatever else the machine is doing; the CPU-time
numbers beside them (``time.process_time``, this process only) do not, which
is what makes an A/B comparison on a shared machine readable.

Hash embeddings exercise chunking, storage and the fusion path, not semantic
quality. ``--dump`` keeps every recall's chunk ids and scores so two trees can
be shown to return the same thing; ``--profile`` writes cProfile stats.
"""
from __future__ import annotations

import argparse
import asyncio
import cProfile
from dataclasses import dataclass
import io
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

#: The corpus revision: origin/main when this benchmark was written.
CORPUS_REV = "acc6492e0fe08e8c83dab11a90b096c38e63ba1e"
CORPUS_ROOTS = ("packages/memory/docs", "packages/memory/src")
TEXT_SUFFIXES = (".md", ".py", ".txt", ".rst")
QUERY_SEED = 20260914
WHEN = "2026-09-01T00:00:00Z"
SPACE = "bench"
BATCH = 32
LIMIT = 10
WARMUP = 5
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


@dataclass(frozen=True)
class Document:
    path: str
    text: str


def corpus(repo: Path, rev: str = CORPUS_REV) -> list[Document]:
    """Every text file under the corpus roots at ``rev``, in path order."""
    archive = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", rev, *CORPUS_ROOTS],
                             check=True, capture_output=True).stdout
    documents: list[Document] = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or not member.name.endswith(TEXT_SUFFIXES):
                continue
            handle = bundle.extractfile(member)
            if handle is None:
                continue
            text = handle.read().decode("utf-8")
            if text.strip():
                documents.append(Document(member.name, text))
    return sorted(documents, key=lambda document: document.path)


def queries(documents: list[Document], count: int, seed: int = QUERY_SEED) -> list[str]:
    """``count`` queries of two to five words taken from one line of a document."""
    chooser = random.Random(seed)
    lines = [words for document in documents for line in document.text.splitlines()
             if len(words := _WORD.findall(line)) >= 3]
    picked: list[str] = []
    while len(picked) < count:
        words = chooser.choice(lines)
        size = chooser.randint(2, min(5, len(words)))
        start = chooser.randint(0, len(words) - size)
        picked.append(" ".join(words[start:start + size]))
    return picked


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: the smallest value at or above ``fraction`` of them."""
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * fraction // 1))
    return ordered[int(rank) - 1]


async def measure(store: str, documents: list[Document], asked: list[str], workdir: Path,
                  profiling: bool = False) -> tuple[dict[str, object], list[dict[str, object]], dict[str, cProfile.Profile]]:
    from scone_memory import HashEmbedder, MemoryEngine
    from scone_memory.backends import (InMemoryDocumentStore, InMemoryVectorIndex, SqliteDocumentStore,
                                       SqliteVectorIndex)
    from scone_memory.ingestion.records import Record

    if store == "memory":
        engine = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=lambda: WHEN)
    else:
        path = workdir / "hot_paths.db"
        engine = MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder(), clock=lambda: WHEN)
    await engine.open()
    try:
        records = [Record(document.text, source=document.path, created_at=WHEN) for document in documents]
        profiles = {"ingest": cProfile.Profile(), "recall": cProfile.Profile()} if profiling else {}
        if profiling:
            profiles["ingest"].enable()
        started, started_cpu = time.perf_counter(), time.process_time()
        chunks = 0
        for offset in range(0, len(records), BATCH):
            chunks += sum(added.chunks for added in await engine.remember_many(SPACE, records[offset:offset + BATCH]))
        ingest_seconds, ingest_cpu = time.perf_counter() - started, time.process_time() - started_cpu
        if profiling:
            profiles["ingest"].disable()
        for query in asked[:WARMUP]:
            await engine.recall(SPACE, query, limit=LIMIT)
        latencies: list[float] = []
        cpu: list[float] = []
        outputs: list[dict[str, object]] = []
        degraded = 0
        if profiling:
            profiles["recall"].enable()
        for query in asked:
            began, began_cpu = time.perf_counter(), time.process_time()
            result = await engine.recall(SPACE, query, limit=LIMIT)
            latencies.append((time.perf_counter() - began) * 1000)
            cpu.append((time.process_time() - began_cpu) * 1000)
            degraded += bool(result.degraded)
            outputs.append({"items": [[item.chunk_id, item.episode_id, repr(item.score),
                                       None if item.similarity is None else repr(item.similarity), item.lanes,
                                       item.start, item.end, item.first_line, item.last_line, item.declaration,
                                       item.superseded]
                                      for item in result.items],
                            "facts": [fact.fact_id for fact in result.facts], "degraded": result.degraded,
                            "top_similarity": result.top_similarity, "space_bytes": result.space_bytes})
        if profiling:
            profiles["recall"].disable()
    finally:
        await engine.close()
    return {
        "store": store,
        "documents": len(documents),
        "chunks": chunks,
        "ingest_seconds": round(ingest_seconds, 4),
        "chunks_per_second": round(chunks / ingest_seconds, 1) if ingest_seconds else None,
        "ingest_cpu_seconds": round(ingest_cpu, 4),
        "chunks_per_cpu_second": round(chunks / ingest_cpu, 1) if ingest_cpu else None,
        "queries": len(asked),
        "recall_p50_ms": round(percentile(latencies, 0.50), 3),
        "recall_p95_ms": round(percentile(latencies, 0.95), 3),
        "recall_mean_ms": round(statistics.fmean(latencies), 3),
        "recall_cpu_p50_ms": round(percentile(cpu, 0.50), 3),
        "recall_cpu_p95_ms": round(percentile(cpu, 0.95), 3),
        #: Recalls that said a lane was degraded or behind: a latency
        #: measured over a lane that did not run is not a latency.
        "degraded_recalls": degraded,
    }, outputs, profiles


async def run(repo: Path, stores: list[str], count: int, rev: str,
              profile_dir: Path | None) -> tuple[list[dict[str, object]], dict[str, list[object]]]:
    documents = corpus(repo, rev)
    asked = queries(documents, count)
    reports: list[dict[str, object]] = []
    dumps: dict[str, list[object]] = {}
    for store in stores:
        with tempfile.TemporaryDirectory(prefix="scone-hot-paths-") as scratch:
            report, outputs, profiles = await measure(store, documents, asked, Path(scratch), profile_dir is not None)
        for phase, profile in profiles.items():
            profile.dump_stats(str(Path(profile_dir or ".") / f"hot_paths-{store}-{phase}.prof"))
        reports.append(report)
        dumps[store] = [{"query": query, **output} for query, output in zip(asked, outputs)]
    return reports, dumps


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--rev", default=CORPUS_REV)
    parser.add_argument("--stores", default="memory,sqlite")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--json", type=Path, help="write the metrics here")
    parser.add_argument("--dump", type=Path, help="write every recall's ids and scores here")
    parser.add_argument("--profile", type=Path, help="write cProfile stats into this directory")
    options = parser.parse_args(argv)
    stores = [store for store in options.stores.split(",") if store]
    unknown = sorted(set(stores) - {"memory", "sqlite"})
    if unknown:
        parser.error(f"unknown stores: {', '.join(unknown)}")
    if options.profile is not None:
        options.profile.mkdir(parents=True, exist_ok=True)
    reports, dumps = asyncio.run(run(options.repo, stores, options.queries, options.rev, options.profile))
    for report in reports:
        print(json.dumps(report), flush=True)
    if options.json is not None:
        import scone_memory

        options.json.write_text(json.dumps({"rev": options.rev, "python": sys.version.split()[0],
                                            "scone_memory": str(Path(scone_memory.__file__).parent),
                                            "reports": reports}, indent=2) + "\n")
    if options.dump is not None:
        options.dump.write_text(json.dumps(dumps, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
