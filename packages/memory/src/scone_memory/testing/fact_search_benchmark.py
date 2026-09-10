"""Measure SQLite fact lookup against the original ledger scan, without models.

Uses disposable synthetic databases only. This measures lexical fact lookup,
not embedding retrieval, end-to-end answer latency, or generation accuracy.
"""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import sqlite3
from statistics import median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Sequence

from .. import HashEmbedder, InMemoryVectorIndex, MemoryEngine
from ..backends.sqlite import SqliteDocumentStore
from ..core.models import Fact
from ..core.ports import TextFilter

SPACE = "synthetic-benchmark"
WHEN = "2025-01-01T00:00:00Z"


class MeasuredStore(SqliteDocumentStore):
    scan_rows = 0
    point_reads = 0

    async def list_facts(self, space: str, include_closed: bool = False) -> list[Fact]:
        result = await super().list_facts(space, include_closed)
        self.scan_rows += len(result)
        return result

    async def get_fact(self, space: str, fact_id: int) -> Fact | None:
        self.point_reads += 1
        return await super().get_fact(space, fact_id)


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    scope: TextFilter | None = None


CASES = (
    Case("sparse-term", "quasar"),
    Case("common-terms", "records routine"),
    # Most higher-ranked candidates fail the source scope. This deliberately
    # exposes work that scales with matching postings, even with an index.
    Case("common-term-with-source-filter", "policy", TextFilter(tags=("blue",), where={"team": "blue"})),
)


def seed(store: SqliteDocumentStore, size: int) -> None:
    for identifier, team in ((1, "red"), (2, "blue")):
        store.conn.execute("""INSERT INTO episodes
            (id,space,kind,content,content_hash,source,tags,metadata,created_at,ingested_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""", (identifier, SPACE, "file", "Synthetic benchmark source.",
            str(identifier), f"fixture/{team}", json.dumps([team]), json.dumps({"team": team}), WHEN, WHEN))

    def rows() -> Iterator[tuple[int, str, str, str, str, float, str, str, int]]:
        for identifier in range(1, size+1):
            allowed = identifier % 10 == 0 or identifier >= size-1
            yield (identifier, SPACE, f"node{identifier}", "records",
                   "quasar release policy" if identifier >= size-2 else "routine maintenance policy",
                   .5 if allowed else .9, "2024-01-01T00:00:00Z", "active", 2 if allowed else 1)

    store.conn.executemany("""INSERT INTO facts
        (id,space,subject,predicate,object,confidence,valid_from,status,source_episode_id)
        VALUES (?,?,?,?,?,?,?,?,?)""", rows())
    store.conn.commit()


async def measure(engine: MemoryEngine, store: MeasuredStore, case: Case, *, indexed: bool,
                  repeats: int) -> dict[str, object]:
    lookup = engine._facts_for_query if indexed else engine._scan_facts_for_query
    samples: list[float] = []
    expected: list[int] | None = None
    store.scan_rows = store.point_reads = 0
    for _ in range(repeats):
        started = perf_counter()
        facts = await lookup(SPACE, case.query, WHEN, scope=case.scope)
        samples.append((perf_counter()-started)*1000)
        identifiers = [fact.fact_id for fact in facts]
        if expected is not None and identifiers != expected:
            raise AssertionError("Fact lookup changed results during repeated reads")
        expected = identifiers
    return {"median_ms": median(samples), "samples_ms": samples, "fact_ids": expected,
            "ledger_rows_materialized": store.scan_rows, "fact_point_reads": store.point_reads}


async def benchmark_size(size: int, repeats: int) -> dict[str, object]:
    if type(size) is not int or not 100 <= size <= 250_000:
        raise ValueError("size must be an integer from 100 to 250000")
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError("repeats must be an integer from 1 to 20")
    with TemporaryDirectory(prefix="scone-fact-benchmark-") as folder:
        store = MeasuredStore(Path(folder)/"synthetic.db")
        try:
            seed(store, size)
            engine = MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder())
            started = perf_counter()
            cold = await store.search_facts(SPACE, "quasar", WHEN, 10)
            cold_ms = (perf_counter()-started)*1000
            if {fact.fact_id for fact in cold} != {size-2, size-1, size}:
                raise AssertionError("Sparse fixture lookup omitted the final ledger records")
            measurements: list[dict[str, object]] = []
            for case in CASES:
                scan = await measure(engine, store, case, indexed=False, repeats=repeats)
                indexed = await measure(engine, store, case, indexed=True, repeats=repeats)
                if scan["fact_ids"] != indexed["fact_ids"]:
                    raise AssertionError(f"Indexed results differ from scan: {case.name}")
                if indexed["ledger_rows_materialized"] != 0:
                    raise AssertionError("Indexed engine unexpectedly fell back to a full ledger scan")
                measurements.append({"case": case.name, "identical_results": True,
                                     "scan": scan, "indexed": indexed})
            return {"facts": size, "repeats": repeats, "cold_index_and_first_query_ms": cold_ms,
                    "postings": store.conn.execute("SELECT count(*) FROM fact_search_postings").fetchone()[0],
                    "cases": measurements}
        finally:
            store.conn.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 50000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path; never overwritten.")
    options = parser.parse_args(argv)
    if len(options.sizes) > 6 or len(set(options.sizes)) != len(options.sizes) or any(not 100 <= n <= 250_000 for n in options.sizes):
        parser.error("provide up to six distinct sizes from 100 to 250000")
    if not 1 <= options.repeats <= 20:
        parser.error("--repeats must be from 1 to 20")
    # Reserve before any workload. Reports cannot overwrite an earlier result.
    with options.output.open("x", encoding="utf-8") as output:
        report: dict[str, object] = {"schema_version": 1, "python": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version, "platform": platform.system(), "machine": platform.machine(),
            "scope": "synthetic SQLite lexical fact lookup; not end-to-end RAG or generation accuracy",
            "status": "running"}
        try:
            report["runs"] = [asyncio.run(benchmark_size(size, options.repeats)) for size in options.sizes]
            report["status"] = "completed"
        except BaseException:
            report["status"] = "failed"
            raise
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
