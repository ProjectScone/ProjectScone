"""The hot-path benchmark measures the same work every time it is run.

Its before/after numbers are only comparable if the corpus, the queries and
the recall outputs cannot drift between two runs; these tests hold it to that
on a corpus small enough for a unit test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from ..paths import PACKAGE_ROOT

SCRIPT = PACKAGE_ROOT / "benchmarks" / "hot_paths.py"
spec = importlib.util.spec_from_file_location("hot_paths_benchmark", SCRIPT)
assert spec is not None and spec.loader is not None
hot_paths = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hot_paths
spec.loader.exec_module(hot_paths)

DOCUMENTS = [
    hot_paths.Document("docs/recall.md", "# Recall\n\nRecall fuses the vector lane with the text lane.\n\n"
                       "Scores are normalised so the top item is one.\n"),
    hot_paths.Document("src/pkg/engine.py", "class Engine:\n    def recall(self, query):\n        return query\n\n"
                       "def fuse(lanes):\n    return sorted(lanes)\n"),
    hot_paths.Document("docs/ingest.md", "Ingestion cuts records into chunks and embeds every chunk.\n"),
]


def test_nearest_rank_percentiles_pick_a_measured_value():
    values = [float(value) for value in range(1, 101)]
    assert hot_paths.percentile(values, 0.50) == 50.0
    assert hot_paths.percentile(values, 0.95) == 95.0
    assert hot_paths.percentile([3.0], 0.95) == 3.0


def test_the_queries_are_the_same_words_on_every_run():
    first = hot_paths.queries(DOCUMENTS, 30)
    assert first == hot_paths.queries(DOCUMENTS, 30)
    assert hot_paths.queries(DOCUMENTS, 12) == first[:12], "a shorter run asks a prefix of a longer one"
    assert all(2 <= len(query.split()) <= 5 for query in first)


async def test_both_stores_report_every_metric_and_recall_the_same_way_twice(tmp_path: Path):
    asked = hot_paths.queries(DOCUMENTS, 8)
    for store in ("memory", "sqlite"):
        runs = []
        for attempt in range(2):
            workdir = tmp_path / f"{store}-{attempt}"
            workdir.mkdir()
            report, outputs, profiles = await hot_paths.measure(store, DOCUMENTS, asked, workdir)
            runs.append(outputs)
        assert profiles == {}
        assert report["store"] == store and report["documents"] == 3 and report["queries"] == 8
        assert report["chunks"] >= 3 and report["degraded_recalls"] == 0
        for metric in ("ingest_seconds", "ingest_cpu_seconds", "recall_p50_ms", "recall_p95_ms", "recall_cpu_p50_ms"):
            assert report[metric] >= 0
        assert runs[0] == runs[1]
        assert any(output["items"] for output in runs[0])
