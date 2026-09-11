"""Graph quality, measured on a versioned synthetic fixture.

The benchmark loads gold-labelled facts, projects them, and scores what a
person would check by hand: whether names of one thing stay one entity,
whether values stay values and things stay things, whether expected
connections are found and absent ones are not, and how large the views
are. The deterministic part of the report hashes to the same artefact on
every run; thresholds turn a regression into a failure.
"""
from __future__ import annotations

import json
from pathlib import Path

from scone_memory.entities import project as project_module
from scone_memory.entities.classify import ObjectClassification
from scone_memory.testing.entity_graph_benchmark import THRESHOLDS_V1, failures, run_entity_graph_benchmark

FIXTURE = Path(__file__).resolve().parents[2] / "benchmarks" / "entity_graph" / "fixtures-v1.jsonl"


async def test_the_fixture_scores_as_recorded_and_passes_its_thresholds():
    report = await run_entity_graph_benchmark(FIXTURE)
    assert report.literal_error_rate == 0.0 and report.connected_claim_share == 1.0
    assert report.path_recall == {2: 1.0, 3: 1.0} and report.path_false_positives == 0
    assert failures(report, THRESHOLDS_V1) == []


async def test_names_of_one_thing_on_different_keys_are_counted_as_fragments():
    """Before identity decisions exist, "dr. alice chen" is its own entity:
    the alice cluster is two entities, which later work must bring to one."""
    report = await run_entity_graph_benchmark(FIXTURE)
    assert report.cluster_fragments["alice chen"] == 2 and report.cluster_fragments["acme robotics"] == 1
    assert 0 < report.alias_bcubed_f1 < 1


async def test_two_runs_give_the_same_artefact():
    first, second = await run_entity_graph_benchmark(FIXTURE), await run_entity_graph_benchmark(FIXTURE)
    assert first.artefact_sha256 == second.artefact_sha256 and len(first.artefact_sha256) == 64


async def test_a_classifier_that_calls_everything_a_thing_fails_the_gate(monkeypatch):
    monkeypatch.setattr(project_module, "classify_object",
                        lambda text, predicate, context: ObjectClassification("entity", None, "name_shape"))
    report = await run_entity_graph_benchmark(FIXTURE)
    assert report.literal_error_rate > 0 and any("literal_error_rate" in failure for failure in failures(report,
                                                                                                          THRESHOLDS_V1))


def test_the_cli_prints_the_report_and_fails_on_a_breach(capsys):
    from scone_memory.runtime.cli import main

    assert main(["bench-graph", "--fixtures", str(FIXTURE), "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["report"]["connected_claim_share"] == 1.0 and printed["failures"] == []


async def test_a_path_found_where_none_should_be_is_a_false_positive(tmp_path):
    fixture = tmp_path / "negative.jsonl"
    fixture.write_text("\n".join(json.dumps(row) for row in [
        {"kind": "meta", "name": "negative", "space": "bench"},
        {"kind": "fact", "subject": "ana", "predicate": "knows", "object": "Ben"},
        {"kind": "fact", "subject": "ben", "predicate": "knows", "object": "Cy"},
        {"kind": "path", "from": "ana", "to": "cy", "hops": 2, "expected": False},
    ]))
    report = await run_entity_graph_benchmark(fixture)
    assert report.path_false_positives == 1
    assert any(failure.startswith("path_false_positives") for failure in failures(report, THRESHOLDS_V1))
