"""Graph quality, measured on a versioned synthetic fixture.

The benchmark loads gold-labelled facts, projects them, and scores what a
person would check by hand: whether names of one thing stay one entity,
whether values stay values and things stay things, whether expected
connections are found and absent ones are not, and how large the views
are. The deterministic part of the report hashes to the same artefact on
every run; thresholds turn a regression into a failure.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from scone_memory.entities import project as project_module
from scone_memory.entities.classify import ObjectClassification
from scone_memory.testing import entity_graph_benchmark as benchmark_module
from scone_memory.testing.entity_graph_benchmark import (
    THRESHOLDS_V1,
    FixtureError,
    failures,
    run_entity_graph_benchmark,
)

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


def _projected_without(monkeypatch, change):
    """Serve the benchmark a projection with something taken out, the way a
    projection bug would lose it."""
    real = benchmark_module.load_projection

    async def lossy(*args, **kwargs):
        projection, coverage = await real(*args, **kwargs)
        return change(projection), coverage

    monkeypatch.setattr(benchmark_module, "load_projection", lossy)


def _breached(report):
    return {failure.split(":")[0] for failure in failures(report, THRESHOLDS_V1)}


async def test_a_labelled_value_the_projection_lost_is_an_error_not_a_value(monkeypatch):
    """Dropping every value leaves no attribute to be wrong about; the claims
    are missing, and missing is not classified correctly."""
    _projected_without(monkeypatch, lambda projection: replace(
        projection, attributes=(), roles=tuple(role for role in projection.roles if role.object_id is not None)))
    report = await run_entity_graph_benchmark(FIXTURE)
    assert report.attributes == 0 and report.claims_missing == 7
    assert report.literal_error_rate == round(7 / 15, 6)
    assert {"literal_error_rate", "claims_missing"} <= _breached(report)


async def test_a_labelled_relation_the_projection_lost_is_not_connected(monkeypatch):
    """A role still says the object is a thing, but no relation carries the
    claim, so nothing in the graph connects the two."""
    _projected_without(monkeypatch, lambda projection: replace(
        projection, relations=tuple(relation for relation in projection.relations if relation.predicate != "leads")))
    report = await run_entity_graph_benchmark(FIXTURE)
    assert report.connected_claim_share == round(7 / 8, 6) and report.claims_missing == 1
    assert {"connected_claim_share", "literal_error_rate", "claims_missing"} <= _breached(report)


async def test_a_gold_name_the_projection_lost_is_missing_not_whole(monkeypatch):
    def without_globex(projection):
        gone = next(entity.entity_id for entity in projection.entities if entity.key == "globex")
        return replace(
            projection,
            entities=tuple(entity for entity in projection.entities if entity.entity_id != gone),
            relations=tuple(relation for relation in projection.relations if gone not in (relation.subject_id,
                                                                                          relation.object_id)),
            attributes=tuple(attribute for attribute in projection.attributes if attribute.entity_id != gone),
            roles=tuple(role for role in projection.roles if gone not in (role.subject_id, role.object_id)))

    _projected_without(monkeypatch, without_globex)
    report = await run_entity_graph_benchmark(FIXTURE)
    assert report.gold_names_missing == 2 and report.path_ends_missing == 1 and report.claims_missing == 2
    # Eleven gold names. The two globex names add nothing; the three alice
    # names sit on two entities (recall 2/3, 2/3, 1/3); six are whole.
    precision, recall = 9 / 11, (2 / 3 + 2 / 3 + 1 / 3 + 6) / 11
    assert report.alias_bcubed_f1 == round(2 * precision * recall / (precision + recall), 6)
    assert {"gold_names_missing", "path_ends_missing", "claims_missing"} <= _breached(report)


async def test_the_fixture_reports_nothing_missing(tmp_path):
    report = await run_entity_graph_benchmark(FIXTURE)
    assert (report.claims_missing, report.gold_names_missing, report.path_ends_missing) == (0, 0, 0)


def _fixture(tmp_path, rows):
    path = tmp_path / "fixture.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [{"kind": "meta", "name": "gold"}, *rows]))
    return path


async def test_a_fixture_whose_gold_names_claims_it_never_loads_is_refused(tmp_path):
    """Gold about a name or claim the fixture never states can only be
    vacuously right, so it is refused instead of scored."""
    fixture = _fixture(tmp_path, [
        {"kind": "same_entity", "names": ["Missing Person"]},
        {"kind": "literal", "subject": "Missing Person", "predicate": "age", "object": "34", "value": True},
        {"kind": "path", "from": "Missing Person", "to": "Missing Other", "hops": 2, "expected": False},
    ])
    with pytest.raises(FixtureError) as refused:
        await run_entity_graph_benchmark(fixture)
    message = str(refused.value)
    assert "Missing Person" in message and "Missing Other" in message and "age" in message


def test_the_cli_refuses_a_broken_fixture_with_its_reasons(tmp_path, capsys):
    from scone_memory.runtime.cli import main

    fixture = _fixture(tmp_path, [{"kind": "same_entity", "names": ["Nobody"]}])
    assert main(["bench-graph", "--fixtures", str(fixture)]) == 2
    assert "Nobody" in capsys.readouterr().err


async def test_a_score_with_no_gold_is_unmeasured_and_fails_its_threshold(tmp_path):
    """Facts alone give nothing to be right about: every score is None and
    each threshold says so, instead of passing on zero cases."""
    fixture = _fixture(tmp_path, [{"kind": "fact", "subject": "ana", "predicate": "knows", "object": "Ben"}])
    report = await run_entity_graph_benchmark(fixture)
    assert (report.fragmentation, report.alias_bcubed_f1, report.literal_error_rate,
            report.connected_claim_share, report.path_false_positives) == (None,) * 5
    unmeasured = {failure.split(":")[0] for failure in failures(report, THRESHOLDS_V1) if "unmeasured" in failure}
    assert unmeasured == {"fragmentation", "alias_bcubed_f1", "literal_error_rate", "connected_claim_share",
                          "path_recall", "path_false_positives"}


async def test_the_recorded_artefact_is_the_one_the_fixture_gives():
    """results.md records the baseline; a change to what the report
    measures must come with a new recording, not leave a stale hash."""
    import re

    recorded = re.findall(r"Artefact: `([0-9a-f]{64})`", (FIXTURE.parent / "results.md").read_text())
    report = await run_entity_graph_benchmark(FIXTURE)
    assert recorded == [report.artefact_sha256]
