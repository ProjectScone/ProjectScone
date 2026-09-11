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


async def test_a_value_carried_by_the_wrong_attribute_is_not_carried(monkeypatch):
    """Six values dropped and their fact ids pinned onto the one left: an id
    is evidence only on an attribute with the claim's subject, predicate
    and value."""
    _projected_without(monkeypatch, lambda projection: replace(projection, attributes=(replace(
        projection.attributes[0],
        fact_ids=tuple(fact_id for attribute in projection.attributes for fact_id in attribute.fact_ids)),)))
    report = await run_entity_graph_benchmark(FIXTURE)
    assert report.claims_missing == 6 and report.literal_error_rate == round(6 / 15, 6)
    assert {"claims_missing", "literal_error_rate"} <= _breached(report)


async def test_a_relation_carried_between_the_wrong_entities_is_not_carried(monkeypatch):
    def misplaced(projection):
        leads = next(relation for relation in projection.relations if relation.predicate == "leads")
        rest = [relation for relation in projection.relations if relation is not leads]
        rest[0] = replace(rest[0], fact_ids=rest[0].fact_ids + leads.fact_ids)
        return replace(projection, relations=tuple(rest))

    _projected_without(monkeypatch, misplaced)
    report = await run_entity_graph_benchmark(FIXTURE)
    assert report.claims_missing == 1 and report.connected_claim_share == round(7 / 8, 6)


def _with(tmp_path, *extra):
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    path = tmp_path / "extended.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [*rows, *extra]))
    return path


async def test_a_fact_that_begins_after_as_of_is_out_of_view_not_missing(tmp_path):
    report = await run_entity_graph_benchmark(_with(tmp_path, {
        "kind": "fact", "subject": "future person", "predicate": "works_at", "object": "Future Company",
        "valid_from": "2030-01-01T00:00:00Z"}))
    assert report.claims_missing == 0 and report.claims_out_of_view == 1
    assert failures(report, THRESHOLDS_V1) == []


async def test_a_superseded_fact_is_out_of_view_not_missing(tmp_path):
    report = await run_entity_graph_benchmark(_with(tmp_path, {
        "kind": "fact", "subject": "bob stone", "predicate": "lives_in", "object": "Porto",
        "valid_from": "2023-01-01T00:00:00Z"}))
    assert report.claims_missing == 0 and report.claims_out_of_view == 1


async def test_the_fixture_has_nothing_out_of_view():
    assert (await run_entity_graph_benchmark(FIXTURE)).claims_out_of_view == 0


async def test_a_label_on_a_claim_the_view_does_not_hold_is_refused(tmp_path):
    later = {"kind": "fact", "subject": "future person", "predicate": "age", "object": "40",
             "valid_from": "2030-01-01T00:00:00Z"}
    with pytest.raises(FixtureError, match="does not hold at as_of; its fact begins later"):
        await run_entity_graph_benchmark(_with(tmp_path, later, {
            "kind": "literal", "subject": "future person", "predicate": "age", "object": "40", "value": True}))
    earlier = {"kind": "fact", "subject": "bob stone", "predicate": "lives_in", "object": "Porto",
               "valid_from": "2023-01-01T00:00:00Z"}
    with pytest.raises(FixtureError, match="does not hold at as_of; a later row for its subject and predicate supersedes it"):
        await run_entity_graph_benchmark(_with(tmp_path, earlier, {
            "kind": "literal", "subject": "bob stone", "predicate": "lives_in", "object": "Porto", "value": False}))


@pytest.mark.parametrize("row, complaint", [
    ({"kind": "literal", "subject": "alice chen", "predicate": "age", "object": "34", "value": None}, "value"),
    ({"kind": "literal", "subject": "alice chen", "predicate": "age", "object": "34", "value": "true"}, "value"),
    ({"kind": "literal", "subject": "alice chen"}, "predicate"),
    ({"kind": "path", "from": "alice chen", "to": "lisbon", "hops": 2, "expected": "false"}, "expected"),
    ({"kind": "path", "from": "alice chen", "to": "lisbon", "hops": "2"}, "hops"),
    ({"kind": "path", "from": "alice chen", "to": "lisbon", "hops": True}, "hops"),
    ({"kind": "path", "from": "alice chen", "to": "lisbon", "hops": 9}, "hops"),
    ({"kind": "path", "from": "alice chen", "to": "lisbon", "hops": 2, "expect": False}, "expect"),
    ({"kind": "same_entity", "names": "alice chen"}, "names"),
    ({"kind": "same_entity", "names": ["alice chen", ""]}, "names"),
    ({"kind": "fact", "subject": "alice chen", "predicate": "age", "object": 34}, "object"),
    ({"kind": "fact", "subject": "x", "predicate": "p", "object": "y", "valid_from": "someday"}, "valid_from"),
    ({"kind": "meta", "name": "second"}, "meta"),
    (["not", "an", "object"], "object"),
])
async def test_malformed_gold_is_refused_with_its_line(tmp_path, row, complaint):
    """Gold is never reinterpreted: a wrong type, a missing or unknown field,
    or a second meta row is refused, naming the line and the field."""
    with pytest.raises(FixtureError, match=complaint) as refused:
        await run_entity_graph_benchmark(_with(tmp_path, row))
    assert "line 41" in str(refused.value)


async def test_a_line_that_is_not_json_is_refused(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"kind": "meta", "name": "broken"}\n{"kind": "fact",\n')
    with pytest.raises(FixtureError, match="line 2"):
        await run_entity_graph_benchmark(path)


async def test_a_meta_as_of_that_is_not_a_time_is_refused(tmp_path):
    path = tmp_path / "when.jsonl"
    path.write_text(json.dumps({"kind": "meta", "name": "when", "as_of": "June"}))
    with pytest.raises(FixtureError, match="as_of"):
        await run_entity_graph_benchmark(path)


async def test_a_gold_name_only_a_later_fact_states_is_refused(tmp_path):
    with pytest.raises(FixtureError, match="'future person' is in no fact due by as_of"):
        await run_entity_graph_benchmark(_with(tmp_path, {
            "kind": "fact", "subject": "future person", "predicate": "works_at", "object": "Globex",
            "valid_from": "2030-01-01T00:00:00Z"}, {"kind": "same_entity", "names": ["future person"]}))


async def test_a_claim_the_store_lost_is_missing_not_out_of_view(monkeypatch):
    """With no stored fact to say otherwise, a claim due by as_of is one the
    view should hold, so losing it counts against the graph."""
    from scone_memory.backends.memory import InMemoryDocumentStore

    listed = InMemoryDocumentStore.list_facts

    async def losing(self, *args, **kwargs):
        return [fact for fact in await listed(self, *args, **kwargs) if fact.predicate != "leads"]

    monkeypatch.setattr(InMemoryDocumentStore, "list_facts", losing)
    _projected_without(monkeypatch, lambda projection: replace(
        projection, relations=tuple(relation for relation in projection.relations if relation.predicate != "leads")))
    report = await run_entity_graph_benchmark(FIXTURE)
    assert report.claims_missing == 1 and report.claims_out_of_view == 0


async def test_a_lost_claim_that_begins_after_as_of_is_still_out_of_view(monkeypatch, tmp_path):
    from scone_memory.backends.memory import InMemoryDocumentStore

    listed = InMemoryDocumentStore.list_facts

    async def losing(self, *args, **kwargs):
        return [fact for fact in await listed(self, *args, **kwargs) if fact.subject != "future person"]

    monkeypatch.setattr(InMemoryDocumentStore, "list_facts", losing)
    report = await run_entity_graph_benchmark(_with(tmp_path, {
        "kind": "fact", "subject": "future person", "predicate": "works_at", "object": "Future Company",
        "valid_from": "2030-01-01T00:00:00Z"}))
    assert report.claims_missing == 0 and report.claims_out_of_view == 1


@pytest.mark.parametrize("as_of", ["2023-06-01T00:00:00Z", "2025-06-01T00:00:00Z"])
async def test_a_claim_restated_in_another_interval_is_its_own_fact(tmp_path, as_of):
    """34, then 35, then 34 again: three facts, not two. The label on 34 is
    about whichever of its facts holds at as_of, and each fact is in view or
    out of it by its own interval."""
    path = tmp_path / "restated.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"kind": "meta", "name": "restated", "as_of": as_of},
        *({"kind": "fact", "subject": "returning person", "predicate": "age", "object": value,
           "valid_from": f"{year}-01-01T00:00:00Z"} for year, value in ((2023, "34"), (2024, "35"), (2025, "34"))),
        {"kind": "literal", "subject": "returning person", "predicate": "age", "object": "34", "value": True},
    ]))
    report = await run_entity_graph_benchmark(path)
    assert report.facts == 3 and report.claims_out_of_view == 1 and report.claims_missing == 0
    assert report.literal_error_rate == 0.0


@pytest.mark.parametrize("years, stored, out_of_view", [((2023, 2024), 1, 0), ((2024, 2023), 2, 0)])
async def test_a_value_reaffirmed_later_is_the_same_fact_not_a_lost_one(tmp_path, years, stored, out_of_view):
    """34 from 2023, then 34 again from 2024: one claim, and it holds, in
    whichever order the ledger was told, however many facts it kept."""
    path = tmp_path / "reaffirmed.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"kind": "meta", "name": "reaffirmed", "as_of": "2025-06-01T00:00:00Z"},
        *({"kind": "fact", "subject": "alice", "predicate": "age", "object": "34",
           "valid_from": f"{year}-01-01T00:00:00Z"} for year in years),
        {"kind": "literal", "subject": "alice", "predicate": "age", "object": "34", "value": True},
    ]))
    report = await run_entity_graph_benchmark(path)
    assert (report.facts, report.claims_missing, report.claims_out_of_view) == (stored, 0, out_of_view)
    assert report.literal_error_rate == 0.0



def _rows(tmp_path, *rows, as_of="2025-06-01T00:00:00Z"):
    path = tmp_path / "timeline.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [{"kind": "meta", "name": "timeline", "as_of": as_of}, *rows]))
    return path


def _works(year, company, predicate="works_at", subject="alice"):
    return {"kind": "fact", "subject": subject, "predicate": predicate, "object": company,
            "valid_from": f"{year}-01-01T00:00:00Z"}


@pytest.mark.parametrize("rows, missing", [
    ((_works(2020, "Acme"), _works(2023, "Acme"), _works(2021, "Globex")), 1),
    ((_works(2020, "Acme"), _works(2021, "Globex"), _works(2023, "Acme")), 0),
    ((_works(2021, "Globex"), _works(2023, "Acme")), 0),
    ((_works(2023, "Globex"), _works(2023, "Acme")), 0),
])
async def test_what_holds_is_the_fixtures_latest_claim_whatever_order_it_was_told(tmp_path, rows, missing):
    """Acme from 2020, Globex from 2021, Acme again from 2023: at 2025 the
    fixture says Acme. Told in the first order, the ledger folds the 2023
    Acme into the 2020 one and a late Globex then cuts it short, so the
    view says Globex; the bench must call that missing, not rightly out of
    view."""
    report = await run_entity_graph_benchmark(_rows(tmp_path, *rows))
    assert (report.claims_missing, report.claims_out_of_view) == (missing, 1)


async def test_a_label_in_another_spelling_of_its_claim_is_scored(tmp_path):
    report = await run_entity_graph_benchmark(_rows(
        tmp_path, _works(2023, "34", predicate="age"), _works(2024, "34", predicate="Age"),
        {"kind": "literal", "subject": "Alice", "predicate": " Age", "object": "34 ", "value": True}))
    assert report.literal_error_rate == 0.0 and report.claims_missing == 0


async def test_a_predicate_written_in_another_case_is_still_carried(tmp_path):
    report = await run_entity_graph_benchmark(_rows(
        tmp_path, _works(2024, "Acme", predicate="Works_At"),
        {"kind": "literal", "subject": "alice", "predicate": "Works_At", "object": "Acme", "value": False}))
    assert (report.claims_missing, report.literal_error_rate, report.connected_claim_share) == (0, 0.0, 1.0)


async def test_a_label_on_a_claim_a_later_row_supersedes_is_refused(tmp_path):
    with pytest.raises(FixtureError, match="a later row for its subject and predicate supersedes it"):
        await run_entity_graph_benchmark(_rows(
            tmp_path, _works(2020, "Acme"), _works(2022, "Globex"),
            {"kind": "literal", "subject": "alice", "predicate": "works_at", "object": "Acme", "value": False}))
