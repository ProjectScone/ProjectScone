"""The entity graph from the command line: report, path, context, entity,
timeline and export, on the configured store, reading the same projection
the HTTP routes and MCP tools read."""

from __future__ import annotations

import io
import sys
import json
import xml.etree.ElementTree as ElementTree

import pytest

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryEventLog,
                          InMemoryVectorIndex, MemoryEngine)
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime import cli
from scone_memory.runtime.cli import build_parser, run

DAY = "2024-01-01T00:00:00Z"


@pytest.fixture
async def engine():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.assert_fact("default", "alice chen", "works_at", "Acme Robotics", valid_from=DAY)
    await memory.assert_fact("default", "acme robotics", "based_in", "Lisbon", valid_from=DAY)
    await memory.assert_fact("default", "alice park", "knows", "Bob", valid_from=DAY)
    return memory


async def graph(engine, *arguments: str) -> tuple[int, str]:
    out = io.StringIO()
    code = await run(build_parser().parse_args(["graph", *arguments]), engine, io.StringIO(""), out)
    return code, out.getvalue()


async def test_path_prints_the_route_between_two_names(engine):
    code, text = await graph(engine, "path", "alice chen", "lisbon")
    assert code == 0 and "path: alice chen -works_at-> Acme Robotics -based_in-> Lisbon" in text


async def test_an_ambiguous_name_lists_candidates_and_exits_nonzero(engine):
    code, text = await graph(engine, "entity", "alice")
    assert code == 1 and text.count("candidate: ") == 2


async def test_context_takes_names_or_a_question(engine):
    code, text = await graph(engine, "context", "--question", "who works in lisbon?")
    assert code == 0 and "entity: Lisbon" in text
    code, text = await graph(engine, "context", "alice chen", "--max-bytes", "512")
    assert code == 0 and len(text.encode()) <= 513


async def test_report_prints_markdown_or_json(engine):
    code, text = await graph(engine, "report", "--markdown")
    assert code == 0 and text.startswith("# Knowledge report")
    code, text = await graph(engine, "report", "--json")
    assert code == 0 and json.loads(text)["summary"]["entities"] >= 4


async def test_timeline_prints_an_entitys_items(engine):
    code, text = await graph(engine, "timeline", "alice chen", "--json")
    assert code == 0 and [item["predicate"] for item in json.loads(text)["items"]] == ["works_at"]


async def test_export_writes_any_format_to_a_file(engine, tmp_path):
    target = tmp_path / "graph.graphml"
    code, text = await graph(engine, "export", "--format", "graphml", "--out", str(target))
    assert code == 0 and str(target) in text
    assert ElementTree.fromstring(target.read_bytes()).tag.endswith("graphml")


@pytest.mark.parametrize("arguments, code", [(("path", "alice chen", "lisbon"), 0),
                                             (("context", "alice chen"), 0),
                                             (("entity", "alice chen"), 0),
                                             (("entity", "alice"), 1)])
async def test_json_prints_the_packet_as_the_http_route_does(engine, arguments, code):
    """--json is honoured, not ignored: the same packet text, with its
    status, seeds, candidates, coverage and the instant it was read at."""
    engine.clock = lambda: "2025-06-01T00:00:00.000Z"
    plain_code, plain = await graph(engine, *arguments)
    json_code, printed = await graph(engine, *arguments, "--json")
    body = json.loads(printed)
    assert plain_code == json_code == code
    assert body["text"] == plain.rstrip("\n") and body["status"] == ("prepared" if code == 0 else "ambiguous")
    assert body["filters"] == {"status": "current", "as_of": "2025-06-01T00:00:00.000Z"}
    assert {"seeds", "candidates", "coverage"} <= set(body) and "reasons" in body["coverage"]
    assert len(body["candidates"]) == (2 if code else 0)


@pytest.fixture
async def capped(monkeypatch):
    """Three people at Acme, with a read that holds only the newest two."""
    from scone_memory.entities import read

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for n in range(3):
        await memory.assert_fact("default", f"Alice {n}", "works_at", "Acme", valid_from=DAY)
    monkeypatch.setattr(read, "MAX_FACTS", 2)
    yield memory
    await memory.close()


async def test_a_timeline_name_missing_from_a_capped_read_may_still_exist(capped):
    code, text = await graph(capped, "timeline", "Alice 0")
    body = json.loads(text)
    assert code == 1 and body["complete"] is False and body["coverage"]["truncated"] is True
    assert "may exist" in body["error"] and body["name"] == "Alice 0"


async def test_a_timeline_ambiguity_in_a_capped_read_says_its_candidates_are_partial(capped):
    code, text = await graph(capped, "timeline", "Alice")
    body = json.loads(text)
    assert code == 1 and body["complete"] is False and body["truncated"] is True
    assert body["coverage"]["truncated"] is True and len(body["candidates"]) == 2


@pytest.fixture
async def moved():
    """Alice left Acme for Beta at the turn of 2025, and the clock is read
    once just before it and ever after just past it."""
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.assert_fact("default", "alice", "works_at", "Acme", valid_from="2024-01-01T00:00:00Z")
    await memory.assert_fact("default", "alice", "works_at", "Beta", valid_from="2025-01-01T00:00:00Z")
    readings = iter(["2024-12-31T23:59:59.000Z"])
    memory.clock = lambda: next(readings, "2025-01-01T00:00:01.000Z")
    yield memory
    await memory.close()


async def test_a_report_is_read_and_labelled_at_one_instant(moved):
    code, text = await graph(moved, "report", "--json")
    report = json.loads(text)
    keys = {entity["key"] for entity in report["central_entities"]}
    assert code == 0 and report["filters"]["as_of"] == "2024-12-31T23:59:59.000Z"
    assert "acme" in keys and "beta" not in keys


async def test_an_export_says_the_instant_it_was_read_at(moved):
    code, text = await graph(moved, "export", "--format", "json")
    about = json.loads(text)["graph"]["about"]
    assert code == 0 and about["as_of"] == "2024-12-31T23:59:59.000Z" and about["status"] == "current"
    assert about["coverage"]["truncated"] is False


@pytest.mark.parametrize("arguments, named", [
    (("report", "--resolution", "0"), "--resolution"),
    (("report", "--resolution", "nan"), "--resolution"),
    (("report", "--resolution", "11"), "--resolution"),
    (("context", "alice", "--max-bytes", "1"), "--max-bytes"),
    (("timeline", "alice", "--as-of", "not-a-date"), "--as-of"),
])
def test_an_invalid_option_is_an_input_error(arguments, named, capsys):
    """Refused at the command line the way the HTTP route refuses it: exit
    2 with the option named, never a traceback."""
    from scone_memory.runtime.cli import main

    code = main(["graph", *arguments], env={"SCONE_DOCUMENTS": "memory", "SCONE_VECTORS": "memory"},
                out=io.StringIO(), stdin=io.StringIO())
    assert code == 2 and named in capsys.readouterr().err


@pytest.mark.parametrize("arguments, named", [
    (("context", *(f"person {n}" for n in range(25))), "names"),
    (("context", "x" * 201), "names"),
    (("entity", "x" * 201), "names"),
    (("path", "alice", "x" * 201), "names"),
    (("path", "alice", "bob", "--max-hops", "5"), "--max-hops"),
    (("context", "--question", "q" * 2001), "--question"),
    (("schema", "--limit", "0"), "--limit"),
    (("schema", "--max-bytes", "100"), "--max-bytes"),
    (("walk", "lisbon", "--hops", "9"), "--hops"),
    (("context", "--question", "q", "--min-similarity", "nan"), "--min-similarity"),
    (("walk", *(f"person {n}" for n in range(25))), "names"),
    (("timeline", "alice", "--as-of", "0001-01-01T00:00:00+01:00"), "--as-of"),
    (("timeline", "alice", "--as-of", "9999-12-31T23:59:59-01:00"), "--as-of"),
])
def test_a_value_past_the_routes_bounds_is_an_input_error(arguments, named, capsys):
    from scone_memory.runtime.cli import main

    code = main(["graph", *arguments], env={"SCONE_DOCUMENTS": "memory", "SCONE_VECTORS": "memory"},
                out=io.StringIO(), stdin=io.StringIO())
    assert code == 2 and named in capsys.readouterr().err


async def test_schema_prints_the_kinds_and_predicates_as_json(engine):
    code, text = await graph(engine, "schema")
    body = json.loads(text)
    assert code == 0 and {entry["predicate"] for entry in body["predicates"]} == {"works_at", "based_in", "knows"}
    code, text = await graph(engine, "schema", "--limit", "1")
    assert code == 0 and json.loads(text)["truncated"] is True


def test_export_offers_every_format_the_library_writes():
    from scone_memory.entities.export import EXPORT_FORMATS

    graph_parser = next(action for action in build_parser()._subparsers._group_actions[0].choices["graph"]._actions
                        if action.dest == "graph_command")
    export = graph_parser.choices["export"]
    assert next(action.choices for action in export._actions if action.dest == "format") == list(EXPORT_FORMATS)


async def test_export_writes_a_timeline_graph(engine):
    code, text = await graph(engine, "export", "--format", "gexf")
    assert code == 0 and 'mode="dynamic"' in text and 'timeformat="dateTime"' in text


@pytest.mark.parametrize("arguments", [("report",), ("report", "--markdown"), ("schema",),
                                       ("timeline", "zed"), ("context", "zed", "--json"), ("entity", "zed")])
async def test_stored_text_utf8_cannot_encode_still_prints(arguments):
    """A real terminal is a strict UTF-8 stream, unlike StringIO: a lone
    surrogate from the ledger is printed as its escape, never a crash."""
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.assert_fact("default", "zed \ud800", "odd\ud800predicate", "Acme", valid_from=DAY)
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="utf-8", errors="strict", write_through=True)
    code = await run(build_parser().parse_args(["graph", *arguments]), memory, io.StringIO(""), out)
    await memory.close()
    assert code in (0, 1) and raw.getvalue().decode("utf-8")


async def test_walk_prints_what_depends_on_an_entity_hop_by_hop(engine):
    code, text = await graph(engine, "walk", "lisbon", "--direction", "in")
    view = json.loads(text)
    hops = {entity["key"]: entity["hop"] for entity in view["entities"]}
    assert code == 0 and hops == {"lisbon": 0, "acme robotics": 1, "alice chen": 2}
    assert view["filters"]["direction"] == "in"
    code, text = await graph(engine, "walk", "lisbon", "--hops", "1")
    assert code == 0 and "hop_limit" in json.loads(text)["coverage"]["reasons"]


async def test_walk_names_an_unknown_or_ambiguous_seed(engine):
    code, text = await graph(engine, "walk", "alice")
    assert code == 1 and len(json.loads(text)["candidates"]) == 2
    code, text = await graph(engine, "walk", "nobody")
    assert code == 1 and "no entity is named" in json.loads(text)["error"]


async def test_context_can_seed_by_resemblance(engine):
    code, text = await graph(engine, "context", "--question", "which robotics firm?", "--similar")
    assert code == 0 and " similar " in text


async def test_report_can_say_what_recall_uses():
    from scone_memory.observability.events import InMemoryEventLog

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=InMemoryEventLog()).open()
    await memory.assert_fact("default", "alice chen", "works_at", "Acme Robotics", valid_from=DAY)
    await memory.recall("default", "alice chen")
    code, text = await graph(memory, "report", "--markdown", "--usage")
    await memory.close()
    assert code == 0 and "## What recall uses" in text and "Alice Chen (1)" in text.replace("alice chen (1)", "Alice Chen (1)")


async def test_match_answers_a_structured_question_and_exits_zero_on_a_row(engine):
    code, text = await graph(engine, "match", "--pattern", "?who", "works_at", "?org",
                             "--pattern", "?org", "based_in", "Lisbon", "--returns", "?who")
    assert code == 0 and "row: ?who = alice chen (person) ent:" in text
    code, text = await graph(engine, "match", "--pattern", "?who", "works_at", "Lisbon")
    assert code == 1 and "result: no match" in text


async def test_match_answers_json_with_the_routes_record(engine):
    """Carl left Acme in 2021, before Acme's Lisbon office (2024): only
    --apart joins them."""
    await engine.assert_fact("default", "carl", "works_at", "Acme Robotics", valid_from="2020-01-01T00:00:00Z")
    await engine.assert_fact("default", "carl", "works_at", "Globex", valid_from="2021-01-01T00:00:00Z")
    question = ["--pattern", "?who", "works_at", "?org", "--pattern", "?org", "based_in", "Lisbon", "--returns", "?who",
                "--json", "--status", "history", "--limit", "5"]
    code, text = await graph(engine, "match", *question, "--apart")
    body = json.loads(text)
    assert code == 0 and body["filters"]["status"] == "history" and body["filters"]["together"] is False
    assert body["filters"]["limit"] == 5
    assert [row["bindings"]["?who"]["key"] for row in body["rows"]] == ["alice chen", "carl"]
    code, text = await graph(engine, "match", *question)
    assert [row["bindings"]["?who"]["key"] for row in json.loads(text)["rows"]] == ["alice chen"]


@pytest.mark.parametrize("arguments, complaint", [
    (["--pattern", "?1x", "works_at", "?org"], "variable"),
    (["--pattern", "?who", "works_at", "?org", "--returns", "?nobody"], "?nobody"),
    (["--pattern", "?who", "works_at", "?org", "--limit", "0"], "limit"),
    (["--pattern", "?who", "works_at", "?org", "--max-bytes", "100"], "max_bytes"),
    (["--pattern", "?who", "works_at", "?org", "--as-of", "soon"], "RFC 3339"),
])
async def test_a_malformed_match_is_refused_before_anything_is_read(engine, arguments, complaint):
    with pytest.raises(InvalidInput, match=complaint.replace("?", r"\?")):
        await graph(engine, "match", *arguments)


async def test_overview_prints_the_digests_and_json_on_request(engine):
    code, text = await graph(engine, "overview", "--question", "who works where?")
    assert code == 0 and text.startswith("overview: space default, ") and 'matched "works"' in text
    code, text = await graph(engine, "overview", "--json", "--limit", "1", "--facts", "0")
    body = json.loads(text)
    assert code == 0 and len(body["communities"]) == 1 and body["communities"][0]["fact_ids"] == []
    with pytest.raises(InvalidInput, match="limit"):
        await graph(engine, "overview", "--limit", "0")


async def test_changes_prints_what_changed_and_json_on_request(engine):
    code, text = await graph(engine, "changes", "--since", "2023-01-01T00:00:00Z")
    assert code == 0 and text.startswith("changes: space default, ") and "began: alice chen works_at" in text
    code, text = await graph(engine, "changes", "--since", "2023-01-01T00:00:00Z", "--until", "2023-06-01T00:00:00Z",
                             "--json")
    assert code == 1 and json.loads(text)["status"] == "unchanged"
    with pytest.raises(InvalidInput, match="before"):
        await graph(engine, "changes", "--since", "2030-01-01T00:00:00Z")


async def test_duplicates_prints_the_pairs_and_json_on_request(engine):
    await engine.assert_fact("default", "dr. alice chen", "leads", "Robotics Lab", valid_from=DAY)
    code, text = await graph(engine, "duplicates")
    assert code == 0 and "pair: alice chen" in text
    code, text = await graph(engine, "duplicates", "--json", "--min-score", "0.99", "--limit", "3")
    assert code == 0 and json.loads(text)["pairs"][0]["score"] == 1.0
    with pytest.raises(InvalidInput, match="min_score"):
        await graph(engine, "duplicates", "--min-score", "2")


async def test_health_prints_what_wants_attention(engine):
    code, text = await graph(engine, "health", "--limit", "2")
    assert code == 0 and "health: space default" in text
    assert "unsourced:" in text and "(claims nothing cites a source for" in text
    assert "see scone facts," in text
    code, shown = await graph(engine, "health", "--json")
    assert code == 0 and json.loads(shown)["status"] == "concerns"


async def temporal(engine, *arguments: str) -> tuple[int, str]:
    out = io.StringIO()
    code = await run(build_parser().parse_args(["when", *arguments]), engine, io.StringIO(""), out)
    return code, out.getvalue()


async def test_when_computes_the_answer_and_shows_its_working():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.remember("default", "I met Emma for coffee near the river.", created_at="2023-04-11T12:00:00Z")
    code, text = await temporal(memory, "How many days ago did I meet Emma?", "--now", "2023-04-20T10:12:00Z")
    assert code == 0 and "answer: 9 days ago" in text and "event: meet emma → 2023-04-11" in text
    code, shown = await temporal(memory, "How many days ago did I meet Emma?", "--now", "2023-04-20T10:12:00Z",
                                 "--json")
    assert code == 0 and json.loads(shown)["value"]["days"] == 9


async def test_when_answers_a_question_about_a_day_from_that_day():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.remember("default", "I met Emma for coffee near the river.", created_at="2023-04-11T12:00:00Z")
    code, text = await temporal(memory, "What did I do 9 days ago?", "--now", "2023-04-20T10:12:00Z")
    assert code == 0, "a day's passages are an answer, not a failure"
    assert "on: 2023-04-11" in text and "passage: 2023-04-11" in text


async def test_when_leaves_a_question_it_cannot_read_to_the_caller():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    code, text = await temporal(memory, "What did I drink?")
    assert code == 1 and "not a temporal question" in text


def test_bench_temporal_scores_a_file_of_dated_questions(tmp_path):
    from tests.benchmarks.test_bench_temporal import ITEMS

    path = tmp_path / "items.json"
    path.write_text(json.dumps(ITEMS), encoding="utf-8")
    out = io.StringIO()
    code = cli.main(["bench-temporal", str(path)], env={}, stdin=io.StringIO(""), out=out)
    assert code == 0 and "temporal: 5 questions; computed 1 of 5" in out.getvalue(), out.getvalue()


async def test_a_listed_fact_says_which_episode_it_came_from():
    """A claim with no episode rests on whoever wrote it, and a list is
    where that difference shows."""
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    said = await memory.remember("default", "Alice Chen works at Acme Robotics.")
    await memory.assert_fact("default", "alice chen", "works_at", "Acme Robotics", valid_from=DAY,
                             source_episode_id=said.episode_id, quote="Alice Chen works at Acme Robotics.")
    await memory.assert_fact("default", "bob stone", "works_at", "Globex", valid_from=DAY)
    out = io.StringIO()
    code = await run(build_parser().parse_args(["facts"]), memory, io.StringIO(""), out)
    lines = sorted(line for line in out.getvalue().splitlines() if line.startswith("#"))
    assert code == 0 and f"from episode {said.episode_id}" in lines[0]
    assert "from episode" not in lines[1], "nothing cites a source for this one"


def test_bench_code_measures_a_corpus_of_source_files(tmp_path):
    from tests.benchmarks.test_bench_code import PLANNER

    (tmp_path / "planner.py").write_text(PLANNER, encoding="utf-8")
    out = io.StringIO()
    code = cli.main(["bench-code", str(tmp_path), "--asked", "name"], env={}, stdin=io.StringIO(""), out=out)
    assert code == 0, out.getvalue()
    assert "2 question(s) over 1 file(s), asked by name" in out.getvalue(), out.getvalue()


def test_bench_code_refuses_a_root_that_is_not_a_directory(tmp_path):
    missing = tmp_path / "nowhere"
    out = io.StringIO()
    code = cli.main(["bench-code", str(missing)], env={}, stdin=io.StringIO(""), out=out)
    assert code == 2, out.getvalue()


def test_tune_measures_the_settings_it_is_given(tmp_path):
    import json as _json

    from tests.benchmarks.test_bench import DATASET

    path = tmp_path / "items.json"
    path.write_text(_json.dumps(DATASET[:2]), encoding="utf-8")
    out = io.StringIO()
    code = cli.main(["tune", str(path), "--sample", "2", "--candidates", "100"], env={},
                    stdin=io.StringIO(""), out=out)
    assert code == 0, out.getvalue()
    said = out.getvalue()
    assert "2 question(s) at k=5" in said and "the defaults: recall_any" in said
    assert "SCONE_RECALL_CANDIDATES=100" in said and "take:" in said


def test_tune_refuses_a_candidate_limit_that_is_not_a_number(tmp_path):
    out = io.StringIO()
    code = cli.main(["tune", str(tmp_path / "nothing.json"), "--candidates", "many"], env={},
                    stdin=io.StringIO(""), out=out)
    assert code == 2


async def test_graph_meanings_says_what_the_predicates_are_taken_to_mean():
    from scone_memory.entities.meanings import RelationMeanings

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                relation_meanings=RelationMeanings(inverse={"works_at": "employs"},
                                                                   symmetric=["married_to"],
                                                                   transitive=["part_of"])).open()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["graph", "meanings"]), memory, io.StringIO(""), out)
    said = out.getvalue()
    assert code == 0
    assert "employs ↔ works_at" in said, "a pair reads the same either way round, so it is shown in one order"
    assert "reads both ways: married_to" in said and "carries through: part_of" in said
    assert "4 claims" in said and "200000" in said.replace(",", "")


async def test_graph_meanings_says_plainly_when_nothing_is_configured():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["graph", "meanings"]), memory, io.StringIO(""), out)
    assert code == 0 and "nothing is configured" in out.getvalue()
    assert "only what was said" in out.getvalue()


async def test_graph_match_takes_what_follows_only_when_asked():
    from scone_memory.entities.meanings import RelationMeanings

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                relation_meanings=RelationMeanings(inverse={"works_at": "employs"})).open()
    await memory.assert_fact("default", "alice chen", "works_at", "Acme Robotics", valid_from=DAY)
    args = ["graph", "match", "--pattern", "?who", "employs", "?whom"]
    out = io.StringIO()
    code = await run(build_parser().parse_args(args), memory, io.StringIO(""), out)
    assert code == 1 and "not found: predicate" in out.getvalue()
    out = io.StringIO()
    code = await run(build_parser().parse_args([*args, "--follows"]), memory, io.StringIO(""), out)
    said = out.getvalue()
    assert code == 0, said
    assert "Acme Robotics" in said and "follows: inverse" in said, said


def test_bench_route_says_where_a_file_of_questions_went(tmp_path):
    """A measurement only anyone can run is one the CLI can run."""
    from tests.benchmarks.test_bench_route import DATASET

    path = tmp_path / "items.json"
    path.write_text(json.dumps(DATASET), encoding="utf-8")
    out = io.StringIO()
    code = cli.main(["bench-route", str(path)], env={}, stdin=io.StringIO(""), out=out)
    assert code == 0, out.getvalue()
    assert "routing: 2 question(s)" in out.getvalue(), out.getvalue()
    assert "1 agree with the file" in out.getvalue(), out.getvalue()


def test_bench_route_reports_as_json_when_asked(tmp_path):
    from tests.benchmarks.test_bench_route import DATASET

    path = tmp_path / "items.json"
    path.write_text(json.dumps(DATASET), encoding="utf-8")
    out = io.StringIO()
    code = cli.main(["bench-route", str(path), "--limit", "1", "--json"],
                    env={}, stdin=io.StringIO(""), out=out)
    assert code == 0, out.getvalue()
    assert json.loads(out.getvalue())["questions"] == 1


async def test_recall_parts_gives_every_part_of_a_question_a_turn():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.remember("default", "We reverted the billing change after the invoices came out wrong.")
    await memory.remember("default", "At the Thursday meeting were Priya, Tomas and the auditor.")
    out = io.StringIO()
    asked = ["recall", "What did I decide about billing, and who was at the meeting?", "--parts", "--limit", "2"]
    code = await run(build_parser().parse_args(asked), memory, io.StringIO(""), out)
    shown = out.getvalue()
    assert code == 0, shown
    assert "[What did I decide about billing]" in shown and "[who was at the meeting?]" in shown, shown
    assert "not evidence" in shown, "a search with no floor must not read as an answered question"


async def test_recall_without_parts_is_unchanged():
    """The flag is opt-in: it was measured to change nothing on LongMemEval,
    so it does not become the default search."""
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.remember("default", "We reverted the billing change.")
    out = io.StringIO()
    asked = ["recall", "What did I decide about billing, and who was at the meeting?", "--limit", "2"]
    code = await run(build_parser().parse_args(asked), memory, io.StringIO(""), out)
    assert code == 0 and "[" not in out.getvalue(), out.getvalue()


def test_bench_parts_compares_both_ways_on_the_questions_that_split(tmp_path):
    from tests.benchmarks.test_bench_parts import SPLIT, WHOLE

    path = tmp_path / "items.json"
    path.write_text(json.dumps([SPLIT, WHOLE]), encoding="utf-8")
    out = io.StringIO()
    code = cli.main(["bench-parts", str(path), "--k", "2"], env={}, stdin=io.StringIO(""), out=out)
    assert code == 0, out.getvalue()
    assert "1 of 2 question(s)" in out.getvalue(), out.getvalue()


def test_bench_parts_refuses_a_k_it_cannot_search(tmp_path):
    path = tmp_path / "items.json"
    path.write_text("[]", encoding="utf-8")
    out = io.StringIO()
    assert cli.main(["bench-parts", str(path), "--k", "0"], env={}, stdin=io.StringIO(""), out=out) == 2


async def test_sync_plans_before_it_writes(tmp_path):
    (tmp_path / "a.py").write_text("def a(): pass", encoding="utf-8")
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=InMemoryEventLog()).open()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["sync", str(tmp_path)]), memory, io.StringIO(""), out)
    assert code == 0 and "would sync" in out.getvalue() and "last sync: never" in out.getvalue()
    assert (await memory.status("default")).episodes == 0, "a plan writes nothing"
    out = io.StringIO()
    code = await run(build_parser().parse_args(["sync", str(tmp_path), "--apply", "--marker", "repo"]),
                     memory, io.StringIO(""), out)
    assert code == 0 and "synced repo" in out.getvalue() and "1 added" in out.getvalue()
    assert (await memory.status("default")).episodes == 1


async def test_sync_says_a_removal_needs_apply(tmp_path):
    (tmp_path / "a.py").write_text("def a(): pass", encoding="utf-8")
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=InMemoryEventLog()).open()
    out, err = io.StringIO(), io.StringIO()
    saved, sys.stderr = sys.stderr, err
    try:
        code = await run(build_parser().parse_args(["sync", str(tmp_path), "--remove"]),
                         memory, io.StringIO(""), out)
    finally:
        sys.stderr = saved
    assert code == 0 and "needs --apply" in err.getvalue(), err.getvalue()
    assert (await memory.status("default")).episodes == 0


async def test_map_reads_a_directory_reached_through_a_hidden_one(tmp_path):
    """The same filter fault as sync had: judged on the whole path it skips
    the entire tree whenever the root itself sits under a dot-segment, which
    is what "a map that quietly skipped half a repository" describes."""
    root = tmp_path / ".cache" / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("def a(): pass", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "hooks.py").write_text("def hook(): pass", encoding="utf-8")
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["map", str(root)]), memory, io.StringIO(""), out)
    assert code == 0, out.getvalue()
    assert (await memory.status("default")).episodes == 1, out.getvalue()
    assert ".git" not in out.getvalue(), "a hidden directory under the root is still skipped"


async def test_recall_can_join_neighbouring_chunks_and_say_what_it_joined():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                chunk_target=90).open()
    await memory.remember("default", (
        "The harbour crane was repainted in May after the survey found rust on the jib. "
        "The survey also found the slew ring needed grease, which the yard did that week. "
        "The crane went back into service on the first of June, a day late."))
    out = io.StringIO()
    asked = ["recall", "crane survey rust jib slew grease", "--merge", "--limit", "5"]
    code = await run(build_parser().parse_args(asked), memory, io.StringIO(""), out)
    shown = out.getvalue()
    assert code == 0, shown
    assert "passage(s) joined from" in shown and "joined" in shown, shown
    plain = io.StringIO()
    code = await run(build_parser().parse_args(asked[:2] + ["--limit", "5"]),
                     memory, io.StringIO(""), plain)
    assert code == 0 and "joined" not in plain.getvalue(), "merging stays opt-in"


async def test_recall_can_widen_a_single_hit():
    """The case --merge cannot serve: one chunk matched, and the answer is
    in the sentence after it."""
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                chunk_target=70).open()
    await memory.remember("default", (
        "The survey of the harbour crane was booked for the third of May. "
        "It found rust on the jib and a slew ring that needed grease. "
        "The yard did both in the same week."))
    out = io.StringIO()
    asked = ["recall", "rust jib slew grease", "--window", "80", "--limit", "1"]
    code = await run(build_parser().parse_args(asked), memory, io.StringIO(""), out)
    assert code == 0 and "item(s) widened" in out.getvalue(), out.getvalue()
    plain = io.StringIO()
    code = await run(build_parser().parse_args(asked[:2] + ["--limit", "1"]),
                     memory, io.StringIO(""), plain)
    assert code == 0 and "widened" not in plain.getvalue(), "widening stays opt-in"
