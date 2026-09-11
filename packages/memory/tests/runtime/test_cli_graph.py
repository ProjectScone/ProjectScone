"""The entity graph from the command line: report, path, context, entity,
timeline and export, on the configured store, reading the same projection
the HTTP routes and MCP tools read."""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ElementTree

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
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
