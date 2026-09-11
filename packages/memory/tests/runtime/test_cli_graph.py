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
