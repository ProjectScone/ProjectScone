"""Narrowing a search from the command line by what was recorded."""

from __future__ import annotations

import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.cli import build_parser, run


@pytest.fixture
async def engine():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for n in range(20):
        await memory.remember("default", f"quarterly planning note {n}", metadata={"status": "draft"})
    await memory.remember("default", "quarterly planning note, the one that shipped",
                          metadata={"status": "published"})
    return memory


async def ask(engine, conditions):
    args = build_parser().parse_args(
        ["recall", "quarterly planning note", "--conditions", conditions, "--json"])
    out = io.StringIO()
    await run(args, engine, io.StringIO(""), out)
    return json.loads(out.getvalue())


async def test_the_command_line_narrows_the_same_way_the_route_does(engine):
    answer = await ask(engine, json.dumps({"field": "status", "is": "published"}))
    assert [item["text"] for item in answer["items"]] == \
        ["quarterly planning note, the one that shipped"]


async def test_a_filter_the_command_line_cannot_read_stops_the_search(engine):
    """Nothing is printed. A search that answered from everything after
    silently dropping the filter would look exactly like a wide result."""
    with pytest.raises(InvalidInput, match="conditions must be"):
        await ask(engine, "{not json")
