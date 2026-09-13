"""`forget --with-claims exclude` from the command line, and its receipt line."""
from __future__ import annotations

import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.runtime.cli import build_parser, run


@pytest.fixture
async def engine():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    source = await memory.remember("default", "Alice works at Acme")
    fact = await memory.assert_fact("default", "Alice", "works_at", "Acme", source_episode_id=source.episode_id, quote="Alice works at Acme")
    return memory, source.episode_id, fact.fact_id


async def forget(engine, *arguments: str) -> tuple[int, str]:
    out = io.StringIO()
    code = await run(build_parser().parse_args(["forget", *arguments]), engine, io.StringIO(""), out)
    return code, out.getvalue()


async def test_the_default_line_still_says_the_claims_stand(engine):
    memory, episode_id, fact_id = engine
    code, text = await forget(memory, str(episode_id))
    assert code == 0 and "1 claim(s)" in text and "stand" in text and "excluded" not in text
    assert not (await memory.documents.get_fact("default", fact_id)).excluded


async def test_exclude_is_named_on_the_line_and_in_json(engine):
    memory, episode_id, fact_id = engine
    code, text = await forget(memory, str(episode_id), "--with-claims", "exclude")
    assert code == 0 and f"1 claim(s) excluded" in text, text
    assert (await memory.documents.get_fact("default", fact_id)).excluded


async def test_exclude_in_json(engine):
    memory, episode_id, fact_id = engine
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "forget", str(episode_id), "--with-claims", "exclude"]), memory, io.StringIO(""), out)
    body = json.loads(out.getvalue())
    assert code == 0 and body["claims_policy"] == "exclude" and body["claims_excluded"] == [fact_id]


async def test_an_unknown_policy_is_refused_by_the_parser(engine):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["forget", "1", "--with-claims", "erase"])
