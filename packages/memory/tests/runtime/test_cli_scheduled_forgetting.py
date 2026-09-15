"""`remember --forget-after` and `forget-due` from the command line."""
from __future__ import annotations

import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.cli import build_parser, run
from scone_memory.testing import Clock


@pytest.fixture
async def memory():
    clock = Clock("2026-09-15T12:00:00.000Z")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock).open()
    engine.test_clock = clock
    yield engine
    await engine.close()


async def cli(engine, *arguments: str, stdin: str = "") -> tuple[int, str]:
    out = io.StringIO()
    code = await run(build_parser().parse_args(list(arguments)), engine, io.StringIO(stdin), out)
    return code, out.getvalue()


async def test_remember_takes_a_schedule_and_says_when(memory):
    code, text = await cli(memory, "remember", "--forget-after", "3d", stdin="the spare key is under the mat")
    assert code == 0 and "forgotten after 2026-09-18T12:00:00.000Z" in text, text
    code, text = await cli(memory, "--json", "remember", "--forget-after", "2026-10-01", stdin="a second note")
    assert code == 0 and json.loads(text)["forget_after"] == "2026-10-01T00:00:00.000Z"


async def test_remember_over_an_overdue_memory_says_what_it_forgot(memory):
    await cli(memory, "remember", "--forget-after", "1h", stdin="secret words")
    memory.test_clock.now = "2026-09-15T14:00:00.000Z"
    code, text = await cli(memory, "remember", stdin="secret words")
    assert code == 0 and "episode 1 was past its forget_after 2026-09-15T13:00:00.000Z and was forgotten" in text, text


async def test_remember_refuses_a_past_schedule(memory):
    with pytest.raises(InvalidInput, match="forget_after"):
        await cli(memory, "remember", "--forget-after", "2026-01-01", stdin="too late")
    assert (await memory.status("default")).episodes == 0


async def test_a_jsonl_record_carries_its_own_schedule_and_the_flag_is_refused_with_it(memory):
    lines = json.dumps({"content": "line one", "forget_after": "1h"}) + "\n" + json.dumps({"content": "line two"})
    code, text = await cli(memory, "--json", "remember", "--jsonl", stdin=lines)
    rows = [json.loads(line) for line in text.splitlines()]
    assert code == 0 and [row["forget_after"] for row in rows] == ["2026-09-15T13:00:00.000Z", None]
    with pytest.raises(InvalidInput, match="--forget-after"):
        await cli(memory, "remember", "--jsonl", "--forget-after", "1h", stdin=lines)


async def test_forget_due_sweeps_and_reports_what_went_and_why(memory):
    await cli(memory, "remember", "--forget-after", "1h", stdin="due one")
    await cli(memory, "remember", "--forget-after", "2h", stdin="due two")
    await cli(memory, "remember", stdin="kept")
    memory.test_clock.now = "2026-09-15T15:00:00.000Z"
    code, text = await cli(memory, "--json", "forget-due", "--dry-run", "--now", "2026-09-15T13:30:00Z")
    assert json.loads(text)["due"] == 1
    code, text = await cli(memory, "forget-due", "--dry-run")
    assert code == 0 and "would forget 2 episode(s)" in text and (await memory.status("default")).episodes == 3
    code, text = await cli(memory, "forget-due", "--limit", "1")
    assert code == 0 and "forgot 1 episode(s)" in text and "1 due left for the next pass" in text, text
    code, text = await cli(memory, "--json", "forget-due", "--with-claims", "exclude")
    body = json.loads(text)
    assert code == 0 and len(body["forgotten"]) == 1 and body["items"][0]["reason"].startswith("forget_after 2026-09-15T14:00:00.000Z")
    assert body["with_claims"] == "exclude" and (await memory.status("default")).episodes == 1


async def test_forget_due_says_when_its_walk_was_cut(memory, monkeypatch):
    from scone_memory.memory import scheduled_forget

    monkeypatch.setattr(scheduled_forget, "MAX_SCANNED", 1)
    for n in range(2):
        await cli(memory, "remember", stdin=f"note {n}")
    code, text = await cli(memory, "forget-due")
    assert code == 0 and "walk stopped after 1 episode(s)" in text and "--before" in text, text
    code, text = await cli(memory, "--json", "forget-due", "--before", "2")
    body = json.loads(text)
    assert body["scanned"] == 1 and body["scan_complete"] is True


async def test_forget_due_refuses_a_now_later_than_the_clock(memory):
    with pytest.raises(InvalidInput, match="later than the engine clock"):
        await cli(memory, "forget-due", "--now", "2030-01-01T00:00:00Z")
