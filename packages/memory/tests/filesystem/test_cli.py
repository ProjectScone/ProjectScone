"""The tree at the command line: ls, cat, find and write.

The same tree, the same policy and the same refusals, in the form a
person at a terminal expects. Writing needs the flag that says so, so
that a mistyped path cannot write where a person only meant to look.
"""

from __future__ import annotations

import io

import pytest

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex,
                          MemoryEngine)
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime import cli
from scone_memory.runtime.cli import build_parser, run
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
DAY = "2024-01-01T00:00:00Z"


async def memory() -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z"), events=InMemoryEventLog()).open()
    note = await engine.remember("default", "The Lisbon office opened in March and holds forty desks.")
    await engine.assert_fact("default", "lisbon office", "opened_on", "March 2024", valid_from=DAY,
                             source_episode_id=note.episode_id, quote="The Lisbon office opened in March")
    return engine


async def said(engine, *args, stdin: str = "") -> tuple[int, str]:
    out = io.StringIO()
    code = await run(build_parser().parse_args(list(args)), engine, io.StringIO(stdin), out)
    return code, out.getvalue()


async def test_ls_lists_a_directory():
    code, text = await said(await memory(), "fs", "ls", "/")
    assert code == 0
    assert "/episodes" in text and "/facts" in text and "/notes" in text


async def test_cat_prints_a_file_and_nothing_else():
    code, text = await said(await memory(), "fs", "cat", "/episodes/1.md")
    assert code == 0 and text.strip() == "The Lisbon office opened in March and holds forty desks."


async def test_find_answers_in_paths():
    code, text = await said(await memory(), "fs", "find", "desks")
    assert code == 0 and ".md" in text and "desks" in text


async def test_writing_needs_the_flag_that_says_so():
    """Without --writable the tree is read only, and a mistyped path cannot
    write where somebody only meant to look."""
    engine = await memory()
    with pytest.raises(InvalidInput, match="read only"):
        await said(engine, "fs", "write", "/notes/plan.md", stdin="Ship on Friday.")
    assert (await engine.documents.counts("default")).episodes == 1


async def test_a_note_is_written_from_what_was_piped_in():
    engine = await memory()
    code, text = await said(engine, "fs", "write", "/notes/plan.md", "--writable", stdin="Ship on Friday.")
    assert code == 0, text
    _, back = await said(engine, "fs", "cat", "/notes/plan.md")
    assert back.strip() == "Ship on Friday."


async def test_a_path_that_tries_to_leave_is_refused_at_the_command_line():
    with pytest.raises(InvalidInput, match="leave"):
        await said(await memory(), "fs", "cat", "/episodes/../../etc/passwd")


def test_a_refusal_leaves_the_command_line_with_a_failing_exit_code(tmp_path):
    """What run raises, main turns into an exit code and a message."""
    out = io.StringIO()
    code = cli.main(["fs", "cat", "/episodes/../../etc/passwd"],
                    env={"SCONE_SQLITE_PATH": str(tmp_path / "m.db")}, stdin=io.StringIO(""), out=out)
    assert code == 2
