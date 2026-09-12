"""Pointing the framework at a directory and asking it questions after.

A codebase is the case where "remember this, then ask about it" has an
obvious shape: walk the files, store each one, and — when asked — record
what each says about itself. What it read and what it did not is said
plainly, because a map that quietly skipped half a repository is worse
than no map.
"""

from __future__ import annotations

import io

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.runtime.cli import build_parser, run
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio

PLANNER = '''import json


def plan(question: str) -> str:
    return tidy(question)


def tidy(text: str) -> str:
    return text.strip()
'''
STORE = '''def keep(record: str) -> None:
    return None
'''


@pytest.fixture()
def tree(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "planner.py").write_text(PLANNER, encoding="utf-8")
    (tmp_path / "app" / "store.py").write_text(STORE, encoding="utf-8")
    (tmp_path / "app" / "broken.py").write_text("def half(:\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# not code\n", encoding="utf-8")
    return tmp_path


async def memory() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              clock=Clock("2025-06-01T00:00:00.000Z")).open()


async def mapped(engine, *args) -> str:
    out = io.StringIO()
    code = await run(build_parser().parse_args(["map", *args]), engine, io.StringIO(""), out)
    assert code == 0, out.getvalue()
    return out.getvalue()


async def test_mapping_a_directory_remembers_every_source_file(tree):
    engine = await memory()
    said = await mapped(engine, str(tree))
    counts = await engine.documents.counts("default")
    assert counts.episodes == 3, "three .py files; the readme is not code"
    assert "3 file(s)" in said


async def test_a_file_is_stored_under_the_path_it_was_read_from(tree):
    engine = await memory()
    await mapped(engine, str(tree))
    sources = {episode.source for episode in await engine.documents.recent_episodes("default", 10)}
    assert sources == {"app/planner.py", "app/store.py", "app/broken.py"}


async def test_mapping_can_record_what_each_file_says_about_itself(tree):
    engine = await memory()
    said = await mapped(engine, str(tree), "--graph")
    facts = await engine.documents.list_facts("default", include_closed=True)
    triples = {(f.subject, f.predicate, f.object) for f in facts}
    assert ("app/planner.py:plan", "calls", "app/planner.py:tidy") in triples
    assert ("app/planner.py", "imports", "json") in triples
    assert "claim(s)" in said


async def test_without_the_flag_nothing_is_claimed(tree):
    engine = await memory()
    await mapped(engine, str(tree))
    assert await engine.documents.list_facts("default", include_closed=True) == []


async def test_a_file_that_cannot_be_read_is_counted_apart_from_one_with_nothing_to_say(tree):
    """A file this cannot read and a file that simply holds no claims are
    different things, and a count that adds them together says neither."""
    (tree / "app" / "empty.py").write_text("X = 1\n", encoding="utf-8")
    engine = await memory()
    said = await mapped(engine, str(tree), "--graph")
    assert "1 could not be read" in said, said
    assert "1 had nothing to say" in said, said


async def test_a_directory_that_is_not_one_is_refused(tmp_path):
    from scone_memory.core.errors import InvalidInput

    engine = await memory()
    with pytest.raises(InvalidInput, match="directory"):
        await run(build_parser().parse_args(["map", str(tmp_path / "nowhere")]), engine,
                  io.StringIO(""), io.StringIO())


async def test_mapping_the_same_directory_again_changes_nothing(tree):
    engine = await memory()
    await mapped(engine, str(tree), "--graph")
    facts = len(await engine.documents.list_facts("default", include_closed=True))
    episodes = (await engine.documents.counts("default")).episodes
    said = await mapped(engine, str(tree), "--graph")
    assert (await engine.documents.counts("default")).episodes == episodes
    assert len(await engine.documents.list_facts("default", include_closed=True)) == facts
    assert "3 already here" in said, said


async def test_mapping_resolves_the_imports_between_the_files_it_saw(tmp_path):
    """The edges that matter most in a codebase are the ones inside it."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "shared.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "worker.py").write_text(
        "from .shared import VALUE\nfrom . import shared\nimport json\n", encoding="utf-8")
    engine = await memory()
    await mapped(engine, str(tmp_path), "--graph")
    triples = {(f.subject, f.predicate, f.object)
               for f in await engine.documents.list_facts("default", include_closed=True)}
    assert ("pkg/worker.py", "imports", "pkg/shared.py") in triples
    assert ("pkg/worker.py", "imports", "json") in triples


async def test_mapping_leaves_out_an_import_of_something_it_did_not_see(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "worker.py").write_text("from .missing import thing\n", encoding="utf-8")
    engine = await memory()
    await mapped(engine, str(tmp_path), "--graph")
    assert not [f for f in await engine.documents.list_facts("default", include_closed=True)
                if f.predicate == "imports"], "nothing points at a file that is not there"
