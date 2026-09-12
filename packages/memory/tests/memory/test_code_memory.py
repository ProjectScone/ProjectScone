"""Code remembered as its declarations, and recall that says which one."""

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine

pytestmark = pytest.mark.asyncio

SOURCE = '''"""The planner."""

from __future__ import annotations

LIMIT = 12


def plan(question: str) -> str:
    """Work out what is being asked, and say so in one line."""
    asked = question.strip()
    if not asked:
        raise ValueError("a question is needed")
    return asked.lower()


class Engine:
    """Holds the parts a plan is run with."""

    def recall(self, query: str) -> list[str]:
        """Answer a query from what has been stored, best first."""
        found = [line for line in self.lines if query in line]
        return sorted(found, key=len)

    def forget(self, query: str) -> int:
        """Take out every line that matches, and say how many went."""
        keeping = [line for line in self.lines if query not in line]
        gone = len(self.lines) - len(keeping)
        self.lines = keeping
        return gone
'''


async def engine(**options):
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              chunk_target=200, **options).open()


async def test_a_source_file_is_cut_into_its_declarations():
    memory = await engine()
    added = await memory.remember("alpha", SOURCE, kind="file", source="src/app/planner.py")
    chunks = await memory.documents.chunks_of("alpha", added.episode_id)
    texts = [chunk.text for chunk in chunks]
    assert any(t.startswith("def plan(") and "return asked.lower()" in t for t in texts), texts
    assert all(t == t.rstrip() or t.endswith("\n") for t in texts)


async def test_prose_is_not_treated_as_code():
    memory = await engine()
    prose = "A paragraph about planning. " * 40
    added = await memory.remember("alpha", prose, kind="note", source="notes/planning.md")
    chunks = await memory.documents.chunks_of("alpha", added.episode_id)
    assert len(chunks) > 1 and all("def " not in c.text for c in chunks)


async def test_declaration_cutting_can_be_turned_off():
    ordinary = await engine(code_aware=False)
    aware = await engine()
    plain = await ordinary.remember("alpha", SOURCE, kind="file", source="src/app/planner.py")
    cut = await aware.remember("alpha", SOURCE, kind="file", source="src/app/planner.py")
    before = [(c.start, c.end) for c in await ordinary.documents.chunks_of("alpha", plain.episode_id)]
    after = [(c.start, c.end) for c in await aware.documents.chunks_of("alpha", cut.episode_id)]
    assert before != after


async def test_a_recalled_chunk_says_where_in_the_file_it_came_from():
    memory = await engine()
    await memory.remember("alpha", SOURCE, kind="file", source="src/app/planner.py")
    result = await memory.recall("alpha", "take out every line that matches")
    [item] = [i for i in result.items if "def forget(" in i.text]
    assert item.source == "src/app/planner.py"
    assert item.declaration == "Engine.forget"
    assert SOURCE.encode()[item.start : item.end].decode() == item.text
    assert item.first_line < item.last_line
    assert SOURCE.split("\n")[item.first_line - 1].strip().startswith("def forget(")


async def test_a_recalled_paragraph_names_no_declaration():
    memory = await engine()
    await memory.remember("alpha", "Planning is what we do on Tuesdays.", kind="note")
    result = await memory.recall("alpha", "planning")
    [item] = result.items
    assert item.declaration is None and item.first_line == 1


CR_SOURCE = "def plan():\r    return 1\r\r\rdef widen():\r    return 2\r"


@pytest.mark.parametrize("code_aware", [True, False])
async def test_a_file_written_with_carriage_returns_is_remembered_and_recalled(code_aware):
    """Old Mac line endings are lines to Python's parser, and must be lines
    here too, whether or not the source is cut at its declarations."""
    memory = await engine(code_aware=code_aware)
    await memory.remember("alpha", CR_SOURCE, kind="file", source="legacy/cr.py")
    result = await memory.recall("alpha", "widen")
    [item] = [i for i in result.items if "def widen" in i.text]
    assert item.first_line >= 1 and item.last_line >= item.first_line
    assert CR_SOURCE.encode()[item.start : item.end].decode() == item.text


async def test_lines_are_counted_the_way_the_file_is_written():
    memory = await engine()
    await memory.remember("alpha", CR_SOURCE, kind="file", source="legacy/cr.py")
    result = await memory.recall("alpha", "widen")
    [item] = result.items
    assert (item.first_line, item.last_line) == (1, 6), (item.first_line, item.last_line)
