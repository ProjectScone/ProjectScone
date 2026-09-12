"""A recalled function body, with the two things that make it readable.

A chunk of code says which declaration it came from -- `Engine.forget` --
and that is where our answer stopped. It did not say the declaration's
signature, so a caller saw a body without its parameters, and it did not
say what the file imported, so a name in the body could not be traced to
where it came from. For "how is this done here", a body without its
signature and its imports is a fragment.

The reference that has this prepends the context into the chunk text.
Ours must not: invariant I1 says `content[span.start:span.end]` is the
source unchanged. So the context sits beside the chunk, quoted from the
source with line numbers, and a caller can check every line of it
against the file.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.code_context import MAX_IMPORTS, code_context

pytestmark = pytest.mark.asyncio

MODULE = '''"""A shelf of papers."""

from __future__ import annotations

import json
from pathlib import Path

from .base import Store


class Shelf(Store):
    """Holds papers."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def keep(
        self,
        paper: str,
        *,
        tag: str = "unsorted",
    ) -> Path:
        """Write a paper to the shelf and return where it went.

        The body is deliberately long enough that a chunk of it does not
        also contain the signature above, which is the whole point of
        this fixture: the caller gets the body and needs the head.
        """
        target = self._root / f"{tag}.json"
        payload = json.dumps({"paper": paper, "tag": tag})
        target.write_text(payload, encoding="utf-8")
        for _ in range(3):
            target.touch()
        return target
'''


async def memory(content=MODULE, source="pkg/shelf.py", target=200):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                               chunk_target=target).open()
    await engine.remember("default", content, source=source)
    return engine


async def test_a_recalled_body_is_given_its_signature_and_its_imports():
    engine = await memory()
    try:
        found = await engine.recall("default", "write a paper to the shelf payload dumps", limit=3)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    assert context.by_chunk, context.record()
    holding = [c for c in context.by_chunk.values()
               if any("keep" in s.name for s in c.holders)]
    assert holding, {k: [s.name for s in v.holders] for k, v in context.by_chunk.items()}
    one = holding[0]
    signature = next(s for s in one.holders if "keep" in s.name)
    # The whole signature, not its first line: `def keep(` alone says
    # nothing about the parameters, which is what a caller wants.
    assert "def keep(" in signature.text, signature.text
    assert "tag: str = \"unsorted\"" in signature.text, signature.text
    assert signature.text in MODULE, "the signature is quoted, never assembled"
    assert [i.text for i in one.imports] == [
        "from __future__ import annotations", "import json",
        "from pathlib import Path", "from .base import Store"], [i.text for i in one.imports]
    for line in one.imports:
        assert MODULE.splitlines()[line.line - 1] == line.text, (line.line, line.text)


async def test_the_holders_run_outermost_first():
    engine = await memory()
    try:
        found = await engine.recall("default", "write a paper to the shelf payload dumps", limit=3)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    chains = [[s.name for s in c.holders] for c in context.by_chunk.values() if c.holders]
    assert any(chain == ["Shelf", "Shelf.keep"] for chain in chains), chains


async def test_prose_is_not_given_code_context():
    """A file called notes.md holding a code block is prose that quotes
    code. Nothing is guessed from the content."""
    engine = await memory("Some notes about json and Path.\n\n" + MODULE, source="notes.md")
    try:
        found = await engine.recall("default", "notes json path", limit=3)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    assert context.by_chunk == {}, context.record()
    assert context.not_code == len(found.items), context.record()
    assert "does not say" in context.why, context.why


async def test_a_file_with_more_imports_than_the_bound_says_so():
    """A count a reader could mistake for the file's imports must not be
    a count of the ones we listed."""
    lines = "".join(f"import module{n}\n" for n in range(40))
    engine = await memory(lines + "\n\ndef work():\n    return module1\n", source="many.py")
    try:
        found = await engine.recall("default", "work return module", limit=3)
        context = await code_context(engine, "default", found.items, imports=5)
    finally:
        await engine.close()
    assert context.by_chunk, context.record()
    one = next(iter(context.by_chunk.values()))
    assert len(one.imports) == 5 and one.more_imports is True, one
    assert context.capped == len(context.by_chunk), context.record()
    assert "not all of them" in context.why, context.why
    assert MAX_IMPORTS > 5


async def test_a_source_that_is_gone_is_not_answered_from_stale_text():
    engine = await memory()
    try:
        found = await engine.recall("default", "write a paper shelf", limit=2)
        await engine.forget("default", found.items[0].episode_id)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    assert context.by_chunk == {} and context.gone > 0, context.record()
    assert "no longer there" in context.why, context.why


async def test_the_episode_budget_is_checked_and_bounds_the_reads():
    engine = await memory()
    try:
        found = await engine.recall("default", "write a paper shelf", limit=2)
        for bad in (0, -1, 1.5, True, "2"):
            with pytest.raises(InvalidInput):
                await code_context(engine, "default", found.items, episodes=bad)
        for bad in (0, -1, 1.5, True):
            with pytest.raises(InvalidInput):
                await code_context(engine, "default", found.items, imports=bad)
    finally:
        await engine.close()


async def test_the_cli_can_ask_for_it_and_does_not_by_default():
    """A feature nobody can reach is not a feature. `--code-context` is
    the reachable surface, and it stays opt-in like `--merge` and
    `--window` beside it."""
    import io

    from scone_memory.runtime.cli import build_parser, run

    engine = await memory()
    try:
        asked = ["recall", "write a paper to the shelf payload dumps", "--code-context",
                 "--limit", "3"]
        out = io.StringIO()
        code = await run(build_parser().parse_args(asked), engine, io.StringIO(""), out)
        shown = out.getvalue()
        assert code == 0, shown
        assert "inside Shelf.keep" in shown, shown
        assert "from pathlib import Path" in shown, shown
        plain = io.StringIO()
        code = await run(build_parser().parse_args(asked[:2] + ["--limit", "3"]),
                         engine, io.StringIO(""), plain)
        assert code == 0 and "from pathlib import Path" not in plain.getvalue(), \
            "code context stays opt-in"
    finally:
        await engine.close()
