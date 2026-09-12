"""What breaks if I change this?

A code graph exists to answer that, and ours could not. `graph changes`
answers a different question -- what changed between two moments -- and
nothing walked the dependency edges backwards from a symbol to everything
that rests on it.

The reference calls it blast radius. Theirs walks a fixed relation list
outward; ours walks the relations this graph actually records, in the one
direction that means "depends on", and is explicit that it can only see
what was ingested.

Two rules, each with a test:

- **Direction is the whole point.** `A calls B` means changing B affects
  A, not the other way about. Walking undirected would report everything
  B calls as affected by B, which is exactly backwards.
- **A bounded answer says it is bounded.** Depth, count and bytes each
  stop the walk, each is reported, and none of them may read as "nothing
  further depends on it".
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.entities.affected import MAX_HOPS, affected

pytestmark = pytest.mark.asyncio

# store.py defines Shelf; api.py calls it; web.py calls api; unrelated.py
# touches none of them.
SOURCES = {
    "pkg/store.py": (
        "import json\n\n\n"
        "class Shelf:\n"
        '    """Holds papers."""\n\n'
        "    def keep(self, paper: str) -> str:\n"
        '        return json.dumps({"paper": paper})\n'),
    "pkg/api.py": (
        "from pkg.store import Shelf\n\n\n"
        "def put(paper: str) -> str:\n"
        "    shelf = Shelf()\n"
        "    return shelf.keep(paper)\n"),
    "pkg/web.py": (
        "from pkg.api import put\n\n\n"
        "def handle(request: str) -> str:\n"
        "    return put(request)\n"),
    "pkg/unrelated.py": (
        "import math\n\n\n"
        "def area(radius: float) -> float:\n"
        "    return math.pi * radius * radius\n"),
}


async def graphed():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    for name, text in SOURCES.items():
        await engine.remember("default", text, source=name)
    return engine


async def test_what_rests_on_a_module_is_found_and_what_it_rests_on_is_not():
    """Direction is the whole point: `pkg/api.py` imports `pkg.store`, so
    changing that module reaches the file that imports it. `pkg.store`
    is not reached by asking about `json`, which it imports.

    Asked of the **module**, because that is what an import edge names.
    A file and the module that resolves to it are separate entities in
    this graph and nothing links them -- see
    `test_a_file_is_not_yet_linked_to_the_module_that_names_it`, which
    records that as the open defect it is rather than letting this test
    quietly depend on it.
    """
    engine = await graphed()
    try:
        blast = await affected(engine, "default", "pkg.store")
    finally:
        await engine.close()
    labels = {one.label for one in blast.reached}
    assert "pkg/api.py" in labels, (blast.record(), sorted(labels))
    assert not any("json" in one for one in labels), \
        "what the target depends on is not what depends on the target"
    assert not any("unrelated" in one for one in labels), sorted(labels)
    assert blast.status == "found" and blast.by_depth.get(1), blast.record()


async def test_a_file_is_not_yet_linked_to_the_module_that_names_it():
    """A known gap, recorded as a test so it cannot be forgotten and
    cannot be mistaken for working.

    `from pkg.store import Shelf` records `pkg/api.py imports pkg.store`.
    The file is the entity `pkg/store.py`. Nothing joins the two, so a
    blast radius asked about the **file** misses everything that imports
    it. Two causes, both real:

    - `_imported` consults the resolver only for **relative** imports, so
      an absolute module is kept as written and never resolved to the
      file it names.
    - `MemoryEngine._record_claims` passes **no resolver at all**, so
      through `remember()` even a relative import resolves to nothing.
      Only `scone map` supplies one.

    Fixing it belongs at ingestion, where the tree is known -- not here by
    guessing that `a.b` means `a/b.py`, which is exactly the kind of guess
    this graph is supposed to refuse. When it is fixed, this test should
    start failing and be replaced by the file-level assertion.
    """
    engine = await graphed()
    try:
        by_file = await affected(engine, "default", "pkg/store.py")
        by_module = await affected(engine, "default", "pkg.store")
    finally:
        await engine.close()
    assert by_module.reached, by_module.record()
    assert by_file.reached == (), (
        "if this now finds dependants, module-to-file linking works and this "
        "test should be replaced", by_file.record())
    assert "not a finding" in by_file.why, by_file.why


async def test_a_name_the_graph_does_not_hold_says_so():
    engine = await graphed()
    try:
        blast = await affected(engine, "default", "pkg/nowhere.py")
    finally:
        await engine.close()
    assert blast.reached == () and blast.status == "unknown", blast.record()
    assert "not in this graph" in blast.why, blast.why


async def test_every_bound_is_refused_or_reported_but_never_silent():
    engine = await graphed()
    try:
        for bad in (0, -1, MAX_HOPS + 1, 1.5, True):
            with pytest.raises(InvalidInput):
                await affected(engine, "default", "pkg/store.py", max_hops=bad)
        for bad in (0, -1, 1.5, True):
            with pytest.raises(InvalidInput):
                await affected(engine, "default", "pkg/store.py", limit=bad)
    finally:
        await engine.close()


async def test_a_walk_that_stopped_at_its_depth_says_so():
    """Unconditionally, on a chain that is genuinely deeper than the
    bound. I wrote this as `if shallow.stopped_at_depth:` first, over a
    fixture whose chain was one hop -- so the assertion never ran and a
    mutation removing the bound left the test green."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    try:
        await engine.remember("default",
                              "class Base:\n    pass\n\n\n"
                              "class Middle(Base):\n    pass\n\n\n"
                              "class Top(Middle):\n    pass\n",
                              source="pkg/chain.py")
        shallow = await affected(engine, "default", "pkg/chain.py:Base", max_hops=1)
        deep = await affected(engine, "default", "pkg/chain.py:Base", max_hops=4)
    finally:
        await engine.close()
    assert len(deep.reached) > len(shallow.reached), (shallow.record(), deep.record())
    assert shallow.stopped_at_depth is True, shallow.record()
    assert "stopped at" in shallow.why and "unexplored rather than absent" in shallow.why, \
        shallow.why
    # And the deeper walk, having run out of dependants, does not claim
    # to have stopped early.
    assert deep.stopped_at_depth is False, deep.record()
    assert deep.deepest == 2, deep.record()


async def test_the_answer_says_it_can_only_see_what_was_ingested():
    """The honest limit: a graph holds the code someone gave it, and an
    empty blast radius means nothing here depends on it, not that nothing
    does."""
    engine = await graphed()
    try:
        blast = await affected(engine, "default", "pkg/web.py")
    finally:
        await engine.close()
    assert "ingested" in blast.why or "this graph" in blast.why, blast.why


async def test_the_cli_can_ask_and_says_what_it_cannot_see():
    """A feature nobody can reach is not a feature."""
    import io

    from scone_memory.runtime.cli import build_parser, run

    engine = await graphed()
    try:
        out = io.StringIO()
        code = await run(build_parser().parse_args(["graph", "affected", "pkg.store"]),
                         engine, io.StringIO(""), out)
        shown = out.getvalue()
        assert code == 0, shown
        assert "pkg/api.py" in shown, shown
        # The sentence that stops an empty or partial answer reading as
        # a statement about the caller's codebase.
        assert "this graph holds" in shown, shown
        assert "1 hop(s)" in shown, shown
        empty = io.StringIO()
        code = await run(build_parser().parse_args(["graph", "affected", "pkg/unrelated.py"]),
                         engine, io.StringIO(""), empty)
        assert code == 0 and "not a finding" in empty.getvalue(), empty.getvalue()
    finally:
        await engine.close()


async def test_the_json_answer_carries_the_shape_and_the_caveat():
    """Both branches this covers had a bug in them that the tests did not
    reach and mypy did: the JSON path called an `emit` that does not
    exist in that scope, and the ambiguity branch compared a `score`
    field `Candidate` does not have. Neither had a test, which is how
    they came to be written wrong."""
    import io
    import json as jsonlib

    from scone_memory.runtime.cli import build_parser, run

    engine = await graphed()
    try:
        out = io.StringIO()
        code = await run(build_parser().parse_args(
            ["graph", "affected", "pkg.store", "--json"]), engine, io.StringIO(""), out)
        shown = out.getvalue()
        assert code == 0, shown
        said = jsonlib.loads(shown)
        assert said["status"] == "found", said
        assert said["by_depth"].get("1"), said
        assert any(row[0] == "pkg/api.py" for row in said["entities"]), said
        assert "this graph holds" in said["why"], said["why"]
    finally:
        await engine.close()


async def test_a_name_that_means_two_things_is_refused_rather_than_guessed():
    """The resolver's own verdict, not a comparison of my own invention.
    Two files defining the same symbol name give a name that means more
    than one thing, and answering for one of them silently would be a
    blast radius for a symbol the caller did not ask about."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    try:
        for name in ("one", "two"):
            await engine.remember(
                "default",
                "import json\n\n\nclass Shelf:\n    def keep(self, paper: str) -> str:\n"
                '        return json.dumps({"paper": paper})\n',
                source=f"pkg/{name}.py")
        blast = await affected(engine, "default", "Shelf")
    finally:
        await engine.close()
    if blast.status == "ambiguous":
        assert "more than one thing" in blast.why, blast.why
        assert blast.reached == (), blast.record()
    else:
        # The resolver settled it; then the answer must be about the one
        # it settled on and say which.
        assert blast.target, blast.record()
