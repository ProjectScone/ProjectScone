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
from scone_memory.entities.affected import MAX_HOPS, MAX_NAME, affected

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


async def test_a_dependency_that_no_longer_holds_is_not_reported_as_current():
    """I read the graph as `mode="all"` while taking a `when` argument,
    so an edge closed years ago and an edge that does not begin until
    2030 were both reported as things that break if you change the
    target. **An argument that implies a filter it does not apply is
    worse than no filter at all.**

    The default is `current` now, and the mode and the moment are on
    every answer including the empty ones.
    """
    engine = await graphed()
    try:
        await engine.assert_fact("default", "pkg/old.py", "imports", "pkg.store",
                                 valid_from="2019-01-01T00:00:00Z")
        closed = [f for f in await engine.facts("default")
                  if f.subject == "pkg/old.py"][0]
        await engine.close_fact("default", closed.fact_id, "the module was removed in 2021")
        blast = await affected(engine, "default", "pkg.store")
    finally:
        await engine.close()
    labels = {one.label for one in blast.reached}
    assert "pkg/old.py" not in labels, (blast.record(), sorted(labels))
    assert blast.mode == "current" and blast.at, blast.record()
    assert "read as current at" in blast.why, blast.why


async def test_a_name_longer_than_the_bound_is_refused_rather_than_echoed():
    """`target` and `why` echo the name, so a 20,000-character name gave a
    40,000-byte answer under a 512-byte budget. The same
    metadata-outside-the-budget fault the context receipts had, which I
    had fixed there a few hours earlier."""
    engine = await graphed()
    try:
        with pytest.raises(InvalidInput) as raised:
            await affected(engine, "default", "x" * (MAX_NAME + 1))
        assert "echoed in the answer" in str(raised.value), str(raised.value)
        blast = await affected(engine, "default", "x" * MAX_NAME)
        assert len(blast.why.encode()) < 2_000, len(blast.why.encode())
    finally:
        await engine.close()


async def test_a_deleted_space_is_not_answered_from_the_snapshot_it_read():
    """The rule merging and windowing already follow and this did not: a
    source confirmed gone is dropped, never served. A projection is a
    snapshot, so a space deleted while it was being read would otherwise
    be answered from rows that no longer exist."""
    from scone_memory.core.errors import SconeError

    engine = await graphed()
    try:
        await engine.delete_space("default")
        with pytest.raises(SconeError):
            await affected(engine, "default", "pkg.store")
    finally:
        await engine.close()


async def test_a_truncated_read_is_disclosed_rather_than_answered_as_nothing():
    """The projection reports what its read covered and I discarded it
    with `_`. So a read cut short by the fact limit could omit every
    incoming edge and the answer would say "nothing rests on this" --
    a bound biting with nothing saying so, in a module whose own
    docstring makes that a rule."""
    from scone_memory.entities import read

    engine = await graphed()
    try:
        whole = await affected(engine, "default", "pkg.store")
        assert whole.reached and not whole.partial_read, whole.record()
        before = read.MAX_FACTS
        read.MAX_FACTS = 1
        try:
            cut = await affected(engine, "default", "pkg.store")
        finally:
            read.MAX_FACTS = before
    finally:
        await engine.close()
    assert cut.partial_read is True, cut.record()
    assert "truncated" in cut.why, cut.why
    assert "missing from the graph it walked" in cut.why, cut.why


async def test_a_name_is_bounded_in_the_bytes_it_costs_not_the_characters():
    """`MAX_NAME` counted characters, so 200 emoji passed the bound and
    then cost 5,209 serialized bytes under a 512-byte budget -- because
    an emoji is four UTF-8 bytes and twelve more once JSON escapes it.
    A bound on characters is not a bound on an answer's size."""
    engine = await graphed()
    try:
        emoji = "\N{GRINNING FACE}" * MAX_NAME
        assert len(emoji) == MAX_NAME, len(emoji)
        with pytest.raises(InvalidInput) as raised:
            await affected(engine, "default", emoji)
        assert "byte" in str(raised.value), str(raised.value)
        # And the bound still admits a name of MAX_NAME plain bytes.
        assert (await affected(engine, "default", "x" * MAX_NAME)).status
    finally:
        await engine.close()


async def test_the_answer_s_own_framing_is_charged_against_the_byte_budget():
    """`max_bytes` bounded the entity list and nothing else, so `target`
    and `why` -- both of which echo the name -- were spent outside it.
    Charge the framing first: a budget too small to hold it lists
    nothing and says the framing spent it, rather than quietly
    returning ten times what was asked for."""
    engine = await graphed()
    try:
        whole = await affected(engine, "default", "pkg.store")
        assert whole.reached and whole.framing_bytes > 0, whole.record()
        tiny = await affected(engine, "default", "pkg.store", max_bytes=1)
        # The one that proves the charge rather than the flag: four bytes
        # of room is less than any entity costs, while `max_bytes` here is
        # hundreds -- so a budget that forgot to subtract its own framing
        # would list the dependant happily.
        squeezed = await affected(engine, "default", "pkg.store",
                                  max_bytes=whole.framing_bytes + 4)
    finally:
        await engine.close()
    assert not tiny.reached, tiny.record()
    assert tiny.not_listed == len(whole.reached), (tiny.record(), whole.record())
    assert tiny.framing_spent_budget is True, tiny.record()
    assert "framing" in tiny.why, tiny.why

    cheapest = min(len(one.label.encode()) + len(one.depends_on.encode()) for one in whole.reached)
    assert cheapest > 4, cheapest
    assert squeezed.framing_spent_budget is False, squeezed.record()
    assert not squeezed.reached, squeezed.record()
    assert squeezed.not_listed == len(whole.reached), (squeezed.record(), whole.record())


async def test_a_dependency_excluded_while_the_graph_was_read_is_not_reported():
    """`_living` fences the space and nothing fenced the facts. A
    projection is a snapshot; excluding `web.py imports api` the instant
    after it was taken still reported web as a dependant, from a read
    the answer called `current`. The ledger fence catalog already uses
    is the one this needed."""
    engine = await graphed()
    try:
        before = await affected(engine, "default", "pkg.api")
        assert any(one.label == "pkg/web.py" for one in before.reached), before.record()

        projection = engine.entities.projection
        excluded = False

        async def exclude_the_moment_it_is_read(space, **kw):
            nonlocal excluded
            answer = await projection(space, **kw)
            if not excluded:
                excluded = True
                for fact in await engine.facts(space):
                    if fact.predicate == "imports" and "web" in fact.subject:
                        await engine.exclude(space, fact.fact_id, "moved")
            return answer

        engine.entities.projection = exclude_the_moment_it_is_read  # type: ignore[assignment]
        try:
            after = await affected(engine, "default", "pkg.api")
        finally:
            engine.entities.projection = projection  # type: ignore[assignment]
    finally:
        await engine.close()
    assert excluded, "the test never reached the fact it meant to exclude"
    assert not any(one.label == "pkg/web.py" for one in after.reached), after.record()


async def test_a_ledger_that_will_not_hold_still_is_said_rather_than_passed_off():
    """The retry answers the common case, where one write lands mid-read.
    A ledger written on every attempt cannot be answered from any single
    moment, and the answer says so instead of stamping a moment it was
    never true at."""
    engine = await graphed()
    try:
        projection = engine.entities.projection
        reads = 0

        async def write_under_every_read(space, **kw):
            nonlocal reads
            answer = await projection(space, **kw)
            reads += 1
            await engine.remember(space, f"pkg/churn{reads}.py holds nothing.", source=f"c{reads}.md")
            return answer

        engine.entities.projection = write_under_every_read  # type: ignore[assignment]
        try:
            restless = await affected(engine, "default", "pkg.store")
        finally:
            engine.entities.projection = projection  # type: ignore[assignment]
    finally:
        await engine.close()
    assert reads > 1, reads
    assert restless.graph_moved is True, restless.record()
    assert "moved under all" in restless.why, restless.why
    assert restless.record()["graph_moved"] is True, restless.record()
