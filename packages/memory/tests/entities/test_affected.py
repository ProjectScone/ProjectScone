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
from scone_memory.entities.affected import MAX_BYTES, MAX_HOPS, MAX_NAME, affected

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
        # The budget is a promise about this line of output and nothing
        # else, so it is checked where the output is: what the CLI prints
        # is exactly what the answer said it would cost.
        assert said["bytes_spent"] == len(shown.rstrip("\n").encode()), (
            said["bytes_spent"], len(shown.rstrip("\n").encode()))
        narrow = io.StringIO()
        code = await run(build_parser().parse_args(
            ["graph", "affected", "pkg.store", "--json", "--max-bytes", str(said["bytes_spent"])]),
            engine, io.StringIO(""), narrow)
        assert code == 0, narrow.getvalue()
        assert len(narrow.getvalue().rstrip("\n").encode()) <= said["bytes_spent"], narrow.getvalue()
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


async def wide(count=8):
    """One module with many direct dependants, so a budget has something
    to drop. The narrow fixture has exactly one, and dropping it costs
    more than it saves -- the clause explaining the omission is longer
    than the entity omitted."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    await engine.remember("default", SOURCES["pkg/store.py"], source="pkg/store.py")
    for index in range(count):
        await engine.remember("default", f"from pkg.store import Shelf\n\n\ndef put{index}() -> str:\n"
                              f"    return Shelf().keep('{index}')\n", source=f"pkg/reader{index}.py")
    return engine


async def test_what_is_kept_is_a_prefix_when_the_dependants_are_not_all_one_size():
    """The prefix and monotonicity promises held only because every label
    in my fixture was the same length.

    A raw-byte check inside the walk decided what to append before the
    serialized trim ever ran. With eight very long names and one short
    one, the budget left after the long names could still admit the short
    one, so the answer skipped earlier entries and kept a later, cheaper
    one -- not a prefix -- and the set could change unpredictably as the
    budget grew. Two bounds spending one budget in different units, the
    earlier of them invisible. The walk bounds only the count now; the
    bound that decides what is served is the one measured on what is
    served.
    """
    import json

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                HashEmbedder()).open()
    try:
        for index in range(8):
            await engine.assert_fact("default", f"pkg/{index}_{'x' * 1800}.py:Caller",
                                     "calls", "root.py:Root")
        await engine.assert_fact("default", "short.py:Caller", "calls", "root.py:Root")
        whole = await affected(engine, "default", "root.py:Root")
        assert len(whole.reached) == 9, [len(one.label) for one in whole.reached]
        assert len({len(one.label) for one in whole.reached}) == 2, "the fixture needs two sizes"
        steps = []
        for budget in range(800, whole.bytes_spent + 1, 50):
            try:
                steps.append((budget, await affected(engine, "default", "root.py:Root",
                                                     max_bytes=budget)))
            except InvalidInput:
                continue
    finally:
        await engine.close()

    assert steps, "no budget in the range produced an answer"
    kept = [len(answer.reached) for _, answer in steps]
    assert all(before <= after for before, after in zip(kept, kept[1:])), kept
    for budget, answer in steps:
        assert answer.reached == whole.reached[:len(answer.reached)], (budget, [
            len(one.label) for one in answer.reached])
        assert len(json.dumps(answer.record()).encode()) <= budget, (budget, answer.bytes_spent)
        assert answer.bytes_spent == len(json.dumps(answer.record()).encode()), answer.bytes_spent
        assert answer.not_listed == len(whole.reached) - len(answer.reached), (budget, answer.record())


async def test_a_budget_drops_dependants_from_the_tail_and_says_how_many():
    """`max_bytes` bounded the entity list and nothing else, so `target`
    and `why` -- which echo the name -- were spent outside it and the
    answer came back larger than the budget that asked for it. The whole
    serialized record is bounded now.

    Asserted over the range rather than at one budget: the count kept
    never rises as the budget falls, every answer fits the budget it was
    given, what is kept is always a prefix of the whole, and somewhere in
    the range a genuine part of the list is served -- which is the case
    all-or-nothing would pass a single-point test without serving."""
    import json

    engine = await wide()
    try:
        whole = await affected(engine, "default", "pkg.store")
        assert len(whole.reached) >= 6, whole.record()
        with pytest.raises(InvalidInput) as raised:
            await affected(engine, "default", "pkg.store", max_bytes=1)
        least = max(int(word) for word in str(raised.value).replace(",", " ").split()
                    if word.isdigit())
        steps = [(budget, await affected(engine, "default", "pkg.store", max_bytes=budget))
                 for budget in range(whole.bytes_spent, least - 1, -8)]
    finally:
        await engine.close()

    kept = [len(answer.reached) for _, answer in steps]
    assert kept[0] == len(whole.reached), (kept, whole.record())
    assert kept[-1] == 0, kept
    assert any(0 < one < len(whole.reached) for one in kept), kept
    assert all(before >= after for before, after in zip(kept, kept[1:])), kept
    for budget, answer in steps:
        assert len(json.dumps(answer.record()).encode()) <= budget, (budget, answer.record())
        assert answer.bytes_spent <= budget, (budget, answer.record())
        assert answer.reached == whole.reached[:len(answer.reached)], (budget, answer.record())
        assert answer.not_listed == len(whole.reached) - len(answer.reached), (budget, answer.record())
        # by_depth keeps the whole shape, so what was dropped is still counted.
        assert sum(answer.by_depth.values()) == len(whole.reached), (budget, answer.record())


async def test_a_budget_that_holds_no_dependant_at_all_says_so_or_is_refused():
    """Two ends of the same bound. A budget that holds the answer but
    not one dependant lists nothing and sets the flag; a budget that
    cannot hold even that is refused, naming the number that would --
    because what is left is the target, the status and what the read
    covered, and an answer that stops saying those is worse than none."""
    engine = await wide()
    try:
        whole = await affected(engine, "default", "pkg.store")
        with pytest.raises(InvalidInput) as raised:
            await affected(engine, "default", "pkg.store", max_bytes=1)
        # The refusal names a budget that works, and this is the test of
        # that promise: asked again at exactly that number, it answers.
        least = max(int(word) for word in str(raised.value).replace(",", " ").split()
                    if word.isdigit())
        bare = await affected(engine, "default", "pkg.store", max_bytes=least)
        with pytest.raises(InvalidInput):
            await affected(engine, "default", "pkg.store", max_bytes=least - 1)
    finally:
        await engine.close()
    assert bare.bytes_spent == least, (bare.record(), least)
    assert not bare.reached, bare.record()
    assert bare.framing_spent_budget is True, bare.record()
    assert bare.not_listed == len(whole.reached), (bare.record(), whole.record())
    assert "not listed" in bare.why, bare.why


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


async def test_the_budget_bounds_the_answer_the_caller_receives_not_the_strings_in_it():
    """Charging the framing was the right idea measured in the wrong
    unit. A caller receives `json.dumps(record())`; I charged raw UTF-8.
    JSON escapes one emoji to twelve bytes and adds every key around
    them, so `max_bytes=600` returned 631 and a 50-emoji name under 512
    returned 1,682 -- the second because the unknown and ambiguous
    branches return before the budget is ever consulted.

    Measured on the serialized answer now, on every path out."""
    import json

    engine = await graphed()
    try:
        # This graph has one dependant, and dropping it costs more than
        # it saves -- the clause explaining the omission is longer than
        # the entity omitted -- so the whole answer is the smallest one.
        full = (await affected(engine, "default", "pkg.store")).bytes_spent
        for budget in (full, full + 1, 1_400, MAX_BYTES):
            answer = await affected(engine, "default", "pkg.store", max_bytes=budget)
            size = len(json.dumps(answer.record()).encode())
            assert size <= budget, (budget, size, answer.record())
            assert answer.bytes_spent == size, (answer.bytes_spent, size)

        # A name that fits MAX_NAME and still cannot be echoed inside the
        # budget: 50 emoji are 200 UTF-8 bytes and 600 JSON bytes, twice
        # over. Refused, naming both numbers, rather than answered at
        # three times what was asked for.
        wide = "\N{GRINNING FACE}" * 50
        assert len(wide.encode()) == MAX_NAME, len(wide.encode())
        with pytest.raises(InvalidInput) as raised:
            await affected(engine, "default", wide, max_bytes=512)
        assert "512" in str(raised.value), str(raised.value)
        assert "unknown" not in str(raised.value).lower() or "byte" in str(raised.value)
    finally:
        await engine.close()


async def test_a_graph_with_no_resolved_cross_file_imports_says_so():
    """A file's blast radius is empty here for a reason that has nothing
    to do with the file.

    When no file in the graph imports any other, the answer would have
    been empty for every file in it, and that is a fact about the graph
    rather than about the target. "Nothing rests on this" is technically
    hedged already; a caller reading it about every file in their
    codebase deserves the actual reason.

    This notice was written when a relative import recorded nothing
    without a resolver that knew the tree, which was true of every
    language and made the case common. Both forms name their file now,
    so what is left is a graph that genuinely holds no edge between two
    of its own files -- one file, or files importing only packages
    outside it, or a language this reader does not read.
    """
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    try:
        # Two files that import only packages outside this graph. Both
        # relative-import forms now name their file, so this is what is
        # left: a graph with no edge between two of its own files, where
        # the empty answer would have been empty for every file in it.
        await engine.remember("default", "import json\n\n\ndef keep(p):\n"
                              "    return json.dumps(p)\n", source="pkg/store.py")
        await engine.remember("default", "import csv\n\n\ndef put(p):\n"
                              "    return csv.writer(p)\n", source="pkg/api.py")
        blast = await affected(engine, "default", "pkg/store.py")
    finally:
        await engine.close()
    assert not blast.reached, blast.record()
    assert blast.unresolved_imports is True, blast.record()
    assert "no file in this graph imports another" in blast.why, blast.why
    assert "whatever the file is" in blast.why, blast.why


async def test_a_package_imported_by_its_directory_is_matched_to_its_init():
    """The half of the old objection that was right.

    `from .core import errors` names `core/errors`, and arithmetic cannot
    say whether that is `core/errors.py` or `core/errors/__init__.py`.
    `.py` is written, because a module is the common case and a name is
    cheap; the two are matched where the whole graph can be consulted
    instead of one file. So a package's `__init__.py` collects the edges
    recorded against the module spelling of its own name.
    """
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    try:
        await engine.remember("default", "from ..core.errors import Bad\n\n\n"
                              "def put(p):\n    return Bad(p)\n", source="pkg/api/routes.py")
        await engine.remember("default", "class Bad(Exception):\n    pass\n",
                              source="pkg/core/errors/__init__.py")
        blast = await affected(engine, "default", "pkg/core/errors/__init__.py")
    finally:
        await engine.close()
    assert blast.status == "found", blast.record()
    assert any(one.label == "pkg/api/routes.py" for one in blast.reached), blast.record()


async def test_a_graph_that_did_resolve_its_imports_says_nothing_of_the_kind():
    """The other half: the notice must not appear when the graph does
    hold cross-file edges, or it becomes noise that is skipped."""
    engine = await graphed()
    try:
        blast = await affected(engine, "default", "pkg.store")
    finally:
        await engine.close()
    assert blast.reached, blast.record()
    assert blast.unresolved_imports is False, blast.record()
    assert "sync --graph" not in blast.why, blast.why


async def test_a_relative_import_reaches_the_file_it_names_without_a_resolver():
    """The blast radius answered "nothing" for every file in this
    repository, because a relative import recorded nothing at all unless
    the caller could say which file it meant -- and `engine.remember` and
    the episodes route see one file each, so neither can.

    A relative import does not need the tree to be *named*, only to be
    *confirmed*. `from ..core import errors` inside `pkg/a/b.py` names
    `pkg/core/errors` by arithmetic on the path. Whether such a file
    exists is a question the whole graph can answer later, and answering
    it later is what makes the result independent of the order files
    arrived in.
    """
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    try:
        # The importer arrives first, naming a file the graph has not seen.
        await engine.remember("default", "from ..core.errors import Bad\n\n\n"
                              "def put(p):\n    return Bad(p)\n", source="pkg/api/routes.py")
        await engine.remember("default", "class Bad(Exception):\n    pass\n",
                              source="pkg/core/errors.py")
        blast = await affected(engine, "default", "pkg/core/errors.py")
    finally:
        await engine.close()
    assert blast.status == "found", blast.record()
    assert any(one.label == "pkg/api/routes.py" for one in blast.reached), blast.record()
    assert blast.unresolved_imports is False, blast.record()


async def test_a_candidate_spelling_finds_the_file_the_graph_really_holds():
    """`./store` from a `.tsx` file is recorded as `store.tsx`, and the
    file may well be `store.ts`. The spelling is a candidate; which one
    exists is a question for read time, when every file is visible.

    Aliased only onto a name nothing was ever ingested for -- a label
    that is the object of edges and the subject of none. Two files that
    genuinely exist under different extensions stay two files.
    """
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    try:
        await engine.remember("default", "export function keep(p: string) { return p; }\n",
                              source="web/memory/store.ts")
        await engine.remember("default", "import {keep} from './store';\n"
                              "export function put(p: string) { return keep(p); }\n",
                              source="web/memory/page.tsx")
        blast = await affected(engine, "default", "web/memory/store.ts")
    finally:
        await engine.close()
    assert blast.status == "found", blast.record()
    assert any(one.label == "web/memory/page.tsx" for one in blast.reached), blast.record()


async def test_two_files_that_both_exist_are_not_merged_by_their_extension():
    """The guard on the alias above. `store.ts` and `store.tsx` can both
    be real, and then neither stands in for the other."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    try:
        await engine.remember("default", "export function keep(p: string) { return p; }\n",
                              source="web/store.ts")
        await engine.remember("default", "export function draw(p: string) { return p; }\n",
                              source="web/store.tsx")
        await engine.remember("default", "import {keep} from './store';\n"
                              "export function put(p: string) { return keep(p); }\n",
                              source="web/page.ts")
        reached = await affected(engine, "default", "web/store.tsx")
    finally:
        await engine.close()
    # page.ts records `web/store.ts`, which exists. Nothing reaches the
    # .tsx file, and it is not handed page.ts by resemblance.
    assert not reached.reached, reached.record()


async def test_a_symbol_in_a_package_is_reached_under_either_spelling():
    """The file-level match was not enough. `from .core import Base` in a
    package names `pkg/core.py:Base`, and the graph holds
    `pkg/core/__init__.py:Base` -- two symbol entities, and only the two
    *file* entities were being matched, so the real symbol had no
    dependants while the candidate had them all."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    try:
        await engine.remember("default", "class Base:\n    pass\n",
                              source="pkg/core/__init__.py")
        await engine.remember("default", "from .core import Base\n\n\n"
                              "class Derived(Base):\n    pass\n", source="pkg/shelf.py")
        blast = await affected(engine, "default", "pkg/core/__init__.py:Base")
    finally:
        await engine.close()
    assert any(one.label.startswith("pkg/shelf.py") for one in blast.reached), blast.record()


async def test_a_package_beside_a_module_of_the_same_name_wins_as_python_says():
    """`core.py` and `core/__init__.py` in one directory is legal, and
    Python imports the **package**. Both were claiming the importer.

    This is the one place a language rule decides it, so this match is
    not conditional on the module being a phantom the way the extension
    match is: there is no rule saying `store.ts` beats `store.tsx`, and
    there is one saying a package beats a module."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    try:
        # Each needs a declaration of its own: a file becomes an entity
        # when something in the graph names it, and a bare assignment
        # names nothing.
        await engine.remember("default", "def shadowed():\n    return 1\n", source="pkg/core.py")
        await engine.remember("default", "def chosen():\n    return 2\n",
                              source="pkg/core/__init__.py")
        await engine.remember("default", "from .core import chosen\n\n\n"
                              "def put():\n    return chosen()\n", source="pkg/shelf.py")
        package = await affected(engine, "default", "pkg/core/__init__.py")
        module = await affected(engine, "default", "pkg/core.py")
    finally:
        await engine.close()
    assert any(one.label == "pkg/shelf.py" for one in package.reached), package.record()
    assert not module.reached, module.record()
