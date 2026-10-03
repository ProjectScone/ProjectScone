"""What runs when this runs, and what reaches it.

``graph affected`` answers "what breaks if I change this" -- a set, walked
backwards over eight kinds of dependency. A call flow answers a different
question that a code graph should also answer: follow the ``calls`` edges
only, from one declaration, and show the chain in both directions -- what
it reaches when it runs, and who reaches it.

The rules, each with a test:

- **One predicate.** Only ``calls``. An import is not a call, and mixing
  them in would draw a flow that never happens at run time.
- **Direction is kept, both ways.** Downstream is what the root calls;
  upstream is who calls the root; neither is allowed to leak into the
  other.
- **A cycle is drawn once and ends.** Recursion is ordinary code.
- **Every bound says it bit**, and an empty flow says "nothing here",
  never "nothing".
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.entities.callflow import MAX_HOPS, MAX_REACHED, call_flow, mermaid

pytestmark = pytest.mark.asyncio

# handle -> put -> keep -> dumps; and a recursive walk; and an untouched one.
SOURCES = {
    "pkg/store.py": (
        "import json\n\n\n"
        "class Shelf:\n"
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
    "pkg/walk.py": (
        "def walk(depth: int) -> int:\n"
        "    return walk(depth - 1) if depth else 0\n"),
    # Mutual recursion: a real cycle in the call graph, which a walk must
    # draw once and leave, rather than follow round for ever.
    "pkg/loop.py": (
        "def ping(n: int) -> int:\n"
        "    return pong(n)\n\n\n"
        "def pong(n: int) -> int:\n"
        "    return ping(n - 1) if n else 0\n"),
    "pkg/unrelated.py": (
        "def area(radius: float) -> float:\n"
        "    return radius * radius\n"),
}


async def graphed():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    for name, text in SOURCES.items():
        await engine.remember("default", text, source=name)
    return engine


def labels(steps):
    return [step.label for step in steps]


async def test_what_a_function_reaches_when_it_runs():
    engine = await graphed()
    try:
        flow = await call_flow(engine, "default", "pkg/web.py:handle")
    finally:
        await engine.close()
    downstream = labels(flow.downstream)
    assert any("put" in one for one in downstream), flow.record()
    assert flow.status == "found"
    assert all(step.hop >= 1 for step in flow.downstream)


async def test_who_reaches_a_function_is_the_other_direction_and_they_do_not_mix():
    engine = await graphed()
    try:
        flow = await call_flow(engine, "default", "pkg/api.py:put")
    finally:
        await engine.close()
    up, down = labels(flow.upstream), labels(flow.downstream)
    assert any("handle" in one for one in up), flow.record()
    assert not any("handle" in one for one in down), "who calls it is never what it calls"
    assert not any("put" in one for one in up + down), "the root is not its own neighbour"


async def test_an_import_is_not_a_call():
    """`pkg/api.py` imports `pkg.store`; that edge must not appear in a
    flow, which is about what happens when code runs."""
    engine = await graphed()
    try:
        flow = await call_flow(engine, "default", "pkg/api.py:put")
    finally:
        await engine.close()
    for step in flow.downstream + flow.upstream:
        assert step.predicate == "calls", step.record()


async def test_a_cycle_is_drawn_once_and_ends():
    """`ping` calls `pong` calls `ping`. A walk that did not remember
    where it had been would go round until a bound stopped it, and would
    report the root as something the root reaches."""
    engine = await graphed()
    try:
        flow = await call_flow(engine, "default", "pkg/loop.py:ping", downstream_hops=8)
    finally:
        await engine.close()
    seen = labels(flow.downstream)
    assert seen, flow.record()
    assert len(seen) == len(set(seen)), seen
    assert "pkg/loop.py:ping" not in seen, "the root is not reported as something it reaches"
    assert any("pong" in one for one in seen)
    assert not flow.bounds, "it ended because the calls ran out, not because a bound bit"


async def test_a_hop_bound_that_bites_says_so_and_never_reads_as_the_end():
    engine = await graphed()
    try:
        near = await call_flow(engine, "default", "pkg/web.py:handle", downstream_hops=1)
        far = await call_flow(engine, "default", "pkg/web.py:handle", downstream_hops=4)
    finally:
        await engine.close()
    assert len(near.downstream) <= len(far.downstream)
    if len(far.downstream) > len(near.downstream):
        assert near.bounds["downstream_hops"] is True, near.record()
        assert "further" in near.why.lower() or "bound" in near.why.lower(), near.why


async def test_a_name_this_graph_never_heard_of_says_nothing_here():
    engine = await graphed()
    try:
        flow = await call_flow(engine, "default", "pkg/nowhere.py:absent")
    finally:
        await engine.close()
    assert flow.status == "unknown" and not flow.downstream and not flow.upstream
    assert "this graph" in flow.why or "here" in flow.why


async def test_an_isolated_function_is_empty_not_unknown():
    engine = await graphed()
    try:
        flow = await call_flow(engine, "default", "pkg/unrelated.py:area")
    finally:
        await engine.close()
    assert flow.status == "empty", flow.record()
    assert "nothing" in flow.why.lower()


@pytest.mark.parametrize("bad", [
    {"downstream_hops": 0}, {"downstream_hops": MAX_HOPS + 1}, {"upstream_hops": -1},
    {"max_reached": 0}, {"max_reached": MAX_REACHED + 1}, {"downstream_hops": True},
])
async def test_a_bound_outside_its_own_bounds_is_refused(bad):
    engine = await graphed()
    try:
        with pytest.raises(InvalidInput):
            await call_flow(engine, "default", "pkg/web.py:handle", **bad)
    finally:
        await engine.close()


# -- the drawing ---------------------------------------------------------------------


async def test_the_flow_draws_as_mermaid_with_the_root_marked_and_arrows_the_right_way():
    engine = await graphed()
    try:
        flow = await call_flow(engine, "default", "pkg/api.py:put")
    finally:
        await engine.close()
    drawn = mermaid(flow)
    assert drawn.startswith("flowchart"), drawn[:60]
    assert "-->" in drawn
    assert "classDef root" in drawn and ":::root" in drawn, "the root is marked, not just listed"
    lines = [line.strip() for line in drawn.splitlines() if "-->" in line]
    assert lines, drawn
    assert len(lines) == len(set(lines)), "an edge is drawn once"


def test_a_label_with_a_quote_or_a_bracket_cannot_break_the_drawing():
    from scone_memory.entities.callflow import Flow, Step

    flow = Flow(root='pkg/odd.py:say["hi"]', status="found",
                downstream=[Step('pkg/odd.py:other[)"', 1, "calls", 7, "x()", 'pkg/odd.py:say["hi"]')],
                upstream=[],
                bounds={}, why="")
    drawn = mermaid(flow)
    import re

    texts = re.findall(r'\["(.*?)"\]', drawn)
    assert texts, drawn
    for text in texts:
        assert '"' not in text and "[" not in text and "]" not in text, \
            f"a raw quote or bracket inside a node's text ends the node early: {text!r}"
    assert "#quot;" in drawn and "#91;" in drawn, "the characters are escaped, not dropped"
    assert "hi" in drawn and "other" in drawn, "and the name is still readable"


async def test_a_call_into_another_file_joins_the_imported_name_to_the_declaration():
    """The chain that could not cross a file.

    `pkg/web.py` calls `put`, and the graph records that as a call to
    `pkg.api.put` -- the name the caller imported -- while the function
    itself is `pkg/api.py:put`. Two labels for one thing, joined nowhere,
    so a walk reached the edge of its file and stopped. Joined, the flow
    runs web -> api -> store, which is what the code does.
    """
    engine = await graphed()
    try:
        flow = await call_flow(engine, "default", "pkg/web.py:handle", downstream_hops=4)
    finally:
        await engine.close()
    reached = labels(flow.downstream)
    assert any(one == "pkg/api.py:put" for one in reached), flow.record()
    assert any("Shelf" in one for one in reached), \
        "and past it: what put calls is two hops from handle, which is the point of joining"
    assert "joined" in flow.why, flow.why
    hops = {step.label: step.hop for step in flow.downstream}
    assert hops["pkg/api.py:put"] == 1, "the imported name is not an extra hop of its own"


def test_the_join_is_refused_where_two_declarations_answer_to_one_name():
    """The single-definition rule this graph holds everywhere else."""
    from scone_memory.entities.callflow import _joined, module_form

    assert module_form("pkg/api.py:put") == "pkg.api.put"
    assert module_form("pkg/api/__init__.py:put") == "pkg.api.put"
    assert module_form("pkg/api.py") == "", "a file is not a declaration"
    assert module_form("Shelf:keep") == "", "a name with no file behind it names no module"
    assert module_form("json.dumps") == "", "an imported name is already in module form"
    one = {"a": "pkg/api.py:put", "b": "pkg.api.put"}
    assert _joined(one) == {"b": "a"}, "one declaration answers, so the name joins to it"
    two = {"a": "pkg/api.py:put", "c": "pkg/api/__init__.py:put", "b": "pkg.api.put"}
    assert _joined(two) == {}, "two declarations answer to pkg.api.put, so neither is chosen"
