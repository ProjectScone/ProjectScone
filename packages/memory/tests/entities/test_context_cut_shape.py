"""What the relation cut left out, by where it points.

An entity with three hundred relations shows the strongest ``max_relations``
and says ``relations_cut 236``. A reader asking who calls a hub then has a
number and nowhere to look. The reference's ``explain`` groups what it cut
by direction and file. Here the cut is grouped by the entity it was cut
from, the direction, the predicate and, for code, the file the far end is
declared in; each group is one line with its count, the largest first, and
the groups past ``MAX_CUT_GROUPS`` are counted in coverage, never dropped
in silence. Nothing is read for them: the grouping uses the projection the
walk already holds, so the cut costs no facts from the re-read budget.
"""

from __future__ import annotations

from collections import Counter

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities.context import ContextLimits, graph_context
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"

pytestmark = pytest.mark.asyncio


async def engine_with(facts) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    for subject, predicate, object_ in facts:
        await engine.assert_fact("alpha", subject, predicate, object_, valid_from=DAY)
    return engine


CALLS = ([(f"pkg/router.py:route_{n}", "calls", "pkg/recall.py:recall") for n in range(4)]
         + [(f"pkg/cli.py:main_{n}", "calls", "pkg/recall.py:recall") for n in range(2)]
         + [("pkg/recall.py:recall", "calls", "pkg/lexical.py:tokens"),
            ("pkg/recall.py", "defines", "pkg/recall.py:recall")])


def prefixed(text: str, prefix: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(prefix)]


def cut_count(packet) -> int:
    [reason] = [reason for reason in packet.coverage["reasons"] if reason.startswith("relations_cut ")]
    return int(reason.split()[1])


async def test_the_callers_a_hub_left_out_are_counted_by_the_file_they_are_in():
    engine = await engine_with(CALLS)
    try:
        packet = await graph_context(engine, "alpha", names=["pkg/recall.py:recall"],
                                     limits=ContextLimits(max_hops=1, max_relations=2))
    finally:
        await engine.close()
    shown = prefixed(packet.text, "hop 1: ")
    assert len(shown) == 2 and cut_count(packet) == 6
    # Every relation is either shown or counted in exactly one group.
    left: Counter[str] = Counter()
    for subject, predicate, object_ in CALLS:
        if any(line.startswith(f"hop 1: {subject} {predicate} {object_} [") for line in shown):
            continue
        if object_ == "pkg/recall.py:recall":
            left[f"cut: {{count}} in {subject.partition(':')[0]} -{predicate}-> pkg/recall.py:recall"] += 1
        else:
            left[f"cut: pkg/recall.py:recall -{predicate}-> {{count}} in {object_.partition(':')[0]}"] += 1
    expected = [shape.format(count=count) for shape, count in sorted(left.items(), key=lambda item: (-item[1], item[0]))]
    assert prefixed(packet.text, "cut: ") == expected
    groups = packet.record("alpha", "current", DAY)["coverage"]["relations_cut_by"]
    assert sum(group["count"] for group in groups) == 6
    assert {group["file"] for group in groups} == {shape.split(" in ")[1].split(" ")[0] for shape in left}


async def test_the_largest_group_comes_first():
    engine = await engine_with([(f"pkg/router.py:route_{n}", "calls", "pkg/recall.py:recall") for n in range(5)]
                               + [("pkg/cli.py:main", "calls", "pkg/recall.py:recall"),
                                  ("pkg/recall.py", "defines", "pkg/recall.py:recall")])
    try:
        packet = await graph_context(engine, "alpha", names=["pkg/recall.py:recall"],
                                     limits=ContextLimits(max_hops=1, max_relations=1))
    finally:
        await engine.close()
    groups = packet.record("alpha", "current", DAY)["coverage"]["relations_cut_by"]
    assert [group["count"] for group in groups] == sorted((group["count"] for group in groups), reverse=True)
    assert groups[0]["file"] == "pkg/router.py" and groups[0]["direction"] == "in" and groups[0]["count"] >= 4


async def test_a_group_outside_code_names_no_file():
    """A person is not declared in a file, even when a name holds a colon and a dot."""
    engine = await engine_with([(name, "knows", "alice chen") for name in ("bob stone", "carol ruiz", "dave kim")]
                               + [("notes.md: chapter 1", "mentions", "alice chen")])
    try:
        packet = await graph_context(engine, "alpha", names=["alice chen"],
                                     limits=ContextLimits(max_hops=1, max_relations=1))
    finally:
        await engine.close()
    groups = packet.record("alpha", "current", DAY)["coverage"]["relations_cut_by"]
    assert groups and all(group["file"] is None for group in groups)
    assert sum(group["count"] for group in groups) == cut_count(packet) == 3
    assert all(group["direction"] == "in" for group in groups)
    assert sorted(prefixed(packet.text, "cut: ")) == sorted(
        f"cut: {group['count']} {'entity' if group['count'] == 1 else 'entities'} -{group['predicate']}-> alice chen"
        for group in groups)


async def test_a_code_relation_places_its_far_end_only_by_a_path_or_a_qualified_declaration():
    """``Node.js`` at the end of a code relation is a name shaped like a file,
    not a file; ``setup.py:main`` is declared in ``setup.py``."""
    engine = await engine_with([("pkg/app.py:main", "develops_with", "Node.js"),
                                ("pkg/app.py:main", "calls", "setup.py:main"),
                                ("pkg/app.py:main", "calls", "pkg/util/io.py"),
                                ("pkg/app.py", "defines", "pkg/app.py:main")])
    try:
        packet = await graph_context(engine, "alpha", names=["pkg/app.py:main"],
                                     limits=ContextLimits(max_hops=1, max_relations=1))
    finally:
        await engine.close()
    shown = "\n".join(prefixed(packet.text, "hop 1: "))
    groups = packet.record("alpha", "current", DAY)["coverage"]["relations_cut_by"]
    placed = {(group["predicate"], group["file"]) for group in groups}
    by_far_end = {"Node.js": ("develops_with", None), "setup.py:main": ("calls", "setup.py"),
                  "pkg/util/io.py": ("calls", "pkg/util/io.py"), "pkg/app.py ": ("defines", "pkg/app.py")}
    walked = {pair for far, pair in by_far_end.items() if far in shown}
    assert len(walked) == 1 and placed == set(by_far_end.values()) - walked


async def test_nothing_cut_means_no_cut_lines():
    engine = await engine_with(CALLS)
    try:
        packet = await graph_context(engine, "alpha", names=["pkg/recall.py:recall"], limits=ContextLimits(max_hops=1))
    finally:
        await engine.close()
    assert not prefixed(packet.text, "cut: ")
    assert packet.record("alpha", "current", DAY)["coverage"]["relations_cut_by"] == []


@pytest.mark.parametrize("bound, written, beyond", [(2, 2, 1), (3, 3, 0)], ids=["one-past", "exactly"])
async def test_groups_past_the_bound_are_counted_not_dropped(monkeypatch, bound, written, beyond):
    """This cut falls into three groups: calls from the router, from the command line, and out to lexical."""
    import scone_memory.entities.context as module

    monkeypatch.setattr(module, "MAX_CUT_GROUPS", bound)
    engine = await engine_with(CALLS)
    try:
        packet = await graph_context(engine, "alpha", names=["pkg/recall.py:recall"],
                                     limits=ContextLimits(max_hops=1, max_relations=2))
    finally:
        await engine.close()
    assert len(prefixed(packet.text, "cut: ")) == written
    assert len(packet.record("alpha", "current", DAY)["coverage"]["relations_cut_by"]) == written
    assert [reason for reason in packet.coverage["reasons"] if reason.startswith("cut_groups_cut")] == (
        [f"cut_groups_cut {beyond}"] if beyond else [])


async def test_on_a_hub_the_shape_of_the_cut_survives_the_byte_budget():
    """The strongest relations alone fill the budget on a hub, so the cut is
    written before them: the budget takes the weakest relations, not the only
    line saying where the rest are."""
    engine = await engine_with([(f"person {number:02d}", "knows", "alice chen") for number in range(80)])
    try:
        packet = await graph_context(engine, "alpha", names=["alice chen"],
                                     limits=ContextLimits(max_hops=1, max_bytes=1_500))
    finally:
        await engine.close()
    assert cut_count(packet) == 16 and "omitted:" in packet.text
    assert prefixed(packet.text, "cut: ") == ["cut: 16 entities -knows-> alice chen"]
