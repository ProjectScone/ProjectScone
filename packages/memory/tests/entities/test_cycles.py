"""Files that cannot load without each other are found; loops the code works around are shown apart."""
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities import cycles as cycles_module
from scone_memory.entities.cycles import CyclesError, graph_cycles, loop_through, shortest_loop, strongly_connected
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


async def engine_with(*triples) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    for subject, predicate, obj in triples:
        await engine.assert_fact("alpha", subject, predicate, obj, valid_from=DAY, origin="extracted")
    return engine


def test_components_and_shortest_loops_over_a_small_graph():
    edges = {"a": {"b": [1]}, "b": {"c": [2]}, "c": {"a": [3], "d": [4]}, "d": {"e": [5]}, "e": {"d": [6]}, "f": {"f": [7]}}
    assert strongly_connected(edges) == [["a", "b", "c"], ["d", "e"]], "a lone self-loop is not a component of two"
    walk, hops = shortest_loop(edges, ["a", "b", "c"])
    assert walk == ("a", "b", "c", "a") and hops == ((1,), (2,), (3,))
    assert shortest_loop(edges, ["d", "e"]) == (("d", "e", "d"), ((5,), (6,)))
    assert shortest_loop({"x": {"y": [1]}}, ["x", "y"]) == ((), ()), "no way back within the component is no loop"
    assert shortest_loop(edges, ["a", "b", "c"], max_length=2) == ((), ()), "a loop longer than the bound is not walked"
    merged = {"a": {"b": [1]}, "b": {"a": [2], "c": [3]}, "c": {"b": [4]}}
    assert loop_through(merged, ["a", "b", "c"], {"c": {"b": [4]}}) == (("c", "b", "c"), ((4,), (3,))), \
        "the example of a group held apart crosses the deferred edge, not the load-time loop beside it"
    assert loop_through(merged, ["a", "b", "c"], {"c": {"b": [4]}}, max_length=1) == ((), ())


async def test_a_load_time_loop_is_a_cycle_and_a_function_level_import_holds_one_apart():
    engine = await engine_with(
        ("app/a.py", "imports", "app/b.py"), ("app/b.py", "imports", "app/c.py"), ("app/c.py", "imports", "app/a.py"),
        ("lib/x.py", "imports", "lib/y.py"), ("lib/y.py", "imports_when_called", "lib/x.py"),
        ("tools/t.py", "imports", "json"), ("app/a.py", "imports", "app/a.py"),
    )
    try:
        found = await graph_cycles(engine, "alpha")
        assert found.status == "cycles" and found.totals["cycles"] == 1 and found.totals["held_apart"] == 1
        [cycle] = found.cycles
        assert cycle["size"] == 3 and cycle["example"][0] == cycle["example"][-1] and len(cycle["hops"]) == 3
        assert all(hop for hop in cycle["hops"]), "every hop cites the facts that made it"
        [apart] = found.held_apart
        assert sorted(apart["members"]) == ["lib/x.py", "lib/y.py"] and apart["deferred"], "the deferred import that closes the loop is named"
        assert "cannot load without each other" in found.text and "joined only through deferred imports" in found.text
        assert "json" not in "".join(str(c["members"]) for c in found.cycles), "a module that imports nothing closes no loop"
        assert found.record("alpha", status="current", as_of="2025-06-01T00:00:00Z")["totals"]["load_edges"] == 5, "a self-import is not an edge"
    finally:
        await engine.close()


async def test_a_type_only_loop_is_held_apart_and_a_clean_graph_says_none():
    engine = await engine_with(("m/p.py", "imports_for_types", "m/q.py"), ("m/q.py", "imports", "m/p.py"),
                               ("m/q.py", "depends_on", "requests"))
    try:
        found = await graph_cycles(engine, "alpha")
        assert found.status == "held_apart" and found.totals["cycles"] == 0 and found.totals["held_apart"] == 1
        with pytest.raises(CyclesError, match="limit must be from 1 to 100"):
            await graph_cycles(engine, "alpha", limit=0)
        with pytest.raises(CyclesError, match="max_bytes"):
            await graph_cycles(engine, "alpha", max_bytes=10)
    finally:
        await engine.close()
    clean = await engine_with(("a.py", "imports", "b.py"), ("b.py", "imports", "c.py"))
    try:
        none = await graph_cycles(clean, "alpha")
        assert none.status == "none" and "result: no cycle" in none.text and none.cycles == () and none.held_apart == ()
    finally:
        await clean.close()


async def test_a_held_apart_group_around_a_cycle_shows_the_deferred_hop_and_the_bounds_are_said(monkeypatch):
    engine = await engine_with(("a.py", "imports", "b.py"), ("b.py", "imports", "a.py"), ("b.py", "imports", "c.py"),
                               ("c.py", "imports_when_called", "b.py"))
    try:
        found = await graph_cycles(engine, "alpha")
        [cycle], [apart] = found.cycles, found.held_apart
        assert sorted(cycle["members"]) == ["a.py", "b.py"] and sorted(apart["members"]) == ["a.py", "b.py", "c.py"]
        assert apart["example"] == ["c.py", "b.py", "c.py"] and apart["hops"][0] == apart["deferred"], \
            "the loop shown for a group held apart crosses the deferred import, not the load-time cycle inside the group"
        monkeypatch.setattr(cycles_module, "MAX_LENGTH", 1)
        bounded = await graph_cycles(engine, "alpha")
        assert bounded.status == "cycles" and bounded.cycles[0]["example"] == [] and bounded.held_apart[0]["example"] == []
        assert "loops_over_bound 2 longer than 1 hops" in bounded.coverage["reasons"] and "no loop within 1 hops" in bounded.text
        monkeypatch.setattr(cycles_module, "MAX_LENGTH", 32)
        monkeypatch.setattr(cycles_module, "MAX_ENTITIES", 2)
        declined = await graph_cycles(engine, "alpha")
        assert declined.status == "none" and declined.totals["cycles"] == 0 and "not searched" in declined.text
        assert any(reason.startswith("entities_over_bound 3 > 2") for reason in declined.coverage["reasons"])
    finally:
        await engine.close()


async def test_a_module_imported_at_load_anywhere_counts_as_load():
    engine = await engine_with(("a.py", "imports", "b.py"), ("a.py", "imports_when_called", "b.py"),
                               ("b.py", "imports_for_types", "a.py"), ("b.py", "imports", "a.py"))
    try:
        found = await graph_cycles(engine, "alpha")
        assert found.status == "cycles" and found.totals == {**found.totals, "cycles": 1, "held_apart": 0}, \
            "a file that also imports the module inside a function still needs it to load"
    finally:
        await engine.close()


async def test_the_shown_bound_is_said_and_the_rest_are_counted():
    triples = []
    for n in range(4):
        triples += [(f"g{n}/a.py", "imports", f"g{n}/b.py"), (f"g{n}/b.py", "imports", f"g{n}/a.py")]
    engine = await engine_with(*triples)
    try:
        found = await graph_cycles(engine, "alpha", limit=2)
        assert found.totals["cycles"] == 4 and len(found.cycles) == 2 and "cycles_shown 2 of 4" in found.text
    finally:
        await engine.close()
