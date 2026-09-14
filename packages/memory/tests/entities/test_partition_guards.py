"""Communities a reader can use: no community swallows the graph, none is a loose bag.

Modularity partitions have two known failures. One community can take a
large share of a graph, which a map by community then draws as one blob;
and over a large graph modularity merges small modules into one large
community that holds several. The reference splits a community over a
quarter of the graph, re-splits a large one whose member pairs are rarely
linked, and can leave hubs out while communities are found. Linked pairs
fall with size in a sparse graph, so on this project's own code graph that
test fired on 12 of the 17 communities of 50 or more; here a large
community is split when its own partition is strong. Each guard
re-partitions only the community it looks at, on its own links, and the
coverage says how often it split and how often it kept one whole.
"""

from __future__ import annotations

import pytest

from scone_memory.core.models import Fact
from scone_memory.entities.analysis import _guarded, analyze_projection
from scone_memory.entities.project import project_entities


def clique(prefix: str, size: int) -> list[tuple[str, str]]:
    names = [f"{prefix}{n}" for n in range(size)]
    return [(a, b) for index, a in enumerate(names) for b in names[index + 1:]]


def adjacency(edges) -> dict[str, dict[str, int]]:
    graph: dict[str, dict[str, int]] = {}
    for left, right in edges:
        graph.setdefault(left, {})[right] = 1
        graph.setdefault(right, {})[left] = 1
    return graph


def facts(edges) -> list[Fact]:
    return [Fact(fact_id=n + 1, space="alpha", subject=left, predicate="knows", object=right,
                 valid_from="2025-01-01T00:00:00Z") for n, (left, right) in enumerate(edges)]


def test_a_community_over_a_quarter_of_the_graph_is_split_on_its_own_links():
    edges = clique("a", 12) + clique("b", 12) + [("a0", "b0")] + clique("c", 4) + clique("d", 4)
    graph = adjacency(edges)
    lumped = sorted(f"{p}{n}" for p in "ab" for n in range(12))
    rest = [sorted(f"c{n}" for n in range(4)), sorted(f"d{n}" for n in range(4))]
    parts, fired = _guarded(graph, [lumped, *rest], 1.0)
    assert sorted(map(sorted, parts)) == sorted([sorted(f"a{n}" for n in range(12)), sorted(f"b{n}" for n in range(12)),
                                                 *rest])
    assert fired == {"split_oversized": 1, "split_nested": 0, "unsplittable": 0}


def test_a_small_graph_is_never_split_for_being_a_large_share_of_itself():
    graph = adjacency(clique("a", 6) + clique("b", 3) + [("a0", "b0")])
    parts = [sorted(f"a{n}" for n in range(6)), sorted(f"b{n}" for n in range(3))]
    assert _guarded(graph, parts, 1.0) == (parts, {"split_oversized": 0, "split_nested": 0, "unsplittable": 0})


def test_an_oversized_community_that_is_one_tight_clique_stays_whole_and_is_counted():
    graph = adjacency(clique("a", 14) + clique("b", 4) + [("a0", "b0")])
    parts = [sorted(f"a{n}" for n in range(14)), sorted(f"b{n}" for n in range(4))]
    kept, fired = _guarded(graph, parts, 1.0)
    assert kept == parts and fired["unsplittable"] == 1 and fired["split_oversized"] == 0


def test_a_large_community_holding_communities_of_its_own_is_split_even_under_the_share():
    # Six groups of ten, each a ring with one chord, joined in a chain: 60 members, cohesion under 5%.
    edges = []
    for g in range(6):
        ring = [f"g{g}n{n}" for n in range(10)]
        edges += [(ring[n], ring[(n + 1) % 10]) for n in range(10)] + [(ring[0], ring[5])]
        if g:
            edges.append((f"g{g - 1}n3", f"g{g}n7"))
    others = [(f"x{g}n{n}", f"x{g}n{m}") for g in range(60) for n in range(4) for m in range(n + 1, 4)]
    graph = adjacency(edges + others)
    loose = sorted({node for edge in edges for node in edge})
    fringe = [sorted({f"x{g}n{n}" for n in range(4)}) for g in range(60)]
    parts, fired = _guarded(graph, [loose, *fringe], 1.0)
    assert len(loose) * 4 < len(graph), "under a quarter of the graph, so only the nested guard can fire"
    assert fired["split_nested"] == 1 and fired["split_oversized"] == 0
    assert len([part for part in parts if set(part) <= set(loose)]) > 1


def test_the_analysis_reports_how_often_each_guard_fired():
    # A star is one community of every entity, and has no finer structure to split into.
    edges = [(f"person{n:02d}", "hub") for n in range(30)]
    analysis = analyze_projection(project_entities("alpha", facts(edges), revision=1))
    record = analysis.coverage.record()
    assert record["unsplittable"] == 1 and record["split_oversized"] == 0 and len(analysis.communities) == 1
    assert record["modularity_before_guards"] is None, "nothing was split, so there is no other number to give"


def test_after_a_split_the_modularity_before_the_guards_is_given_beside_the_final_one(monkeypatch):
    """The first partition is made to lump two cliques together, as modularity
    does over a large graph; the guard's own partition of that community runs."""
    import scone_memory.entities.analysis as module

    edges = clique("a", 12) + clique("b", 12) + [("a0", "b0")] + clique("c", 4) + clique("d", 4)
    projection = project_entities("alpha", facts(edges), revision=1)
    ids = {entity.label: entity.entity_id for entity in projection.entities}
    real = module._partition
    calls = []

    def lumping(graph, resolution):
        calls.append(len(graph))
        if len(calls) == 1:
            group = lambda prefixes, size: sorted(ids[f"{p}{n}"] for p in prefixes for n in range(size))
            return [group("ab", 12), group("c", 4), group("d", 4)], 1
        return real(graph, resolution)

    monkeypatch.setattr(module, "_partition", lumping)
    analysis = analyze_projection(projection)
    record = analysis.coverage.record()
    assert record["split_oversized"] == 1 and len(analysis.communities) == 4
    assert record["modularity_before_guards"] is not None
    assert record["modularity_before_guards"] < analysis.modularity, "here the split was the better partition"


def test_with_hubs_detached_both_modularities_are_measured_on_the_whole_graph(monkeypatch):
    """Communities are found without the hubs, but the final modularity is
    taken over the whole graph with each hub rejoined. The number before the
    guards must be taken the same way, or the two are not comparable: here the
    hub-less graph scores the lumped partition higher than the whole graph does."""
    import scone_memory.entities.analysis as module

    edges = (clique("a", 12) + clique("b", 12) + [("a0", "b0")] + clique("c", 4) + clique("d", 4)
             + [("hub", f"a{n}") for n in range(12)] + [("hub", f"b{n}") for n in range(4)])
    projection = project_entities("alpha", facts(edges), revision=1)
    ids = {entity.label: entity.entity_id for entity in projection.entities}
    real = module._partition
    calls = []

    def lumping(graph, resolution):
        calls.append(len(graph))
        if len(calls) == 1:
            group = lambda prefixes, size: sorted(ids[f"{p}{n}"] for p in prefixes for n in range(size))
            return [group("ab", 12), group("c", 4), group("d", 4)], 1
        return real(graph, resolution)

    monkeypatch.setattr(module, "_partition", lumping)
    analysis = analyze_projection(projection, detach_hubs=95)
    assert analysis.coverage.hubs_detached == 1 and analysis.coverage.split_oversized == 1
    whole = adjacency([(ids[left], ids[right]) for left, right in edges])
    lumped = {ids[f"{p}{n}"]: "ab" for p in "ab" for n in range(12)} | {ids["hub"]: "ab"}
    lumped |= {ids[f"{p}{n}"]: p for p in "cd" for n in range(4)}
    assert analysis.coverage.modularity_before_guards == pytest.approx(module._modularity(whole, lumped))
    final = {member: community.community_id for community in analysis.communities for member in community.members}
    assert analysis.modularity == pytest.approx(module._modularity(whole, final)), "the fixture's weights match the analysis"


def test_a_hub_rejoining_a_community_the_guards_left_alone_is_a_member_once(monkeypatch):
    """The partition before the guards and the one after share the communities
    no guard touched; the hub is added to each partition, never twice to one."""
    import scone_memory.entities.analysis as module

    edges = (clique("a", 12) + clique("b", 12) + [("a0", "b0")] + clique("c", 8) + clique("d", 4)
             + [("hub", f"c{n}") for n in range(8)] + [("hub", f"a{n}") for n in range(4)]
             + [("hub", f"b{n}") for n in range(3)])
    projection = project_entities("alpha", facts(edges), revision=1)
    ids = {entity.label: entity.entity_id for entity in projection.entities}
    labels = {entity_id: label for label, entity_id in ids.items()}
    real = module._partition
    calls = []

    def lumping(graph, resolution):
        calls.append(len(graph))
        if len(calls) == 1:
            group = lambda prefixes, size: sorted(ids[f"{p}{n}"] for p in prefixes for n in range(size))
            return [group("ab", 12), group("c", 8), group("d", 4)], 1
        return real(graph, resolution)

    monkeypatch.setattr(module, "_partition", lumping)
    analysis = analyze_projection(projection, detach_hubs=95)
    assert analysis.coverage.hubs_detached == 1 and analysis.coverage.split_oversized == 1
    members = [labels[member] for community in analysis.communities for member in community.members]
    assert sorted(members) == sorted(ids), "every entity is in exactly one community, once"
    [home] = [community for community in analysis.communities if ids["hub"] in community.members]
    assert sorted(labels[member] for member in home.members) == sorted([*(f"c{n}" for n in range(8)), "hub"])


def test_a_community_under_fifty_is_not_looked_at_for_nesting():
    path = [(f"p{n}", f"p{n + 1}") for n in range(44)]
    others = [(f"x{g}n{n}", f"x{g}n{m}") for g in range(50) for n in range(4) for m in range(n + 1, 4)]
    graph = adjacency(path + others)
    loose = sorted({node for edge in path for node in edge})
    fringe = [sorted({f"x{g}n{n}" for n in range(4)}) for g in range(50)]
    assert len(loose) < 50 and len(loose) * 4 < len(graph)
    parts, fired = _guarded(graph, [loose, *fringe], 1.0)
    assert loose in parts and fired == {"split_oversized": 0, "split_nested": 0, "unsplittable": 0}


def test_detached_hubs_are_left_out_while_communities_are_found_then_join_where_most_links_go():
    # A hub linked to all nine of the smaller clique and four of the larger: it joins the smaller.
    edges = clique("a", 9) + clique("b", 12) + [("hub", f"a{n}") for n in range(9)] + [("hub", f"b{n}") for n in range(4)]
    projection = project_entities("alpha", facts(edges), revision=1)
    labels = {entity.entity_id: entity.label for entity in projection.entities}
    detached = analyze_projection(projection, detach_hubs=95)
    of = {labels[member]: community.community_id for community in detached.communities for member in community.members}
    assert of["hub"] == of["a0"] and of["a0"] != of["b0"]
    assert detached.coverage.hubs_detached == 1 and detached.coverage.record()["detach_hubs"] == 95
    plain = analyze_projection(projection)
    assert plain.coverage.hubs_detached == 0 and plain.coverage.record()["detach_hubs"] is None


@pytest.mark.parametrize("percentile", [49, 100.5, True])
def test_a_hub_percentile_outside_fifty_to_a_hundred_is_refused(percentile):
    projection = project_entities("alpha", facts(clique("a", 4)), revision=1)
    with pytest.raises(ValueError):
        analyze_projection(projection, detach_hubs=percentile)


async def test_the_report_detaches_hubs_over_http_and_the_command_line():
    import io
    import json

    from fastapi.testclient import TestClient

    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.api import create_app
    from scone_memory.runtime.cli import build_parser, run

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    edges = clique("a", 9) + clique("b", 12) + [("hub", f"a{n}") for n in range(9)] + [("hub", f"b{n}") for n in range(4)]
    for left, right in edges:
        await engine.assert_fact("default", left, "knows", right, valid_from="2025-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key-a": "default"})) as client:
        headers = {"authorization": "Bearer key-a"}
        made = client.get("/v1/graph/report", params={"detach_hubs": 95}, headers=headers)
        refused = client.get("/v1/graph/report", params={"detach_hubs": 40}, headers=headers)
    out = io.StringIO()
    code = await run(build_parser().parse_args(["graph", "report", "--detach-hubs", "95"]), engine, io.StringIO(""), out)
    from scone_memory.core.errors import InvalidInput

    with pytest.raises(InvalidInput, match="detach-hubs"):
        await run(build_parser().parse_args(["graph", "report", "--detach-hubs", "40"]), engine, io.StringIO(""),
                  io.StringIO())
    await engine.close()
    assert made.status_code == 200, made.text
    assert made.json()["analysis"]["coverage"]["hubs_detached"] == 1
    assert made.json()["analysis"]["coverage"]["detach_hubs"] == 95 and refused.status_code == 422
    assert code == 0 and json.loads(out.getvalue())["analysis"]["coverage"]["hubs_detached"] == 1


def test_a_large_community_whose_own_split_is_weak_is_kept_whole_and_counted():
    import random

    draw = random.Random(3)
    nodes = [f"r{n}" for n in range(60)]
    dense = [(a, b) for index, a in enumerate(nodes) for b in nodes[index + 1:] if draw.random() < 0.3]
    others = [(f"x{g}n{n}", f"x{g}n{m}") for g in range(60) for n in range(4) for m in range(n + 1, 4)]
    graph = adjacency(dense + others)
    fringe = [sorted({f"x{g}n{n}" for n in range(4)}) for g in range(60)]
    assert len(nodes) * 4 < len(graph)
    parts, fired = _guarded(graph, [sorted(nodes), *fringe], 1.0)
    assert sorted(nodes) in parts, "several pieces at modularity 0.14 are not structure"
    assert fired == {"split_oversized": 0, "split_nested": 0, "unsplittable": 1}


def test_an_oversized_community_is_split_even_when_its_own_split_is_weak():
    """Size alone is the reason for this guard: a community drawn as one blob
    is split into what its links give, strong or not."""
    import random

    draw = random.Random(3)
    nodes = [f"r{n}" for n in range(60)]
    dense = [(a, b) for index, a in enumerate(nodes) for b in nodes[index + 1:] if draw.random() < 0.3]
    graph = adjacency(dense + clique("c", 4))
    parts, fired = _guarded(graph, [sorted(nodes), sorted(f"c{n}" for n in range(4))], 1.0)
    assert sorted(nodes) not in parts and fired == {"split_oversized": 1, "split_nested": 0, "unsplittable": 0}
