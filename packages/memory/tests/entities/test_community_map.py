"""A map of the communities, for a graph too large to draw entity by entity.

The drawings show the 200 most connected entities and leave the rest out.
The community map draws every community instead, up to its own bound: one
circle per community sized by its members, one line per pair of
communities that links join, weighted and titled by how many. A link is a
pair of entities one or more relations join, as the analysis counts
them. What it leaves out, and how much of the graph the analysis read, it
says.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree

import pytest

from scone_memory.core.models import Fact
from scone_memory.entities.analysis import cached_analysis
from scone_memory.entities.export import EXPORT_FORMATS, export_graph
from scone_memory.entities.project import project_entities

SVG = "{http://www.w3.org/2000/svg}"


def fact(number: int, subject: str, predicate: str, object_: str) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject.casefold(), predicate=predicate, object=object_,
                valid_from="2025-01-01T00:00:00Z")


def ledger():
    triples = [(f"worker {n}", "works_at", "Acme") for n in range(4)] + [
        (f"resident {n}", "lives_in", "Porto") for n in range(3)] + [
        ("worker 0", "lives_in", "Porto"), ("worker 1", "lives_in", "Porto"), ("Ann", "age", "34"),
        # A second relation between one pair: still one link.
        ("worker 0", "moved_to", "Porto")]
    return [fact(n + 1, *triple) for n, triple in enumerate(triples)]


def drawn(projection, **about):
    exported = export_graph(projection, "communities", about=about)
    assert exported.media_type == "image/svg+xml" and exported.filename == "graph-communities.svg"
    return ElementTree.fromstring(exported.body)


def circles(root):
    """Each community circle's group: its community id, and its title text."""
    return {group.get("data-group"): group.find(f"{SVG}circle").find(f"{SVG}title").text
            for group in root.iter(f"{SVG}g") if group.get("data-group") is not None}


def test_the_map_is_offered_as_a_format():
    assert "communities" in EXPORT_FORMATS


def test_each_community_is_a_circle_naming_its_members_count():
    projection = project_entities("alpha", ledger(), revision=1)
    root = drawn(projection)
    communities = cached_analysis(projection).communities
    titles = circles(root)
    for community in communities:
        assert titles[community.community_id].startswith(f"{community.label}: {len(community.members)} entities"), \
            titles[community.community_id]
    radius = {group.get("data-group"): float(group.find(f"{SVG}circle").get("r")) for group in root.iter(f"{SVG}g")
              if group.get("data-group") is not None}
    sizes = sorted(communities, key=lambda community: len(community.members))
    assert radius[sizes[0].community_id] < radius[sizes[-1].community_id], "a larger community is drawn larger"
    # Ann has only a value, no relation: counted as an entity in no community.
    assert titles["unlinked"] == "no relation: 1 entity"


def test_communities_joined_by_links_are_drawn_joined_by_how_many():
    projection = project_entities("alpha", ledger(), revision=1)
    root = drawn(projection)
    communities = {community.community_id: community for community in cached_analysis(projection).communities}
    of = {member: community_id for community_id, community in communities.items() for member in community.members}
    pairs = {frozenset((relation.subject_id, relation.object_id)) for relation in projection.relations
             if relation.subject_id != relation.object_id}
    crossing: dict[frozenset[str], int] = {}
    for pair in pairs:
        a, b = (of.get(end) for end in pair)
        if a and b and a != b:
            crossing[frozenset((a, b))] = crossing.get(frozenset((a, b)), 0) + 1
    assert crossing, "the fixture joins its communities"
    lines = {frozenset((line.get("data-from-group"), line.get("data-to-group"))): line
             for line in root.iter(f"{SVG}line")}
    assert set(lines) == set(crossing)
    for pair, count in crossing.items():
        a, b = sorted(pair, key=lambda community_id: communities[community_id].label)
        title = lines[pair].find(f"{SVG}title").text
        assert re.fullmatch(rf"{count} links? between .+ and .+", title), title
        assert communities[a].label in title and communities[b].label in title


def test_a_community_title_counts_links_inside_and_across():
    projection = project_entities("alpha", ledger(), revision=1)
    titles = circles(drawn(projection))
    communities = cached_analysis(projection).communities
    for community in communities:
        inside = f"{community.internal_links} link{'' if community.internal_links == 1 else 's'} inside"
        assert (f"{inside}, {community.boundary_links} to other communities"
                in titles[community.community_id]), titles[community.community_id]


def test_the_map_says_what_it_and_the_analysis_left_out(monkeypatch):
    import scone_memory.entities.export as module

    many = [fact(n * 10 + k, f"member {n}-{k}", "knows", f"Hub {n}") for n in range(6) for k in range(n + 2)]
    projection = project_entities("alpha", many, revision=1)
    monkeypatch.setattr(module, "_MAP_COMMUNITIES", 3)
    root = drawn(projection, status="current", as_of="2025-06-01T00:00:00.000Z")
    kept = sorted(cached_analysis(projection).communities, key=lambda c: (-len(c.members), c.community_id))
    assert set(circles(root)) == {community.community_id for community in kept[:3]}, "the largest are drawn"
    left = kept[3:]
    desc = root.find(f"{SVG}desc").text
    assert (f"{len(left)} communities ({sum(len(c.members) for c in left)} entities) left out of the map"
            in desc), desc
    assert "current facts as of 2025-06-01T00:00:00.000Z" in desc and desc.startswith("projection ")
    assert "communities of every entity with a relation to another" in desc
    assert "a link is a pair of entities one or more relations join" in desc


def _crossing(projection):
    """Links between each pair of communities, as the map should count them."""
    communities = cached_analysis(projection).communities
    of = {member: community.community_id for community in communities for member in community.members}
    crossing: dict[tuple[str, str], int] = {}
    for pair in {frozenset((r.subject_id, r.object_id)) for r in projection.relations if r.subject_id != r.object_id}:
        ends = sorted({of.get(end) for end in pair} - {None})
        if len(ends) == 2:
            crossing[(ends[0], ends[1])] = crossing.get((ends[0], ends[1]), 0) + 1
    return crossing


def test_the_map_keeps_the_strongest_lines_and_counts_the_rest(monkeypatch):
    import scone_memory.entities.export as module

    stars = [(f"{hub}{k}", "knows", f"Hub {hub.upper()}") for hub in "abcd" for k in range(6)]
    # Hub A and Hub B are joined by two relations: one link, not two.
    bridges = [("Hub A", "knows", "Hub B"), ("Hub A", "met", "Hub B"), ("a0", "knows", "Hub B"),
               ("Hub A", "knows", "b0"), ("Hub B", "knows", "Hub C"), ("Hub C", "knows", "Hub D")]
    projection = project_entities("alpha", [fact(n + 1, *t) for n, t in enumerate(stars + bridges)], revision=1)
    crossing = sorted(_crossing(projection).items(), key=lambda item: (-item[1], item[0]))
    assert len(crossing) >= 2 and crossing[0][1] > crossing[-1][1], crossing
    of = {member: community.community_id for community in cached_analysis(projection).communities
          for member in community.members}
    hubs = {entity.label: entity.entity_id for entity in projection.entities if entity.label.startswith("Hub")}
    assert of[hubs["Hub A"]] != of[hubs["Hub B"]], "the doubled pair crosses between communities"
    monkeypatch.setattr(module, "_MAP_LINKS", 1)
    root = drawn(projection)
    [line] = list(root.iter(f"{SVG}line"))
    assert {line.get("data-from-group"), line.get("data-to-group")} == set(crossing[0][0])
    assert line.find(f"{SVG}title").text.startswith(f"{crossing[0][1]} links between "), "the strongest is kept"
    rest = crossing[1:]
    links = sum(count for _, count in rest)
    said = (f"{links} link{'' if links == 1 else 's'} between {len(rest)} pair{'' if len(rest) == 1 else 's'} "
            f"of communities left out of the map")
    assert said in root.find(f"{SVG}desc").text


def test_a_map_of_a_truncated_analysis_says_so(monkeypatch):
    import scone_memory.entities.analysis as analysis

    projection = project_entities("alpha", ledger(), revision=1)
    analysis._ANALYSES.clear()
    real = analysis.analyze_projection
    monkeypatch.setattr(analysis, "analyze_projection", lambda p, **kw: real(p, **{**kw, "max_entities": 5}))
    try:
        desc = drawn(projection).find(f"{SVG}desc").text
    finally:
        analysis._ANALYSES.clear()
    assert re.search(r"communities found among \d+ of the \d+ entities with a relation to another", desc), desc


def test_names_on_the_map_are_text():
    hostile = '</title><script>alert(1)</script>'
    projection = project_entities("alpha", [fact(1, hostile, "knows", "Bob"), fact(2, "Bob", "knows", hostile)],
                                  revision=1)
    body = export_graph(projection, "communities").body
    assert b"<script" not in body
    assert any(hostile in (title.text or "") for title in ElementTree.fromstring(body).iter(f"{SVG}title"))


def test_the_drawings_point_to_the_map_when_they_leave_entities_out(monkeypatch):
    import scone_memory.entities.export as module

    projection = project_entities("alpha", ledger(), revision=1)
    monkeypatch.setattr(module, "_SVG_NODES", 3)
    desc = ElementTree.fromstring(export_graph(projection, "svg").body).find(f"{SVG}desc").text
    assert "left out of the drawing" in desc and "the communities format draws the graph by community" in desc
    monkeypatch.setattr(module, "_SVG_NODES", 200)
    whole = ElementTree.fromstring(export_graph(projection, "svg").body).find(f"{SVG}desc").text
    assert "communities format" not in whole


@pytest.mark.parametrize("entities", [0, 1])
def test_an_empty_or_single_entity_graph_still_draws_a_map(entities):
    facts = [fact(1, "Ann", "age", "34")][:entities]
    root = drawn(project_entities("alpha", facts, revision=1))
    assert root.tag == f"{SVG}svg" and float(root.get("width")) > 0
    assert len(circles(root)) == entities


def code_ledger():
    """Two rings of modules, every one importing `typing`: what the graph
    names and never reads is attached to a community for reading, and joins
    no two communities."""
    rows = []
    for side in ("a", "b"):
        for n in range(4):
            rows.append((f"{side}/m{n}.py", "defines", f"{side}/m{n}.py:run"))
            rows.append((f"{side}/m{n}.py", "imports", f"{side}/m{(n + 1) % 4}.py"))
            rows.append((f"{side}/m{n}.py", "imports", "typing"))
    return [fact(n + 1, *row) for n, row in enumerate(rows)]


def test_a_line_counts_the_links_the_analysis_counts_and_none_through_what_the_graph_only_names():
    projection = project_entities("alpha", code_ledger(), revision=1)
    analysis = cached_analysis(projection)
    assert analysis.external, "the fixture has an entity the graph only names"
    attached = [community for community in analysis.communities if analysis.external & set(community.members)]
    assert attached and len(attached[0].members) > 1, "and it is attached to a community of the graph's own"
    root = drawn(projection)
    across: dict[str, int] = {community.community_id: 0 for community in analysis.communities}
    for line in root.iter(f"{SVG}line"):
        count = int(line.find(f"{SVG}title").text.split(" ", 1)[0])
        across[line.get("data-from-group")] += count
        across[line.get("data-to-group")] += count
    for community in analysis.communities:
        assert across[community.community_id] == community.boundary_links, community.label
