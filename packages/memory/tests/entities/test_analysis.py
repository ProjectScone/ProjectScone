"""Whole-space graph analysis: communities, central things, and surprises.

Every result is computed from the projection's recorded relations, carries
the relation and fact ids behind it, and is deterministic: the same facts in
any order give the same communities, ranks and suggestions.
"""
from __future__ import annotations

import random
from collections import Counter

import pytest

from scone_memory.core.models import Fact
from scone_memory.entities.analysis import analyze_projection
from scone_memory.entities.project import project_entities


def fact(number: int, subject: str, predicate: str, object_: str) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject.casefold(), predicate=predicate, object=object_,
                valid_from="2025-01-01T00:00:00Z")


def two_teams() -> list[Fact]:
    """Two tight groups that share exactly one link."""
    rows, number = [], 0
    for team in (("Ana", "Ben", "Cho", "Dev"), ("Eli", "Fay", "Gus", "Hal")):
        for left in team:
            for right in team:
                if left < right:
                    number += 1
                    rows.append(fact(number, left, "knows", right))
    number += 1
    rows.append(fact(number, "Dev", "mentors", "Eli"))
    return rows


def labels(projection):
    return {entity.entity_id: entity.key for entity in projection.entities}


def test_two_tight_groups_become_two_communities():
    projection = project_entities("alpha", two_teams(), revision=1)
    analysis = analyze_projection(projection)
    names = labels(projection)
    groups = sorted(sorted(names[member] for member in community.members) for community in analysis.communities)
    assert groups == [["ana", "ben", "cho", "dev"], ["eli", "fay", "gus", "hal"]]
    assert analysis.modularity > 0.3


def test_the_only_link_between_communities_is_the_surprise_and_cites_its_fact():
    rows = two_teams()
    projection = project_entities("alpha", rows, revision=1)
    analysis = analyze_projection(projection)
    names = labels(projection)
    top = analysis.surprising_connections[0]
    assert {names[top.subject_id], names[top.object_id]} == {"dev", "eli"}
    assert top.fact_ids == (rows[-1].fact_id,) and top.links_between_communities == 1


def test_the_bridging_entities_rank_highest_for_betweenness():
    projection = project_entities("alpha", two_teams(), revision=1)
    analysis = analyze_projection(projection)
    names = labels(projection)
    ranked = sorted(analysis.importance, key=lambda item: -item.betweenness)
    assert {names[ranked[0].entity_id], names[ranked[1].entity_id]} == {"dev", "eli"}
    # Every shortest path between the groups runs through the bridge's ends,
    # and no path within a group needs a third member.
    assert {names[item.entity_id] for item in analysis.importance if item.betweenness > 0} == {"dev", "eli"}
    assert abs(sum(item.pagerank for item in analysis.importance) - 1.0) < 1e-6


def test_a_hub_held_apart_lets_the_groups_it_glued_come_apart():
    rows = two_teams()
    number = len(rows)
    for person in ("Ana", "Ben", "Cho", "Dev", "Eli", "Fay"):
        number += 1
        rows.append(fact(number, "Hub", "knows", person))
    projection = project_entities("alpha", rows, revision=1)
    names = labels(projection)
    glued = analyze_projection(projection)
    assert glued.hubs == frozenset() and glued.coverage.hubs_held_apart == 0 and glued.coverage.exclude_hubs is None
    apart = analyze_projection(projection, exclude_hubs=80)
    [hub] = apart.hubs
    assert names[hub] == "hub" and apart.coverage.hubs_held_apart == 1 and apart.coverage.exclude_hubs == 80
    own = sorted(sorted(names[m] for m in community.members if m not in apart.hubs) for community in apart.communities)
    assert own == [["ana", "ben", "cho", "dev"], ["eli", "fay", "gus", "hal"]], "the partition is found without the hub"
    [attached] = [community for community in apart.communities if hub in community.members]
    assert {names[m] for m in attached.members} == {"ana", "ben", "cho", "dev", "hub"}, \
        "the hub is attached to the community it links most: four links against two"
    assert all(hub not in community.top_entities for community in apart.communities), "a hub names no community"
    assert not any(hub in (s.subject_id, s.object_id) for s in apart.surprising_connections), "a link to a hub is no surprise"
    assert next(item for item in apart.importance if item.entity_id == hub).participation == 0.0
    assert apart.coverage.record()["hubs_held_apart"] == 1
    with pytest.raises(ValueError, match="50 to 100"):
        analyze_projection(projection, exclude_hubs=10)
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    again = analyze_projection(project_entities("alpha", shuffled, revision=1), exclude_hubs=80)
    assert again.hubs == apart.hubs and [c.label for c in again.communities] == [c.label for c in apart.communities]


def test_a_community_of_files_is_named_by_the_directory_they_share():
    rows = [fact(1, "src/pkg/a.py", "imports", "src/pkg/b.py"), fact(2, "src/pkg/b.py", "imports", "src/pkg/c.py"),
            fact(3, "src/pkg/a.py", "defines", "src/pkg/a.py:Thing"), fact(4, "src/pkg/c.py", "imports", "src/pkg/a.py"),
            fact(5, "web/app.ts", "imports", "web/lib/util.ts"), fact(6, "web/lib/util.ts", "imports", "web/lib/fmt.ts"),
            fact(7, "web/app.ts", "imports", "web/lib/fmt.ts"),
            fact(8, "top.py", "imports", "other.py"), fact(9, "other.py", "imports", "third.py"), fact(10, "third.py", "imports", "top.py")]
    analysis = analyze_projection(project_entities("alpha", rows, revision=1))
    by_label = sorted(community.label for community in analysis.communities)
    placed = [label for label in by_label if label.split(" · ")[0].endswith("/")]
    assert [label.split(" · ")[0] for label in placed] == ["src/pkg/", "web/"], "files under one directory are named by it"
    [root] = [label for label in by_label if label not in placed]
    assert set(root.split(" · ")) <= {"top.py", "other.py", "third.py"}, "files at the root keep their central members' names"
    assert all(" · src/pkg/" not in label and "src/pkg/a.py" not in label for label in by_label), \
        "the shared directory is not repeated on each member"
    assert any(label.startswith("src/pkg/ · ") and "a.py" in label for label in placed)
    spread = [fact(1, "app/a.py", "imports", "app/b.py"), fact(2, "app/b.py", "imports", "app/c.py"), fact(3, "app/c.py", "imports", "app/a.py"),
              fact(4, "app/a.py", "imports", "api/routes.py"), fact(5, "api/routes.py", "imports", "app/b.py"),
              fact(6, "app/sub/d.py", "imports", "app/a.py")]
    [community] = analyze_projection(project_entities("alpha", spread, revision=1)).communities
    assert community.label.startswith("app/ · "), "a community that spans two packages is named by the one most of it sits in"
    assert "app/a.py" not in community.label and "api/routes.py" in community.label or "routes.py" in community.label


def test_communities_are_named_after_their_most_central_members():
    projection = project_entities("alpha", two_teams(), revision=1)
    analysis = analyze_projection(projection)
    for community in analysis.communities:
        assert community.label and len(community.top_entities) <= 3
        assert community.top_entities[0] in community.members


def test_suggested_questions_use_real_names_and_cite_their_basis():
    rows = two_teams()
    projection = project_entities("alpha", rows, revision=1)
    analysis = analyze_projection(projection)
    question = next(q for q in analysis.suggestions if q.kind == "connection")
    assert "dev" in question.text.casefold() and "eli" in question.text.casefold()
    assert question.fact_ids == (rows[-1].fact_id,)


def test_the_same_facts_in_any_order_give_the_same_analysis():
    rows = two_teams()
    first = analyze_projection(project_entities("alpha", rows, revision=1))
    for seed in range(10):
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        again = analyze_projection(project_entities("alpha", shuffled, revision=1))
        assert again == first


def test_an_empty_projection_is_analysed_without_error():
    analysis = analyze_projection(project_entities("alpha", [], revision=1))
    assert analysis.communities == () and analysis.importance == () and analysis.modularity == 0.0


def test_a_budget_on_entities_is_reported():
    rows = two_teams()
    analysis = analyze_projection(project_entities("alpha", rows, revision=1), max_entities=5)
    assert analysis.coverage.truncated and "entity_limit" in analysis.coverage.reasons
    assert sum(len(community.members) for community in analysis.communities) == 5


# Relation ids are hashes; these pairs put the rare link's id both first and
# not first, so the ranking must come from rarity rather than id order.
@pytest.mark.parametrize(("funder", "funded"), [("Hal", "Ivy"), ("Gus", "Jon"), ("Eli", "Lou")])
def test_a_rare_link_between_communities_outranks_a_common_one(funder, funded):
    rows, number = [], 0
    for team in (("Ana", "Ben", "Cho", "Dev"), ("Eli", "Fay", "Gus", "Hal"), ("Ivy", "Jon", "Kim", "Lou")):
        for left in team:
            for right in team:
                if left < right:
                    number += 1
                    rows.append(fact(number, left, "knows", right))
    for left, right in (("Ana", "Eli"), ("Ben", "Fay"), ("Cho", "Gus")):
        number += 1
        rows.append(fact(number, left, "advises", right))
    number += 1
    rows.append(fact(number, funder, "funds", funded))
    projection = project_entities("alpha", rows, revision=1)
    analysis = analyze_projection(projection)
    names = labels(projection)
    top = analysis.surprising_connections[0]
    assert {names[top.subject_id], names[top.object_id]} == {funder.casefold(), funded.casefold()}
    links = [s.links_between_communities for s in analysis.surprising_connections]
    assert links == sorted(links) and links[0] == 1 and set(links) == {1, 3}


def test_a_budget_that_strands_a_kept_hub_still_analyses():
    """A hub heavier than every other entity keeps its place in the budget
    while all its light leaves fall out, leaving it with no neighbours. The
    groups around it must still merge over several levels without losing it."""
    rng, pairs, strength = random.Random(0), [], Counter()
    for i in range(14):
        for j in range(i + 1, 14):
            if rng.random() < 0.2:
                weight = rng.randint(1, 10) * 10
                pairs.append((f"person{chr(97 + i)}", f"person{chr(97 + j)}", weight))
                strength[i] += weight
                strength[j] += weight
    pairs += [("hub", f"leaf{chr(97 + n // 26)}{chr(97 + n % 26)}", 1) for n in range(min(strength.values()) + 1)]
    rows = [fact(0, left, "knows", right) for left, right, weight in pairs for _ in range(weight)]
    rows = [row.model_copy(update={"fact_id": number}) for number, row in enumerate(rows, 1)]
    projection = project_entities("alpha", rows, revision=1)
    analysis = analyze_projection(projection, max_entities=len(strength) + 1)
    kept = {member for community in analysis.communities for member in community.members}
    hub = next(entity.entity_id for entity in projection.entities if entity.key == "hub")
    assert hub in kept and len(kept) == len(strength) + 1 and analysis.coverage.truncated


def test_sampled_betweenness_is_marked_and_never_passes_its_maximum():
    """A 501-entity star is past the exact limit. Scaling 64 sampled sources
    up to the whole graph overshoots the hub's true score of 1; an estimate
    above the largest possible value is clamped, and the method says sampled."""
    rows = [fact(number, f"person{number:03d}", "knows", "Hub") for number in range(1, 501)]
    analysis = analyze_projection(project_entities("alpha", rows, revision=1))
    assert analysis.coverage.betweenness == "sampled:64"
    hub = max(analysis.importance, key=lambda item: item.betweenness)
    assert labels_of(rows, hub.entity_id) == "hub" and hub.betweenness == 1.0


def labels_of(rows, entity_id):
    return labels(project_entities("alpha", rows, revision=1))[entity_id]


def ring_of_cliques(cliques: int = 6, size: int = 4) -> list[Fact]:
    """Tight groups joined in a ring by single links."""
    rows, number = [], 0
    for group in range(cliques):
        members = [f"g{group} m{member}" for member in range(size)]
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                number += 1
                rows.append(fact(number, left, "knows", right))
        number += 1
        rows.append(fact(number, members[0], "knows", f"g{(group + 1) % cliques} m1"))
    return rows


def test_resolution_sets_how_fine_the_communities_are():
    projection = project_entities("alpha", ring_of_cliques(), revision=1)
    coarse = analyze_projection(projection, resolution=0.05)
    usual = analyze_projection(projection)
    # A clique member's three links outweigh the size penalty until about 10.
    fine = analyze_projection(projection, resolution=10.0)
    assert len(coarse.communities) < len(usual.communities) == 6 < len(fine.communities)
    assert usual.coverage.resolution == 1.0 and fine.coverage.resolution == 10.0


def test_resolution_must_be_positive():
    with pytest.raises(ValueError):
        analyze_projection(project_entities("alpha", ring_of_cliques(), revision=1), resolution=0)
