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
