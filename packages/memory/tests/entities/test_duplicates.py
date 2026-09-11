"""Entities that may be one thing under two names, suggested with why.

The graph joins names only by the one identity rule: case and spacing
aside, nothing looser, because deciding that two spellings are one thing
is a decision with evidence behind it. This finds the pairs worth that
decision and says why each is one: the same name once titles and
punctuation are set aside, names that share words or letters, one the
initials of the other, and the neighbours they share, cited. It merges
nothing.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities.duplicates import DuplicatesError, likely_duplicates
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


async def engine_with(*triples: tuple[str, str, str]) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    for subject, predicate, obj in triples:
        await engine.assert_fact("alpha", subject, predicate, obj, valid_from=DAY)
    return engine


def pairs(found) -> list[tuple[str, str]]:
    return [tuple(sorted((pair["a"]["key"], pair["b"]["key"]))) for pair in found.pairs]


async def test_a_title_set_aside_leaves_the_same_name():
    engine = await engine_with(("alice chen", "works_at", "Acme Robotics"), ("dr. alice chen", "leads", "Robotics Lab"))
    found = await likely_duplicates(engine, "alpha")
    assert pairs(found) == [("alice chen", "dr. alice chen")]
    [pair] = found.pairs
    assert pair["score"] == 1.0 and "the same name once titles and punctuation are set aside" in pair["reasons"]


async def test_a_misspelling_is_found_by_its_letters():
    engine = await engine_with(("alice chen", "works_at", "Acme Robotics"), ("bob stone", "works_at", "Acme Robtics"),
                               ("acme robotics", "based_in", "Lisbon"))
    found = await likely_duplicates(engine, "alpha")
    assert ("acme robotics", "acme robtics") in pairs(found)
    pair = next(p for p in found.pairs if {p["a"]["key"], p["b"]["key"]} == {"acme robotics", "acme robtics"})
    assert any(reason.startswith("names alike by their letters") for reason in pair["reasons"])


async def test_a_one_word_misspelling_is_found_though_no_word_is_shared():
    engine = await engine_with(("alice chen", "lives_in", "Wellington"), ("bob stone", "lives_in", "Welington"))
    found = await likely_duplicates(engine, "alpha")
    assert pairs(found) == [("welington", "wellington")]


async def test_initials_meet_the_name_they_stand_for():
    engine = await engine_with(("alice chen", "works_at", "International Business Machines"),
                               ("bob stone", "works_at", "IBM"))
    found = await likely_duplicates(engine, "alpha")
    assert ("ibm", "international business machines") in pairs(found)
    pair = next(p for p in found.pairs if "ibm" in (p["a"]["key"], p["b"]["key"]))
    assert "one name is the initials of the other" in pair["reasons"]


async def test_shared_neighbours_are_cited_as_evidence():
    engine = await engine_with(("acme robotics", "based_in", "Lisbon"), ("acme robotics", "founded_by", "Dana Ruiz"),
                               ("acme robotics inc", "based_in", "Lisbon"), ("acme robotics inc", "founded_by", "Dana Ruiz"))
    found = await likely_duplicates(engine, "alpha")
    pair = next(p for p in found.pairs if {p["a"]["key"], p["b"]["key"]} == {"acme robotics", "acme robotics inc"})
    assert "neighbours in common: Dana Ruiz, Lisbon" in pair["reasons"] and len(pair["fact_ids"]) == 4
    from scone_memory.entities.duplicates import _alike, _letters

    by_name = max(2 / 3, _alike(_letters("acme robotics"), _letters("acme robotics inc")))
    assert pair["score"] == round(min(1.0, by_name + 0.15), 3), "every neighbour shared adds to the likelihood"


async def test_things_of_different_kinds_are_not_suggested():
    engine = await engine_with(("paris", "capital_of", "France"), ("paris hilton", "works_at", "Hilton Hotels"),
                               ("bob stone", "lives_in", "Paris"))
    found = await likely_duplicates(engine, "alpha")
    assert ("paris", "paris hilton") not in pairs(found)


async def test_two_things_related_to_each_other_are_suggested_less():
    engine = await engine_with(("acme", "parent_of", "Acme Labs"), ("acme labs", "based_in", "Lisbon"))
    related = await likely_duplicates(engine, "alpha", min_score=0.0)
    pair = next(p for p in related.pairs if {p["a"]["key"], p["b"]["key"]} == {"acme", "acme labs"})
    assert any(reason.startswith("related to each other: acme parent_of Acme Labs") for reason in pair["reasons"])
    assert pair["score"] < 0.5


async def test_unrelated_names_are_not_suggested():
    engine = await engine_with(("alice chen", "works_at", "Acme Robotics"), ("bob stone", "lives_in", "Porto"))
    found = await likely_duplicates(engine, "alpha")
    assert found.pairs == () and found.status == "none"


async def test_a_word_too_common_to_compare_by_is_counted(monkeypatch):
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "MAX_BLOCK", 3)
    engine = await engine_with(*[(f"worker {n}", "works_at", "Acme") for n in range(5)])
    found = await likely_duplicates(engine, "alpha")
    assert any(reason.startswith("blocks_skipped ") for reason in found.coverage["reasons"])


async def test_more_pairs_than_the_limit_are_counted():
    engine = await engine_with(("alice chen", "works_at", "Acme"), ("dr. alice chen", "leads", "Lab"),
                               ("bob stone", "works_at", "Globex"), ("mr. bob stone", "leads", "Unit"))
    found = await likely_duplicates(engine, "alpha", limit=1)
    assert len(found.pairs) == 1 and "pairs_cut 1" in found.coverage["reasons"]


async def test_a_capped_read_is_said(monkeypatch):
    from scone_memory.entities import read

    monkeypatch.setattr(read, "MAX_FACTS", 1)
    engine = await engine_with(("alice chen", "works_at", "Acme"), ("dr. alice chen", "leads", "Lab"))
    found = await likely_duplicates(engine, "alpha")
    assert found.coverage["read"]["truncated"] is True
    assert [line for line in found.text.splitlines() if line.startswith("result: ")] == [
        "result: no likely duplicate among the facts read"]


async def test_the_text_says_each_pair_and_why():
    engine = await engine_with(("alice chen", "works_at", "Acme Robotics"), ("dr. alice chen", "leads", "Robotics Lab"))
    found = await likely_duplicates(engine, "alpha")
    [line] = [line for line in found.text.splitlines() if line.startswith("pair: ")]
    assert line.startswith("pair: alice chen (person) ent:") and " ~ dr. alice chen" in line and "1.00" in line


@pytest.mark.parametrize("options, message", [
    ({"limit": 0}, "limit"), ({"limit": 501}, "limit"), ({"min_score": -0.1}, "min_score"),
    ({"min_score": 1.1}, "min_score"), ({"max_bytes": 100}, "max_bytes"),
])
async def test_bounds_are_refused_before_anything_is_read(options, message):
    engine = await engine_with(("alice chen", "works_at", "Acme"))
    with pytest.raises(DuplicatesError, match=message):
        await likely_duplicates(engine, "alpha", **options)


async def test_only_names_that_could_reach_the_score_are_compared():
    """Pairs are drawn from each name's rarest words and letters only: two
    names that cannot be alike enough are never compared, so a large graph
    costs its likely pairs, not all of them."""
    import random

    chosen = random.Random(3)

    def word() -> str:
        return "".join(chosen.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(7))

    names = sorted({f"{word()} {word()} company" for _ in range(120)})
    engine = await engine_with(*[(name, "knows", "Zed") for name in names])
    found = await likely_duplicates(engine, "alpha")
    everything = len(names) * (len(names) - 1) // 2
    assert found.coverage["compared"] < everything // 20


async def test_candidates_past_the_bound_are_counted(monkeypatch):
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "MAX_CANDIDATES", 5)
    engine = await engine_with(*[(f"{first} chen", "knows", "Zed") for first in (
        "alice", "bruno", "clara", "dmitri", "elena", "farid", "greta", "hiro", "ines", "jonas")])
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    assert found.coverage["compared"] == 5 and any(r.startswith("candidates_cut ") for r in found.coverage["reasons"])


async def test_names_that_differ_by_a_number_are_different_things():
    engine = await engine_with(("worker 12", "works_at", "Acme"), ("worker 13", "works_at", "Acme"),
                               ("room 101", "part_of", "Block A"), ("room 101b", "part_of", "Block A"))
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    assert ("worker 12", "worker 13") not in pairs(found)
    assert ("room 101", "room 101b") not in pairs(found)


async def test_the_same_name_once_folded_meets_however_common_its_other_blocks(monkeypatch):
    """Names of single letters have no word or letter run to file them by,
    and here their initials are shared too widely to compare by; the same
    name once folded still meets."""
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "MAX_BLOCK", 2)
    engine = await engine_with(("j k l", "knows", "Zed"), ("dr. j k l", "knows", "Yves"), ("j k lee", "knows", "Uma"))
    found = await likely_duplicates(engine, "alpha")
    assert ("dr. j k l", "j k l") in pairs(found)
