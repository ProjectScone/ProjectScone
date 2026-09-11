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
    assert "robotics and robtics are one letter apart" in pair["reasons"]


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
    by_words = 2 / 3  # acme and robotics, of acme, robotics and inc
    assert pair["score"] == round(min(1.0, by_words + 0.15), 3), "every neighbour shared adds to the likelihood"


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
    """"worker" alone is compared with the names holding it only while few
    do; past that the word is too common to compare by, and that is said."""
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "MAX_BLOCK", 3)
    engine = await engine_with(("worker", "works_at", "Acme"), *[(f"{first} worker", "works_at", "Acme")
                                                                for first in ("alpha", "bravo", "charlie", "delta")])
    found = await likely_duplicates(engine, "alpha")
    assert any(reason.startswith("blocks_skipped ") for reason in found.coverage["reasons"])
    monkeypatch.setattr(duplicates_module, "MAX_BLOCK", 200)
    found = await likely_duplicates(engine, "alpha")
    assert ("alpha worker", "worker") in pairs(found) and found.coverage["reasons"] == []


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
    engine = await engine_with(("chen", "knows", "Zed"), *[(f"{first} chen", "knows", "Zed") for first in (
        "alice", "bruno", "clara", "dmitri", "elena", "farid", "greta", "hiro", "ines", "jonas")])
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    assert found.coverage["compared"] == 5 and any(r.startswith("candidates_cut ") for r in found.coverage["reasons"])


async def test_names_that_differ_by_a_number_are_different_things():
    """A name's numbers are its runs of digits, in order: 12 is not 13, and
    "Studio54" and "Studio 54" hold the same 54."""
    engine = await engine_with(("worker 12", "works_at", "Acme"), ("worker 13", "works_at", "Acme"),
                               ("room 101", "part_of", "Block A"), ("room 102", "part_of", "Block A"),
                               ("studio 54", "based_in", "New York"), ("studio54", "based_in", "New York"),
                               ("version 1.23", "part_of", "Scone"), ("version 12.3", "part_of", "Scone"))
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    assert ("worker 12", "worker 13") not in pairs(found) and ("room 101", "room 102") not in pairs(found)
    assert ("version 1.23", "version 12.3") not in pairs(found)
    assert ("studio 54", "studio54") in pairs(found)
    studio = next(p for p in found.pairs if {p["a"]["key"], p["b"]["key"]} == {"studio 54", "studio54"})
    assert studio["score"] == 1.0 and "the same name once spacing is set aside" in studio["reasons"]


@pytest.mark.parametrize("left, right", [
    ("Acme Robotics Company", "Zcme Robtics Company"),
    ("J K L M Alpha", "J K L M"),
    ("IT", "Information Technology"),
    ("Acme Compnay", "Acme Company"),
])
async def test_every_way_a_pair_can_score_is_a_way_it_is_found(left, right):
    """A long name alike letter by letter, and names alike by short words,
    are compared as surely as those sharing a rare word."""
    engine = await engine_with((left, "based_in", "Lisbon"), (right, "based_in", "Lisbon"))
    found = await likely_duplicates(engine, "alpha")
    assert pairs(found) == [tuple(sorted((left.lower(), right.lower())))]


async def test_names_with_nothing_left_once_folded_are_not_alike(monkeypatch):
    """Nothing is not a name two things can share, even compared outright."""
    from itertools import combinations
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "_candidates", lambda named, words: (set(combinations(sorted(named), 2)), {}))
    engine = await engine_with(("...", "knows", "Zed"), ("!!!", "knows", "Zed"))
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    assert found.pairs == ()


@pytest.mark.parametrize("left, right", [
    ("University of Lisbon", "University of Porto"), ("Alpha Holdings", "Aldi Holdings"),
    ("Acme Company", "Apex Company"), ("Bank of America", "Bank of Canada"), ("John Smith", "Jane Smith"),
    ("J K L M Shared Alpha Beta Gamma", "J K L M Shared Delta Epsilon Zeta"),
])
async def test_names_that_each_keep_a_word_the_other_lacks_are_two_things(left, right):
    """However much they share, two names each holding a word the other has
    no spelling of name two things, even with a neighbour in common."""
    engine = await engine_with((left, "based_in", "Lisbon"), (right, "based_in", "Lisbon"))
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    assert tuple(sorted((left.lower(), right.lower()))) not in pairs(found)


async def test_one_name_within_the_other_is_suggested():
    engine = await engine_with(("acme", "based_in", "Lisbon"), ("acme robotics", "based_in", "Lisbon"))
    found = await likely_duplicates(engine, "alpha")
    [pair] = found.pairs
    assert pairs(found) == [("acme", "acme robotics")] and pair["reasons"][0] == "names share the words acme"


async def test_short_words_must_be_spelt_alike(monkeypatch):
    """One letter makes another word of a short one: Bob is not Rob, nor
    Jon John, even compared outright."""
    from itertools import combinations
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "_candidates", lambda named, words: (set(combinations(sorted(named), 2)), {}))
    engine = await engine_with(("bob stone", "based_in", "Lisbon"), ("rob stone", "based_in", "Lisbon"),
                               ("jon ruiz", "based_in", "Porto"), ("john ruiz", "based_in", "Porto"))
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    assert found.pairs == ()


async def test_every_pair_suggested_says_why():
    engine = await engine_with(("wellington", "based_in", "Lisbon"), ("welington", "based_in", "Porto"),
                               ("acme", "based_in", "Lisbon"), ("acme labs", "based_in", "Porto"),
                               ("ibm", "based_in", "Oslo"), ("international business machines", "based_in", "Oslo"))
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    assert len(found.pairs) >= 3 and all(pair["reasons"] for pair in found.pairs)


def test_neighbours_in_common_add_to_a_likeness_by_name_and_never_make_one():
    """Two people who work at one firm and live in one city are two people:
    names alike in nothing score nothing however many neighbours they
    share, so no pair scores that the blocks do not find."""
    from scone_memory.entities.duplicates import _Pair

    shared = [("Acme", [1], [2]), ("Porto", [3], [4])]
    assert _Pair("a", "b", 0.0, [], shared, 2, []).score()[0] == 0.0
    assert _Pair("a", "b", 0.4, [], shared, 2, []).score()[0] == pytest.approx(0.55)


@pytest.mark.parametrize("seed", range(4))
async def test_the_blocks_find_every_pair_comparing_all_of_them_would(monkeypatch, seed):
    """Whatever route a pair scores by, the blocks find it: over names that
    mix shared and misspelt words, short and long, initials, spacing, small
    words and numbers, comparing every pair finds exactly the same."""
    import random
    from itertools import combinations
    from scone_memory.entities import duplicates as duplicates_module

    chosen = random.Random(seed)
    stems = ["acme", "robotics", "company", "j", "k", "shared", "alpha", "studio", "54", "lisbon", "of", "the",
             "international", "business", "machines", "ibm", "it", "information", "technology", "bob", "rob"]

    def spelt(word: str) -> str:
        roll = chosen.random()
        if len(word) < 4 or roll < 0.55:
            return word
        at = chosen.randrange(len(word) - 1)
        if roll < 0.7:
            return word[:at] + word[at + 1] + word[at] + word[at + 2:]
        if roll < 0.85:
            return word[:at] + word[at + 1:]
        return word[:at] + chosen.choice("aeiouz") + word[at + 1:]

    names = set()
    while len(names) < 40:
        words = [spelt(chosen.choice(stems)) for _ in range(chosen.randint(1, 4))]
        names.add("".join(words) if chosen.random() < 0.1 else " ".join(words))
    engine = await engine_with(*[(name, "knows", chosen.choice(["Zed", "Yves", "Uma"])) for name in sorted(names)])
    for min_score in (0.0, 0.5):
        found = await likely_duplicates(engine, "alpha", min_score=min_score, limit=500)
        with monkeypatch.context() as patched:
            patched.setattr(duplicates_module, "_candidates",
                            lambda named, words: (set(combinations(sorted(named), 2)), {}))
            everything = await likely_duplicates(engine, "alpha", min_score=min_score, limit=500)
        assert found.coverage["reasons"] == everything.coverage["reasons"] == []
        assert found.pairs == everything.pairs, min_score


async def test_the_same_name_once_folded_meets_however_common_its_other_blocks(monkeypatch):
    """Names of single letters have no word or letter run to file them by,
    and here their initials are shared too widely to compare by; the same
    name once folded still meets."""
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "MAX_BLOCK", 2)
    engine = await engine_with(("j k l", "knows", "Zed"), ("dr. j k l", "knows", "Yves"), ("j k lee", "knows", "Uma"))
    found = await likely_duplicates(engine, "alpha")
    assert ("dr. j k l", "j k l") in pairs(found)


async def test_names_with_different_numbers_are_never_even_compared(monkeypatch):
    """A family of two hundred numbered members shares a word, but no two of
    them can be one thing, so no pair of them is compared; and each name's
    letters are worked out once, however many pairs it is in."""
    from scone_memory.entities import duplicates as duplicates_module

    built = []
    real = duplicates_module._letters
    monkeypatch.setattr(duplicates_module, "_letters", lambda text: built.append(text) or real(text))
    engine = await engine_with(*[(f"family{n // 200:03} member{n:04}", "value", "yes") for n in range(200)])
    found = await likely_duplicates(engine, "alpha", limit=1)
    assert found.coverage["compared"] == 0 and len(built) <= 201


async def test_initials_count_the_small_words_too():
    for name, initials in (("Bank of America", "BOA"), ("Museum of Modern Art", "MOMA")):
        engine = await engine_with(("alice chen", "works_at", name), ("bob stone", "works_at", initials))
        found = await likely_duplicates(engine, "alpha")
        assert pairs(found) == [tuple(sorted((initials.lower(), name.lower())))], name


class Withdraws(InMemoryDocumentStore):
    """Withdraws every fact of the space the first time one is re-read."""
    engine = None
    armed = False

    async def get_fact(self, space, fact_id):
        if self.armed:
            self.armed = False
            for fact in await self.list_facts(space, include_closed=True):
                await self.engine.exclude(space, fact.fact_id, "withdrawn")
        return await super().get_fact(space, fact_id)


async def test_a_suggestion_is_made_again_when_its_evidence_changes_while_read():
    store = Withdraws()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from=DAY)
    await engine.assert_fact("alpha", "acme robotics inc", "based_in", "Lisbon", valid_from=DAY)
    store.engine, store.armed = engine, True
    found = await likely_duplicates(engine, "alpha")
    fresh = await likely_duplicates(engine, "alpha")
    assert found.pairs == fresh.pairs == ()


class StillWithdraws(Withdraws):
    """As Withdraws, in a store whose revision does not move."""

    async def revision(self, space):
        return 1


async def test_evidence_that_stopped_counting_is_not_cited_where_the_revision_cannot_tell():
    store = StillWithdraws()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from=DAY)
    await engine.assert_fact("alpha", "acme robotics inc", "based_in", "Lisbon", valid_from=DAY)
    store.engine, store.armed = engine, True
    found = await likely_duplicates(engine, "alpha")
    [pair] = found.pairs
    assert pair["fact_ids"] == [] and not any(reason.startswith("neighbours in common") for reason in pair["reasons"])
    assert "stale_evidence 2" in found.coverage["reasons"]


class KeepsMoving(InMemoryDocumentStore):
    """Writes an unrelated fact each time the answer re-reads one."""
    engine = None
    moving = False

    async def get_fact(self, space, fact_id):
        if self.moving:
            self.moving = False
            await self.engine.assert_fact(space, f"visitor {await self.revision(space)}", "passed_by", "Zed")
            self.moving = True
        return await super().get_fact(space, fact_id)


async def test_a_ledger_that_keeps_moving_is_said():
    store = KeepsMoving()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from=DAY)
    await engine.assert_fact("alpha", "acme robotics inc", "based_in", "Lisbon", valid_from=DAY)
    store.engine, store.moving = engine, True
    found = await likely_duplicates(engine, "alpha")
    assert "ledger_moved_during_read" in found.coverage["reasons"] and found.pairs


async def test_each_words_letters_are_worked_out_once(monkeypatch):
    from scone_memory.entities import duplicates as duplicates_module

    built = []
    real = duplicates_module._letters
    monkeypatch.setattr(duplicates_module, "_letters", lambda text: built.append(text) or real(text))
    engine = await engine_with(*[(f"acme {word}", "knows", "Zed") for word in ("robotics", "robtics", "rbotics",
                                                                               "roboticz")])
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    # robotics is one letter from each other spelling, and rbotics from
    # robtics (two swapped); the others are two apart.
    assert len(found.pairs) == 4 and sorted(built) == ["rbotics", "robotics", "roboticz", "robtics"]


def test_one_edit_is_a_letter_changed_added_dropped_or_two_neighbours_swapped():
    """Checked against the edit distance itself, worked out the long way,
    over every pair of a few thousand short words."""
    import random
    from scone_memory.entities.duplicates import _one_apart, _spellings

    def distance(left: str, right: str) -> int:
        rows = [[i + j if not i * j else 0 for j in range(len(right) + 1)] for i in range(len(left) + 1)]
        for i in range(1, len(left) + 1):
            for j in range(1, len(right) + 1):
                rows[i][j] = min(rows[i - 1][j] + 1, rows[i][j - 1] + 1,
                                 rows[i - 1][j - 1] + (left[i - 1] != right[j - 1]))
                if i > 1 and j > 1 and left[i - 1] == right[j - 2] and left[i - 2] == right[j - 1]:
                    rows[i][j] = min(rows[i][j], rows[i - 2][j - 2] + 1)
        return rows[-1][-1]

    chosen = random.Random(7)
    words = sorted({"".join(chosen.choice("abc") for _ in range(chosen.randint(1, 5))) for _ in range(400)})
    for left in words:
        for right in words:
            assert _one_apart(left, right) is (distance(left, right) == 1), (left, right)
            if distance(left, right) == 1 and min(len(left), len(right)) >= 4:
                assert _spellings(left) & _spellings(right), (left, right)


async def test_a_name_is_compared_only_with_names_holding_every_one_of_its_words():
    """Thirty-six names, each two words of six: every name shares a word
    with ten others, but none holds both words of another, so none is
    compared."""
    engine = await engine_with(*[(f"{first} {second}", "knows", "Zed") for first in (
        "amber", "basil", "cedar", "delta", "eagle", "fjord") for second in (
        "harbor", "island", "jungle", "kettle", "lantern", "meadow")])
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    assert found.coverage["compared"] == 0 and found.pairs == ()


async def test_a_name_searches_under_its_least_common_word(monkeypatch):
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "MAX_BLOCK", 3)
    engine = await engine_with(*[(f"{first} worker", "works_at", "Acme")
                                 for first in ("alpha", "bravo", "charlie", "delta")], ("alpha", "knows", "Zed"))
    found = await likely_duplicates(engine, "alpha")
    assert found.coverage["reasons"] == [] and ("alpha", "alpha worker") in pairs(found)


async def test_a_misspelt_word_is_paired_with_the_spelling_most_alike():
    engine = await engine_with(("acme robotics", "based_in", "Lisbon"), ("acme robotica robotic", "based_in", "Lisbon"))
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    [pair] = found.pairs
    paired = [set(reason.removesuffix(" are one letter apart").split(" and ")) for reason in pair["reasons"]
              if reason.endswith(" are one letter apart")]
    assert paired == [{"robotic", "robotics"}]


async def test_looking_past_the_bound_is_counted(monkeypatch):
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "MAX_EXAMINED", 10)
    engine = await engine_with(("chen", "knows", "Zed"), *[(f"{first} chen", "knows", "Zed") for first in (
        "alice", "bruno", "clara", "dmitri", "elena", "farid", "greta", "hiro", "ines", "jonas")])
    found = await likely_duplicates(engine, "alpha")
    assert any(reason.startswith("candidates_cut ") for reason in found.coverage["reasons"])
    assert ("chen", "alice chen") not in pairs(found) and found.pairs == ()


async def test_initials_too_common_to_compare_by_are_counted(monkeypatch):
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "MAX_BLOCK", 3)
    engine = await engine_with(("ibm", "knows", "Zed"), ("international business machines", "knows", "Yves"),
                               ("indian bureau of mines", "knows", "Uma"), ("irish beef marketing", "knows", "Tom"))
    found = await likely_duplicates(engine, "alpha")
    assert found.coverage["reasons"] == ["blocks_skipped 1"] and found.pairs == ()


async def test_a_misspelt_word_counts_for_how_alike_its_letters_are():
    from scone_memory.entities.duplicates import _alike, _letters

    engine = await engine_with(("wellington", "based_in", "Lisbon"), ("welington", "based_in", "Porto"),
                               ("acme robotics", "knows", "Zed"), ("acme robtics", "knows", "Yves"))
    found = {pair: p["score"] for pair, p in zip(pairs(await likely_duplicates(engine, "alpha")),
                                                  (await likely_duplicates(engine, "alpha")).pairs)}
    city = _alike(_letters("wellington"), _letters("welington"))
    firm = _alike(_letters("robotics"), _letters("robtics"))
    assert found[("welington", "wellington")] == round(city / (2 - city), 3)
    assert found[("acme robotics", "acme robtics")] == round((1 + firm) / (3 - firm), 3)


async def test_facts_not_read_again_for_want_of_budget_count_for_nothing_and_are_said(monkeypatch):
    """Past the re-read budget a cited fact is unknown: it supports no pair,
    it is not cited, and the answer says how many went unread."""
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "MAX_REREADS", 3)
    engine = await engine_with(*[(name, f"link_{index}", "Lisbon") for index in range(4)
                                 for name in ("acme robotics", "acmerobotics")],
                               *[(name, f"link_{index}", f"City {index}") for index in range(4)
                                 for name in ("acme robotics", "acmerobotics")])
    reads: list[int] = []
    original = engine.documents.get_fact

    async def counted(space, fact_id):
        reads.append(fact_id)
        return await original(space, fact_id)

    monkeypatch.setattr(engine.documents, "get_fact", counted)
    found = await likely_duplicates(engine, "alpha")
    [pair] = found.pairs
    assert set(pair["fact_ids"]) <= set(reads)
    assert any(reason.startswith("rereads_cut ") for reason in found.coverage["reasons"])
    assert "coverage: limited: " in found.text


async def test_a_relation_not_read_again_still_counts_against_a_pair(monkeypatch):
    """What could make a pair less likely is not dropped for being unread."""
    from scone_memory.entities import duplicates as duplicates_module

    engine = await engine_with(("acme", "parent_of", "Acme Labs"), ("acme labs", "based_in", "Lisbon"))
    monkeypatch.setattr(duplicates_module, "MAX_REREADS", 0)
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    pair = next(p for p in found.pairs if {p["a"]["key"], p["b"]["key"]} == {"acme", "acme labs"})
    assert pair["score"] < 0.5 and "related to each other: acme parent_of Acme Labs (not read again)" in pair["reasons"]


async def test_a_word_too_long_to_spell_out_must_be_spelt_alike(monkeypatch):
    """Spelling out a word's misspellings costs its length squared, so past
    NEAR_MAX_LETTERS a word matches only itself, in the search and the
    score alike: nothing that long is expanded, and none is suggested
    misspelt."""
    from itertools import combinations
    from scone_memory.entities import duplicates as duplicates_module

    assert duplicates_module._spellings("a" * 4000) == frozenset({"a" * 4000})
    long, longer = "b" * 40, "b" * 39 + "c"
    fitting, fits = "d" * 20 + "robotics", "d" * 20 + "robtics"
    engine = await engine_with((f"acme {long}", "based_in", "Lisbon"), (f"acme {longer}", "based_in", "Lisbon"),
                               (f"acme {fitting}", "based_in", "Porto"), (f"acme {fits}", "based_in", "Porto"))
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    assert pairs(found) == [(f"acme {fitting}", f"acme {fits}")]
    monkeypatch.setattr(duplicates_module, "_candidates", lambda named, words: (set(combinations(sorted(named), 2)), {}))
    assert pairs(await likely_duplicates(engine, "alpha", min_score=0.0)) == pairs(found)


async def test_spellings_past_the_budget_are_not_spelt_out_and_are_said(monkeypatch):
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "MAX_SPELLINGS", 12)
    engine = await engine_with(("acme robotics", "based_in", "Lisbon"), ("acme robtics", "based_in", "Lisbon"),
                               ("zeta works", "based_in", "Porto"), ("zeta wroks", "based_in", "Porto"))
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    assert any(reason.startswith("spellings_cut ") for reason in found.coverage["reasons"])
    assert "coverage: limited: " in found.text


async def test_names_past_the_spelling_budget_are_found_by_their_words_alone(monkeypatch):
    """Past MAX_SPELLINGS nothing more is spelt out: the names left are filed
    under their words alone, their misspellings are not looked for, and how
    many there were is said."""
    from scone_memory.entities import duplicates as duplicates_module

    monkeypatch.setattr(duplicates_module, "MAX_SPELLINGS", 0)
    spelt = []
    real = duplicates_module._spellings
    monkeypatch.setattr(duplicates_module, "_spellings", lambda word: spelt.append(word) or real(word))
    # Different initials, so only a spelling could bring the two together.
    engine = await engine_with(("acme robotics", "based_in", "Lisbon"), ("acme gobotics", "based_in", "Lisbon"),
                               ("acme", "based_in", "Lisbon"))
    found = await likely_duplicates(engine, "alpha", min_score=0.0)
    assert spelt == [], "nothing past the budget is spelt out"
    assert ("acme gobotics", "acme robotics") not in pairs(found) and ("acme", "acme robotics") in pairs(found)
    assert "spellings_cut 4 names" in found.coverage["reasons"]  # Lisbon is a name too
    monkeypatch.setattr(duplicates_module, "MAX_SPELLINGS", 250_000)
    assert ("acme gobotics", "acme robotics") in pairs(await likely_duplicates(engine, "alpha", min_score=0.0))
