"""Structured questions over the graph: triple patterns joined by variables.

"Who works at an organisation based in Lisbon?" is two patterns sharing a
variable. LlamaIndex answers such a question by having a model write
Cypher for an outside graph store; here the pattern itself is the query,
matched over the space's own projection: bounded, deterministic, every
row citing the facts it rests on and re-read before it is shown, and, in
history, only facts that held at one moment are joined.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities import match as match_module
from scone_memory.entities.match import MatchQueryError, graph_match
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


async def seeded(store=None) -> MemoryEngine:
    engine = await MemoryEngine(store or InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    note = await engine.remember("alpha", "Dr. Alice Chen joined Acme Robotics. Acme Robotics is based in Lisbon.")
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", source_episode_id=note.episode_id,
                             quote="Dr. Alice Chen joined Acme Robotics", valid_from=DAY)
    await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from=DAY)
    await engine.assert_fact("alpha", "bob stone", "lives_in", "Lisbon", valid_from=DAY)
    await engine.assert_fact("alpha", "alice chen", "joined_on", "May 2021", valid_from=DAY)
    await engine.assert_fact("alpha", "alice park", "knows", "bob stone", valid_from=DAY)
    return engine


def where(*patterns: str) -> list[dict[str, str]]:
    return [dict(zip(("subject", "predicate", "object"), pattern.split(" | "))) for pattern in patterns]


def keys(result, variable: str) -> list[str]:
    return [row["bindings"][variable].get("key", row["bindings"][variable].get("value")) for row in result.rows]


def lines(text: str, prefix: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(prefix)]


async def test_a_join_finds_who_works_somewhere_based_in_a_city():
    engine = await seeded()
    found = await graph_match(engine, "alpha", where("?who | works_at | ?org", "?org | based_in | Lisbon"))
    assert found.status == "matched" and found.variables == ("?who", "?org")
    assert keys(found, "?who") == ["alice chen"] and keys(found, "?org") == ["acme robotics"]
    facts = {fact.predicate: fact.fact_id for fact in await engine.facts("alpha")}
    assert found.rows[0]["fact_ids"] == sorted([facts["works_at"], facts["based_in"]])
    [row] = lines(found.text, "row: ")
    assert row.startswith("row: ?who = Alice Chen (person) ent:") and "; ?org = Acme Robotics (organisation) ent:" in row
    assert 'quote verified: "Dr. Alice Chen joined Acme Robotics"' in row
    assert lines(found.text, "pattern: ") == ["pattern: ?who works_at ?org", 'pattern: ?org based_in "Lisbon"']
    assert lines(found.text, "coverage: ") == ["coverage: complete"]


async def test_an_object_constant_matches_a_value_as_well_as_an_entity():
    engine = await seeded()
    by_value = await graph_match(engine, "alpha", where("?who | joined_on | May 2021"))
    assert keys(by_value, "?who") == ["alice chen"]
    bound = await graph_match(engine, "alpha", where("alice chen | joined_on | ?when"))
    assert bound.rows[0]["bindings"]["?when"]["value"] == "May 2021"


async def test_a_case_meaningful_value_matches_only_as_written():
    engine = await seeded()
    await engine.assert_fact("alpha", "laptop", "memory", "512 MB", valid_from=DAY)
    assert (await graph_match(engine, "alpha", where("?thing | memory | 512 mb"))).status == "not_found"
    assert keys(await graph_match(engine, "alpha", where("?thing | memory | 512 MB")), "?thing") == ["laptop"]


async def test_a_pattern_without_variables_asks_whether_it_holds():
    engine = await seeded()
    held = await graph_match(engine, "alpha", where("alice chen | works_at | acme robotics"))
    assert held.status == "matched" and held.variables == () and len(held.rows) == 1
    assert held.rows[0]["bindings"] == {} and len(held.rows[0]["fact_ids"]) == 1
    assert lines(held.text, "row: ")[0].startswith("row: holds [fact ")
    assert (await graph_match(engine, "alpha", where("bob stone | works_at | acme robotics"))).status == "none"


async def test_a_variable_binds_one_kind_of_thing():
    engine = await seeded()
    found = await graph_match(engine, "alpha", where("?x | joined_on | ?v", "?v | based_in | ?y"))
    assert found.status == "none" and found.rows == ()
    assert lines(found.text, "result: ") == ["result: no match"]


async def test_a_variable_used_twice_names_the_same_thing_both_times():
    """Bob works at Acme and lives in Lisbon: not somewhere he both works
    and lives."""
    engine = await seeded()
    await engine.assert_fact("alpha", "bob stone", "works_at", "Acme Robotics", valid_from=DAY)
    found = await graph_match(engine, "alpha", where("?who | works_at | ?place", "?who | lives_in | ?place"))
    assert found.status == "none"
    await engine.assert_fact("alpha", "dana ruiz", "works_at", "Lisbon", valid_from=DAY)
    await engine.assert_fact("alpha", "dana ruiz", "lives_in", "Lisbon", valid_from=DAY)
    both = await graph_match(engine, "alpha", where("?who | works_at | ?place", "?who | lives_in | ?place"))
    assert keys(both, "?who") == ["dana ruiz"]


async def test_a_value_carried_into_another_pattern_meets_only_the_same_text():
    engine = await seeded()
    await engine.assert_fact("alpha", "bob stone", "joined_on", "May 2021", valid_from=DAY)
    await engine.assert_fact("alpha", "carol diaz", "joined_on", "may 2021", valid_from=DAY)
    found = await graph_match(engine, "alpha", where("alice chen | joined_on | ?day", "?who | joined_on | ?day"),
                              returns=["?who"])
    assert keys(found, "?who") == ["alice chen", "bob stone"]


async def test_rows_past_the_reread_budget_are_shown_unverified(monkeypatch):
    monkeypatch.setattr(match_module, "MAX_REREADS", 1)
    engine = await seeded()
    found = await graph_match(engine, "alpha", where("?who | works_at | ?org", "?org | based_in | Lisbon"))
    assert len(found.rows) == 1 and "unverified 1" in found.coverage["reasons"]


async def test_a_predicate_can_be_a_variable():
    engine = await seeded()
    found = await graph_match(engine, "alpha", where("alice chen | ?p | ?o"))
    assert [row["bindings"]["?p"]["predicate"] for row in found.rows] == ["joined_on", "works_at"]


async def test_rows_are_distinct_over_the_variables_returned():
    engine = await seeded()
    await engine.assert_fact("alpha", "bob stone", "works_at", "Acme Robotics", valid_from=DAY)
    both = await graph_match(engine, "alpha", where("?who | works_at | ?org", "?org | based_in | ?city"))
    cities = await graph_match(engine, "alpha", where("?who | works_at | ?org", "?org | based_in | ?city"),
                               returns=["?city"])
    assert len(both.rows) == 2 and cities.variables == ("?city",) and keys(cities, "?city") == ["lisbon"]
    facts = {(fact.subject, fact.predicate) for fact in await engine.facts("alpha")}
    assert len(cities.rows[0]["fact_ids"]) == 3 and len(facts) == 6


async def test_in_history_only_facts_that_held_together_are_joined():
    """Alice worked at Acme until 2021; Acme was in Lisbon from 2024. They
    never held at one moment, so history does not say Alice worked at a
    Lisbon firm, unless asked to join across time."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from="2020-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "alice chen", "works_at", "Globex", valid_from="2021-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from="2024-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "carol diaz", "works_at", "Acme Robotics", valid_from="2024-06-01T00:00:00Z")
    query = where("?who | works_at | ?org", "?org | based_in | Lisbon")
    together = await graph_match(engine, "alpha", query, status="history")
    assert keys(together, "?who") == ["carol diaz"]
    assert together.rows[0]["during"] == [{"from": "2024-06-01T00:00:00.000Z", "until": None}]
    apart = await graph_match(engine, "alpha", query, status="history", together=False)
    assert keys(apart, "?who") == ["alice chen", "carol diaz"] and apart.rows[0]["during"] == []
    assert lines(apart.text, "row: ")[0].endswith(" never held together")
    assert " held from 2024-06-01T00:00:00.000Z" in lines(apart.text, "row: ")[1]
    alone = await graph_match(engine, "alpha", where("alice chen | works_at | ?org", "?org | based_in | Lisbon"),
                              status="history")
    assert lines(alone.text, "result: ") == [
        "result: no match whose facts held at one moment; 1 joins across times that never met (together=false shows them)"]


async def test_a_constant_naming_several_entities_answers_candidates():
    engine = await seeded()
    await engine.assert_fact("alpha", "acme inc", "based_in", "Porto", valid_from=DAY)
    await engine.assert_fact("alpha", "acme, inc.", "based_in", "Faro", valid_from=DAY)
    found = await graph_match(engine, "alpha", where("Acme, Inc | based_in | ?city"))
    assert found.status == "ambiguous" and found.rows == ()
    assert sorted(candidate["key"] for candidate in found.candidates) == ["acme inc", "acme, inc."]
    assert len(lines(found.text, "candidate: ")) == 2


async def test_a_near_miss_is_not_matched_but_suggested():
    """A pattern means what it names: "alice" is neither Alice Chen nor
    Alice Park, though both are offered."""
    engine = await seeded()
    found = await graph_match(engine, "alpha", where("alice | works_at | ?org"))
    assert found.status == "not_found" and found.rows == ()
    assert found.not_found == ({"position": "subject", "term": "alice"},)
    assert sorted(candidate["key"] for candidate in found.candidates) == ["alice chen", "alice park"]


async def test_a_predicate_no_fact_uses_is_named():
    engine = await seeded()
    found = await graph_match(engine, "alpha", where("?x | works_for | ?y"))
    assert found.status == "not_found" and found.not_found == ({"position": "predicate", "term": "works_for"},)


async def test_more_rows_than_the_limit_are_counted():
    engine = await seeded()
    found = await graph_match(engine, "alpha", where("?who | ?p | lisbon"), limit=1)
    assert len(found.rows) == 1 and "rows_cut 1" in found.coverage["reasons"]
    assert lines(found.text, "coverage: ") == ["coverage: limited: rows_cut 1"]


async def test_a_search_cut_short_never_says_no_match(monkeypatch):
    monkeypatch.setattr(match_module, "MAX_WORK", 3)
    engine = await seeded()
    found = await graph_match(engine, "alpha", where("?a | ?p | ?b", "?c | ?q | ?d"))
    assert "search_cut" in found.coverage["reasons"] and found.rows == ()
    assert lines(found.text, "result: ") == ["result: no match within the search budget"]


async def test_every_candidate_examined_is_charged_even_when_it_is_refused(monkeypatch):
    monkeypatch.setattr(match_module, "MAX_WORK", 3)
    monkeypatch.setattr(match_module, "_extend", lambda *arguments: None)
    engine = await seeded()
    found = await graph_match(engine, "alpha", where("?a | ?p | ?b"))
    assert "search_cut" in found.coverage["reasons"] and found.coverage["searched"] > 3


async def test_a_variable_bound_to_an_entity_matches_no_predicate_and_costs_nothing():
    """?org is an organisation after the first pattern; as a predicate in
    the second it can match nothing, so nothing is tried."""
    engine = await seeded()
    for n in range(50):
        await engine.assert_fact("alpha", f"person {n}", "works_at", "Acme Robotics", valid_from=DAY)
    found = await graph_match(engine, "alpha", where("?who | works_at | ?org", "?x | ?org | ?y"))
    assert found.status == "none" and found.coverage["searched"] == 51


async def test_a_value_and_a_predicate_of_one_name_never_stand_for_each_other_and_cost_nothing():
    """Badges are labelled with predicate names. A label bound as a value
    looks at no predicate's facts, and a predicate bound first looks at no
    label, though the names are the same."""
    engine = await seeded()
    await engine.assert_fact("alpha", "badge 7", "label", "works_at", valid_from=DAY)
    await engine.assert_fact("alpha", "badge 8", "label", "joined_on", valid_from=DAY)
    for n in range(20):
        await engine.assert_fact("alpha", f"person {n}", "works_at", "Acme Robotics", valid_from=DAY)
    as_predicate = await graph_match(engine, "alpha", where("?thing | label | ?v", "?x | ?v | ?y"))
    assert as_predicate.status == "none" and as_predicate.coverage["searched"] == 2
    as_value = await graph_match(engine, "alpha", where("alice chen | ?p | May 2021", "?thing | label | ?p"))
    assert as_value.status == "none" and as_value.coverage["searched"] == 1


async def test_each_pattern_is_matched_from_its_smallest_index():
    engine = await seeded()
    for n in range(10):
        await engine.assert_fact("alpha", f"person {n}", "works_at", "Globex", valid_from=DAY)
    found = await graph_match(engine, "alpha", where("?who | works_at | acme robotics"))
    assert keys(found, "?who") == ["alice chen"] and found.coverage["searched"] == 1


async def test_comparing_when_facts_held_is_charged_too(monkeypatch):
    monkeypatch.setattr(match_module, "MAX_WORK", 2)
    engine = await seeded()
    found = await graph_match(engine, "alpha", where("?who | works_at | ?org", "?org | based_in | ?city"))
    assert found.rows == () and "search_cut" in found.coverage["reasons"]


class Changing(InMemoryDocumentStore):
    """Runs an action the first time the answer re-reads one of its facts."""
    targets: set[int] = set()
    action = None

    async def get_fact(self, space, fact_id):
        if fact_id in self.targets:
            self.targets.discard(fact_id)
            await self.action(fact_id)
        return await super().get_fact(space, fact_id)


async def test_when_facts_held_together_is_judged_on_the_facts_as_reread():
    """Alice's job at Acme looked open when matched; before the row was
    shown a backfill ended it in 2021, three years before Acme's Lisbon
    office. The row is dropped: they never held at one moment."""
    store = Changing()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    job = await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from="2020-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from="2024-01-01T00:00:00Z")

    async def backfill(_):
        await engine.assert_fact("alpha", "alice chen", "works_at", "Globex", valid_from="2021-01-01T00:00:00Z")

    store.targets, store.action = {job.fact_id}, backfill
    found = await graph_match(engine, "alpha", where("?who | works_at | ?org", "?org | based_in | Lisbon"),
                              status="history")
    assert found.rows == () and "stale_evidence 1" in found.coverage["reasons"]


async def friends():
    """Alice knew Bob from 2020, then Carol from 2021; both live in Lisbon.
    The store withdraws a fact the first time the answer re-reads it."""
    store = Changing()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    facts = {
        "knows bob": (await engine.assert_fact("alpha", "alice chen", "knows", "Bob Stone",
                                               valid_from="2020-01-01T00:00:00Z")).fact_id,
        "knows carol": (await engine.assert_fact("alpha", "alice chen", "knows", "Carol Diaz",
                                                 valid_from="2021-01-01T00:00:00Z")).fact_id,
        "bob lisbon": (await engine.assert_fact("alpha", "bob stone", "lives_in", "Lisbon",
                                                valid_from="2020-01-01T00:00:00Z")).fact_id,
        "carol lisbon": (await engine.assert_fact("alpha", "carol diaz", "lives_in", "Lisbon",
                                                  valid_from="2020-01-01T00:00:00Z")).fact_id}

    async def withdraw(fact_id):
        await engine.exclude("alpha", fact_id, "withdrawn")

    store.action = withdraw
    return engine, store, facts


async def test_a_row_stands_only_while_one_whole_witness_does():
    """Alice knew Bob, then Carol, who both live in Lisbon. While the answer
    is read, Alice-knows-Bob and Carol-lives-in-Lisbon are withdrawn: what
    is left (Alice-knows-Carol, Bob-lives-in-Lisbon) no longer joins."""
    for together in (True, False):
        engine, store, facts = await friends()
        store.targets = {facts["knows bob"], facts["carol lisbon"]}
        found = await graph_match(engine, "alpha", where("?who | knows | ?friend", "?friend | lives_in | Lisbon"),
                                  returns=["?who"], status="history", together=together)
        assert found.rows == () and "stale_evidence 1" in found.coverage["reasons"], together


async def test_a_row_held_whenever_any_of_its_witnesses_did():
    engine, _, facts = await friends()
    found = await graph_match(engine, "alpha", where("?who | knows | ?friend", "?friend | lives_in | Lisbon"),
                              returns=["?who"], status="history")
    assert found.rows[0]["during"] == [{"from": "2020-01-01T00:00:00.000Z", "until": None}]
    assert len(found.rows[0]["fact_ids"]) == 4


async def test_a_fact_that_stopped_counting_is_not_cited_though_its_witness_stands():
    """Carol lived in Lisbon twice; one of those facts is withdrawn while
    the answer is read. The other still joins, and only it is cited."""
    store = Changing()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "alice chen", "knows", "Carol Diaz", valid_from="2019-01-01T00:00:00Z")
    first = await engine.assert_fact("alpha", "carol diaz", "lives_in", "Lisbon", valid_from="2020-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "carol diaz", "lives_in", "Porto", valid_from="2021-01-01T00:00:00Z")
    again = await engine.assert_fact("alpha", "carol diaz", "lives_in", "Lisbon", valid_from="2022-01-01T00:00:00Z")

    async def withdraw(fact_id):
        await engine.exclude("alpha", fact_id, "withdrawn")

    store.targets, store.action = {first.fact_id}, withdraw
    found = await graph_match(engine, "alpha", where("?who | knows | ?friend", "?friend | lives_in | Lisbon"),
                              status="history")
    assert again.fact_id in found.rows[0]["fact_ids"] and first.fact_id not in found.rows[0]["fact_ids"]


async def test_a_row_cites_only_the_witnesses_that_still_hold():
    engine, store, facts = await friends()
    store.targets = {facts["knows bob"]}
    found = await graph_match(engine, "alpha", where("?who | knows | ?friend", "?friend | lives_in | Lisbon"),
                              returns=["?who"], status="history")
    assert found.rows[0]["fact_ids"] == sorted([facts["knows carol"], facts["carol lisbon"]])
    assert found.rows[0]["during"] == [{"from": "2021-01-01T00:00:00.000Z", "until": None}]


async def test_a_capped_read_never_says_no_match(monkeypatch):
    from scone_memory.entities import read

    monkeypatch.setattr(read, "MAX_FACTS", 2)
    engine = await seeded()
    found = await graph_match(engine, "alpha", where("?x | knows | ?y", "?y | joined_on | ?z"))
    assert found.coverage["read"]["truncated"] is True
    assert lines(found.text, "result: ") == ["result: no match among the facts read"]
    missing = await graph_match(engine, "alpha", where("?x | works_at | ?y"))
    assert lines(missing.text, "not found: ") == ['not found: predicate "works_at" among the facts read']


class ExcludesOnReread(InMemoryDocumentStore):
    target = None
    engine = None

    async def get_fact(self, space, fact_id):
        if fact_id == self.target:
            self.target = None
            await self.engine.exclude(space, fact_id, "retracted")
        return await super().get_fact(space, fact_id)


async def test_a_row_resting_on_a_fact_that_stopped_counting_is_dropped():
    store = ExcludesOnReread()
    engine = await seeded(store)
    works = next(fact for fact in await store.list_facts("alpha", include_closed=True) if fact.predicate == "works_at")
    store.engine, store.target = engine, works.fact_id
    found = await graph_match(engine, "alpha", where("?who | works_at | ?org", "?org | based_in | Lisbon"))
    assert found.rows == () and "stale_evidence 1" in found.coverage["reasons"]
    assert not lines(found.text, "row: ") and lines(found.text, "result: ") == [
        "result: no match among the facts that still hold"]


async def test_the_text_fits_its_budget():
    engine = await seeded()
    for index in range(40):
        await engine.assert_fact("alpha", f"person {index}", "lives_in", "Lisbon", valid_from=DAY)
    found = await graph_match(engine, "alpha", where("?who | lives_in | lisbon"), limit=100, max_bytes=1024)
    assert len(found.text.encode("utf-8")) <= 1024 and lines(found.text, "omitted: ")


@pytest.mark.parametrize("patterns, options, message", [
    ([], {}, "between 1 and 6 patterns"),
    (where(*["?a | knows | ?b"] * 7), {}, "between 1 and 6 patterns"),
    ([{"subject": "?a", "predicate": "knows"}], {}, "subject, predicate and object"),
    ([{"subject": "?a", "predicate": "knows", "object": 3}], {}, "text"),
    (where("?1a | knows | ?b"), {}, "variable"),
    (where("  | knows | ?b"), {}, "empty"),
    (where(f"{'x' * 201} | knows | ?b"), {}, "200 characters"),
    (where("?a | knows | ?b"), {"returns": ["?c"]}, "?c"),
    (where("?a | knows | ?b"), {"limit": 0}, "limit"),
    (where("?a | knows | ?b"), {"limit": 101}, "limit"),
    (where("?a | knows | ?b"), {"max_bytes": 100}, "max_bytes"),
])
async def test_a_malformed_query_is_refused_before_anything_is_read(patterns, options, message):
    engine = await seeded()
    with pytest.raises(MatchQueryError, match=message.replace("?", r"\?")):
        await graph_match(engine, "alpha", patterns, **options)
