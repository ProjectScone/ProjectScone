"""What changed in the graph between two moments, cited.

"What changed since we last spoke?" is asked of time, not of names. The
graph as it held at one moment is set beside the graph at another: the
relations that began and ended, a claim that moved from one object to
another, the values that changed and the entities that came and went.
Every change cites the facts behind it, re-read now, and no change is
claimed of a read that was cut short.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities import changes as changes_module
from scone_memory.entities.changes import ChangesError, graph_changes
from scone_memory.testing import Clock

JAN = "2024-01-01T00:00:00Z"
MAR = "2024-03-01T00:00:00Z"
SINCE = "2024-02-01T00:00:00.000Z"
UNTIL = "2024-06-01T00:00:00.000Z"


async def history(store=None) -> MemoryEngine:
    """Alice moved from Acme to Globex and was promoted; Bob met Carol; Dan
    left Initech for nowhere; Erin arrived. All in March."""
    engine = await MemoryEngine(store or InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-01-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from=JAN)
    await engine.assert_fact("alpha", "alice chen", "works_at", "Globex", valid_from=MAR)
    await engine.assert_fact("alpha", "alice chen", "title", "engineer", valid_from=JAN)
    await engine.assert_fact("alpha", "alice chen", "title", "manager", valid_from=MAR)
    await engine.assert_fact("alpha", "bob stone", "knows", "Carol Diaz", valid_from=MAR)
    dan = await engine.assert_fact("alpha", "dan wu", "works_at", "Initech", valid_from=JAN)
    await engine.assert_fact("alpha", "dan wu", "lives_in", "Porto", valid_from=JAN)
    await engine.assert_fact("alpha", "erin fox", "works_at", "Globex", valid_from=MAR)
    engine.clock.now = "2024-03-01T00:00:00.000Z"
    await engine.close_fact("alpha", dan.fact_id, "left")
    engine.clock.now = "2025-01-01T00:00:00.000Z"
    return engine


def lines(text: str, prefix: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(prefix)]


async def test_a_claim_that_moved_is_one_change_citing_both_ends():
    engine = await history()
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL)
    moved = [change for change in found.changes if change["kind"] == "moved"]
    assert [(c["subject"]["key"], c["predicate"], c["before"]["label"], c["after"]["label"]) for c in moved] == [
        ("alice chen", "works_at", "Acme Robotics", "Globex")]
    assert len(moved[0]["fact_ids"]) == 2
    assert "moved: alice chen works_at Acme Robotics → Globex [facts 1, 2]" in found.text
    assert not any(c["kind"] in ("began", "ended") and c["subject"]["key"] == "alice chen" and c["predicate"] == "works_at"
                   for c in found.changes)


async def test_relations_that_began_and_ended_are_said_with_their_facts():
    engine = await history()
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL)
    began = {(c["subject"]["key"], c["predicate"]) for c in found.changes if c["kind"] == "began"}
    ended = {(c["subject"]["key"], c["predicate"]) for c in found.changes if c["kind"] == "ended"}
    assert began == {("bob stone", "knows"), ("erin fox", "works_at")} and ended == {("dan wu", "works_at")}
    assert lines(found.text, "ended: ") == ["ended: dan wu works_at Initech [fact 6]"]


async def test_a_value_that_changed_says_what_it_was_and_is():
    engine = await history()
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL)
    [value] = [c for c in found.changes if c["kind"] == "value"]
    assert (value["before"], value["after"]) == (["engineer"], ["manager"])
    assert lines(found.text, "value: ") == ["value: alice chen title engineer → manager [facts 3, 4]"]


async def test_entities_that_came_and_went_are_counted():
    engine = await history()
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL)
    assert sorted(found.entities["appeared"]) == ["bob stone", "carol diaz", "erin fox", "globex"]
    assert sorted(found.entities["gone"]) == ["acme robotics", "initech"]


async def test_an_unchanged_stretch_says_no_change():
    engine = await history()
    found = await graph_changes(engine, "alpha", since="2024-04-01T00:00:00Z", until=UNTIL)
    assert found.status == "unchanged" and found.changes == ()
    assert lines(found.text, "result: ") == ["result: no change"]


async def test_no_change_is_claimed_of_a_capped_read(monkeypatch):
    from scone_memory.entities import read

    monkeypatch.setattr(read, "MAX_FACTS", 2)
    engine = await history()
    found = await graph_changes(engine, "alpha", since="2024-04-01T00:00:00Z", until=UNTIL)
    assert found.status == "unknown" and found.changes == ()
    assert lines(found.text, "result: ") == [
        "result: unknown: the reads were cut short, and what changed cannot be told from what they left out"]


async def test_a_capped_read_invents_no_change_from_what_it_left_out(monkeypatch):
    """Alice worked at Acme from 2020, Globex from 2023 and Acme again from
    2024. Read whole, 2021 and 2025 agree; read cut to the newest fact, the
    2021 side lacks Acme, which is no evidence Acme began."""
    from scone_memory.entities import read

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-01-01T00:00:00.000Z")).open()
    for year, firm in ((2020, "Acme Company"), (2023, "Globex Company"), (2024, "Acme Company")):
        await engine.assert_fact("alpha", "alice chen", "works_at", firm, valid_from=f"{year}-01-01T00:00:00Z")
    whole = await graph_changes(engine, "alpha", since="2021-01-01T00:00:00Z", until="2025-01-01T00:00:00Z")
    assert whole.status == "unchanged"
    monkeypatch.setattr(read, "MAX_FACTS", 1)
    cut = await graph_changes(engine, "alpha", since="2021-01-01T00:00:00Z", until="2025-01-01T00:00:00Z")
    assert cut.status == "unknown" and cut.changes == ()
    assert cut.entities == {"appeared": None, "appeared_total": None, "gone": None, "gone_total": None}
    assert cut.coverage["withheld"] == ["began", "ended", "moved", "value", "appeared", "gone"]


async def test_a_whole_side_still_confirms_what_its_absence_shows(monkeypatch):
    """Only the read at until was cut short: what began is still known, for
    its absence at since was read whole; what ended is not."""
    engine = await history()
    real = changes_module.load_projection
    calls = []

    async def until_cut(*arguments, **options):
        projection, coverage = await real(*arguments, **options)
        calls.append(1)
        if len(calls) == 2:
            coverage = {**coverage, "reasons": ["fact_limit"]}
        return projection, coverage

    monkeypatch.setattr(changes_module, "load_projection", until_cut)
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL)
    kinds = {c["kind"] for c in found.changes}
    assert kinds == {"began"} and found.coverage["withheld"] == ["ended", "moved", "value", "gone"]
    assert found.entities["appeared_total"] == 4 and found.entities["gone"] is None
    assert lines(found.text, "withheld: ") == [
        "withheld: ended, moved and value changes and entities gone, for the read at until was cut short"]


async def test_changes_past_the_limit_are_counted():
    engine = await history()
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL, limit=2)
    assert len(found.changes) == 2 and "changes_cut 3" in found.coverage["reasons"]


async def test_the_changes_come_in_a_fixed_order():
    engine = await history()
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL)
    assert [c["kind"] for c in found.changes] == ["moved", "began", "began", "ended", "value"]


class ExcludesOnReread(InMemoryDocumentStore):
    targets: set[int] = set()
    engine = None

    async def get_fact(self, space, fact_id):
        if fact_id in self.targets:
            self.targets.discard(fact_id)
            await self.engine.exclude(space, fact_id, "retracted")
        return await super().get_fact(space, fact_id)


async def test_a_fact_withdrawn_while_the_answer_is_read_is_answered_as_it_now_stands():
    store = ExcludesOnReread()
    engine = await history(store)
    bob = next(fact for fact in await store.list_facts("alpha", include_closed=True) if fact.subject == "bob stone")
    store.engine, store.targets = engine, {bob.fact_id}
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL)
    assert not any(c["subject"]["key"] == "bob stone" for c in found.changes)
    assert found.coverage["reasons"] == [], "read again, and whole"


class StillRevision(ExcludesOnReread):
    """A store whose revision does not move when a fact is withdrawn."""

    async def revision(self, space):
        return 1


async def test_a_change_resting_on_a_fact_that_stopped_counting_is_dropped_where_the_revision_cannot_tell():
    store = StillRevision()
    engine = await history(store)
    bob = next(fact for fact in await store.list_facts("alpha", include_closed=True) if fact.subject == "bob stone")
    store.engine, store.targets = engine, {bob.fact_id}
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL)
    assert not any(c["subject"]["key"] == "bob stone" for c in found.changes)
    assert "stale_evidence 1" in found.coverage["reasons"]


class Backfills(InMemoryDocumentStore):
    """Asserts a backfill the first time (or, with ``every``, each time) the
    answer re-reads the target fact."""
    target = None
    engine = None
    every = False

    async def get_fact(self, space, fact_id):
        if fact_id == self.target:
            self.target = self.target if self.every else None
            year = 2020 if not self.every else 2000 + (await self.revision(space)) % 20
            await self.engine.assert_fact("alpha", "alice chen", "works_at", f"Acme {year}",
                                          valid_from=f"{year}-01-01T00:00:00Z")
        return await super().get_fact(space, fact_id)


async def test_a_backfill_written_while_the_answer_is_read_is_answered_again():
    """Acme from 2024 looks like it began since 2021, until a backfill
    written during the re-read says Acme held from 2020. The answer is
    computed again, and agrees with a fresh one: nothing changed."""
    store = Backfills()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-01-01T00:00:00.000Z")).open()
    late = await engine.assert_fact("alpha", "alice chen", "works_at", "Acme 2020", valid_from="2024-01-01T00:00:00Z")
    store.target, store.engine = late.fact_id, engine
    found = await graph_changes(engine, "alpha", since="2021-01-01T00:00:00Z", until="2025-01-01T00:00:00Z")
    fresh = await graph_changes(engine, "alpha", since="2021-01-01T00:00:00Z", until="2025-01-01T00:00:00Z")
    assert found.status == fresh.status == "unchanged" and found.changes == ()


async def test_a_ledger_that_never_stops_moving_is_answered_unknown():
    store = Backfills()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-01-01T00:00:00.000Z")).open()
    late = await engine.assert_fact("alpha", "alice chen", "works_at", "Acme 2020", valid_from="2024-01-01T00:00:00Z")
    store.target, store.engine, store.every = late.fact_id, engine, True
    found = await graph_changes(engine, "alpha", since="2021-01-01T00:00:00Z", until="2025-01-01T00:00:00Z")
    assert found.status == "unknown" and found.changes == ()
    assert "ledger_moved_during_read" in found.coverage["reasons"]
    assert found.coverage["withheld"] == ["began", "ended", "moved", "value", "appeared", "gone"]
    assert lines(found.text, "result: ") == ["result: unknown: the ledger changed while it was read; ask again"]


async def test_a_ledger_that_moved_between_the_two_reads_is_said(monkeypatch):
    engine = await history()
    real = changes_module.load_projection
    calls = []

    async def moving(*arguments, **options):
        projection, coverage = await real(*arguments, **options)
        calls.append(1)
        if len(calls) == 2:
            from dataclasses import replace

            projection = replace(projection, revision=projection.revision + 1)
        return projection, coverage

    monkeypatch.setattr(changes_module, "load_projection", moving)
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL)
    assert len(calls) == 4, "read again once the ledger was seen to move"
    assert found.status == "changed", "the second pair of reads agreed"


@pytest.mark.parametrize("options, message", [
    ({"since": UNTIL, "until": SINCE}, "before"), ({"since": UNTIL, "until": UNTIL}, "before"),
    ({"since": "soon"}, "RFC 3339"), ({"since": SINCE, "until": "later"}, "RFC 3339"),
    ({"since": SINCE, "limit": 0}, "limit"), ({"since": SINCE, "limit": 501}, "limit"),
    ({"since": SINCE, "max_bytes": 100}, "max_bytes"),
])
async def test_bounds_are_refused_before_anything_is_read(options, message):
    engine = await history()
    with pytest.raises(ChangesError, match=message):
        await graph_changes(engine, "alpha", **{"until": UNTIL, **options})


async def test_a_claim_that_only_became_an_entity_has_not_changed():
    """Alice has liked jazz since 2020. In 2023 a fact about jazz itself
    made it an entity, so the same claim is drawn as a relation where it
    was a value. That is how the graph shows it, not a change in it."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-01-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "alice chen", "likes", "jazz", valid_from="2020-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "jazz", "genre_of", "music", valid_from="2023-01-01T00:00:00Z")
    found = await graph_changes(engine, "alpha", since="2021-01-01T00:00:00Z", until="2024-01-01T00:00:00Z")
    assert [(c["kind"], c["subject"]["key"], c["predicate"]) for c in found.changes] == [("value", "jazz", "genre_of")]


async def test_a_move_from_a_value_to_an_entity_is_one_change():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-01-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "alice chen", "lives_in", "n/a", valid_from="2020-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "alice chen", "lives_in", "Lisbon", valid_from="2023-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "bob stone", "lives_in", "Lisbon", valid_from="2019-01-01T00:00:00Z")
    found = await graph_changes(engine, "alpha", since="2021-01-01T00:00:00Z", until="2024-01-01T00:00:00Z")
    [moved] = found.changes
    assert moved["kind"] == "moved" and moved["before"] == {"value": "n/a"} and moved["after"]["key"] == "lisbon"
    assert lines(found.text, "moved: ") == ['moved: alice chen lives_in "n/a" → Lisbon [facts 1, 2]']


async def test_a_value_whose_case_carries_meaning_changes_with_its_case():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-01-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "laptop", "memory", "512 MB", valid_from="2020-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "laptop", "memory", "512 mb", valid_from="2023-01-01T00:00:00Z")
    found = await graph_changes(engine, "alpha", since="2021-01-01T00:00:00Z", until="2024-01-01T00:00:00Z")
    assert lines(found.text, "value: ") == ["value: laptop memory 512 MB → 512 mb [facts 1, 2]"]


async def test_a_new_value_of_a_many_valued_predicate_begins_rather_than_moves():
    """Nothing moved: Alice knew Bob before and knows Carol as well now, so
    the change is one that began, and Bob's claim is not reported at all."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-01-01T00:00:00.000Z"), many_valued=["knows"]).open()
    await engine.assert_fact("alpha", "alice chen", "knows", "Bob Stone", valid_from=JAN)
    await engine.assert_fact("alpha", "alice chen", "knows", "Carol Diaz", valid_from=MAR)
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL)
    assert [(change["kind"], change["predicate"]) for change in found.changes] == [("began", "knows")]
    assert "began: alice chen knows Carol Diaz" in found.text


async def test_one_value_ending_and_another_beginning_is_not_a_claim_that_moved():
    """With a many-valued predicate the two are separate claims: Bob ended,
    Carol began, and Dana holds throughout. Nothing moved."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-01-01T00:00:00.000Z"), many_valued=["knows"]).open()
    bob = await engine.assert_fact("alpha", "alice chen", "knows", "Bob Stone", valid_from=JAN)
    await engine.assert_fact("alpha", "alice chen", "knows", "Dana Ruiz", valid_from=JAN)
    engine.clock.now = MAR
    await engine.close_fact("alpha", bob.fact_id, "lost touch")
    await engine.assert_fact("alpha", "alice chen", "knows", "Carol Diaz", valid_from=MAR)
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL)
    assert sorted((change["kind"], change["predicate"]) for change in found.changes) == [
        ("began", "knows"), ("ended", "knows")]
    assert "ended: alice chen knows Bob Stone" in found.text and "began: alice chen knows Carol Diaz" in found.text
    assert "moved" not in found.text and "Dana" not in found.text
