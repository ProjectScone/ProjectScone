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
    assert found.status == "unchanged" and lines(found.text, "result: ") == ["result: no change among the facts read"]


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


async def test_a_change_resting_on_a_fact_that_stopped_counting_is_dropped():
    store = ExcludesOnReread()
    engine = await history(store)
    bob = next(fact for fact in await store.list_facts("alpha", include_closed=True) if fact.subject == "bob stone")
    store.engine, store.targets = engine, {bob.fact_id}
    found = await graph_changes(engine, "alpha", since=SINCE, until=UNTIL)
    assert not any(c["subject"]["key"] == "bob stone" for c in found.changes)
    assert "stale_evidence 1" in found.coverage["reasons"]


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
    assert "ledger_moved_between_reads" in found.coverage["reasons"]


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
