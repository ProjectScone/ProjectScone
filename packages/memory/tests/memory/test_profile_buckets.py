"""A profile in two buckets: what stays true, and what is true lately.

A profile answers "who is this about" in one list. Some of what it holds
is settled -- a name, a role, a diet -- and some is what the subject is in
the middle of -- this quarter's project, this week's city. The two are
used differently: a settled claim colours every answer, a current one
matters while it is current. So a profile can be read in two buckets.

Which bucket a claim goes in is decided by a rule the ledger can check,
never by what its words seem to mean:

- an explicit per-predicate override, first;
- then how often the subject's value for that predicate changed lately:
  a slot that keeps changing is dynamic, however long this value held;
- then how long this claim has held: long enough is static, less is dynamic.

The dynamic bucket is ordered by weight: how often a claim was stated,
decaying with the time since it was last stated; the static bucket is not. Each bucket has its own
count and byte bound and says when either cut it, and each shown claim
names the rule that placed it. Off unless asked for: today's profile is
unchanged.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.core.timeutil import parse_rfc3339
from scone_memory.core.validation import normalise_time
from scone_memory.memory.catalog import BucketBounds, BucketRules
from scone_memory.testing import Clock

SPACE = "alpha"
NOW = "2025-06-01T00:00:00.000Z"


def ago(days: float, clock: Clock | None = None) -> str:
    when = parse_rfc3339(clock.now if clock else NOW) - timedelta(days=days)
    return normalise_time(when.isoformat())


async def engine(clock: Clock | None = None, **options) -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              clock=clock or Clock(NOW), **options).open()


def placed(profile, fact) -> dict:
    return profile.buckets.coverage["placed"][str(fact.fact_id)]


async def test_buckets_are_off_unless_asked_for_and_the_profile_is_as_it_was():
    memory = await engine()
    await memory.assert_fact(SPACE, "mark", "name", "Mark", valid_from=ago(1000))
    profile = await memory.profile(SPACE)
    assert profile.buckets is None
    assert sorted(profile.coverage) == ["candidates", "facts_read", "policy", "reasons", "restatements", "revision"]


async def test_a_claim_held_long_is_static_and_a_new_one_is_dynamic():
    memory = await engine()
    name = await memory.assert_fact(SPACE, "mark", "name", "Mark", valid_from=ago(1000))
    project = await memory.assert_fact(SPACE, "mark", "works_on", "the crane survey", valid_from=ago(10))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert [f.fact_id for f in profile.buckets.static] == [name.fact_id]
    assert [f.fact_id for f in profile.buckets.dynamic] == [project.fact_id]
    assert placed(profile, name) == {"bucket": "static", "rule": "tenure", "held_days": 1000.0, "changes": 0}
    assert placed(profile, project)["rule"] == "tenure" and placed(profile, project)["held_days"] == 10.0
    third = await memory.assert_fact(SPACE, "mark", "reads", "a novel", valid_from=ago(1 / 3))
    fresh = placed(await memory.profile(SPACE, buckets=BucketBounds()), third)
    assert fresh["held_days"] == 0.33 and fresh["weight"] == 0.9923
    # The profile itself is the same profile.
    assert {f.fact_id for f in profile.static_facts} == {name.fact_id, project.fact_id}


async def test_a_claim_is_static_from_the_day_it_has_held_long_enough():
    """The testing clock moves; the claim does not. It settles on the day
    its tenure reaches the threshold, not the day after."""
    clock = Clock(NOW)
    memory = await engine(clock, profile_bucket_rules=BucketRules.of(static_after_days=30))
    fact = await memory.assert_fact(SPACE, "mark", "lives_in", "Lisbon", valid_from=ago(29))
    assert [f.fact_id for f in (await memory.profile(SPACE, buckets=BucketBounds())).buckets.dynamic] == [fact.fact_id]
    clock.now = ago(-1)
    settled = await memory.profile(SPACE, buckets=BucketBounds())
    assert [f.fact_id for f in settled.buckets.static] == [fact.fact_id]
    assert placed(settled, fact)["held_days"] == 30.0


async def test_a_slot_that_keeps_changing_is_dynamic_however_long_this_value_held():
    memory = await engine(profile_bucket_rules=BucketRules.of(static_after_days=90, dynamic_changes=2))
    await memory.assert_fact(SPACE, "mark", "works_on", "the bridge", valid_from=ago(330))
    await memory.assert_fact(SPACE, "mark", "works_on", "the tunnel", valid_from=ago(300))
    current = await memory.assert_fact(SPACE, "mark", "works_on", "the crane survey", valid_from=ago(200))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert [f.fact_id for f in profile.buckets.dynamic] == [current.fact_id]
    assert placed(profile, current) == {"bucket": "dynamic", "rule": "changes", "held_days": 200.0, "changes": 2,
                                        "last_stated": ago(200), "weight": placed(profile, current)["weight"]}


async def test_the_first_value_a_slot_holds_is_not_a_change():
    memory = await engine(profile_bucket_rules=BucketRules.of(static_after_days=90, dynamic_changes=2))
    await memory.assert_fact(SPACE, "mark", "works_on", "the tunnel", valid_from=ago(300))
    current = await memory.assert_fact(SPACE, "mark", "works_on", "the crane survey", valid_from=ago(200))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert [f.fact_id for f in profile.buckets.static] == [current.fact_id]
    assert placed(profile, current)["changes"] == 1 and placed(profile, current)["rule"] == "tenure"


async def test_changes_before_the_window_are_not_counted():
    memory = await engine(profile_bucket_rules=BucketRules.of(dynamic_changes=1, change_window_days=365))
    await memory.assert_fact(SPACE, "mark", "lives_in", "Porto", valid_from=ago(900))
    current = await memory.assert_fact(SPACE, "mark", "lives_in", "Lisbon", valid_from=ago(400))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert [f.fact_id for f in profile.buckets.static] == [current.fact_id]
    assert placed(profile, current)["changes"] == 0


async def test_a_change_on_the_first_day_of_the_window_is_not_counted():
    memory = await engine(profile_bucket_rules=BucketRules.of(dynamic_changes=1, change_window_days=100))
    await memory.assert_fact(SPACE, "mark", "lives_in", "Porto", valid_from=ago(300))
    current = await memory.assert_fact(SPACE, "mark", "lives_in", "Lisbon", valid_from=ago(100))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert placed(profile, current) == {"bucket": "static", "rule": "tenure", "held_days": 100.0, "changes": 0}
    later = await engine(profile_bucket_rules=BucketRules.of(dynamic_changes=1, change_window_days=100.5))
    await later.assert_fact(SPACE, "mark", "lives_in", "Porto", valid_from=ago(300))
    moved = await later.assert_fact(SPACE, "mark", "lives_in", "Lisbon", valid_from=ago(100))
    assert placed(await later.profile(SPACE, buckets=BucketBounds()), moved)["changes"] == 1


async def test_a_value_backfilled_later_counts_where_it_falls_in_time():
    """Changes are counted in the order values held, not the order they
    were written: a value recorded afterwards for an earlier time is a
    change at its own time."""
    memory = await engine(profile_bucket_rules=BucketRules.of(dynamic_changes=1, change_window_days=365))
    current = await memory.assert_fact(SPACE, "mark", "lives_in", "Lisbon", valid_from=ago(100))
    await memory.assert_fact(SPACE, "mark", "lives_in", "Porto", valid_from=ago(400))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert placed(profile, current)["changes"] == 1 and placed(profile, current)["rule"] == "changes"


async def test_the_changes_rule_reads_only_the_slots_of_the_claims_it_places(monkeypatch):
    """A ledger read holds up to 20,000 facts and a profile places at most
    200 claims: history is read for those claims' slots, not for all."""
    from scone_memory.memory import profile_buckets

    memory = await engine()
    for index in range(30):
        ended = await memory.assert_fact(SPACE, "mark", f"once_{index}", "yes", valid_from=ago(500))
        await memory.close_fact(SPACE, ended.fact_id, "over")
    kept = await memory.assert_fact(SPACE, "mark", "name", "Mark", valid_from=ago(500))
    parsed: list[str] = []
    real = profile_buckets.parse_rfc3339
    monkeypatch.setattr(profile_buckets, "parse_rfc3339", lambda text: parsed.append(text) or real(text))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert [f.fact_id for f in profile.buckets.static] == [kept.fact_id]
    assert len(parsed) < 30, "every closed slot's history was read"


async def test_an_excluded_value_is_still_history():
    """Exclusion hides a claim; it does not unmake the change it was."""
    memory = await engine(profile_bucket_rules=BucketRules.of(dynamic_changes=1))
    old = await memory.assert_fact(SPACE, "mark", "lives_in", "Porto", valid_from=ago(300))
    current = await memory.assert_fact(SPACE, "mark", "lives_in", "Lisbon", valid_from=ago(200))
    await memory.exclude(SPACE, old.fact_id, "private")
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert placed(profile, current)["changes"] == 1


async def test_a_value_superseded_at_the_instant_it_began_never_held_and_is_not_a_change():
    """Corrected twice on the same day it was stated: the ledger closes the
    first two at the instant they began, so only the last value ever held."""
    memory = await engine()
    for city in ("Porto", "Oporto"):
        await memory.assert_fact(SPACE, "mark", "lives_in", city, valid_from=ago(200))
    current = await memory.assert_fact(SPACE, "mark", "lives_in", "Lisbon", valid_from=ago(200))
    ledger = await memory.documents.list_facts(SPACE, include_closed=True)
    assert sorted((f.valid_from == f.valid_until, f.object) for f in ledger) == [
        (False, "Lisbon"), (True, "Oporto"), (True, "Porto")]
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert placed(profile, current) == {"bucket": "static", "rule": "tenure", "held_days": 200.0, "changes": 0}


async def test_a_value_is_held_since_it_began_across_rows_that_meet():
    """The ledger may split one unbroken value into rows that meet: backfilled
    to an earlier start, or stated again from the instant it was closed."""
    memory = await engine()
    current = await memory.assert_fact(SPACE, "mark", "lives_in", "Lisbon", valid_from=ago(10))
    await memory.assert_fact(SPACE, "mark", "lives_in", "Lisbon", valid_from=ago(500))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert placed(profile, current) == {"bucket": "static", "rule": "tenure", "held_days": 500.0, "changes": 0}

    clock = Clock(ago(300))
    memory = await engine(clock)
    first = await memory.assert_fact(SPACE, "mark", "lives_in", "Lisbon", valid_from=ago(500))
    await memory.close_fact(SPACE, first.fact_id, "ended")
    clock.now = NOW
    again = await memory.assert_fact(SPACE, "mark", "lives_in", "Lisbon", valid_from=ago(300))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert placed(profile, again)["held_days"] == 500.0


async def test_a_value_is_held_since_it_resumed_not_since_it_first_held():
    """Vegetarian, vegan, then vegetarian again from 100 days ago, closed and
    stated again from that instant 20 days ago: the two latest rows meet, so
    the value has held 100 days; the vegan row breaks it from the first."""
    clock = Clock(ago(20))
    memory = await engine(clock, profile_bucket_rules=BucketRules.of(dynamic_changes=3))
    await memory.assert_fact(SPACE, "mark", "diet", "vegetarian", valid_from=ago(500))
    await memory.assert_fact(SPACE, "mark", "diet", "vegan", valid_from=ago(300))
    resumed = await memory.assert_fact(SPACE, "mark", "diet", "vegetarian", valid_from=ago(100))
    await memory.close_fact(SPACE, resumed.fact_id, "lapsed")
    clock.now = NOW
    current = await memory.assert_fact(SPACE, "mark", "diet", "vegetarian", valid_from=ago(20))
    ledger = await memory.documents.list_facts(SPACE, include_closed=True)
    assert sorted((f.valid_from, f.valid_until, f.object) for f in ledger) == [
        (ago(500), ago(300), "vegetarian"), (ago(300), ago(100), "vegan"), (ago(100), ago(20), "vegetarian"),
        (ago(20), None, "vegetarian")]
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert placed(profile, current) == {"bucket": "static", "rule": "tenure", "held_days": 100.0, "changes": 2}


async def test_a_proposed_value_is_not_a_change():
    memory = await engine(profile_bucket_rules=BucketRules.of(dynamic_changes=1))
    current = await memory.assert_fact(SPACE, "mark", "lives_in", "Lisbon", valid_from=ago(400))
    await memory.assert_fact(SPACE, "mark", "lives_in", "Faro", valid_from=ago(10), proposed=True)
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert placed(profile, current) == {"bucket": "static", "rule": "tenure", "held_days": 400.0, "changes": 0}


async def test_the_same_value_stated_again_after_a_gap_is_not_a_change():
    clock = Clock(ago(200))
    memory = await engine(clock, profile_bucket_rules=BucketRules.of(dynamic_changes=1))
    first = await memory.assert_fact(SPACE, "mark", "diet", "vegetarian", valid_from=ago(300))
    await memory.close_fact(SPACE, first.fact_id, "lapsed")
    clock.now = NOW
    again = await memory.assert_fact(SPACE, "mark", "diet", "vegetarian", valid_from=ago(100))
    assert again.fact_id != first.fact_id
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert placed(profile, again) == {"bucket": "static", "rule": "tenure", "held_days": 100.0, "changes": 0}


async def test_each_value_of_a_many_valued_predicate_is_its_own_slot():
    memory = await engine(many_valued=["knows"], profile_bucket_rules=BucketRules.of(dynamic_changes=2))
    await memory.assert_fact(SPACE, "mark", "knows", "Alice", valid_from=ago(300))
    await memory.assert_fact(SPACE, "mark", "knows", "Bob", valid_from=ago(250))
    carol = await memory.assert_fact(SPACE, "mark", "knows", "Carol", valid_from=ago(200))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert len(profile.buckets.static) == 3
    assert placed(profile, carol)["changes"] == 0


async def test_an_override_beats_changes_and_tenure_both_ways():
    rules = BucketRules.of(static_predicates=["works_on"], dynamic_predicates=["name"], dynamic_changes=1)
    memory = await engine(profile_bucket_rules=rules)
    name = await memory.assert_fact(SPACE, "mark", "name", "Mark", valid_from=ago(1000))
    await memory.assert_fact(SPACE, "mark", "works_on", "the tunnel", valid_from=ago(20))
    project = await memory.assert_fact(SPACE, "mark", "works_on", "the crane survey", valid_from=ago(5))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert [f.fact_id for f in profile.buckets.static] == [project.fact_id]
    assert [f.fact_id for f in profile.buckets.dynamic] == [name.fact_id]
    assert placed(profile, project)["rule"] == placed(profile, name)["rule"] == "override"
    assert profile.buckets.coverage["rules"]["static_predicates"] == ["works_on"]


async def test_the_dynamic_bucket_decays_by_recency_and_the_static_one_does_not():
    """Stated three times two months ago against once yesterday: static
    order is by restatement, dynamic order lets the older evidence fade."""
    both = ["often", "fresh"]
    for bucket in ("static", "dynamic"):
        clock = Clock(ago(80))
        rules = BucketRules.of(**{f"{bucket}_predicates": both}, half_life_days=30)
        memory = await engine(clock, profile_bucket_rules=rules)
        often = await memory.assert_fact(SPACE, "mark", "often", "said", valid_from=ago(80))
        for days in (70, 60):
            clock.now = ago(days)
            await memory.assert_fact(SPACE, "mark", "often", "said", valid_from=ago(days))
        clock.now = NOW
        fresh = await memory.assert_fact(SPACE, "mark", "fresh", "said", valid_from=ago(1))
        profile = await memory.profile(SPACE, buckets=BucketBounds())
        shown = [f.fact_id for f in getattr(profile.buckets, bucket)]
        if bucket == "static":
            assert shown == [often.fact_id, fresh.fact_id]
            assert "weight" not in placed(profile, often)
        else:
            assert shown == [fresh.fact_id, often.fact_id]
            assert placed(profile, often)["weight"] == 0.75 and placed(profile, often)["last_stated"] == ago(60)


async def test_a_dynamic_claim_stated_again_lately_is_recent_again():
    clock = Clock(ago(200))
    memory = await engine(clock, profile_bucket_rules=BucketRules.of(dynamic_predicates=["old", "new"]))
    old = await memory.assert_fact(SPACE, "mark", "old", "said", valid_from=ago(200))
    clock.now = NOW
    await memory.assert_fact(SPACE, "mark", "old", "said", valid_from=ago(2))
    new = await memory.assert_fact(SPACE, "mark", "new", "said", valid_from=ago(1))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert [f.fact_id for f in profile.buckets.dynamic] == [old.fact_id, new.fact_id]
    assert placed(profile, old)["last_stated"] == ago(2)


async def test_a_restatement_dated_after_now_is_not_when_a_claim_was_last_stated():
    memory = await engine(profile_bucket_rules=BucketRules.of(dynamic_predicates=["plans"]))
    fact = await memory.assert_fact(SPACE, "mark", "plans", "a trip", valid_from=ago(10))
    for days in (-5, -30):
        await memory.assert_fact(SPACE, "mark", "plans", "a trip", valid_from=ago(days))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert placed(profile, fact)["last_stated"] == ago(10)
    # Neither is a restatement yet, so neither adds weight: stated once, ten days ago.
    assert placed(profile, fact)["weight"] == round(0.5 ** (10 / 30), 4)


async def test_each_bucket_has_its_own_count_bound_and_says_when_it_cut():
    memory = await engine()
    for index in range(3):
        await memory.assert_fact(SPACE, "mark", f"settled_{index}", "yes", valid_from=ago(500 + index))
    await memory.assert_fact(SPACE, "mark", "lately", "yes", valid_from=ago(3))
    profile = await memory.profile(SPACE, buckets=BucketBounds(static_limit=2, dynamic_limit=2))
    static, dynamic = profile.buckets.coverage["static"], profile.buckets.coverage["dynamic"]
    assert len(profile.buckets.static) == 2 and len(profile.buckets.dynamic) == 1
    assert (static["candidates"], static["shown"], static["omitted"], static["cut"]) == (3, 2, 1, "count")
    assert (dynamic["shown"], dynamic["omitted"], dynamic["cut"]) == (1, 0, None)
    assert static["limit"] == 2 and dynamic["limit"] == 2
    exact = await memory.profile(SPACE, buckets=BucketBounds(static_limit=3))
    assert exact.buckets.coverage["static"]["cut"] is None


async def test_each_bucket_has_its_own_byte_bound_and_says_when_it_cut():
    memory = await engine()
    for index in range(3):
        await memory.assert_fact(SPACE, "mark", f"lately_{index}", "x" * 40, valid_from=ago(index + 1))
    await memory.assert_fact(SPACE, "mark", "settled", "y" * 40, valid_from=ago(500))
    # Each record is 161 bytes, so two fill 325 exactly: 2 brackets, 161, a comma, 161.
    profile = await memory.profile(SPACE, buckets=BucketBounds(dynamic_max_bytes=325))
    dynamic, static = profile.buckets.coverage["dynamic"], profile.buckets.coverage["static"]
    assert (dynamic["shown"], dynamic["omitted"], dynamic["cut"]) == (2, 1, "bytes")
    assert dynamic["bytes"] == 325 and dynamic["max_bytes"] == 325
    records = [profile.buckets.record(fact) for fact in profile.buckets.dynamic]
    assert dynamic["bytes"] == len(json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode())
    assert static["cut"] is None and static["shown"] == 1
    roomy = await memory.profile(SPACE, buckets=BucketBounds(dynamic_max_bytes=16000))
    assert roomy.buckets.coverage["dynamic"]["cut"] is None and roomy.buckets.coverage["dynamic"]["shown"] == 3


async def test_a_bucket_record_names_its_rule():
    memory = await engine()
    fact = await memory.assert_fact(SPACE, "mark", "name", "Mark", valid_from=ago(1000))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert profile.buckets.record(fact) == {"fact_id": fact.fact_id, "subject": "mark", "predicate": "name",
                                            "object": "Mark", "valid_from": ago(1000), "rule": "tenure"}


@pytest.mark.parametrize("include, static, dynamic", [("static", 1, 0), ("dynamic", 0, 1), ("both", 1, 1)])
async def test_a_reader_may_ask_for_one_bucket_or_both(include, static, dynamic):
    memory = await engine()
    await memory.assert_fact(SPACE, "mark", "name", "Mark", valid_from=ago(1000))
    await memory.assert_fact(SPACE, "mark", "works_on", "the crane survey", valid_from=ago(10))
    profile = await memory.profile(SPACE, buckets=BucketBounds(include=include))
    assert (len(profile.buckets.static), len(profile.buckets.dynamic)) == (static, dynamic)
    assert ("static" in profile.buckets.coverage, "dynamic" in profile.buckets.coverage) == (bool(static), bool(dynamic))
    assert len(profile.buckets.coverage["placed"]) == static + dynamic


async def test_claims_past_the_candidate_bound_are_said_to_be_cut(monkeypatch):
    from scone_memory.memory import catalog

    memory = await engine()
    await memory.assert_fact(SPACE, "mark", "name", "Mark", valid_from=ago(1000))
    await memory.assert_fact(SPACE, "mark", "works_on", "the crane survey", valid_from=ago(10))
    assert (await memory.profile(SPACE, buckets=BucketBounds())).buckets.coverage["candidates_truncated"] is False
    monkeypatch.setattr(catalog, "MAX_PROFILE_CANDIDATES", 2)
    assert (await memory.profile(SPACE, buckets=BucketBounds())).buckets.coverage["candidates_truncated"] is False
    monkeypatch.setattr(catalog, "MAX_PROFILE_CANDIDATES", 1)
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert profile.buckets.coverage["candidates_truncated"] is True
    assert len(profile.buckets.static) + len(profile.buckets.dynamic) == 1


async def test_only_claims_that_hold_are_bucketed():
    memory = await engine()
    closed = await memory.assert_fact(SPACE, "mark", "lives_in", "Porto", valid_from=ago(900))
    await memory.close_fact(SPACE, closed.fact_id, "moved")
    hidden = await memory.assert_fact(SPACE, "mark", "met", "Mallory", valid_from=ago(900))
    await memory.exclude(SPACE, hidden.fact_id, "private")
    kept = await memory.assert_fact(SPACE, "mark", "name", "Mark", valid_from=ago(900))
    profile = await memory.profile(SPACE, buckets=BucketBounds())
    assert [f.fact_id for f in profile.buckets.static + profile.buckets.dynamic] == [kept.fact_id]


@pytest.mark.parametrize("options, message", [
    ({"static_predicates": ["name"], "dynamic_predicates": [" Name "]}, "both"),
    ({"static_predicates": "name"}, "collection"),
    ({"static_after_days": -1}, "static_after_days"),
    ({"static_after_days": float("inf")}, "static_after_days"),
    ({"dynamic_changes": 0}, "dynamic_changes"),
    ({"dynamic_changes": 1.5}, "dynamic_changes"),
    ({"dynamic_changes": True}, "dynamic_changes"),
    ({"change_window_days": 0}, "change_window_days"),
    ({"half_life_days": 0}, "half_life_days"),
    ({"half_life_days": float("nan")}, "half_life_days"),
    ({"half_life_days": True}, "half_life_days"),
    ({"change_window_days": "365"}, "change_window_days"),
])
def test_malformed_rules_are_refused(options, message):
    with pytest.raises(InvalidInput, match=message):
        BucketRules.of(**options)


@pytest.mark.parametrize("options, message", [
    ({"include": "all"}, "include"),
    ({"static_limit": 0}, "static_limit"),
    ({"dynamic_limit": 51}, "dynamic_limit"),
    ({"static_limit": True}, "static_limit"),
    ({"static_max_bytes": 99}, "static_max_bytes"),
    ({"dynamic_max_bytes": 16001}, "dynamic_max_bytes"),
    ({"static_max_bytes": 2000.0}, "static_max_bytes"),
])
def test_malformed_bounds_are_refused(options, message):
    with pytest.raises(InvalidInput, match=message):
        BucketBounds(**options)


def test_bounds_at_their_limits_are_accepted():
    BucketBounds(static_limit=1, dynamic_limit=50, static_max_bytes=100, dynamic_max_bytes=16000)
    BucketRules.of(static_after_days=0, dynamic_changes=1)
