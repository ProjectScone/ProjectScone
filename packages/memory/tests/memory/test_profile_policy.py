"""What a profile is made of, and in what order.

A profile answers "who is this space about" without being asked a
question. What belongs in one is the owner's choice, so which predicates
count is configured, never guessed; and what leads it is what the space
says most often and most recently, not whichever claim was written
first. Every item keeps the evidence it rests on, and the read behind it
is bounded and said.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.memory.catalog import ProfilePolicy
from scone_memory.testing import Clock

SPACE = "alpha"


async def engine(**options) -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              clock=Clock("2025-06-01T00:00:00.000Z"), **options).open()


async def test_what_the_space_says_most_often_leads_the_profile():
    """A claim restated three times leads one stated once, however new the
    second is: a profile is what holds, by how often it is said."""
    memory = await engine()
    for day in ("2021-01-01", "2022-01-01", "2023-01-01"):
        await memory.assert_fact(SPACE, "mark", "prefers", "dark mode", valid_from=f"{day}T00:00:00Z")
    await memory.assert_fact(SPACE, "mark", "uses", "vim", valid_from="2024-01-01T00:00:00Z")
    profile = await memory.profile(SPACE)
    assert [(fact.predicate, fact.object) for fact in profile.static_facts] == [
        ("prefers", "dark mode"), ("uses", "vim")]
    assert profile.coverage["restatements"]["1"] == 2, "the leading claim was stated twice again"


async def test_the_newer_of_two_claims_said_as_often_leads():
    memory = await engine()
    await memory.assert_fact(SPACE, "mark", "uses", "vim", valid_from="2021-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "mark", "drinks", "coffee", valid_from="2024-01-01T00:00:00Z")
    profile = await memory.profile(SPACE)
    assert [fact.predicate for fact in profile.static_facts] == ["drinks", "uses"]


async def test_only_the_predicates_a_policy_names_are_profiled():
    memory = await engine(profile_policy=ProfilePolicy.of(predicates=["prefers", "uses"]))
    await memory.assert_fact(SPACE, "mark", "prefers", "dark mode", valid_from="2024-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "mark", "uses", "vim", valid_from="2024-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "mark", "met", "a stranger on the train", valid_from="2024-02-01T00:00:00Z")
    profile = await memory.profile(SPACE)
    assert sorted(fact.predicate for fact in profile.static_facts) == ["prefers", "uses"]
    assert profile.coverage["policy"] == {"predicates": ["prefers", "uses"], "without": []}


async def test_a_policy_may_instead_say_which_predicates_never_count():
    memory = await engine(profile_policy=ProfilePolicy.of(without=["met"]))
    await memory.assert_fact(SPACE, "mark", "prefers", "dark mode", valid_from="2024-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "mark", "met", "a stranger on the train", valid_from="2024-02-01T00:00:00Z")
    profile = await memory.profile(SPACE)
    assert [fact.predicate for fact in profile.static_facts] == ["prefers"]


async def test_a_predicate_is_named_as_the_ledger_stores_it():
    memory = await engine(profile_policy=ProfilePolicy.of(predicates=["  Prefers "]))
    assert memory.profile_policy.predicates == frozenset({"prefers"})
    await memory.assert_fact(SPACE, "mark", "PREFERS", "dark mode", valid_from="2024-01-01T00:00:00Z")
    assert len((await memory.profile(SPACE)).static_facts) == 1


@pytest.mark.parametrize("options, message", [
    ({"predicates": "prefers"}, "collection"),
    ({"without": [3]}, "predicate"),
])
def test_a_malformed_policy_is_refused(options, message):
    with pytest.raises(InvalidInput, match=message):
        ProfilePolicy.of(**options)


async def test_a_claim_that_stopped_holding_is_not_profiled():
    memory = await engine()
    old = await memory.assert_fact(SPACE, "mark", "lives_in", "Porto", valid_from="2020-01-01T00:00:00Z")
    await memory.close_fact(SPACE, old.fact_id, "moved")
    hidden = await memory.assert_fact(SPACE, "mark", "met", "Mallory", valid_from="2024-01-01T00:00:00Z")
    await memory.exclude(SPACE, hidden.fact_id, "private")
    await memory.assert_fact(SPACE, "mark", "knows", "Bob", proposed=True)
    await memory.assert_fact(SPACE, "mark", "uses", "vim", valid_from="2024-01-01T00:00:00Z")
    assert [fact.predicate for fact in (await memory.profile(SPACE)).static_facts] == ["uses"]


async def test_the_read_behind_a_profile_is_bounded_and_said(monkeypatch):
    from scone_memory.memory import catalog

    monkeypatch.setattr(catalog, "MAX_PROFILE_FACTS", 1)
    memory = await engine()
    await memory.assert_fact(SPACE, "mark", "uses", "vim", valid_from="2024-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "mark", "drinks", "coffee", valid_from="2024-02-01T00:00:00Z")
    profile = await memory.profile(SPACE)
    assert profile.coverage["facts_read"] == 1 and "fact_limit" in profile.coverage["reasons"]


async def test_every_item_keeps_the_evidence_it_rests_on():
    memory = await engine()
    said = await memory.remember(SPACE, "Mark prefers dark mode in every editor.")
    await memory.assert_fact(SPACE, "mark", "prefers", "dark mode", valid_from="2024-01-01T00:00:00Z",
                             source_episode_id=said.episode_id, quote="Mark prefers dark mode")
    [fact] = (await memory.profile(SPACE)).static_facts
    assert fact.source_episode_id == said.episode_id and fact.quote == "Mark prefers dark mode"


async def test_the_environment_configures_which_predicates_count():
    from scone_memory.runtime.config import Settings, build_engine

    settings = Settings.from_env({"SCONE_PROFILE_PREDICATES": "prefers, uses", "SCONE_PROFILE_WITHOUT": "met"})
    assert settings.profile_predicates == ("prefers", "uses") and settings.profile_without == ("met",)
    built = await build_engine(settings)
    assert built.profile_policy.predicates == frozenset({"prefers", "uses"})
    assert built.profile_policy.without == frozenset({"met"})
    assert (await build_engine(Settings.from_env({}))).profile_policy == ProfilePolicy()


async def test_the_newest_claims_are_the_ones_whose_restatements_are_counted(monkeypatch):
    """Counting restatements is a read per claim, so only the newest are
    counted; the claims cut are the oldest, never the newest."""
    from scone_memory.memory import catalog

    monkeypatch.setattr(catalog, "MAX_PROFILE_CANDIDATES", 1)
    memory = await engine()
    await memory.assert_fact(SPACE, "mark", "used", "emacs", valid_from="2019-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "mark", "uses", "vim", valid_from="2024-01-01T00:00:00Z")
    profile = await memory.profile(SPACE)
    assert [fact.predicate for fact in profile.static_facts] == ["uses"]
    assert profile.coverage["candidates"] == 1
