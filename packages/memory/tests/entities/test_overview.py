"""The graph at a glance, for questions about the whole of it.

"What are the main groups here?" names nothing, so no walk from a seed
answers it. GraphRAG answers such global questions from summaries of each
community, written in advance by a model. Here each community is digested
from the graph itself: its size and kinds, the predicates it is made of,
its central entities, and a few of its facts, cited and re-read now. A
question ranks the communities it concerns first and says why; the
caller's model reads the digests.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities.overview import OverviewError, graph_overview
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


async def seeded(store=None) -> MemoryEngine:
    """Three groups: a firm, a city and a lab."""
    engine = await MemoryEngine(store or InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    note = await engine.remember("alpha", "Alice Chen joined Acme Robotics.")
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from=DAY,
                             source_episode_id=note.episode_id, quote="Alice Chen joined Acme Robotics")
    for person in ("bob stone", "carol diaz", "dan wu"):
        await engine.assert_fact("alpha", person, "works_at", "Acme Robotics", valid_from=DAY)
    for person in ("erin fox", "frank li", "gina ro"):
        await engine.assert_fact("alpha", person, "lives_in", "Porto", valid_from=DAY)
    for person in ("hana kim", "ivan oz"):
        await engine.assert_fact("alpha", person, "studies_at", "Quantum Lab", valid_from=DAY)
    await engine.assert_fact("alpha", "hana kim", "knows", "ivan oz", valid_from=DAY)
    return engine


def lines(text: str, prefix: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(prefix)]


async def test_without_a_question_the_communities_come_largest_first_with_their_facts():
    engine = await seeded()
    overview = await graph_overview(engine, "alpha")
    assert overview.status == "prepared"
    labels = [community["label"].lower() for community in overview.communities]
    assert "acme robotics" in labels[0] and len(labels) == 3
    [first] = [line for line in lines(overview.text, "community: ") if "Acme Robotics" in line]
    assert "5 entities" in first and "works_at 4" in first
    assert lines(overview.text, "communities: ")[0].startswith("communities: 3 over 12 entities")
    assert 'quote verified: "Alice Chen joined Acme Robotics"' in overview.text
    assert all(community["fact_ids"] for community in overview.communities)
    assert lines(overview.text, "coverage: ") == ["coverage: complete"]


async def test_a_question_ranks_first_the_communities_it_concerns_and_says_why():
    engine = await seeded()
    named = await graph_overview(engine, "alpha", question="who studies with Hana Kim?")
    assert "hana kim" in named.communities[0]["label"].lower()
    assert named.communities[0]["matched"] == ["hana kim", "studies"]
    worded = await graph_overview(engine, "alpha", question="who lives in which city?")
    assert "porto" in worded.communities[0]["label"].lower() and worded.communities[0]["matched"] == ["lives"]
    assert "ordered by the question" in lines(worded.text, "communities: ")[0]


async def test_an_entity_the_question_names_outweighs_a_word_it_shares():
    """Hana Kim is named; "lives" is only a word Porto's predicates share.
    Porto is the larger, but the lab comes first."""
    engine = await seeded()
    overview = await graph_overview(engine, "alpha", question="where does hana kim spend time, and who lives nearby?")
    assert "hana kim" in overview.communities[0]["label"].lower()
    assert "porto" in overview.communities[1]["label"].lower()


async def test_an_unchanged_graph_is_analysed_once(monkeypatch):
    from scone_memory.entities import overview as overview_module

    calls = []
    real = overview_module.analyze_projection
    monkeypatch.setattr(overview_module, "analyze_projection", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(overview_module, "_ANALYSES", type(overview_module._ANALYSES)())
    engine = await seeded()
    await graph_overview(engine, "alpha")
    await graph_overview(engine, "alpha", question="who works at acme robotics?")
    assert len(calls) == 1
    await engine.assert_fact("alpha", "zed", "works_at", "Acme Robotics", valid_from=DAY)
    await graph_overview(engine, "alpha")
    assert len(calls) == 2, "a changed graph is analysed again"


async def test_entities_in_no_community_are_counted():
    engine = await seeded()
    await engine.assert_fact("alpha", "zed", "joined_on", "May 2021", valid_from=DAY)
    overview = await graph_overview(engine, "alpha")
    assert lines(overview.text, "communities: ")[0].endswith("; 1 more entity has no relation to another")
    assert overview.coverage["isolated_entities"] == 1


async def test_a_question_that_concerns_no_community_keeps_them_by_size():
    engine = await seeded()
    overview = await graph_overview(engine, "alpha", question="what are the main themes?")
    assert "acme robotics" in overview.communities[0]["label"].lower()
    assert all(community["matched"] == [] for community in overview.communities)
    assert "none named by the question; by size" in lines(overview.text, "communities: ")[0]


async def test_the_communities_left_out_are_counted():
    engine = await seeded()
    overview = await graph_overview(engine, "alpha", limit=1)
    assert len(overview.communities) == 1 and "communities_cut 2" in overview.coverage["reasons"]


class ExcludesOnReread(InMemoryDocumentStore):
    targets: set[int] = set()
    engine = None

    async def get_fact(self, space, fact_id):
        if fact_id in self.targets:
            self.targets.discard(fact_id)
            await self.engine.exclude(space, fact_id, "retracted")
        return await super().get_fact(space, fact_id)


async def test_a_fact_that_stopped_counting_is_not_cited():
    store = ExcludesOnReread()
    engine = await seeded(store)
    works = [fact.fact_id for fact in await store.list_facts("alpha", include_closed=True)
             if fact.predicate == "works_at"]
    store.engine, store.targets = engine, set(works)
    overview = await graph_overview(engine, "alpha", facts_each=2)
    acme = next(community for community in overview.communities if "acme robotics" in community["label"].lower())
    assert acme["fact_ids"] == [] and any(reason.startswith("stale_evidence ") for reason in overview.coverage["reasons"])


async def test_the_text_fits_its_budget():
    engine = await seeded()
    overview = await graph_overview(engine, "alpha", facts_each=10, max_bytes=600)
    assert len(overview.text.encode("utf-8")) <= 600 and lines(overview.text, "omitted: ")


async def test_a_graph_with_no_links_has_no_communities_to_show():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock()).open()
    await engine.assert_fact("alpha", "alice chen", "joined_on", "May 2021", valid_from=DAY)
    overview = await graph_overview(engine, "alpha")
    assert overview.status == "empty" and overview.communities == ()
    assert lines(overview.text, "communities: ") == ["communities: none; 1 entity has no relation to another"]


async def test_a_capped_read_is_said(monkeypatch):
    from scone_memory.entities import read

    monkeypatch.setattr(read, "MAX_FACTS", 3)
    overview = await graph_overview(await seeded(), "alpha")
    assert overview.coverage["read"]["truncated"] is True and lines(overview.text, "coverage: ")[0].startswith(
        "coverage: limited: fact_limit")


@pytest.mark.parametrize("options, message", [
    ({"limit": 0}, "limit"), ({"limit": 51}, "limit"), ({"facts_each": 11}, "facts_each"),
    ({"facts_each": -1}, "facts_each"), ({"question": "q" * 2001}, "question"), ({"question": ""}, "question"),
    ({"resolution": 0}, "resolution"), ({"resolution": 11}, "resolution"), ({"max_bytes": 100}, "max_bytes"),
])
async def test_bounds_are_refused_before_anything_is_read(options, message):
    engine = await seeded()
    with pytest.raises(OverviewError, match=message):
        await graph_overview(engine, "alpha", **options)
