"""A community's report: sentences that each quote a record, stale the moment the graph moves."""

from __future__ import annotations

import json
import re

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.entities.analysis import cached_analysis
from scone_memory.entities.read import load_projection
from scone_memory.entities.reports import ReportCache, community_report, space_report
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


class CitingChat:
    """Quotes the opening words of every passage it is shown; folds by citing every note."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if "Notes:" in user:
            ids = re.findall(r"^\[(n\d+)\]", user, flags=re.M)
            return json.dumps({"summary": [{"sentence": "Everything the notes say, together.", "notes": ids}]})
        rows = [{"sentence": f"A note on {m.group(1)}.", "passage": m.group(1), "quote": " ".join(m.group(2).split()[:3])}
                for m in re.finditer(r"^\[((?:fact|quote|report|chunk):[^\]]+)\] (.+)$", user, flags=re.M)]
        return json.dumps({"notes": rows})


async def seeded() -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    note = await engine.remember("alpha", "Alice Chen joined Acme Robotics.")
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from=DAY,
                             source_episode_id=note.episode_id, quote="Alice Chen joined Acme Robotics")
    for person in ("bob stone", "carol diaz", "dan wu"):
        await engine.assert_fact("alpha", person, "works_at", "Acme Robotics", valid_from=DAY)
    for person in ("erin fox", "frank li", "gina ro"):
        await engine.assert_fact("alpha", person, "lives_in", "Porto", valid_from=DAY)
    return engine


async def acme_id(engine: MemoryEngine) -> str:
    projection, _ = await load_projection(engine, "alpha", mode="current")
    analysis = cached_analysis(projection)
    return next(c.community_id for c in analysis.communities if "acme" in c.label.lower())


async def test_a_report_cites_the_community_s_claims_and_the_quotes_the_code_re_read():
    engine = await seeded()
    report = await community_report(engine, CitingChat(), "alpha", await acme_id(engine))
    assert report.status == "synthesized" and "Acme" in report.label
    cited = {c.passage_id for s in report.synthesis.sentences for c in s.citations}
    assert any(p.startswith("fact:") for p in cited) and any(p.startswith("quote:") for p in cited)
    assert report.facts_total == 4 and len(report.facts_cited) == 4 and report.stale_evidence == 0
    record = report.record()
    assert record["verified_accuracy"] is False and record["projection_digest"] and record["revision"] >= 1
    assert "Acme Robotics" in report.question


async def test_a_report_is_stale_the_moment_the_graph_moves_and_the_cache_rebuilds_it():
    engine = await seeded()
    cache = ReportCache()
    community = await acme_id(engine)
    first, cached = await cache.get(engine, CitingChat(), "alpha", community)
    assert cached is False
    again, cached = await cache.get(engine, CitingChat(), "alpha", community)
    assert cached is True and again is first
    await engine.assert_fact("alpha", "eve adams", "works_at", "Acme Robotics", valid_from=DAY)
    projection, _ = await load_projection(engine, "alpha", mode="current")
    assert first.current(projection) is False
    # A community's id is made of its members, so the grown community is a
    # new id; the old report is stale and the old id names nothing now.
    with pytest.raises(InvalidInput, match="no community"):
        await cache.get(engine, CitingChat(), "alpha", community)
    grown = await acme_id(engine)
    assert grown != community
    rebuilt, cached = await cache.get(engine, CitingChat(), "alpha", grown)
    assert cached is False and rebuilt.revision > first.revision and rebuilt.facts_total == 5


async def test_an_unknown_community_and_bounds_are_refused():
    engine = await seeded()
    with pytest.raises(InvalidInput, match="no community"):
        await community_report(engine, CitingChat(), "alpha", "nope")
    with pytest.raises(InvalidInput, match="facts_each"):
        await community_report(engine, CitingChat(), "alpha", await acme_id(engine), facts_each=0)
    with pytest.raises(InvalidInput, match="over the bound"):
        await space_report(engine, CitingChat(), "alpha", max_reports=1)


async def test_a_space_report_is_a_synthesis_over_the_community_reports_cited_by_community():
    engine = await seeded()
    made = await space_report(engine, CitingChat(), "alpha", max_reports=5)
    assert made.communities_total == 2 and len(made.reports) == 2
    cited = {c.passage_id for s in made.synthesis.sentences for c in s.citations}
    assert cited and all(p.startswith("report:") for p in cited)
    assert made.record()["communities_reported"] == 2 and made.record()["verified_accuracy"] is False


async def test_a_quote_that_no_longer_verifies_is_counted_not_shown():
    engine = await seeded()
    community = await acme_id(engine)
    note = (await engine.documents.recent_episodes("alpha", 1))[0]
    await engine.forget("alpha", note.episode_id)
    report = await community_report(engine, CitingChat(), "alpha", community)
    cited = {c.passage_id for s in report.synthesis.sentences for c in s.citations}
    assert not any(p.startswith("quote:") for p in cited)
    assert report.stale_evidence >= 1 or report.facts_total < 4
