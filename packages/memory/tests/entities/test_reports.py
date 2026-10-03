"""A community's report: sentences that each quote a record, stale the moment the graph moves."""

from __future__ import annotations

import asyncio
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


class NamingChat:
    """Names a group after its first central member's employer or town; scripted
    answers otherwise: an exception is raised, a number is seconds slept."""

    def __init__(self, answers=None) -> None:
        self.answers, self.asked = list(answers or []), []

    async def complete(self, system: str, user: str) -> str:
        self.asked.append((system, user))
        if self.answers:
            answer = self.answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            if isinstance(answer, float):
                await asyncio.sleep(answer)
                return "Slow name"
            return answer
        return "Acme Robotics staff" if "Acme" in user else "People in Porto"


async def test_a_model_names_the_largest_communities_beside_their_computed_labels():
    from scone_memory.entities.reports import MAX_NAMES, as_community_name, name_communities

    engine = await seeded()
    chat = NamingChat()
    named = await name_communities(engine, chat, "alpha")
    projection, _ = await load_projection(engine, "alpha", mode="current")
    assert named.projection_digest == projection.digest and named.communities_total == len(cached_analysis(projection).communities)
    by_label = {name.label: name for name in named.names}
    acme = next(name for name in named.names if "acme" in name.label.lower())
    assert acme.name == "Acme Robotics staff" and acme.why == "named" and "Acme Robotics" in acme.shown
    assert all(name.label for name in named.names), "the computed label is always there beside the model's name"
    assert all("name alone" in system for system, _ in chat.asked) and all("Relations inside it" in user for _, user in chat.asked)
    record = named.record()
    assert record["communities_named"] == len(named.names) and record["verified_accuracy"] is False and "not facts" in record["note"]
    scripted = NamingChat(["", "Two lines\nof answer", "This is a long explanation of what the group is about, in prose",
                           "\"Quoted name.\"", "Fine"])
    second = await name_communities(engine, scripted, "alpha", max_names=2)
    assert [n.name for n in second.names] == [None, None] and "nothing" in second.names[0].why and "one line" in second.names[1].why
    assert as_community_name("This is a long explanation of what the group is about, in prose") == (None, "the model answered 13 words, longer than a name")
    long_words = "Antidisestablishmentarianism Counterrevolutionary Extraterritoriality"
    assert as_community_name(long_words) == (None, f"the model answered {len(long_words)} characters, longer than a name"), \
        "three words over the character bound blame the characters, not the words"
    assert as_community_name("Acme staff\rPeople in Porto") == (None, "the model answered more than one line"), \
        "any line break is more than one line, not only a newline"
    assert as_community_name("\"Quoted name.\"") == ("Quoted name", "named") and as_community_name(" Fine ") == ("Fine", "named")
    assert len(second.names) == 2, "the bound holds"
    assert record["communities_asked"] == len(named.names) and record["communities_timed_out"] == 0 == record["communities_failed"]
    with pytest.raises(InvalidInput, match=f"1 to {MAX_NAMES}"):
        await name_communities(engine, chat, "alpha", max_names=MAX_NAMES + 1)
    with pytest.raises(InvalidInput, match="timeout_s"):
        await name_communities(engine, chat, "alpha", timeout_s=0)
    await engine.close()


async def test_a_model_failure_or_the_deadline_leaves_that_community_unnamed_and_the_rest_named():
    from scone_memory.entities.reports import name_communities

    engine = await seeded()
    broken = NamingChat(["Acme staff", RuntimeError("boom")])
    named = await name_communities(engine, broken, "alpha", max_names=2)
    assert [n.name for n in named.names] == ["Acme staff", None] and named.names[1].why == "the model failed: RuntimeError"
    record = named.record()
    assert record["communities_asked"] == 2 and record["communities_named"] == 1 and record["communities_failed"] == 1
    slow = NamingChat([0.5, "Fine"])
    named = await name_communities(engine, slow, "alpha", max_names=2, timeout_s=0.1)
    assert [n.name for n in named.names] == [None, None]
    assert named.names[0].why == "timeout: the 0.1s deadline passed while the model was asked"
    assert named.names[1].why == "timeout: the 0.1s deadline passed before the model was asked"
    assert len(slow.asked) == 1, "a pass past its deadline asks the model nothing more"
    record = named.record()
    assert record["communities_asked"] == 1 and record["communities_timed_out"] == 2 and record["timeout_s"] == 0.1
    await engine.close()


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
