"""What an agent may narrow a search by, and what it is told when it cannot.

The engine has always been able to answer "only files, only since March,
only where the status is published". A model calling search_memory could
ask for none of it, so it had to read everything and decide for itself —
which is the expensive, unreliable half of retrieval done in the wrong
place. It can ask now, and a filter this space cannot answer is refused
in words rather than quietly ignored, because a filter ignored is worse
than a filter refused: the model believes it narrowed and it did not.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.integrations.tools import ToolBox
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
SPACE = "alpha"


async def box() -> ToolBox:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    await engine.remember(SPACE, "The harbour crane was repainted in March.", kind="note",
                          source="notes/march.md", created_at="2024-03-02T00:00:00Z",
                          metadata={"status": "published"})
    await engine.remember(SPACE, "The harbour crane was inspected in July.", kind="file",
                          source="reports/july.pdf", created_at="2024-07-02T00:00:00Z",
                          metadata={"status": "draft"})
    return ToolBox(engine, SPACE)


def texts(answer: dict) -> list[str]:
    return [item["text"] for item in answer["items"]]


async def test_an_agent_can_ask_for_one_kind_of_thing():
    found = await (await box()).run("search_memory", {"query": "harbour crane", "kind": "file"})
    assert found["ok"] and texts(found) == ["The harbour crane was inspected in July."]


async def test_an_agent_can_ask_for_a_stretch_of_time():
    tools = await box()
    since = await tools.run("search_memory", {"query": "harbour crane", "since": "2024-06-01T00:00:00Z"})
    assert texts(since) == ["The harbour crane was inspected in July."]
    until = await tools.run("search_memory", {"query": "harbour crane", "until": "2024-06-01T00:00:00Z"})
    assert texts(until) == ["The harbour crane was repainted in March."]


async def test_an_agent_can_ask_for_one_place_it_came_from():
    found = await (await box()).run("search_memory", {"query": "harbour crane",
                                                      "source_prefix": "reports/"})
    assert texts(found) == ["The harbour crane was inspected in July."]


async def test_an_agent_can_ask_by_what_was_recorded_beside_it():
    found = await (await box()).run("search_memory", {"query": "harbour crane",
                                                      "where": {"status": "published"}})
    assert texts(found) == ["The harbour crane was repainted in March."]


async def test_an_agent_asking_for_a_kind_that_is_not_one_is_refused_in_words():
    found = await (await box()).run("search_memory", {"query": "harbour crane", "kind": "telegram"})
    assert found["ok"] is False and "kind" in found["error"]


async def test_an_agent_asking_for_a_moment_that_is_not_one_is_refused():
    found = await (await box()).run("search_memory", {"query": "harbour crane", "since": "last tuesday"})
    assert found["ok"] is False and "since" in found["error"]


async def test_the_answer_says_what_it_was_narrowed_by():
    """A model that cannot see the filter in the answer cannot tell a narrow
    search from an empty space."""
    found = await (await box()).run("search_memory", {"query": "harbour crane", "kind": "file"})
    assert found["narrowed"] == {"kind": "file"}
    wide = await (await box()).run("search_memory", {"query": "harbour crane"})
    assert wide["narrowed"] == {}


async def test_the_command_line_can_ask_for_the_lanes_and_budgets_http_can():
    """The route and the engine both take an entity lane, a candidate
    budget and a reranker switch; the command line could ask for none of
    them. This is plumbing, so it is tested as plumbing: what the command
    line was asked for is what the engine is called with."""
    import io

    from scone_memory.runtime.cli import build_parser, run

    engine = (await box()).engine
    asked: dict = {}
    real = engine.recall

    async def recording(space, query, **options):
        asked.update(options)
        return await real(space, query, **options)

    engine.recall = recording  # type: ignore[method-assign]
    out = io.StringIO()
    code = await run(build_parser().parse_args(
        ["--space", SPACE, "recall", "harbour crane", "--graph-boost",
         "--candidate-limit", "50", "--no-rerank"]), engine, io.StringIO(""), out)
    assert code == 0, out.getvalue()
    assert asked["graph_boost"] is True and asked["candidate_limit"] == 50 and asked["rerank"] is False
    assert "harbour crane" in out.getvalue()


async def test_the_command_line_still_asks_for_the_defaults_when_it_is_not_told():
    import io

    from scone_memory.runtime.cli import build_parser, run

    engine = (await box()).engine
    asked: dict = {}
    real = engine.recall

    async def recording(space, query, **options):
        asked.update(options)
        return await real(space, query, **options)

    engine.recall = recording  # type: ignore[method-assign]
    await run(build_parser().parse_args(["--space", SPACE, "recall", "harbour crane"]),
              engine, io.StringIO(""), io.StringIO())
    assert asked["graph_boost"] is False and asked["candidate_limit"] is None and asked["rerank"] is True
