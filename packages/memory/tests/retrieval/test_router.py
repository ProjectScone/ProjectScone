"""One question, and the machinery that suits it.

A framework that has grown a computed temporal answer, an entity graph
and two retrieval lanes has a new problem: a caller must know which to
ask. LlamaIndex answers that with a model writing a plan. This answers it
with a rule written down, which can be read, argued with and measured —
and which says in every answer which way it went and why, so a wrong
route is visible rather than mysterious.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.retrieval.router import ROUTES, answer_question
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
SPACE = "alpha"
NOW = "2024-07-01T00:00:00Z"


async def memory() -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                               clock=Clock("2024-07-01T00:00:00.000Z")).open()
    told = await engine.remember(SPACE, "I moved to Lisbon on 2024-03-02.",
                                 created_at="2024-03-02T00:00:00Z")
    await engine.remember(SPACE, "The harbour crane was repainted, and it took four days.",
                          created_at="2024-05-01T00:00:00Z")
    await engine.assert_fact(SPACE, "alice chen", "works_at", "Acme Robotics", valid_from="2024-01-01T00:00:00Z",
                             source_episode_id=told.episode_id)
    await engine.assert_fact(SPACE, "acme robotics", "based_in", "Lisbon", valid_from="2024-01-01T00:00:00Z")
    return engine


async def test_a_question_about_dates_goes_to_the_one_that_computes():
    answered = await answer_question(await memory(), SPACE, "How long ago did I move to Lisbon?", now=NOW)
    assert answered.route == "temporal"
    assert "days" in answered.text and answered.why


async def test_a_question_about_something_the_graph_knows_goes_to_the_graph():
    answered = await answer_question(await memory(), SPACE, "What is Alice Chen's employer?", now=NOW)
    assert answered.route == "graph"
    assert "Acme Robotics" in answered.text
    assert "alice chen" in answered.why.lower()


async def test_anything_else_is_an_ordinary_search():
    answered = await answer_question(await memory(), SPACE, "what happened with the crane", now=NOW)
    assert answered.route == "recall"
    assert "crane" in answered.text and "nothing it could compute" in answered.why


async def test_a_date_question_the_ledger_cannot_ground_falls_through_to_search():
    """The rule is not "looks temporal, answer temporally". A question the
    computer cannot ground is better served by the passages than by a
    refusal, and the answer says that is what happened."""
    answered = await answer_question(await memory(), SPACE, "How long ago did I visit Oslo?", now=NOW)
    assert answered.route == "recall"
    assert "could not ground" in answered.why


async def test_every_answer_says_which_way_it_went_and_why():
    engine = await memory()
    for question in ("How long ago did I move to Lisbon?", "What is Alice Chen's employer?",
                     "what happened with the crane"):
        answered = await answer_question(engine, SPACE, question, now=NOW)
        assert answered.route in ROUTES and answered.why
        record = answered.record(SPACE)
        assert record["route"] == answered.route and record["why"] == answered.why
        assert record["question"] == question


async def test_a_caller_can_insist_on_one_route():
    """A rule that cannot be overridden is a rule somebody will work
    around. Naming a route says which, and the answer says it was asked
    for rather than chosen."""
    answered = await answer_question(await memory(), SPACE, "What is Alice Chen's employer?",
                                     now=NOW, route="recall")
    assert answered.route == "recall" and "asked for" in answered.why


async def test_a_route_that_is_not_one_is_refused():
    from scone_memory.core.errors import InvalidInput

    with pytest.raises(InvalidInput, match="route"):
        await answer_question(await memory(), SPACE, "anything", now=NOW, route="telepathy")


async def test_the_command_line_answers_a_question_and_says_how():
    import io

    from scone_memory.runtime.cli import build_parser, run

    engine = await memory()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--space", SPACE, "answer", "How long ago did I move to Lisbon?"]),
                     engine, io.StringIO(""), out)
    said = out.getvalue()
    assert code == 0, said
    assert "route: temporal" in said and "days" in said


async def test_the_command_line_can_insist_on_a_route():
    import io

    from scone_memory.runtime.cli import build_parser, run

    engine = await memory()
    out = io.StringIO()
    await run(build_parser().parse_args(
        ["--space", SPACE, "answer", "How long ago did I move to Lisbon?", "--route", "recall"]),
        engine, io.StringIO(""), out)
    assert "route: recall" in out.getvalue()


def test_a_question_over_http_says_the_route_it_took():
    import asyncio

    from fastapi.testclient import TestClient

    from scone_memory.api import create_app

    async def ready():
        return await memory()

    engine = asyncio.run(ready())
    with TestClient(create_app(engine, {"key-a": SPACE})) as client:
        said = client.get("/v1/answer", params={"q": "What is Alice Chen's employer?", "now": NOW},
                          headers={"Authorization": "Bearer key-a"}).json()
        assert said["route"] == "graph" and said["why"] and "Acme Robotics" in said["text"]
        insisted = client.get("/v1/answer", params={"q": "What is Alice Chen's employer?", "route": "recall"},
                              headers={"Authorization": "Bearer key-a"}).json()
        assert insisted["route"] == "recall" and "asked for" in insisted["why"]
