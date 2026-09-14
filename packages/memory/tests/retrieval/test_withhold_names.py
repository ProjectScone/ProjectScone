"""Withholding the names of people, organisations and places the space's graph knows.

The net withheld addresses, numbers and secrets, and nothing that names
someone: "Alice Chen works at Acme Robotics in Lisbon" came back whole
however it was asked for. A model could guess at names; the space already
knows the ones it holds, in its graph. Here ``person``, ``organisation``
and ``place`` withhold every spelling the graph records for an entity of
that kind, matched on word boundaries without regard to case, across the
same surfaces as every other kind. The report counts the names it knew and
says, as for every kind, that a name the graph does not hold is not found.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.entities.read import load_projection
from scone_memory.retrieval.withhold import NAME_KINDS, chosen_kinds, known_names, withhold

DAY = "2024-01-01T00:00:00Z"
NOTE = "Dr. Alice Chen joined Acme Robotics in Lisbon; ALICE CHEN leads the crane survey. Aliceville is a town."


async def engine_with_names() -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    note = await engine.remember("default", NOTE)
    await engine.assert_fact("default", "alice chen", "works_at", "Acme Robotics", source_episode_id=note.episode_id,
                             quote="Dr. Alice Chen joined Acme Robotics", valid_from=DAY)
    await engine.assert_fact("default", "acme robotics", "based_in", "Lisbon", valid_from=DAY)
    return engine


def test_the_name_kinds_are_chosen_like_any_other_and_need_names_to_apply():
    assert NAME_KINDS == ("person", "organisation", "place")
    assert chosen_kinds(["email", "person"]) == ("email", "person")
    with pytest.raises(InvalidInput, match="graph"):
        withhold([], kinds=["person"])


async def test_every_spelling_the_graph_holds_is_withheld_on_word_boundaries_in_items_and_facts():
    engine = await engine_with_names()
    try:
        projection, _ = await load_projection(engine, "default", mode="current")
        names = known_names(projection, ["person", "organisation", "place"])
        result = await engine.recall("default", "who joined acme robotics", limit=5)
    finally:
        await engine.close()
    assert "alice chen" in {name.lower() for name in names["person"]} and "acme robotics" in {
        name.lower() for name in names["organisation"]}
    held = withhold(result.items, facts=result.facts, kinds=["person", "organisation", "place"], names=names)
    [item] = [item for item in held.items if "crane survey" in item.text]
    assert "Alice Chen" not in item.text and "ALICE CHEN" not in item.text and "Lisbon" not in item.text
    assert item.text.count("[withheld: person]") == 2 and "[withheld: organisation]" in item.text
    assert "Aliceville is a town" in item.text, "a name inside a longer word is left alone"
    assert all("alice chen" not in fact.subject.lower() for fact in held.facts)
    assert held.names_known["person"] >= 1 and held.by_kind["person"] >= 2
    assert "not in the graph" in held.why


def test_a_longer_name_is_withheld_before_a_shorter_one_inside_it():
    from scone_memory.core.models import RecallItem

    item = RecallItem(chunk_id=1, episode_id=1, text="Acme Robotics Lisbon sent Acme a note.", score=1.0,
                      created_at=DAY)
    held = withhold([item], kinds=["organisation"], names={"organisation": ("Acme", "Acme Robotics Lisbon")})
    assert held.items[0].text == "[withheld: organisation] sent [withheld: organisation] a note."


def test_a_name_is_matched_whole_across_any_spacing_and_never_inside_a_word():
    from scone_memory.core.models import RecallItem

    item = RecallItem(chunk_id=1, episode_id=1, text="Acmeville and SuperAcme are towns; ACME\n  Robotics is not.",
                      score=1.0, created_at=DAY)
    held = withhold([item], kinds=["organisation"], names={"organisation": ("Acme", "Acme Robotics")})
    assert held.items[0].text == "Acmeville and SuperAcme are towns; [withheld: organisation] is not."


def test_a_spelling_too_short_to_tell_from_a_word_is_not_used_and_is_counted():
    from scone_memory.core.models import RecallItem

    item = RecallItem(chunk_id=1, episode_id=1, text="Al met Bo at the lab.", score=1.0, created_at=DAY)
    held = withhold([item], kinds=["person"], names={"person": ("Al", "Bo", "Bob")})
    assert held.items[0].text == "Al met Bo at the lab." and held.names_skipped == 2


async def test_names_are_withheld_over_http_and_on_the_command_line():
    from scone_memory.api import create_app
    from scone_memory.runtime.cli import build_parser, run

    engine = await engine_with_names()
    try:
        with TestClient(create_app(engine, {"key": "default"})) as client:
            answer = client.get("/v1/recall", params={"q": "who joined acme robotics", "withhold": "person,place"},
                                headers={"authorization": "Bearer key"})
        out = io.StringIO()
        code = await run(build_parser().parse_args(["recall", "who joined acme robotics", "--withhold", "person",
                                                    "--json"]), engine, io.StringIO(""), out)
    finally:
        await engine.close()
    body = answer.json()
    assert answer.status_code == 200 and "Alice Chen" not in json.dumps(body["items"]) + json.dumps(body["facts"])
    assert body["withheld"]["names_known"]["person"] >= 1
    assert code == 0 and "Alice Chen" not in out.getvalue()
