"""One source, followed through: its sections, chunks, the claims quoting
it with exact byte spans, and the entities those claims name.

Spans are UTF-8 byte offsets into the unchanged source, as chunks are, so
content.encode()[start:end] is exactly the quote even past non-ASCII text.
Mentions are names of known entities found in the text, reported apart
from claims: a name appearing is not the source asserting anything.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.entities.sources import sources_view
from scone_memory.testing import Clock

NOTE = """# Café Nova

Café Nova opened in Lisboa — the owners are Ana Reis and Rui Costa.

## People

Ana Reis runs Café Nova. Ana Reis runs Café Nova. Rui Costa knows Bruno Alves.
"""


def auth() -> dict:
    return {"Authorization": "Bearer key-a"}


@pytest.fixture
async def source():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                chunk_target=64, clock=Clock("2025-06-01T00:00:00.000Z")).open()
    note = await engine.remember("alpha", NOTE)
    runs = await engine.assert_fact("alpha", "ana reis", "runs", "Café Nova", source_episode_id=note.episode_id,
                                    quote="Ana Reis runs Café Nova.", valid_from="2024-01-01T00:00:00Z")
    opened = await engine.assert_fact("alpha", "café nova", "located_in", "Lisboa", source_episode_id=note.episode_id,
                                      quote="Café Nova opened in Lisboa", valid_from="2024-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "bruno alves", "lives_in", "Porto", valid_from="2024-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "owners", "count", "2", valid_from="2024-01-01T00:00:00Z")
    yield engine, note, runs, opened
    await engine.close()


async def test_every_claim_quote_is_located_by_exact_bytes(source):
    engine, note, runs, opened = source
    view = await sources_view(engine, "alpha", note.episode_id)
    body = NOTE.encode()
    claims = {claim["fact_id"]: claim for claim in view["claims"]}
    for claim in claims.values():
        span = claim["span"]
        assert body[span["start"]:span["end"]].decode() == claim["quote"]
    assert claims[runs.fact_id]["occurrences"] == 2 and claims[opened.fact_id]["occurrences"] == 1
    assert claims[runs.fact_id]["section"] == "People"


async def test_sections_come_from_headings_and_chunks_sit_inside_them(source):
    engine, note, _, _ = source
    view = await sources_view(engine, "alpha", note.episode_id)
    assert [section["title"] for section in view["sections"]] == ["Café Nova", "People"]
    assert view["chunks"] and all(chunk["section"] in ("Café Nova", "People") for chunk in view["chunks"])
    plain = await engine.remember("alpha", "Just a line of plain text, no headings.")
    assert (await sources_view(engine, "alpha", plain.episode_id))["sections"] == []


async def test_claims_name_their_entities_and_mentions_stay_apart(source):
    engine, note, runs, _ = source
    view = await sources_view(engine, "alpha", note.episode_id)
    named = {entity["key"] for entity in view["entities"]}
    assert {"ana reis", "café nova", "lisboa"} <= named
    mentioned = {mention["key"] for mention in view["mentions"]}
    assert "bruno alves" in mentioned and "ana reis" not in mentioned
    assert "owners" not in mentioned  # a lowercase single word in running text is not taken as a name
    bruno = next(mention for mention in view["mentions"] if mention["key"] == "bruno alves")
    assert NOTE.encode()[bruno["span"]["start"]:bruno["span"]["end"]].decode() == "Bruno Alves"


async def test_caps_are_reported(source):
    engine, note, _, _ = source
    view = await sources_view(engine, "alpha", note.episode_id, max_chunks=1, max_claims=1)
    assert len(view["chunks"]) == 1 and len(view["claims"]) == 1
    assert {"chunk_limit", "claim_limit"} <= set(view["coverage"]["reasons"])


async def test_the_sources_route_answers_404_and_410_like_an_episode(source):
    engine, note, _, _ = source
    other = await engine.remember("alpha", "Something to forget later.")
    await engine.forget("alpha", other.episode_id)
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        found = client.get("/v1/graph/sources", params={"episode": note.episode_id}, headers=auth())
        missing = client.get("/v1/graph/sources", params={"episode": 9999}, headers=auth())
        gone = client.get("/v1/graph/sources", params={"episode": other.episode_id}, headers=auth())
        features = client.get("/v1/capabilities", headers=auth()).json()["features"]
    assert found.status_code == 200 and found.json()["episode"]["id"] == note.episode_id
    assert missing.status_code == 404 and gone.status_code == 410 and features["graph.sources"] is True



class NoClaimReader(InMemoryDocumentStore):
    facts_for_graph = None


async def test_a_store_that_cannot_list_a_sources_claims_says_so():
    engine = await MemoryEngine(NoClaimReader(), InMemoryVectorIndex(), HashEmbedder()).open()
    note = await engine.remember("alpha", NOTE)
    view = await sources_view(engine, "alpha", note.episode_id)
    assert view["claims"] == [] and "claims_unavailable" in view["coverage"]["reasons"]


async def test_the_view_names_the_exact_content_it_located_spans_in(source):
    import hashlib

    engine, note, _, _ = source
    view = await sources_view(engine, "alpha", note.episode_id)
    assert view["episode"]["content_sha256"] == hashlib.sha256(NOTE.encode()).hexdigest()


class WritesDuringClaims(InMemoryDocumentStore):
    """Adds a claim citing the source just after its claims are read."""
    engine = None
    episode_id = None

    async def facts_for_graph(self, space, source_episode_id, limit):
        rows = await super().facts_for_graph(space, source_episode_id, limit)
        if self.engine is not None:  # just after the claims were read: a view stopping here would miss it
            engine, self.engine = self.engine, None
            await engine.assert_fact(space, "rui costa", "knows", "Bruno Alves", source_episode_id=self.episode_id,
                                     quote="Rui Costa knows Bruno Alves.", valid_from="2024-01-01T00:00:00Z")
        return rows


async def test_a_view_reads_again_when_the_space_moves_while_it_reads():
    store = WritesDuringClaims()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(), chunk_target=64).open()
    note = await engine.remember("alpha", NOTE)
    store.engine, store.episode_id = engine, note.episode_id
    view = await sources_view(engine, "alpha", note.episode_id)
    assert view["consistent"] is True and [claim["predicate"] for claim in view["claims"]] == ["knows"]


async def test_a_capped_projection_read_shows_in_the_views_coverage(source, monkeypatch):
    from scone_memory.entities import read

    engine, note, _, _ = source
    monkeypatch.setattr(read, "MAX_FACTS", 1)
    view = await sources_view(engine, "alpha", note.episode_id)
    assert "fact_limit" in view["coverage"]["reasons"] and view["coverage"]["truncated"] is True
