"""Attributing an answer to stored passages, over HTTP and the command line.

The passages are named by chunk id and read from the space, so a caller
cannot attribute an answer to text the space does not hold, and an id the
space does not hold is named rather than skipped.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.runtime.cli import build_parser, run

NOTE = "Priya moved the launch to March because the audit ran late."
OTHER = "Tomas said the audit found two billing errors in February."


def auth(key: str = "key-a") -> dict:
    return {"Authorization": f"Bearer {key}"}


async def stored():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    first = await engine.remember("alpha", NOTE)
    second = await engine.remember("alpha", OTHER)
    elsewhere = await engine.remember("beta", "The venue in beta is secret and should never be read from alpha.")
    ids = [(await engine.documents.chunks_of(space, added.episode_id))[0].chunk_id
           for space, added in (("alpha", first), ("alpha", second), ("beta", elsewhere))]
    return engine, ids


@pytest.mark.asyncio
async def test_an_answer_is_attributed_to_the_chunks_named():
    engine, (note, other, beta) = await stored()
    answer = "Priya moved the launch to March because the audit ran late. The board was not told."
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        made = client.post("/v1/answers/attribute", json={"answer": answer, "chunk_ids": [note, other, beta, 999]},
                           headers=auth())
        features = client.get("/v1/capabilities", headers=auth()).json()["features"]
        refused = client.post("/v1/answers/attribute", json={"answer": "", "chunk_ids": [note]}, headers=auth())
        empty = client.post("/v1/answers/attribute", json={"answer": answer, "chunk_ids": []}, headers=auth())
    await engine.close()
    assert made.status_code == 200, made.text
    body = made.json()
    assert [s["status"] for s in body["sentences"]] == ["quoted", "unattributed"]
    assert body["sentences"][0]["passage"] == f"chunk:{note}"
    # Another space's chunk is not read, and is named with the one that does not exist.
    assert body["chunks_missing"] == [beta, 999] and body["verified_accuracy"] is False
    assert features["answers.attribution"] is True
    assert refused.status_code == 422 and empty.status_code == 422


@pytest.mark.asyncio
async def test_the_command_line_attributes_an_answer_to_chunks():
    engine, (note, other, _) = await stored()
    out = io.StringIO()
    asked = ["--space", "alpha", "attribute", "--answer", f"{NOTE} Nobody asked the board.", "--chunk", str(note), "--chunk", str(other)]
    code = await run(build_parser().parse_args(asked), engine, io.StringIO(""), out)
    as_json = io.StringIO()
    json_code = await run(build_parser().parse_args([*asked, "--json"]), engine, io.StringIO(""), as_json)
    await engine.close()
    text = out.getvalue()
    assert code == 0 and f"quoted chunk:{note}: {NOTE}" in text and "unattributed: Nobody asked the board." in text
    assert "not a check that the answer is true" in text
    assert json_code == 0 and json.loads(as_json.getvalue())["counts"]["quoted"] == 1


@pytest.mark.asyncio
async def test_naming_no_chunk_is_refused():
    from scone_memory.core.errors import InvalidInput
    from scone_memory.retrieval.attribution import attribute_to_chunks

    engine, _ = await stored()
    try:
        with pytest.raises(InvalidInput):
            await attribute_to_chunks(engine, "alpha", NOTE, [])
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_a_chunk_holding_only_space_is_named_not_dropped():
    """A chunk the space holds whose text is blank cannot be attributed to;
    it is named apart from the ids the space does not hold, never left out."""
    engine, (note, other, _) = await stored()
    reading = engine.documents.get_chunks

    async def blank_other(space, ids):
        return [chunk.model_copy(update={"text": "  \n "}) if chunk.chunk_id == other else chunk for chunk in await reading(space, ids)]

    engine.documents.get_chunks = blank_other
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        made = client.post("/v1/answers/attribute", json={"answer": NOTE, "chunk_ids": [note, other, 999]},
                           headers=auth())
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--space", "alpha", "attribute", "--answer", NOTE, "--chunk",
                                                 str(other)]), engine, io.StringIO(""), out)
    await engine.close()
    assert made.status_code == 200, made.text
    assert made.json()["chunks_empty"] == [other] and made.json()["chunks_missing"] == [999]
    assert made.json()["sentences"][0]["passage"] == f"chunk:{note}"
    assert code == 0 and f"blank in this space: chunk {other}" in out.getvalue()
    assert "unattributed" in out.getvalue()
