"""Compressing a widened recall, over HTTP and the command line.

A window of sentences is asked for with the recall, and compress cuts
what the window added back to the sentences that bear on the question.
Asked for without a window there is nothing it may cut, so the request
is refused rather than answered unchanged as though it had been applied.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.compress import GAP
from scone_memory.runtime.cli import build_parser, run

TEXT = ("The harbour board met on Monday to plan the summer. "
        "The crane survey was booked for the third of May. "
        "Dr. Okafor found rust on the jib. "
        "It needed grease and a new slew ring. "
        "The canteen menu changed in April. "
        "The yard repaired the jib before June. "
        "Parking passes are renewed each winter.")
ASKED = {"q": "slew ring grease crane", "limit": 1, "window": 3, "window_unit": "sentences"}


def auth() -> dict:
    return {"authorization": "Bearer key-a"}


async def engine() -> MemoryEngine:
    # Chunks of a sentence or so, so a window has sentences to add around the hit.
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=45).open()


def check_quoted(body: dict) -> None:
    """Every run is the episode's own bytes, and the text is the runs joined by the marker."""
    [item] = body["items"]
    raw = TEXT.encode()
    [runs] = body["compressed"]["runs"].values()
    assert item["text"] == GAP.join(raw[start:end].decode() for start, end in runs)
    assert body["returned_bytes"] == len(item["text"].encode())


@pytest.mark.asyncio
async def test_a_widened_recall_is_compressed_over_http():
    memory = await engine()
    with TestClient(create_app(memory, {"key-a": "alpha"})) as client:
        assert client.post("/v1/episodes", json={"content": TEXT, "source": "crane.txt"}, headers=auth()).status_code == 200
        hit = client.get("/v1/recall", params={"q": ASKED["q"], "limit": 1}, headers=auth()).json()["items"][0]["text"]
        widened = client.get("/v1/recall", params=ASKED, headers=auth()).json()
        made = client.get("/v1/recall", params={**ASKED, "compress": 0.0}, headers=auth())
        embedded = client.get("/v1/recall", params={**ASKED, "compress": 0.5, "compress_scorer": "embedding"},
                              headers=auth())
        features = client.get("/v1/capabilities", headers=auth()).json()["features"]
    await memory.close()
    assert made.status_code == 200, made.text
    body = made.json()
    assert body["compressed"]["compressed"] == 1 and body["compressed"]["rules"]["measured"] is False
    assert hit.strip() in body["items"][0]["text"], "the sentence retrieved is never cut"
    assert len(body["items"][0]["text"]) < len(widened["items"][0]["text"])
    assert "widened" in body and "items" not in body["compressed"]
    check_quoted(body)
    assert embedded.status_code == 200, embedded.text
    assert embedded.json()["compressed"]["sentences_embedded"] > 0
    assert features["recall.compress"] is True


@pytest.mark.asyncio
async def test_bad_compressions_are_refused_over_http():
    memory = await engine()
    with TestClient(create_app(memory, {"key-a": "alpha"})) as client:
        codes = [client.get("/v1/recall", params=params, headers=auth()).status_code for params in (
            {"q": "crane", "compress": 0.5},
            {"q": "crane", "window": 20, "compress": 0.5},
            {**ASKED, "compress": 1.5},
            {**ASKED, "compress": 0.5, "compress_scorer": "model"})]
        refusal = client.get("/v1/recall", params={"q": "crane", "compress": 0.5}, headers=auth()).text
    await memory.close()
    # Bytes widen too, but not on sentence edges: compress needs a window of sentences.
    assert codes == [422, 422, 422, 422]
    assert "window" in refusal


@pytest.mark.asyncio
async def test_the_command_line_compresses_a_widened_recall():
    memory = await engine()
    await memory.remember("default", TEXT, source="crane.txt")
    out = io.StringIO()
    asked = ["recall", ASKED["q"], "--limit", "1", "--window", "3", "--window-unit", "sentences", "--compress", "0"]
    code = await run(build_parser().parse_args([*asked, "--json"]), memory, io.StringIO(""), out)
    said = io.StringIO()
    text_code = await run(build_parser().parse_args(asked), memory, io.StringIO(""), said)
    for clash in (["--merge"], ["--parts"]):
        with pytest.raises(InvalidInput):
            await run(build_parser().parse_args([*asked, *clash]), memory, io.StringIO(""), io.StringIO())
    with pytest.raises(InvalidInput):
        await run(build_parser().parse_args(["recall", "crane", "--compress", "0.5"]), memory, io.StringIO(""),
                  io.StringIO())
    await memory.close()
    printed = json.loads(out.getvalue())
    assert code == 0 and printed["compressed"]["compressed"] == 1
    assert printed["compressed"]["sentences_dropped"] > 0 and printed["compressed"]["runs"]
    assert text_code == 0 and "sentences left out" in said.getvalue()
