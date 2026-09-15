"""Merging neighbouring fragments on /v1/recall, not only on the command line.

A retrieval feature only the terminal can ask for is one the product
does not have. The route merges after any window and before withholding,
so withholding scans the text a merge reads between fragments; the
command line now merges at the same point, and the refusal it held for
--withhold beside --merge, which existed because it merged after
withholding, is lifted.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.cli import build_parser, run

ADDRESS = "okafor@harbour.example"
# Two paragraphs about the crane with one between them the question does not ask about.
TEXT = ("The harbour crane survey found rust on the jib and grease missing from the slew ring on the crane.\n\n"
        f"Write to {ADDRESS} with any questions about the canteen menu or the winter car park.\n\n"
        "The harbour crane survey also found the jib hoist needed a new rope and a new hook fitted, "
        "and the crane was back in service by June.\n")
QUESTION = "harbour crane survey jib rust grease slew hoist rope hook"
LIMIT = 2


def auth() -> dict:
    return {"authorization": "Bearer key-a"}


async def engine() -> MemoryEngine:
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=100).open()
    await memory.remember("alpha", TEXT, source="crane.txt")
    return memory


@pytest.mark.asyncio
async def test_neighbouring_fragments_are_merged_over_http_and_say_how_much_was_retrieved():
    memory = await engine()
    with TestClient(create_app(memory, {"key-a": "alpha"})) as client:
        plain = client.get("/v1/recall", params={"q": QUESTION, "limit": LIMIT}, headers=auth()).json()
        made = client.get("/v1/recall", params={"q": QUESTION, "limit": LIMIT, "merge": True}, headers=auth())
        sparse = client.get("/v1/recall", params={"q": QUESTION, "limit": LIMIT, "merge": True, "merge_min_share": 1.0},
                            headers=auth())
        features = client.get("/v1/capabilities", headers=auth()).json()["features"]
    await memory.close()
    assert len(plain["items"]) >= 2, "the corpus has to return several fragments to prove anything"
    assert made.status_code == 200, made.text
    body = made.json()
    record = body["merged"]
    assert record["merged"] == 1 and "items" not in record
    [(reported, share)] = record["shares"].items()
    assert 0 < share <= 1 and sorted(record["from_chunks"][reported]) == sorted(i["chunk_id"] for i in plain["items"])
    [whole] = body["items"]
    assert body["returned_bytes"] == len(whole["text"].encode())
    assert features["recall.merge"] is True
    if share < 1.0:
        assert sparse.json()["merged"]["too_sparse"] == 1 and len(sparse.json()["items"]) == len(plain["items"])


@pytest.mark.asyncio
async def test_a_merge_is_withheld_from_like_any_passage_over_http():
    memory = await engine()
    with TestClient(create_app(memory, {"key-a": "alpha"})) as client:
        plain = client.get("/v1/recall", params={"q": QUESTION, "limit": LIMIT}, headers=auth()).json()
        opened = client.get("/v1/recall", params={"q": QUESTION, "limit": LIMIT, "merge": True}, headers=auth()).json()
        held = client.get("/v1/recall", params={"q": QUESTION, "limit": LIMIT, "merge": True, "withhold": "email"},
                          headers=auth())
    await memory.close()
    assert len(plain["items"]) == 2 and not any(ADDRESS in item["text"] for item in plain["items"])
    assert ADDRESS in opened["items"][0]["text"], "the address sits between the fragments, read only by the merge"
    assert held.status_code == 200, held.text
    assert ADDRESS not in held.text and held.json()["withheld"]["count"] >= 1


@pytest.mark.asyncio
async def test_bad_merges_are_refused_over_http():
    memory = await engine()
    with TestClient(create_app(memory, {"key-a": "alpha"})) as client:
        codes = [client.get("/v1/recall", params=params, headers=auth()).status_code for params in (
            {"q": QUESTION, "merge_min_share": 0.5},
            {"q": QUESTION, "merge_min_share": 0},
            {"q": QUESTION, "merge": True, "merge_min_share": 1.5},
            {"q": QUESTION, "merge": True, "window": 2, "window_unit": "sentences", "compress": 0.5})]
    await memory.close()
    assert codes == [422, 422, 422, 422]


@pytest.mark.asyncio
async def test_the_command_line_withholds_from_a_merge_and_takes_a_share_floor():
    memory = await engine()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--space", "alpha", "recall", QUESTION, "--limit", str(LIMIT), "--merge",
                                                "--withhold", "email", "--json"]), memory, io.StringIO(""), out)
    floor = io.StringIO()
    floor_code = await run(build_parser().parse_args(["--space", "alpha", "recall", QUESTION, "--limit", str(LIMIT),
                                                      "--merge", "--merge-min-share", "1", "--json"]),
                           memory, io.StringIO(""), floor)
    with pytest.raises(InvalidInput):
        await run(build_parser().parse_args(["--space", "alpha", "recall", QUESTION, "--merge-min-share", "0.5"]),
                  memory, io.StringIO(""), io.StringIO())
    await memory.close()
    said = json.loads(out.getvalue())
    assert code == 0 and said["merged"]["merged"] == 1 and ADDRESS not in out.getvalue()
    assert said["withheld"]["withheld"] >= 1
    assert floor_code == 0 and json.loads(floor.getvalue())["merged"]["rules"]["min_share"] == 1.0


@pytest.mark.asyncio
async def test_after_a_window_the_share_counts_what_was_retrieved_on_both_surfaces():
    """A byte window makes the two hits overlap, so counting the widened
    spans would call the merge wholly retrieved."""
    memory = await engine()
    with TestClient(create_app(memory, {"key-a": "alpha"})) as client:
        body = client.get("/v1/recall", params={"q": QUESTION, "limit": LIMIT, "merge": True, "window": 100},
                          headers=auth()).json()
    out = io.StringIO()
    await run(build_parser().parse_args(["--space", "alpha", "recall", QUESTION, "--limit", str(LIMIT), "--merge",
                                         "--window", "100", "--json"]), memory, io.StringIO(""), out)
    await memory.close()
    [(_, share)] = body["merged"]["shares"].items()
    [(_, said)] = json.loads(out.getvalue())["merged"]["shares"].items()
    assert share < 0.9 and said == share
