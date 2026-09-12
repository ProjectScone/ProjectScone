"""A batch where one record is wrong, and what becomes of the rest.

Refusing the whole batch is the right default: a caller who sent one bad
record usually wants to fix it and send the lot again, and a half-stored
batch they did not ask for is worse than a clear refusal. But a caller
importing ten thousand records from somewhere messy wants the nine
thousand that are fine and a list of the ones that are not, and until now
there was no way to say so.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.records import Record

pytestmark = pytest.mark.asyncio
SPACE = "alpha"


async def memory() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


def mixed() -> list[Record]:
    return [Record(content="the harbour crane was repainted"),
            Record(content="   "),
            Record(content="the crane was inspected in July"),
            Record(content="a fine note", kind="telegram")]


async def test_one_bad_record_still_refuses_the_whole_batch_by_default():
    engine = await memory()
    with pytest.raises(InvalidInput, match="empty"):
        await engine.remember_many(SPACE, mixed())
    assert (await engine.documents.counts(SPACE)).episodes == 0, "nothing was stored"


async def test_a_caller_can_ask_for_what_could_be_stored():
    engine = await memory()
    outcomes = await engine.remember_many(SPACE, mixed(), partial=True)
    assert [outcome.outcome for outcome in outcomes] == ["accepted", "failed", "accepted", "failed"]
    assert (await engine.documents.counts(SPACE)).episodes == 2


async def test_a_failed_record_says_what_was_wrong_with_it():
    engine = await memory()
    outcomes = await engine.remember_many(SPACE, mixed(), partial=True)
    assert "empty" in (outcomes[1].reason or "")
    assert "kind" in (outcomes[3].reason or "")
    assert outcomes[1].episode_id == -1, "nothing was stored for it"


async def test_the_good_records_are_ordinary_records():
    engine = await memory()
    outcomes = await engine.remember_many(SPACE, mixed(), partial=True)
    found = await engine.recall(SPACE, "harbour crane")
    assert found.items and found.items[0].episode_id == outcomes[0].episode_id


async def test_a_duplicate_among_them_is_still_a_duplicate():
    engine = await memory()
    await engine.remember(SPACE, "the harbour crane was repainted")
    outcomes = await engine.remember_many(SPACE, mixed(), partial=True)
    assert [outcome.outcome for outcome in outcomes] == ["duplicate", "failed", "accepted", "failed"]


async def test_a_batch_that_is_all_wrong_stores_nothing_and_says_so_for_each():
    engine = await memory()
    outcomes = await engine.remember_many(SPACE, [Record(content=" "), Record(content="")], partial=True)
    assert [outcome.outcome for outcome in outcomes] == ["failed", "failed"]
    assert all(outcome.reason for outcome in outcomes)
    assert (await engine.documents.counts(SPACE)).episodes == 0


async def test_a_partial_batch_over_http_answers_each_record(tmp_path):
    from fastapi.testclient import TestClient

    from scone_memory.api import create_app

    engine = await memory()
    app = create_app(engine, {"key-a": SPACE})
    with TestClient(app) as client:
        body = {"partial": True, "records": [{"content": "the harbour crane was repainted"},
                                             {"content": "   "},
                                             {"content": "a fine note", "kind": "telegram"}]}
        said = client.post("/v1/episodes/batch", json=body,
                           headers={"Authorization": "Bearer key-a"})
        assert said.status_code == 200, said.text
        answer = said.json()
        assert [item["outcome"] for item in answer["items"]] == ["accepted", "failed", "failed"]
        assert answer["counts"]["failed"] == 2 and answer["counts"]["accepted"] == 1
        assert "empty" in answer["items"][1]["reason"]


async def test_a_batch_over_http_still_refuses_the_whole_thing_by_default(tmp_path):
    from fastapi.testclient import TestClient

    from scone_memory.api import create_app

    engine = await memory()
    with TestClient(create_app(engine, {"key-a": SPACE})) as client:
        said = client.post("/v1/episodes/batch",
                           json={"records": [{"content": "good"}, {"content": " "}]},
                           headers={"Authorization": "Bearer key-a"})
        assert said.status_code == 422 and "empty" in said.json()["error"]
        assert (await engine.documents.counts(SPACE)).episodes == 0
