"""Scheduled forgetting over the HTTP ingestion routes: an image, a document
and a page by URL take ``forget_after`` as ``POST /v1/episodes`` does, answer a
refused one with 422 before storing anything of their own, and say in the
receipt the schedule the stored episode holds."""
import pytest
from httpx import ASGITransport, AsyncClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.testing import Clock

from ..ingestion.test_image_context import context, picture
from ..ingestion.test_url_import import LAB, server  # noqa: F401 - the page server fixture

AUTH = {"authorization": "Bearer k"}


@pytest.fixture
async def served():
    clock = Clock("2026-09-15T12:00:00.000Z")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock).open()
    async with AsyncClient(transport=ASGITransport(app=create_app(engine, {"k": "alpha"}, url_import=LAB)),
                           base_url="http://fixture") as client:
        yield engine, client
    await engine.close()


async def test_an_image_takes_a_schedule_and_refuses_a_past_one_storing_nothing(served):
    engine, client = served
    image = (await client.post("/v1/attachments", content=picture(), headers={**AUTH, "content-type": "image/png"})).json()
    body = {"attachment_id": image["attachment_id"], "context": context().model_dump(mode="json")}
    refused = await client.post("/v1/images", json={**body, "forget_after": "2020-01-01"}, headers=AUTH)
    assert refused.status_code == 422 and "forget_after" in refused.text
    assert await engine.blobs.held("alpha") == [image["attachment_id"]], "no manifest is stored for a refused schedule"
    saved = await client.post("/v1/images", json={**body, "forget_after": "2d"}, headers=AUTH)
    assert saved.status_code == 200, saved.text
    assert saved.json()["added"]["forget_after"] == "2026-09-17T12:00:00.000Z"
    again = await client.post("/v1/images", json={**body, "forget_after": "9d"}, headers=AUTH)
    assert again.json()["added"]["outcome"] == "duplicate" and again.json()["added"]["forget_after"] == "2026-09-17T12:00:00.000Z"


async def test_a_document_takes_a_schedule_and_refuses_an_unreadable_one_storing_nothing(served, monkeypatch):
    from scone_memory.api import file_documents

    engine, client = served
    parsed: list[str] = []
    real_prepare = file_documents.prepare_document

    async def counted(raw, filename, **kwargs):
        parsed.append(filename)
        return await real_prepare(raw, filename, **kwargs)

    monkeypatch.setattr(file_documents, "prepare_document", counted)
    upload = (await client.post("/v1/attachments", content=b"# Harbour\n\nClosed in November.\n",
                                headers={**AUTH, "content-type": "application/octet-stream", "x-filename": "harbour.md"})).json()
    refused = await client.post("/v1/documents", json={"attachment_id": upload["attachment_id"], "forget_after": "tomorrow"},
                                headers=AUTH)
    assert refused.status_code == 422 and "forget_after" in refused.text
    assert await engine.blobs.held("alpha") == [upload["attachment_id"]], "no manifest is stored for a refused schedule"
    assert parsed == [], "a refused schedule is answered before the file is parsed"
    saved = await client.post("/v1/documents", json={"attachment_id": upload["attachment_id"], "forget_after": "2026-10-01"},
                              headers=AUTH)
    assert saved.status_code == 200, saved.text
    assert saved.json()["added"]["forget_after"] == "2026-10-01T00:00:00.000Z"
    episode = (await client.get(f"/v1/episodes/{saved.json()['added']['episode_id']}", headers=AUTH)).json()
    assert episode["metadata"]["forget_after"] == "2026-10-01T00:00:00.000Z"


async def test_a_page_by_url_takes_a_schedule_and_refuses_a_past_one(served, server):
    engine, client = served
    refused = await client.post("/v1/documents/from-url", json={"url": server + "/page.html", "forget_after": "2020-01-01"},
                                headers=AUTH)
    assert refused.status_code == 422 and "forget_after" in refused.text
    assert await engine.blobs.held("alpha") == []
    made = await client.post("/v1/documents/from-url", json={"url": server + "/page.html", "forget_after": "1h"}, headers=AUTH)
    assert made.status_code == 200, made.text
    assert made.json()["forget_after"] == "2026-09-15T13:00:00.000Z"


@pytest.mark.parametrize("bad", [30, True, ["30d"], {"days": 30}])
async def test_a_schedule_that_is_not_text_is_refused_as_post_episodes_refuses_it(served, bad):
    engine, client = served
    image = (await client.post("/v1/attachments", content=picture(), headers={**AUTH, "content-type": "image/png"})).json()
    upload = (await client.post("/v1/attachments", content=b"# Harbour\n\nClosed in November.\n",
                                headers={**AUTH, "content-type": "application/octet-stream", "x-filename": "harbour.md"})).json()
    answers = {
        "episodes": await client.post("/v1/episodes", json={"content": "the harbour closes", "forget_after": bad}, headers=AUTH),
        "images": await client.post("/v1/images", json={"attachment_id": image["attachment_id"], "forget_after": bad,
                                                         "context": context().model_dump(mode="json")}, headers=AUTH),
        "documents": await client.post("/v1/documents", json={"attachment_id": upload["attachment_id"], "forget_after": bad},
                                       headers=AUTH),
        "from-url": await client.post("/v1/documents/from-url", json={"url": "http://127.0.0.1:9/x.html", "forget_after": bad},
                                      headers=AUTH),
    }
    assert {route: answer.status_code for route, answer in answers.items()} == dict.fromkeys(answers, 422)
    assert set(await engine.blobs.held("alpha")) == {image["attachment_id"], upload["attachment_id"]}, \
        "no manifest is stored for a refused schedule"
