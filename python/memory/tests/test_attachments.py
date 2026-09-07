"""Attachments: bytes an episode carries, addressed by their content.

A screenshot is evidence, and the memory that cannot keep it can only
keep somebody's description of it. Bytes live in a blob store beside the
document store, are named by their SHA-256, and are delivered only to a
key for the space that stored them.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.backends.blobs import FileBlobStore
from scone_memory.core.errors import InvalidInput, NotFound

PNG = b"\x89PNG\r\n\x1a\n" + b"pretend pixels" * 8
SPACE = "alpha"


@pytest.fixture
async def engine():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


@pytest.fixture
def client(engine):
    with TestClient(create_app(engine, {"key-a": SPACE, "key-b": "beta"})) as c:
        yield c


def auth(key: str = "key-a") -> dict:
    return {"authorization": f"Bearer {key}"}


async def test_the_same_bytes_are_stored_once_and_answer_to_their_digest(engine):
    first = await engine.attach(SPACE, PNG, media_type="image/png", filename="review.png")
    again = await engine.attach(SPACE, PNG, media_type="image/png", filename="another-name.png")

    assert first.attachment_id == hashlib.sha256(PNG).hexdigest()
    assert again.attachment_id == first.attachment_id
    assert (first.bytes, first.media_type, first.filename) == (len(PNG), "image/png", "review.png")
    assert (await engine.attachment(SPACE, first.attachment_id))[1] == PNG


async def test_an_episode_carries_the_attachments_it_was_remembered_with(engine):
    stored = await engine.attach(SPACE, PNG, media_type="image/png", filename="review.png")
    added = await engine.remember(SPACE, "The review page, before the fix", attachment_ids=[stored.attachment_id])

    episode = await engine.episode(SPACE, added.episode_id)
    assert [a.attachment_id for a in episode.attachments] == [stored.attachment_id]
    assert episode.attachments[0].filename == "review.png"


async def test_an_attachment_belongs_to_the_space_that_stored_it(engine):
    stored = await engine.attach(SPACE, PNG, media_type="image/png")

    with pytest.raises(NotFound):
        await engine.attachment("beta", stored.attachment_id)


async def test_bytes_too_large_or_of_an_unknown_type_are_refused(engine):
    with pytest.raises(InvalidInput):
        await engine.attach(SPACE, b"x" * (engine.max_attachment_bytes + 1), media_type="image/png")
    with pytest.raises(InvalidInput):
        await engine.attach(SPACE, PNG, media_type="application/x-msdownload")
    with pytest.raises(InvalidInput):
        await engine.attach(SPACE, b"", media_type="image/png")


def test_an_attachment_is_delivered_only_to_its_own_space(client):
    posted = client.post("/v1/attachments", content=PNG,
                         headers={**auth(), "content-type": "image/png", "x-filename": "review.png"})
    stored = posted.json()
    assert stored["attachment_id"] == hashlib.sha256(PNG).hexdigest()

    got = client.get(f"/v1/attachments/{stored['attachment_id']}", headers=auth())
    assert got.status_code == 200 and got.content == PNG
    assert got.headers["content-type"].startswith("image/png")
    assert got.headers["x-content-type-options"] == "nosniff"

    assert client.get(f"/v1/attachments/{stored['attachment_id']}", headers=auth("key-b")).status_code == 404
    assert client.get(f"/v1/attachments/{stored['attachment_id']}").status_code == 401


def test_bytes_that_could_run_in_the_page_are_delivered_as_a_download(client):
    """An SVG is a script in the page's own origin the moment it is served
    inline. It is kept, because it is evidence, and handed back as a file."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    stored = client.post("/v1/attachments", content=svg,
                         headers={**auth(), "content-type": "image/svg+xml"}).json()

    got = client.get(f"/v1/attachments/{stored['attachment_id']}", headers=auth())
    assert got.content == svg
    assert got.headers["content-type"].startswith("application/octet-stream")
    assert got.headers["content-disposition"].startswith("attachment")


def test_an_id_that_is_not_a_digest_never_reaches_the_store(client):
    for bad in ("../../etc/passwd", "not-a-digest", "a" * 63, "A" * 64):
        assert client.get(f"/v1/attachments/{bad}", headers=auth()).status_code in (404, 422)


def test_an_episode_shows_its_attachments_over_http(client):
    stored = client.post("/v1/attachments", content=PNG,
                         headers={**auth(), "content-type": "image/png", "x-filename": "review.png"}).json()
    added = client.post("/v1/episodes",
                        json={"content": "the review page", "attachment_ids": [stored["attachment_id"]]},
                        headers=auth()).json()

    episode = client.get(f"/v1/episodes/{added['episode_id']}", headers=auth()).json()
    assert episode["attachments"] == [{
        "attachment_id": stored["attachment_id"], "media_type": "image/png",
        "bytes": len(PNG), "filename": "review.png",
    }]


async def test_a_file_blob_store_addresses_bytes_by_their_digest(tmp_path):
    """The bytes are on disk, under their own digest, once. Two spaces
    holding the same screenshot are two references, not two copies."""
    blobs = FileBlobStore(tmp_path)
    stored = await blobs.put(SPACE, PNG, media_type="image/png", filename="review.png")

    digest = hashlib.sha256(PNG).hexdigest()
    on_disk = pathlib.Path(tmp_path) / "blobs" / digest[:2] / digest
    assert on_disk.read_bytes() == PNG
    assert (await blobs.get(SPACE, stored.attachment_id))[1] == PNG
    with pytest.raises(NotFound):
        await blobs.get("beta", stored.attachment_id)


async def test_a_configured_server_keeps_attachments_across_a_restart(tmp_path):
    """A server with a database on disk gets a blob directory beside it,
    because an attachment that does not survive a restart is not evidence.
    SCONE_BLOB_DIR moves it; an engine with no database keeps bytes in
    memory rather than writing somewhere it was not asked to."""
    from scone_memory.runtime.config import Settings, build_engine

    env = {"SCONE_DOCUMENTS": "sqlite", "SCONE_SQLITE_PATH": str(tmp_path / "memory.db"),
           "SCONE_EMBEDDER": "hash"}
    engine = await build_engine(Settings.from_env(env))
    stored = await engine.attach(SPACE, PNG, media_type="image/png", filename="review.png")

    again = await build_engine(Settings.from_env(env))
    assert (await again.attachment(SPACE, stored.attachment_id))[1] == PNG
    assert (tmp_path / "attachments" / "blobs" / stored.attachment_id[:2] / stored.attachment_id).exists()

    elsewhere = {**env, "SCONE_BLOB_DIR": str(tmp_path / "elsewhere")}
    moved = await build_engine(Settings.from_env(elsewhere))
    await moved.attach(SPACE, PNG, media_type="image/png")
    assert (tmp_path / "elsewhere" / "blobs" / stored.attachment_id[:2] / stored.attachment_id).exists()


async def test_a_body_over_the_cap_stops_being_read_at_the_cap():
    """The cap has to bind the request, not just the store. Reading the
    whole body and then measuring it lets the caller decide how much
    memory the server spends, which is the cap not existing. Driven
    through a real ASGI receive channel, because an HTTP client buffers
    the body itself and would hide exactly the thing under test."""
    from starlette.requests import Request

    from scone_memory.api.app import read_bounded

    chunks_read = 0

    async def receive():
        nonlocal chunks_read
        chunks_read += 1
        return {"type": "http.request", "body": b"x" * 200, "more_body": chunks_read < 20}

    request = Request({"type": "http", "method": "POST", "path": "/", "headers": [],
                       "query_string": b""}, receive)

    with pytest.raises(InvalidInput):
        await read_bounded(request, 1000)
    assert chunks_read < 20, f"the whole body was read before it was refused ({chunks_read} chunks)"


async def test_a_declared_length_over_the_cap_is_refused_before_a_byte_is_read():
    """A caller that announces 40 MB is told no without sending it."""
    from starlette.requests import Request

    from scone_memory.api.app import read_bounded

    async def receive():  # pragma: no cover - reaching this is the failure
        raise AssertionError("the body was read despite an oversized content-length")

    request = Request({"type": "http", "method": "POST", "path": "/", "query_string": b"",
                       "headers": [(b"content-length", b"40000000")]}, receive)

    with pytest.raises(InvalidInput):
        await read_bounded(request, 1000)


def test_bytes_are_revalidated_so_access_can_be_taken_away(client):
    """An attachment's bytes never change, so a long cache looks free. It
    is not: the thing that changes is whether this key may still read
    them. The client revalidates, and gets a cheap 304 when nothing has."""
    stored = client.post("/v1/attachments", content=PNG,
                         headers={**auth(), "content-type": "image/png"}).json()

    got = client.get(f"/v1/attachments/{stored['attachment_id']}", headers=auth())
    assert got.headers["etag"] == f'"{stored["attachment_id"]}"'
    assert "no-cache" in got.headers["cache-control"]
    assert "31536000" not in got.headers["cache-control"]

    again = client.get(f"/v1/attachments/{stored['attachment_id']}",
                       headers={**auth(), "if-none-match": got.headers["etag"]})
    assert again.status_code == 304 and again.content == b""

    # A key that may not read it gets nothing, whatever it claims to hold.
    assert client.get(f"/v1/attachments/{stored['attachment_id']}",
                      headers={**auth("key-b"), "if-none-match": got.headers["etag"]}).status_code == 404
