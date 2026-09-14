"""A page fetched by URL: read as what the server says it is, refused past every bound, never a private host by default."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.app import create_app
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.files import document_provenance
from scone_memory.ingestion.web import WebLimits, check_url, fetch_page, ingest_url
from scone_memory.runtime import cli

HTML = b"<html><head><title>Harbour notes</title></head><body><h1>Vellmar harbour</h1><p>The harbour closes to sailing boats every November.</p><p>Pilots board arriving ships at the red buoy.</p></body></html>"
MARKDOWN = b"# Orchard\n\nThe orchard grows only Bramley apples.\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401 - quiet
        pass

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/page.html":
            self._send(200, "text/html; charset=utf-8", HTML)
        elif route == "/notes.md":
            self._send(200, "text/markdown", MARKDOWN)
        elif route == "/redirect":
            self.send_response(302); self.send_header("Location", "/page.html"); self.end_headers()
        elif route == "/loop":
            self.send_response(302); self.send_header("Location", "/loop"); self.end_headers()
        elif route == "/away":
            self.send_response(302); self.send_header("Location", "ftp://example.org/secret"); self.end_headers()
        elif route == "/nowhere":
            self.send_response(302); self.send_header("Location", "http://nonexistent.invalid/"); self.end_headers()
        elif route == "/binary":
            self._send(200, "application/octet-stream", b"\x00\x01\x02")
        elif route == "/big":
            # No Content-Length: the bound must be found while reading.
            self.protocol_version = "HTTP/1.0"
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
            self.wfile.write(b"x" * 5000)
        elif route == "/declared":
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.send_header("Content-Length", "999999")
            self.end_headers(); self.wfile.write(b"short")
        else:
            self._send(404, "text/plain", b"nothing here")

    def _send(self, status, media_type, body):
        self.send_response(status); self.send_header("Content-Type", media_type); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


LAB = WebLimits(allow_private=True, max_bytes=4096, max_redirects=2, timeout_seconds=5)


async def test_a_private_host_is_refused_unless_the_caller_says_so(server):
    with pytest.raises(InvalidInput, match="not on the public internet"):
        await fetch_page(server + "/page.html")
    with pytest.raises(InvalidInput, match="only http and https"):
        check_url("ftp://example.org/x")
    with pytest.raises(InvalidInput, match="credentials"):
        check_url("http://user:pass@example.org/x", allow_private=True)
    with pytest.raises(InvalidInput, match="no host"):
        check_url("http:///x", allow_private=True)
    page = await fetch_page(server + "/page.html", LAB)
    assert (page.media_type, page.status, page.data, page.filename, page.redirects) == ("text/html", 200, HTML, "page.html", ())


async def test_every_bound_refuses_rather_than_cuts(server):
    with pytest.raises(InvalidInput, match="more than 4096 bytes"):
        await fetch_page(server + "/big", LAB)
    with pytest.raises(InvalidInput, match="999999 bytes declared"):
        await fetch_page(server + "/declared", LAB)
    with pytest.raises(InvalidInput, match="more than 2 redirect"):
        await fetch_page(server + "/loop", LAB)
    with pytest.raises(InvalidInput, match="not one the document lane reads"):
        await fetch_page(server + "/binary", LAB)
    with pytest.raises(InvalidInput, match="answered 404"):
        await fetch_page(server + "/missing", LAB)
    followed = await fetch_page(server + "/redirect", LAB)
    assert followed.final_url == server + "/page.html" and followed.redirects == (server + "/page.html",)


async def test_a_redirect_is_checked_like_the_first_url(server):
    with pytest.raises(InvalidInput, match="only http and https"):
        await fetch_page(server + "/away", LAB)
    with pytest.raises(InvalidInput, match="does not resolve"):
        await fetch_page(server + "/nowhere", LAB)


async def test_the_page_is_read_as_its_media_type_and_the_episode_says_where_it_came_from(server):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        imported = await ingest_url(engine, "s", server + "/redirect", limits=LAB)
        episode = await engine.episode("s", imported.document.added.episode_id)
        assert episode.kind == "file" and "closes to sailing boats every November" in episode.content
        assert episode.metadata["document_url"] == server + "/redirect"
        assert episode.metadata["document_final_url"] == server + "/page.html"
        assert episode.metadata["document_media_type"] == "text/html" and episode.metadata["document_format"] == "html"
        assert episode.metadata["document_fetched_at"].endswith("+00:00") and imported.bytes == len(HTML)
        provenance = await document_provenance(engine, "s", episode.episode_id)
        assert provenance.format == "html" and provenance.original.attachment_id == imported.document.original.attachment_id
        notes = await ingest_url(engine, "s", server + "/notes.md", limits=LAB)
        assert notes.document.format in ("md", "markdown") and "Bramley" in (await engine.episode("s", notes.document.added.episode_id)).content
        found = await engine.recall("s", "when does the harbour close", limit=2)
        assert found.items and found.items[0].episode_id == episode.episode_id
        again = await ingest_url(engine, "s", server + "/page.html", limits=LAB)
        assert again.document.added.episode_id == episode.episode_id, "the same bytes read the same way are the same document"
    finally:
        await engine.close()


def test_the_route_is_off_by_default_and_reads_a_page_when_on(server):
    import asyncio

    async def engine_for():
        return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    engine = asyncio.run(engine_for())
    with TestClient(create_app(engine, {"key-a": "s"})) as client:
        off = client.post("/v1/documents/from-url", json={"url": server + "/page.html"}, headers={"Authorization": "Bearer key-a"})
        assert off.status_code == 501 and "not enabled" in off.json()["error"]
    with TestClient(create_app(engine, {"key-a": "s"}, url_import=LAB)) as client:
        bad = client.post("/v1/documents/from-url", json={"link": "x"}, headers={"Authorization": "Bearer key-a"})
        assert bad.status_code == 400, bad.text
        made = client.post("/v1/documents/from-url", json={"url": server + "/page.html"}, headers={"Authorization": "Bearer key-a"})
        assert made.status_code == 200, made.text
        body = made.json()
        assert body["media_type"] == "text/html" and body["format"] == "html" and body["episode_id"]
        refused = client.post("/v1/documents/from-url", json={"url": server + "/binary"}, headers={"Authorization": "Bearer key-a"})
        assert refused.status_code == 422 and "not one the document lane reads" in refused.json()["error"], "a refusal is InvalidInput, which the API answers as 422"


def test_the_command_needs_the_setting_and_imports_a_page(server, tmp_path):
    env = {"SCONE_DOCUMENTS": "memory", "SCONE_VECTORS": "memory"}
    out = io.StringIO()
    code = cli.main(["import-url", server + "/page.html"], env=env, stdin=io.StringIO(""), out=out)
    assert code == 2, out.getvalue()
    out = io.StringIO()
    code = cli.main(["--json", "import-url", server + "/page.html"],
                    env={**env, "SCONE_URL_IMPORT": "1", "SCONE_URL_IMPORT_PRIVATE": "1"}, stdin=io.StringIO(""), out=out)
    assert code == 0, out.getvalue()
    payload = json.loads(out.getvalue())
    assert payload["format"] == "html" and payload["url"] == server + "/page.html"
