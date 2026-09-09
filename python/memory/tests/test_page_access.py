"""Bootstrap authority checks use raw Host, never URL parser fallbacks."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.api.conversations import create_conversation_app
from scone_memory.api.page_access import permits_local_bootstrap


KEY = "page-access-synthetic-secret"


@pytest.fixture(params=[False, True], ids=["memory", "composed"])
def page_client(request, tmp_path):
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    if request.param:
        app = create_conversation_app(engine, {KEY: "alpha"}, tmp_path / "journal.db", None,
                                      console=True, local_console_key=KEY)
    else:
        app = create_app(engine, {KEY: "alpha"}, console_key=KEY)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        yield client


@pytest.mark.parametrize("host", [
    "untrusted.example", "untrusted.example/path", "untrusted.example@127.0.0.1",
    "localhost:bad", "localhost:0", "localhost:65536", "localhost:",
    "localhost:12345678901234567890", "localhost#evil", "localhost?evil",
    "localhost\\evil", "localhost,evil", " localhost", "localhost ",
    "[::1", "[::1]evil", "::1", "[::1]:0", "[::1]:65536", "",
])
def test_malformed_or_nonlocal_host_never_receives_bootstrap(page_client, host):
    for path in ("/memory", "/playground", "/conversations/session-one"):
        response = page_client.get(path, headers={"Host": host, "X-Forwarded-Host": "localhost"})
        assert response.status_code == 200
        assert KEY not in response.text
        assert "__SCONE_TOKEN__" in response.text
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("hosts", [["localhost", "localhost"], ["localhost", "untrusted.example"]])
def test_duplicate_host_never_receives_bootstrap(page_client, hosts):
    request = page_client.build_request("GET", "/memory")
    del request.headers["host"]
    for host in hosts:
        request.headers = request.headers.__class__([*request.headers.multi_items(), ("Host", host)])
    response = page_client.send(request)
    assert response.status_code == 200
    assert KEY not in response.text


def test_missing_host_is_rejected_even_with_a_loopback_listening_address():
    # TestClient supplies a Host when absent, so check the actual raw ASGI
    # request that the server receives from a headerless HTTP/1.0 client.
    request = Request({"type": "http", "headers": [], "client": ("127.0.0.1", 50000),
                       "server": ("127.0.0.1", 8123), "scheme": "http", "path": "/memory"})
    assert not permits_local_bootstrap(request)


@pytest.mark.parametrize("host", ["localhost", "LOCALHOST", "localhost:1", "127.0.0.1:65535", "[::1]", "[::1]:8123"])
def test_valid_loopback_authority_still_receives_bootstrap(page_client, host):
    response = page_client.get("/memory", headers={"Host": host})
    assert response.status_code == 200
    assert KEY in response.text
    assert page_client.get("/v1/status").status_code == 401
    for path in ("/learn", "/learn/how-it-works"):
        assert KEY not in page_client.get(path, headers={"Host": host}).text
