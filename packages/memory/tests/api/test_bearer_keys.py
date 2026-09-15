"""The one bearer-key check every HTTP surface uses: the REST API calls it
per route, and the MCP HTTP transport runs it as pure ASGI middleware in
front of the whole app."""

from __future__ import annotations

import json

import pytest

from scone_memory.core.bearer_keys import KEY_STATE, BearerKeys, KeyHolder, Unauthorized, bearer_token, key_holder

KEYS = {"alpha-key": "alpha", "reader-key": "alpha"}
ROLES = {"reader-key": "read"}


@pytest.mark.parametrize("header, token", [
    ("Bearer alpha-key", "alpha-key"),
    ("bearer alpha-key", "alpha-key"),
    ("BEARER   alpha-key  ", "alpha-key"),
    ("Bearer ", None),
    ("Bearer", None),
    ("Basic alpha-key", None),
    ("alpha-key", None),
    ("", None),
], ids=["plain", "lower-scheme", "upper-scheme-padded", "blank-token", "no-token", "other-scheme", "bare", "empty"])
def test_bearer_token_reads_the_scheme_without_case_and_refuses_anything_else(header, token):
    assert bearer_token(header) == token


def test_a_key_reaches_its_space_with_its_role_and_full_when_none_is_recorded():
    assert key_holder("Bearer reader-key", KEYS, ROLES) == KeyHolder("alpha", "read")
    assert key_holder("Bearer alpha-key", KEYS, ROLES) == KeyHolder("alpha", "full")


def test_a_missing_and_an_unknown_key_are_refused_with_different_reasons():
    with pytest.raises(Unauthorized, match="^missing bearer key$"):
        key_holder("Basic alpha-key", KEYS, ROLES)
    with pytest.raises(Unauthorized, match="^unknown key$"):
        key_holder("Bearer nobody", KEYS, ROLES)


def _scope(kind: str = "http", authorization: bytes | None = None) -> dict:
    headers = [(b"host", b"127.0.0.1")]
    if authorization is not None:
        headers.append((b"authorization", authorization))
    return {"type": kind, "path": "/mcp", "headers": headers, "state": {"kept": 1}}


async def _run(middleware: BearerKeys, scope: dict) -> list[dict]:
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    await middleware(scope, receive, send)
    return sent


class Recorder:
    """The app behind the middleware: remembers the scope it was handed."""

    def __init__(self) -> None:
        self.scopes: list[dict] = []

    async def __call__(self, scope, receive, send) -> None:
        self.scopes.append(scope)


@pytest.mark.parametrize("authorization, reason", [
    (None, "missing bearer key"),
    (b"Bearer nobody", "unknown key"),
    (b"Basic alpha-key", "missing bearer key"),
], ids=["no-header", "wrong-key", "wrong-scheme"])
async def test_the_middleware_answers_401_before_the_app_sees_the_request(authorization, reason):
    inner = Recorder()
    sent = await _run(BearerKeys(inner, KEYS, ROLES), _scope(authorization=authorization))
    assert inner.scopes == []
    start, body = sent
    assert start["type"] == "http.response.start" and start["status"] == 401
    headers = dict(start["headers"])
    assert headers[b"content-type"] == b"application/json"
    assert headers[b"www-authenticate"] == b"Bearer"
    assert int(headers[b"content-length"]) == len(body["body"])
    assert json.loads(body["body"]) == {"error": reason}


async def test_an_admitted_request_carries_its_key_holder_and_keeps_the_state_it_had():
    inner = Recorder()
    sent = await _run(BearerKeys(inner, KEYS, ROLES), _scope(authorization=b"Bearer reader-key"))
    assert sent == []
    (scope,) = inner.scopes
    assert scope["state"] == {"kept": 1, KEY_STATE: KeyHolder("alpha", "read")}


async def test_the_lifespan_passes_through_without_a_key():
    inner = Recorder()
    await _run(BearerKeys(inner, KEYS, ROLES), _scope(kind="lifespan"))
    assert [scope["type"] for scope in inner.scopes] == ["lifespan"]


async def test_a_websocket_without_a_key_is_closed_before_it_is_accepted():
    inner = Recorder()
    sent = await _run(BearerKeys(inner, KEYS, ROLES), _scope(kind="websocket"))
    assert inner.scopes == []
    assert sent == [{"type": "websocket.close", "code": 1008, "reason": "missing bearer key"}]


async def test_the_middleware_reads_the_live_key_table():
    """The REST host updates its keys in place; a key removed there is
    refused here from the next request on."""
    keys = dict(KEYS)
    inner = Recorder()
    middleware = BearerKeys(inner, keys, ROLES)
    keys.pop("alpha-key")
    sent = await _run(middleware, _scope(authorization=b"Bearer alpha-key"))
    assert sent[0]["status"] == 401 and inner.scopes == []
