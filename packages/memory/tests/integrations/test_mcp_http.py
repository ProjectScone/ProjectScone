"""The MCP server over Streamable HTTP: JSON-RPC in a POST, answers as a
server-sent event stream, every request behind a bearer key from the
same table the REST API reads, and the same tools and resources as stdio.

Everything runs in process against the ASGI app; no port is opened and
no model is needed.
"""

from __future__ import annotations

import asyncio
import json
import stat
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import httpx
import pytest
from mcp.server.mcpserver import Context

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime import mcp as mcp_module
from scone_memory.core.bearer_keys import KeyHolder
from scone_memory.runtime.mcp import (READING_TOOLS, WRITING_TOOLS, build_parser, create_server,
                                      http_app, main)
from scone_memory.testing import Clock

from .test_mcp import TOOL_ARGUMENTS

PROTOCOL = "2025-06-18"
ORIGIN = "http://127.0.0.1:7438"
KEYS = {"alpha-key": "alpha", "beta-key": "beta", "reader-key": "alpha", "reviewer-key": "alpha",
        "writer-key": "alpha"}
ROLES = {"reader-key": "read", "reviewer-key": "review", "writer-key": "write"}


async def open_engine() -> MemoryEngine:
    return await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=200, clock=Clock()
    ).open()


@pytest.fixture
async def engine():
    memory = await open_engine()
    try:
        yield memory
    finally:
        await memory.close()


@asynccontextmanager
async def running(app) -> AsyncIterator[None]:
    """The ASGI lifespan, driven as a host drives it: the sessions are
    open between startup and shutdown and at no other time."""
    inbox: asyncio.Queue[dict] = asyncio.Queue()
    outbox: asyncio.Queue[dict] = asyncio.Queue()
    task = asyncio.create_task(app({"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}},
                                   inbox.get, outbox.put))
    await inbox.put({"type": "lifespan.startup"})
    started = await asyncio.wait_for(outbox.get(), 10)
    assert started["type"] == "lifespan.startup.complete", started
    try:
        yield
    finally:
        await inbox.put({"type": "lifespan.shutdown"})
        assert (await asyncio.wait_for(outbox.get(), 10))["type"] == "lifespan.shutdown.complete"
        await asyncio.wait_for(task, 10)


def events(text: str) -> list[dict]:
    """The JSON-RPC messages of a server-sent event stream, in order."""
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]


@asynccontextmanager
async def served(app) -> AsyncIterator[httpx.AsyncClient]:
    async with running(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN) as client:
            yield client


class Session:
    """One MCP client session over HTTP, holding a key or none."""

    def __init__(self, client: httpx.AsyncClient, key: Optional[str]) -> None:
        self.client = client
        self.key = key
        self.session_id: Optional[str] = None
        self.next_id = 0

    def headers(self) -> dict[str, str]:
        headers = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
        if self.key is not None:
            headers["authorization"] = f"Bearer {self.key}"
        if self.session_id is not None:
            headers["mcp-session-id"] = self.session_id
            headers["mcp-protocol-version"] = PROTOCOL
        return headers

    async def post(self, message: dict) -> httpx.Response:
        return await self.client.post("/mcp", headers=self.headers(), json=message)

    async def open(self) -> dict:
        response = await self.post({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}})
        assert response.status_code == 200, response.text
        self.session_id = response.headers["mcp-session-id"]
        (answer,) = events(response.text)
        initialized = await self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert initialized.status_code == 202, initialized.text
        return answer["result"]

    async def request(self, method: str, params: Optional[dict] = None) -> dict:
        self.next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            message["params"] = params
        response = await self.post(message)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        (answer,) = [m for m in events(response.text) if m.get("id") == self.next_id]
        return answer

    async def call(self, name: str, **arguments) -> tuple[bool, str]:
        result = (await self.request("tools/call", {"name": name, "arguments": arguments}))["result"]
        return result["isError"], "\n".join(block["text"] for block in result["content"])


async def opened(client: httpx.AsyncClient, key: Optional[str]) -> Session:
    session = Session(client, key)
    await session.open()
    return session


# -- the handshake and the catalogue ------------------------------------------


async def test_initialize_and_tools_list_over_http_offer_what_stdio_offers(engine):
    stdio = create_server(engine, "default")
    async with served(http_app(engine, "default", KEYS, ROLES)) as client:
        session = Session(client, "alpha-key")
        result = await session.open()
        assert result["protocolVersion"] == PROTOCOL
        assert result["serverInfo"]["name"] == "scone-memory"

        listed = (await session.request("tools/list"))["result"]
        assert "nextCursor" not in listed
        tools = {tool["name"]: tool for tool in listed["tools"]}
        assert set(tools) == {tool.name for tool in await stdio.list_tools()} == set(TOOL_ARGUMENTS)
        for name, arguments in TOOL_ARGUMENTS.items():
            assert set(tools[name]["inputSchema"]["properties"]) == arguments, name

        resources = (await session.request("resources/list"))["result"]["resources"]
        assert {r["uri"] for r in resources} == {str(r.uri) for r in await stdio.list_resources()}
        templates = (await session.request("resources/templates/list"))["result"]["resourceTemplates"]
        assert {t["uriTemplate"] for t in templates} == {t.uri_template for t in await stdio.list_resource_templates()}


# -- the key ------------------------------------------------------------------


async def test_no_key_or_a_wrong_bearer_gets_401_and_never_reaches_a_session(engine):
    async with served(http_app(engine, "default", KEYS, ROLES)) as client:
        for key, reason in [(None, "missing bearer key"), ("nobody", "unknown key")]:
            response = await Session(client, key).post({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                                                        "params": {"protocolVersion": PROTOCOL, "capabilities": {},
                                                                   "clientInfo": {"name": "t", "version": "0"}}})
            assert response.status_code == 401, key
            assert response.json() == {"error": reason}

        wrong_scheme = await client.post("/mcp", headers={"authorization": "Basic alpha-key"}, json={})
        assert wrong_scheme.status_code == 401

        # A session opened with a good key is no use to a request without one.
        session = await opened(client, "alpha-key")
        for key in (None, "nobody"):
            session.key = key
            response = await session.post({"jsonrpc": "2.0", "id": 9, "method": "tools/list"})
            assert response.status_code == 401
            ended = await client.delete("/mcp", headers=session.headers())
            assert ended.status_code == 401
        session.key = "alpha-key"
        assert "tools" in (await session.request("tools/list"))["result"], "the session outlived the refusals"


async def test_an_authorised_tool_call_lands_in_the_space_of_its_key(engine):
    async with served(http_app(engine, "default", KEYS, ROLES)) as client:
        session = await opened(client, "alpha-key")
        error, text = await session.call("memory_store", content="Moved to Lisbon in March")
        assert not error, text
        assert text.startswith("stored episode 1 (1 chunks)")

        assert [e.content for e in await engine.documents.recent_episodes("alpha", 5)] == ["Moved to Lisbon in March"]
        assert await engine.documents.recent_episodes("default", 5) == []

        error, text = await session.call("memory_recall", query="where do I live", include_profile=False)
        assert not error
        assert text == "memory [1.00 | 2025-01-01 | episode 1] Moved to Lisbon in March"


async def test_a_key_cannot_reach_another_space_by_argument_address_or_session(engine):
    await engine.assert_fact("alpha", "atlas", "based_in", "lisbon")
    async with served(http_app(engine, "default", KEYS, ROLES)) as client:
        alpha = await opened(client, "alpha-key")

        error, text = await alpha.call("memory_store", content="a note for beta", space="beta")
        assert (error, text) == (True, "space 'beta' is not this key's")
        assert await engine.documents.recent_episodes("beta", 5) == []
        error, text = await alpha.call("memory_recall", query="anything", space="beta")
        assert (error, text) == (True, "space 'beta' is not this key's")
        error, text = await alpha.call("memory_facts_about", entity="atlas", space="alpha")
        assert not error and "based_in" in text, "naming its own space is not a refusal"

        refused = await alpha.request("resources/read", {"uri": "scone://beta/graph/schema"})
        assert "space 'beta' is not this key's" in refused["error"]["message"]
        own = await alpha.request("resources/read", {"uri": "scone://graph/schema"})
        assert "based_in" in own["result"]["contents"][0]["text"], "the fixed address is the key's own space"

        beta = await opened(client, "beta-key")
        theirs = await beta.request("resources/read", {"uri": "scone://graph/schema"})
        assert "based_in" not in theirs["result"]["contents"][0]["text"]

        # Beta's key presenting alpha's session reaches no session at all.
        beta.session_id = alpha.session_id
        response = await beta.post({"jsonrpc": "2.0", "id": 5, "method": "tools/list"})
        assert response.status_code == 404


@pytest.mark.parametrize("key, may_write", [
    ("reader-key", False), ("reviewer-key", False), ("writer-key", True), ("alpha-key", True),
], ids=["read", "review", "write", "full"])
async def test_a_key_role_limits_the_tools_that_write(engine, key, may_write):
    episode = await engine.remember("alpha", "Project Atlas is ready for the review.")
    fact = await engine.assert_fact("alpha", "atlas", "status", "ready")
    role = ROLES.get(key, "full")
    async with served(http_app(engine, "default", KEYS, ROLES)) as client:
        session = await opened(client, key)
        writes = [
            ("memory_store", {"content": "Moved to Lisbon in March"}),
            ("memory_store_facts", {"episode_id": episode.episode_id,
                                    "facts": [{"subject": "atlas", "predicate": "owner", "object": "ana"}]}),
            ("memory_forget", {"fact_id": fact.fact_id, "reason": "retracted"}),
        ]
        for name, arguments in writes:
            error, text = await session.call(name, **arguments)
            if may_write:
                assert not error, (name, text)
            else:
                assert (error, text) == (True, f"key role {role} cannot write"), name
        error, text = await session.call("memory_recall", query="atlas", include_profile=False)
        assert not error, "every role reads"
    if not may_write:
        assert [e.episode_id for e in await engine.documents.recent_episodes("alpha", 5)] == [episode.episode_id]
        assert [f.fact_id for f in await engine.facts("alpha")] == [fact.fact_id]


# -- the stream -----------------------------------------------------------------


def _asgi_scope(headers: list[tuple[bytes, bytes]], method: str = "POST") -> dict:
    return {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
            "path": "/mcp", "raw_path": b"/mcp", "root_path": "", "query_string": b"", "headers": headers,
            "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 7438), "state": {}}


async def asgi_post(app, message: dict, headers: list[tuple[bytes, bytes]],
                    on_send: Optional[Callable[[dict], Awaitable[None]]] = None) -> list[dict]:
    """One POST straight through the ASGI callable, keeping every message
    the app sends in the order it sent them."""
    body = json.dumps(message).encode()
    delivered = False
    sent: list[dict] = []

    async def receive() -> dict:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()  # the client never disconnects mid-answer
        raise AssertionError("unreachable")

    async def send(sent_message: dict) -> None:
        sent.append(sent_message)
        if on_send is not None:
            await on_send(sent_message)

    base = [(b"host", b"127.0.0.1:7438"), (b"accept", b"application/json, text/event-stream"),
            (b"content-type", b"application/json")]
    await app(_asgi_scope(base + headers), receive, send)
    return sent


async def test_a_tool_result_streams_as_server_sent_events_while_the_tool_is_still_running(engine, monkeypatch):
    """The tool reports progress, then waits until the client has that
    report in hand. Were the key middleware (or anything) to hold the
    response back until it was complete, the tool would wait forever."""
    released = asyncio.Event()
    built = create_server

    def with_a_waiting_tool(*args, **kwargs):
        server = built(*args, **kwargs)

        async def wait_for_the_client(ctx: Context) -> str:
            await ctx.report_progress(1, 2, "reported")
            await released.wait()
            return "released"

        server.add_tool(wait_for_the_client, name="wait_for_the_client")
        return server

    monkeypatch.setattr(mcp_module, "create_server", with_a_waiting_tool)
    app = http_app(engine, "default", KEYS, ROLES)
    key = [(b"authorization", b"Bearer alpha-key")]
    async with running(app):
        opening = await asgi_post(app, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}, key)
        session = key + [(b"mcp-session-id", dict(opening[0]["headers"])[b"mcp-session-id"]),
                         (b"mcp-protocol-version", PROTOCOL.encode())]
        await asgi_post(app, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session)

        async def release_on_progress(message: dict) -> None:
            if message["type"] == "http.response.body" and b"notifications/progress" in message.get("body", b""):
                released.set()

        call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "wait_for_the_client", "arguments": {}, "_meta": {"progressToken": "p"}}}
        sent = await asyncio.wait_for(asgi_post(app, call, session, on_send=release_on_progress), timeout=10)

    start, *bodies = sent
    assert start["status"] == 200
    assert dict(start["headers"])[b"content-type"] == b"text/event-stream"
    frames = [events(body["body"].decode()) for body in bodies]
    progress = next(i for i, frame in enumerate(frames) if frame and frame[0].get("method") == "notifications/progress")
    result = next(i for i, frame in enumerate(frames) if frame and frame[0].get("id") == 1)
    assert progress < result
    assert bodies[progress]["more_body"] is True, "the report went out as its own chunk, before the answer"
    assert frames[result][0]["result"]["content"][0]["text"] == "released"
    assert bodies[-1]["more_body"] is False


# -- where it listens ------------------------------------------------------------


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "memory.example.com"])
async def test_a_host_beyond_loopback_is_refused_without_a_key(engine, host):
    with pytest.raises(InvalidInput, match="reaches beyond this machine"):
        http_app(engine, "default", {}, {}, host=host)
    http_app(engine, "default", KEYS, ROLES, host=host)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost", "::1"])
async def test_loopback_without_a_key_serves_as_stdio_does(engine, host):
    app = http_app(engine, "default", {}, {}, host=host)
    async with served(app) as client:
        session = await opened(client, None)
        error, text = await session.call("memory_store", content="a note", space="beta")
        assert not error, "nobody holds a key, so nothing narrows the space: the stdio contract"
    assert len(await engine.documents.recent_episodes("beta", 5)) == 1


async def test_the_lifespan_protocol_opens_and_closes_the_sessions(engine):
    app = http_app(engine, "default", KEYS, ROLES)
    inbox: asyncio.Queue[dict] = asyncio.Queue()
    outbox: asyncio.Queue[dict] = asyncio.Queue()
    lifespan = asyncio.create_task(app({"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}},
                                       inbox.get, outbox.put))
    await inbox.put({"type": "lifespan.startup"})
    assert (await asyncio.wait_for(outbox.get(), 10))["type"] == "lifespan.startup.complete"

    sent = await asgi_post(app, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
        "protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
        [(b"authorization", b"Bearer beta-key")])
    assert sent[0]["status"] == 200

    await inbox.put({"type": "lifespan.shutdown"})
    assert (await asyncio.wait_for(outbox.get(), 10))["type"] == "lifespan.shutdown.complete"
    await asyncio.wait_for(lifespan, 10)


# -- the command line ------------------------------------------------------------


def test_the_transport_defaults_to_stdio_and_http_to_loopback():
    args = build_parser().parse_args([])
    assert args.transport == "stdio"
    args = build_parser().parse_args(["--transport", "http"])
    assert (args.host, args.port) == (None, None)


@pytest.mark.parametrize("argv", [["--host", "0.0.0.0"], ["--port", "9000"]], ids=["host", "port"])
def test_host_and_port_are_refused_for_stdio(argv, capsys):
    with pytest.raises(SystemExit) as refused:
        main(argv, env={})
    assert refused.value.code == 2
    assert "--host and --port apply to --transport http" in capsys.readouterr().err


def test_main_refuses_a_public_bind_asked_to_take_anyone(monkeypatch, capsys):
    async def never(_settings):
        raise AssertionError("stores were opened for a server that must not start")

    monkeypatch.setattr(mcp_module, "build_engine", never)
    assert main(["--transport", "http", "--host", "0.0.0.0", "--allow-anonymous"], env={}) == 2
    refusal = capsys.readouterr().err
    assert "reaches beyond this machine" in refusal and "SCONE_API_KEYS" in refusal


def test_main_serves_http_on_loopback_and_the_port_asked_for(monkeypatch):
    served_with: list[tuple] = []

    async def fake_serve_http(settings, space, host, port):
        served_with.append((dict(settings.keys), space, host, port))

    monkeypatch.setattr(mcp_module, "serve_http", fake_serve_http)
    assert main(["--transport", "http", "--port", "9001", "--space", "notes"], env={"SCONE_API_KEY": "k"}) == 0
    assert main(["--transport", "http"], env={"SCONE_API_KEY": "k"}) == 0
    assert served_with == [({"k": "default"}, "notes", "127.0.0.1", 9001),
                           ({"k": "default"}, "default", "127.0.0.1", mcp_module.DEFAULT_HTTP_PORT)]


# -- the key a self-hoster never has to invent ------------------------------------


def serve_nothing(monkeypatch) -> list[tuple]:
    """Record what main() would have served, without serving it."""
    served: list[tuple] = []

    async def fake_serve_http(settings, space, host, port):
        served.append((dict(settings.keys), dict(settings.roles), space, host, port))

    monkeypatch.setattr(mcp_module, "serve_http", fake_serve_http)
    return served


def test_a_server_with_no_key_makes_one_keeps_it_and_says_it_once(tmp_path, monkeypatch, capsys):
    """Self-hosting takes no key ceremony: the first start issues a key,
    prints it once with the address to paste it into, and saves it, so
    the client configured on Monday still works on Tuesday."""
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "memory.db")}
    served = serve_nothing(monkeypatch)
    assert main(["--transport", "http", "--space", "notes"], env=env) == 0

    saved = tmp_path / "mcp-key"
    (keys, roles, space, host, port) = served[0]
    (key,) = keys
    assert keys == {key: "notes"} and roles == {key: "full"}, "the key opens the space this server serves"
    assert saved.read_text().strip() == key
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600, "a secret nobody else on the box can read"
    said = capsys.readouterr().err
    assert key in said and f"http://{host}:{port}/mcp" in said and "Bearer" in said

    assert main(["--transport", "http", "--space", "notes"], env=env) == 0
    assert served[1][0] == {key: "notes"}, "the same key, so a configured client keeps working"
    again = capsys.readouterr().err
    assert key not in again, "the secret is printed when it is new, not on every start"
    assert str(saved) in again


def test_a_configured_key_is_left_alone_and_nothing_is_written(tmp_path, monkeypatch, capsys):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "memory.db"), "SCONE_API_KEYS": "mine:notes:read"}
    served = serve_nothing(monkeypatch)
    assert main(["--transport", "http", "--space", "notes"], env=env) == 0
    assert served[0][:2] == ({"mine": "notes"}, {"mine": "read"})
    assert not (tmp_path / "mcp-key").exists(), "a host that manages its own keys is not given one"
    assert capsys.readouterr().err == ""


def test_stdio_is_not_given_a_key(tmp_path, monkeypatch):
    """Over stdio the client is the process that started this one; there
    is nothing for a key to prove."""
    async def fake_serve(settings, space):
        assert dict(settings.keys) == {}

    monkeypatch.setattr(mcp_module, "serve", fake_serve)
    assert main(["--space", "notes"], env={"SCONE_SQLITE_PATH": str(tmp_path / "memory.db")}) == 0
    assert not (tmp_path / "mcp-key").exists()


def test_anonymous_on_loopback_writes_no_key_and_serves_without_one(tmp_path, monkeypatch):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "memory.db")}
    served = serve_nothing(monkeypatch)
    assert main(["--transport", "http", "--allow-anonymous"], env=env) == 0
    assert served[0][:2] == ({}, {})
    assert not (tmp_path / "mcp-key").exists()


def test_a_key_file_readable_by_others_is_used_and_said_so(tmp_path, monkeypatch, capsys):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "memory.db")}
    saved = tmp_path / "mcp-key"
    saved.write_text("sk-scone-borrowed\n")
    saved.chmod(0o644)
    served = serve_nothing(monkeypatch)
    assert main(["--transport", "http"], env=env) == 0
    assert served[0][0] == {"sk-scone-borrowed": "default"}
    assert "other users of this machine can read" in capsys.readouterr().err


# -- the narrowing, on every door there is -----------------------------------------


#: Enough arguments for each tool to get past its own input checks, so
#: that what it answers is about the space and nothing else. Every tool
#: the server offers must appear, which is what makes a tool added later
#: fail this file rather than quietly reach other people's spaces.
ENOUGH: dict[str, dict] = {
    "memory_store": {"content": "a note"},
    "memory_recall": {"query": "anything"},
    "memory_facts_about": {"entity": "atlas"},
    "memory_pending": {},
    "memory_store_facts": {"episode_id": 1, "facts": [{"subject": "a", "predicate": "b", "object": "c"}]},
    "memory_forget": {"fact_id": 1, "reason": "retracted"},
    "memory_graph_context": {"names": ["atlas"]},
    "memory_entity": {"name": "atlas"},
    "memory_connections": {"source": "atlas", "target": "ana"},
    "memory_graph_schema": {},
    "memory_graph_match": {"where": [{"subject": "?who", "predicate": "based_in", "object": "lisbon"}]},
    "memory_graph_overview": {},
    "memory_graph_changes": {"since": "2025-01-01T00:00:00Z"},
    "memory_entity_duplicates": {},
    "memory_temporal_answer": {"question": "where does atlas live?"},
    "memory_graph_cycles": {},
    "memory_graph_stats": {},
    "memory_graph_hubs": {},
    "memory_graph_health": {},
    "memory_graph_affected": {"name": "atlas"},
}

SPACE_TEMPLATES = ["scone://{space}/graph/report", "scone://{space}/graph/schema",
                   "scone://{space}/graph/health", "scone://{space}/graph/stats", "scone://{space}/graph/hubs"]


async def test_every_tool_is_classified_as_reading_or_writing(engine):
    """A tool nobody classified is treated as a write, so a read-only key
    loses it -- visibly, here, rather than silently in a release."""
    offered = {tool.name for tool in await create_server(engine, "default").list_tools()}
    assert offered == READING_TOOLS | WRITING_TOOLS
    assert not READING_TOOLS & WRITING_TOOLS
    assert offered == set(ENOUGH)


async def test_every_tool_that_takes_a_space_refuses_one_that_is_not_the_key_s(engine):
    server = create_server(engine, "alpha", None, KeyHolder("alpha", "full"))
    for name, arguments in ENOUGH.items():
        result = await server.call_tool(name, {**arguments, "space": "beta"})
        text = "\n".join(block.text for block in result.content)
        assert (result.is_error, text) == (True, "space 'beta' is not this key's"), name
    assert await engine.documents.recent_episodes("beta", 5) == [], "and nothing was written on the way to refusing"


async def test_every_resource_template_refuses_a_space_that_is_not_the_key_s(engine):
    server = create_server(engine, "alpha", None, KeyHolder("alpha", "full"))
    assert {t.uri_template for t in await server.list_resource_templates()} == set(SPACE_TEMPLATES)
    for template in SPACE_TEMPLATES:
        with pytest.raises(Exception, match="space 'beta' is not this key's"):
            await server.read_resource(template.format(space="beta"))
        assert await server.read_resource(template.format(space="alpha")), "the key's own space is readable"


async def test_every_reading_tool_answers_a_read_only_key(engine):
    """The other half of the classification: a tool wrongly called a
    write would refuse a reader here."""
    server = create_server(engine, "alpha", None, KeyHolder("alpha", "read"))
    for name in sorted(READING_TOOLS):
        result = await server.call_tool(name, ENOUGH[name])
        text = "\n".join(block.text for block in result.content)
        assert "cannot write" not in text, f"{name} is a read, but a reader was refused it"
