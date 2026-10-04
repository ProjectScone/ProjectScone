"""Holder-bound HTTP sessions; host owns the engine and app lifespan."""
import asyncio
import ipaddress
import json
from typing import AsyncIterator, Callable, Mapping, Optional

from contextlib import AsyncExitStack, asynccontextmanager

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from ...core.bearer_keys import KEY_STATE, BearerKeys, KeyHolder
from ...core.errors import InvalidInput
from ...memory.engine import MemoryEngine

from .factory import create_server
DEFAULT_HTTP_PORT = 8765

#: The loopback names the SDK's own DNS-rebinding guard knows. A server
#: on any loopback address is reachable under all of them from the same
#: machine, so all three are allowed as Host.
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")


def is_loopback(host: str) -> bool:
    """Whether a bind address is reachable only from this machine.
    ``0.0.0.0`` and ``::`` are not: they are every address there is."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _host_names(host: str) -> list[str]:
    bracketed = f"[{host}]" if ":" in host else host
    names = list(LOOPBACK_NAMES) if is_loopback(host) else []
    return names + ([bracketed] if bracketed not in names else [])


def check_bind(keys: Mapping[str, str], host: str) -> None:
    """A server with no key answers whoever reaches it, so it may reach
    no further than this machine."""
    if not keys and not is_loopback(host):
        raise InvalidInput(
            f"serving MCP on {host} reaches beyond this machine and would answer anyone: "
            "set SCONE_API_KEYS (or SCONE_API_KEY), or bind a loopback address")


class HttpTransport:
    """The MCP server over Streamable HTTP, one server per key holder.

    A key names a space and a role, and that pair is fixed for the life
    of a session: so rather than re-deciding it inside every tool call,
    each holder gets its own MCP server, built with that space and that
    role, and its own session table. A session id from one holder's
    table means nothing in another's, which is what makes a stolen
    session id useless to a key that did not open it.

    Two keys that name the same space with the same role share a server:
    they are the same authority, and nothing one may do is closed to the
    other. Holders are taken from the key table as it stands when the
    transport is built; a key added later with a space and role no
    existing server covers is told to restart the server, rather than
    served by a server built for somebody else.

    The sessions themselves are opened when the host starts the app and
    closed when it stops it, in the task that runs the lifespan -- a
    session manager's task group must be left by the task that entered
    it, so they cannot be opened lazily inside a request.
    """

    def __init__(self, engine: MemoryEngine, space: str, keys: Mapping[str, str], roles: Mapping[str, str],
                 propose_below: Optional[float] = None, host: str = "127.0.0.1", *, streamable_http_path: str = "/mcp",
                 server_factory: Callable[..., MCPServer] = create_server, dynamic_holders: bool = False) -> None:
        self._dynamic_holders = dynamic_holders
        self._dynamic_tasks: asyncio.TaskGroup | None = None
        self._dynamic_ready: dict[KeyHolder, asyncio.Future[None]] = {}
        self._dynamic_stop: asyncio.Event | None = None
        self._engine, self._factory = engine, server_factory
        self._propose_below, self._host, self._path = propose_below, host, streamable_http_path
        self.anonymous = not keys
        check_bind(keys, host)
        holders = [None] if self.anonymous else sorted(
            {KeyHolder(where, roles.get(key, "full")) for key, where in keys.items()})
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[f"{name}:*" for name in _host_names(host)] + _host_names(host),
            allowed_origins=[f"http://{name}:*" for name in _host_names(host)],
        ) if self.anonymous else None
        self._security = security
        self.servers = {
            holder: server_factory(engine, space if holder is None else holder.space, propose_below, holder)
            for holder in holders
        }
        self.apps = {
            holder: server.streamable_http_app(streamable_http_path=streamable_http_path, host=host, transport_security=security)
            for holder, server in self.servers.items()
        }
        self.sessions: Optional[AsyncExitStack] = None

    async def __call__(self, scope: dict, receive, send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        holder = None if self.anonymous else scope.get("state", {}).get(KEY_STATE)
        app = self.apps.get(holder)
        if app is None and self.sessions is not None and self._dynamic_holders and isinstance(holder, KeyHolder):
            try:
                await self._ensure_holder(holder)
            except (ValueError, RuntimeError):
                await _unavailable(scope, send, "this account's MCP service is unavailable")
                return
            app = self.apps.get(holder)
        if self.sessions is None:
            await _unavailable(scope, send, "the server is still starting")
        elif app is None:
            # A key the host added after this server was built, naming a
            # space or role no server here covers.
            await _unavailable(scope, send, "this key's space is not served yet; restart the server to pick it up")
        else:
            await app(scope, receive, send)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[None]:
        """Open sessions in the host lifespan task, without owning its engine.

        The entering task must also exit: MCP session managers own task groups.
        A host mounting this transport nests this context inside its lifespan.
        """
        if self.sessions is not None:
            raise RuntimeError("MCP transport lifecycle is already running")
        async with AsyncExitStack() as sessions:
            for server in self.servers.values():
                await sessions.enter_async_context(server.session_manager.run())
            async with asyncio.TaskGroup() as tasks:
                self._dynamic_tasks = tasks
                self._dynamic_stop = asyncio.Event()
                self.sessions = sessions
                try:
                    yield
                finally:
                    self.sessions = None
                    self._dynamic_stop.set()
            self._dynamic_tasks = None
            self._dynamic_stop = None
            for holder in self._dynamic_ready:
                self.apps.pop(holder, None)
                self.servers.pop(holder, None)
            self._dynamic_ready.clear()

    async def _ensure_holder(self, holder: KeyHolder) -> None:
        ready = self._dynamic_ready.get(holder)
        if ready is None:
            if len(self._dynamic_ready) >= 128 or self._dynamic_tasks is None or self._dynamic_stop is None:
                raise ValueError("dynamic MCP admission limit")
            ready = asyncio.get_running_loop().create_future()
            self._dynamic_ready[holder] = ready
            self._dynamic_tasks.create_task(self._own_holder(holder, ready, self._dynamic_stop))
        await asyncio.shield(ready)

    async def _own_holder(self, holder: KeyHolder, ready: asyncio.Future[None], stop: asyncio.Event) -> None:
        try:
            server = self._factory(self._engine, holder.space, self._propose_below, holder)
            app = server.streamable_http_app(streamable_http_path=self._path, host=self._host, transport_security=self._security)
            async with server.session_manager.run():
                self.servers[holder], self.apps[holder] = server, app
                ready.set_result(None)
                await stop.wait()
        except Exception:
            if not ready.done():
                ready.set_exception(RuntimeError("dynamic MCP startup failed"))
        finally:
            self.apps.pop(holder, None)
            self.servers.pop(holder, None)
            if not ready.done():
                ready.cancel()

    async def _lifespan(self, receive, send) -> None:
        message = await receive()
        if message["type"] != "lifespan.startup":
            return
        started = False
        try:
            async with self.lifecycle():
                await send({"type": "lifespan.startup.complete"})
                started = True
                while (await receive())["type"] != "lifespan.shutdown":
                    pass
            await send({"type": "lifespan.shutdown.complete"})
        except BaseException as error:
            phase = "shutdown" if started else "startup"
            await send({"type": f"lifespan.{phase}.failed", "message": str(error)})


async def _unavailable(scope: dict, send, reason: str) -> None:
    if scope["type"] != "http":
        await send({"type": "websocket.close", "code": 1013, "reason": reason})
        return
    body = json.dumps({"error": reason}).encode()
    await send({"type": "http.response.start", "status": 503, "headers": [
        (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def http_app(engine: MemoryEngine, space: str = "default", keys: Optional[Mapping[str, str]] = None,
             roles: Optional[Mapping[str, str]] = None, propose_below: Optional[float] = None,
             host: str = "127.0.0.1", *, streamable_http_path: str = "/mcp",
             server_factory: Callable[..., MCPServer] = create_server):
    """The ASGI app that serves MCP over Streamable HTTP.

    With keys, every request carries one and is refused before it
    reaches a session without it. With none, the app serves as stdio
    does -- no key, no narrowing of the space -- and may bind nothing
    but a loopback address, since on a loopback address the caller is
    someone already on this machine.
    """
    transport = HttpTransport(engine, space, keys or {}, roles or {}, propose_below, host,
                              streamable_http_path=streamable_http_path, server_factory=server_factory)
    if transport.anonymous:
        return transport
    return BearerKeys(transport, keys or {}, roles or {})
