"""Optional MCP route over the API's existing engine and key table."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, TYPE_CHECKING, Mapping

from fastapi import FastAPI
from starlette.requests import HTTPConnection
from .authentication import Authentication
from starlette.routing import Route

from ..core.bearer_keys import BearerKeys
from ..memory.engine import MemoryEngine

if TYPE_CHECKING:
    from ..runtime.mcp_modules.transport import HttpTransport


def mount_mcp_routes(
    app: FastAPI, engine: MemoryEngine, *, keys: Mapping[str, str] | None = None,
    roles: Mapping[str, str] | None = None, propose_below: float | None = None,
    authentication: Authentication | None = None,
) -> HttpTransport:
    """Mount exactly /memory; the host retains ownership of engine shutdown.

    Keys are read on every request from the REST API's same mutable table.
    Session-manager entry and exit run in the enclosing app lifespan task.
    """
    try:
        from mcp.server.mcpserver import MCPServer  # noqa: F401: explicit optional-extra check
        from ..runtime.mcp_modules.transport import HttpTransport
    except ModuleNotFoundError as error:
        if error.name and (error.name == 'mcp' or error.name.startswith('mcp.')):
            raise ImportError('SCONE_MCP_ENABLED requires the MCP extra: install scone-memory[mcp]') from error
        raise
    if getattr(app.state, 'mcp_transport', None) is not None:
        raise ValueError('MCP is already mounted')
    memory_app = getattr(app.state, 'memory_app', app)
    key_table = memory_app.state.keys if keys is None else keys
    role_table = memory_app.state.roles if roles is None else roles
    if not key_table:
        raise ValueError('the API MCP route requires at least one bearer key')
    transport = HttpTransport(engine, 'default', key_table, role_table, propose_below,
                              streamable_http_path='/memory', dynamic_holders=authentication is not None)
    authenticated = BearerKeys(transport, key_table, role_table,
        authentication=(lambda scope: authentication(HTTPConnection(scope))) if authentication is not None else None)
    app.router.routes.insert(0, Route('/memory', authenticated, methods=['GET', 'POST', 'DELETE']))
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(host: FastAPI) -> AsyncIterator[Mapping[str, object]]:
        async with previous_lifespan(host) as state:
            async with transport.lifecycle():
                yield state if state is not None else {}

    app.router.lifespan_context = lifespan
    app.state.mcp_transport = transport
    return transport
