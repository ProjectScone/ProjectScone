"""Borrowed-engine MCP mounting, with no network listener or model calls."""
from __future__ import annotations

import builtins

import pytest
from fastapi import FastAPI

from scone_memory.api.app import create_app
from scone_memory.runtime.config import Settings
from .test_mcp_http import KEYS, ROLES, Session, engine, served


class MemorySession(Session):
    async def post(self, message):
        return await self.client.post('/memory', headers=self.headers(), json=message)


def test_mcp_opt_in_environment():
    assert Settings.from_env({}).mcp_enabled is False
    assert Settings.from_env({'SCONE_MCP_ENABLED': 'true'}).mcp_enabled is True
    with pytest.raises(Exception, match='SCONE_MCP_ENABLED'):
        Settings.from_env({'SCONE_MCP_ENABLED': 'perhaps'})


async def test_mount_requires_key_before_mcp_and_borrows_engine(engine):
    from scone_memory.api.mcp_routes import mount_mcp_routes
    app = create_app(engine, KEYS, roles=ROLES)
    transport = mount_mcp_routes(app, engine)
    assert transport.sessions is None
    async with served(app) as client:
        assert transport.sessions is not None
        assert (await client.post('/memory', json={})).status_code == 401
        owner = MemorySession(client, 'alpha-key')
        await owner.open()
        error, _ = await owner.call('memory_store', content='Alpha retained source')
        assert not error
        reader = MemorySession(client, 'reader-key')
        await reader.open()
        error, message = await reader.call('memory_store', content='Forbidden write')
        assert error and 'cannot write' in message
        error, _ = await reader.call('memory_recall', query='source', space='beta')
        assert error
        beta = MemorySession(client, 'beta-key')
        await beta.open()
        error, message = await beta.call('memory_recall', query='Alpha retained source', include_profile=False)
        assert not error and 'Alpha retained source' not in message
        del app.state.keys['alpha-key']
        assert (await owner.post({'jsonrpc':'2.0','id':9,'method':'tools/list'})).status_code == 401
    assert transport.sessions is None
    assert (await engine.episode('alpha', 1)).content == 'Alpha retained source'


def test_missing_extra_has_actionable_failure(monkeypatch):
    from scone_memory.api.mcp_routes import mount_mcp_routes
    original = builtins.__import__
    def no_mcp(name, *args, **kwargs):
        if name.startswith('mcp.server'):
            raise ModuleNotFoundError('no mcp', name='mcp')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_mcp)
    with pytest.raises(ImportError, match='MCP.*extra'):
        mount_mcp_routes(FastAPI(), object(), keys={'key':'default'})


async def test_launcher_disabled_never_imports_mcp_extra(engine, monkeypatch):
    from scone_memory.api.__main__ import build_app
    original = builtins.__import__
    def no_mcp(name, *args, **kwargs):
        if name.startswith('mcp.server'):
            raise ModuleNotFoundError('no mcp', name='mcp')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_mcp)
    app = build_app(Settings(keys=KEYS, roles=ROLES), engine)
    assert not any(getattr(route, 'path', None) == '/memory' for route in app.routes)
    with pytest.raises(ImportError, match='MCP.*extra'):
        build_app(Settings(keys=KEYS, roles=ROLES, mcp_enabled=True), engine)


async def test_launcher_opt_in_mounts_once_and_tears_down_sessions(engine):
    from scone_memory.api.__main__ import build_app
    app = build_app(Settings(keys=KEYS, roles=ROLES, mcp_enabled=True), engine)
    assert sum(getattr(route, 'path', None) == '/memory' for route in app.routes) == 1
    async with served(app) as client:
        session = MemorySession(client, 'alpha-key')
        await session.open()
        assert app.state.mcp_transport.sessions is not None
    assert app.state.mcp_transport.sessions is None
    await engine.remember('alpha', 'Engine remains host-owned after MCP shutdown')


async def test_borrowed_lifecycle_unwinds_partial_startup_in_same_task(engine):
    import asyncio
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from scone_memory.runtime.mcp_modules.transport import HttpTransport
    entered = []
    exited = []
    task = asyncio.current_task()
    @asynccontextmanager
    async def first():
        entered.append(asyncio.current_task())
        try:
            yield
        finally:
            exited.append(asyncio.current_task())
    @asynccontextmanager
    async def failing():
        raise RuntimeError('session startup failed')
        yield
    transport = HttpTransport(engine, 'default', {}, {})
    transport.servers = {None: SimpleNamespace(session_manager=SimpleNamespace(run=first)),
                         'second': SimpleNamespace(session_manager=SimpleNamespace(run=failing))}
    with pytest.raises(RuntimeError, match='session startup failed'):
        async with transport.lifecycle():
            pytest.fail('failed startup must never admit requests')
    assert entered == exited == [task]
    assert transport.sessions is None
    await engine.remember('default', 'No engine ownership transfer')
