"""Explicit CLI launcher. No provider is selected or called on the user's behalf."""

from __future__ import annotations

import asyncio
import importlib
import inspect
from pathlib import Path
import sys

from ..runtime.config import Settings, build_engine
from ..memory.engine import check_space
from ..core.errors import InvalidInput


def load_model_factory(spec: str):
    """Import trusted operator code, without invoking its zero-argument factory."""
    module, separator, name = spec.partition(":")
    if not separator or not all(part.isidentifier() for part in module.split(".")) or not name.isidentifier():
        raise ValueError("model factory must be module:callable")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory) or inspect.iscoroutinefunction(factory) or inspect.isasyncgenfunction(factory):
        raise ValueError("model factory must be a synchronous callable")
    inspect.signature(factory).bind()
    return factory


def journal_path(settings: Settings, value: str) -> Path:
    if value in ("", ":memory:"):
        raise ValueError("journal requires a persistent path")
    path = Path(value).expanduser().resolve()
    if not path.parent.is_dir() or (path.exists() and not path.is_file()):
        raise ValueError("journal must be a file in an existing directory")
    # All native SQLite backends use this path, including an explicit events store.
    if "sqlite" in (settings.documents, settings.vectors, settings.events) and settings.sqlite_path not in ("", ":memory:"):
        memory = Path(settings.sqlite_path).expanduser().resolve()
        if path == memory or (path.exists() and memory.exists() and path.samefile(memory)):
            raise ValueError("journal must be separate from the native memory database")
    return path


async def close_engine(engine):
    """Close every distinct owned backend, even when another backend fails."""
    seen = set()
    failed = False
    for name in ("documents", "vectors", "events", "embedder", "blobs"):
        resource = getattr(engine, name, None)
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if close is not None:
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                failed = True
    if failed:
        raise RuntimeError("conversation server backend cleanup failed")


def create_server(app, *, host: str, port: int):
    """Notify stream readers before Uvicorn waits for open HTTP responses."""
    import uvicorn

    class ConversationServer(uvicorn.Server):
        async def shutdown(self, sockets=None):
            app.state.begin_conversation_shutdown()
            await super().shutdown(sockets=sockets)

    return ConversationServer(uvicorn.Config(app, host=host, port=port,
                                            log_level="warning", access_log=False,
                                            timeout_graceful_shutdown=5))


def main(settings: Settings, *, journal: str, model_factory: str | None = None) -> int:
    if not settings.keys or any(not isinstance(key, str) or not key.strip() for key in settings.keys):
        print("refusing to serve without a key: set SCONE_API_KEY or SCONE_API_KEYS", file=sys.stderr)
        return 2
    try:
        for space in settings.keys.values():
            check_space(space)
    except (InvalidInput, TypeError):
        print("invalid API key space mapping; check SCONE_API_KEYS", file=sys.stderr)
        return 2
    try:
        path = journal_path(settings, journal)
    except (ValueError, OSError):
        print("invalid journal: use a separate file in an existing directory, not the memory database", file=sys.stderr)
        return 2
    try:
        import uvicorn
        from .conversations import create_conversation_app
        # Journal ownership requires flock, even when no model is configured.
        import fcntl  # noqa: F401
    except ImportError:
        print("conversation serving needs the api extra and Linux/macOS journal locking", file=sys.stderr)
        return 2
    runtime_type = factory = None
    if model_factory is not None:
        try:
            factory = load_model_factory(model_factory)
            from ..realtime.text import TextConversation
            runtime_type = TextConversation
        except Exception:
            print("cannot load model factory: check trusted module:callable and zero-argument signature", file=sys.stderr)
            return 2

    async def run():
        # Build database clients on the same loop as the ASGI server.
        engine = await build_engine(settings)
        try:
            scoped = None
            if factory is not None:
                def scoped(space, sid, scope):
                    return runtime_type(engine, space, sid, factory, **scope.kwargs())
            app = create_conversation_app(engine, settings.keys, path, None,
                                          scoped_runtime_factory=scoped,
                                          public_text_streaming=scoped is not None)
            server = create_server(app, host=settings.host, port=settings.port)
            await server.serve()
            if not server.started:
                raise RuntimeError("conversation server did not start")
        finally:
            await close_engine(engine)

    try:
        from ._signals import termination_unwinds
        with termination_unwinds():
            asyncio.run(run())
    except KeyboardInterrupt:
        return 130
    except Exception:
        # Provider/database exception strings may include credentials.
        print("conversation server failed; check storage access, journal ownership and server configuration", file=sys.stderr)
        return 2
    return 0
