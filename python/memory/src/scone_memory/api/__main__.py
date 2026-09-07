"""``scone-memory`` / ``python -m scone_memory.api``: serve the engine
described by the environment. Refuses to start without a key, because
an unauthenticated memory server is a leak waiting for a port scan."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

from ..runtime.config import Settings, build_engine, build_worker
from .app import create_app


def build_app(settings: Settings, engine):
    """The app ``serve`` runs: the memory API alone, or the conversation
    service composed over it on the same origin when the settings name a
    journal (SCONE_CONVERSATIONS_JOURNAL). Raises ValueError for a journal
    or model factory the operator got wrong, before anything is served."""
    worker = build_worker(engine, settings, settings.keys.values())
    if not settings.conversations_journal:
        # One configured key means one space; bake it so the console opens
        # without a prompt. Several keys: the console asks which.
        only_key = next(iter(settings.keys)) if len(settings.keys) == 1 else None
        return create_app(engine, settings.keys, console_key=only_key, worker=worker, reload_pages=settings.reload_pages)
    from .conversation_server import journal_path, load_model_factory
    from .conversations import create_conversation_app

    journal = journal_path(settings, settings.conversations_journal)
    scoped = None
    if settings.conversations_model_factory:
        factory = load_model_factory(settings.conversations_model_factory)
        from ..realtime.text import TextConversation

        def scoped(space, sid, scope):
            return TextConversation(engine, space, sid, factory, **scope.kwargs())
    # The composed shell never carries a key: the tab asks for one.
    return create_conversation_app(engine, settings.keys, journal, None, scoped_runtime_factory=scoped,
                                   console=True, public_text_streaming=scoped is not None,
                                   worker=worker, reload_pages=settings.reload_pages)


def build_server(settings: Settings, app):
    """The uvicorn server for ``app``. A composed host must tell the
    conversation service to end its open streams before uvicorn waits for
    open responses, or a reader holding a stream holds shutdown; the
    memory-only app has no streams to end."""
    import uvicorn

    if settings.conversations_journal:
        from .conversation_server import create_server

        return create_server(app, host=settings.host, port=settings.port)
    return uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.port, log_level="warning"))


def main(settings: Optional[Settings] = None) -> None:
    settings = settings or Settings.from_env()
    if not settings.keys:
        print("refusing to serve without a key: set SCONE_API_KEY or SCONE_API_KEYS", file=sys.stderr)
        sys.exit(2)
    try:
        import uvicorn
    except ImportError:
        print("serving needs uvicorn: pip install 'scone-memory[api]'", file=sys.stderr)
        sys.exit(2)
    async def run() -> None:
        # The engine is built on the loop that serves it. Async database
        # clients (pymongo, psycopg's pool) bind to the loop they were
        # opened on; building on a throwaway loop and serving on uvicorn's
        # made every Mongo request fail with "Cannot use AsyncMongoClient
        # in different event loop" (seen in the compose smoke test).
        engine = await build_engine(settings)
        try:
            app = build_app(settings, engine)
        except (ValueError, OSError, ImportError, AttributeError, TypeError) as error:
            await engine.close()
            print(f"refusing to serve: {error}", file=sys.stderr)
            sys.exit(2)
        worker = app.state.worker
        print(
            f"scone-memory on http://{settings.host}:{settings.port} "
            f"documents={engine.documents.name} vectors={engine.vectors.name} embedder={engine.embedder.id} "
            f"spaces={sorted(set(settings.keys.values()))} "
            + (f"consolidation={settings.chat_model} every {settings.distill_interval_s:g}s" if worker else "consolidation=off")
            + (f" conversations={settings.conversations_journal}" if settings.conversations_journal else ""),
            file=sys.stderr,
        )
        await build_server(settings, app).serve()

    asyncio.run(run())


if __name__ == "__main__":
    main()
