"""``scone-memory`` / ``python -m scone_memory.api``: serve the engine
described by the environment. Refuses to start without a key, because
an unauthenticated memory server is a leak waiting for a port scan."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

from ..config import Settings, build_engine, build_worker
from .app import create_app


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
    engine = asyncio.run(build_engine(settings))
    # One configured key means one space; bake it so the console opens
    # without a prompt. Several keys: the console asks which.
    only_key = next(iter(settings.keys)) if len(settings.keys) == 1 else None
    worker = build_worker(engine, settings, settings.keys.values())
    app = create_app(engine, settings.keys, console_key=only_key, worker=worker)
    print(
        f"scone-memory on http://{settings.host}:{settings.port} "
        f"documents={engine.documents.name} vectors={engine.vectors.name} embedder={engine.embedder.id} "
        f"spaces={sorted(set(settings.keys.values()))} "
        + (f"consolidation={settings.chat_model} every {settings.distill_interval_s:g}s" if worker else "consolidation=off"),
        file=sys.stderr,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
