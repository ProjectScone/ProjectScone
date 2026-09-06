"""``scone-memory`` / ``python -m scone_memory.api``: serve the engine
described by the environment. Refuses to start without a key, because
an unauthenticated memory server is a leak waiting for a port scan."""

from __future__ import annotations

import asyncio
import sys

from ..config import Settings, build_engine
from .app import create_app


def main() -> None:
    settings = Settings.from_env()
    if not settings.keys:
        print("refusing to serve without a key: set SCONE_API_KEY or SCONE_API_KEYS", file=sys.stderr)
        sys.exit(2)
    try:
        import uvicorn
    except ImportError:
        print("serving needs uvicorn: pip install 'scone-memory[api]'", file=sys.stderr)
        sys.exit(2)
    engine = asyncio.run(build_engine(settings))
    app = create_app(engine, settings.keys)
    print(
        f"scone-memory on http://{settings.host}:{settings.port} "
        f"documents={engine.documents.name} vectors={engine.vectors.name} embedder={engine.embedder.id} "
        f"spaces={sorted(set(settings.keys.values()))}",
        file=sys.stderr,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
