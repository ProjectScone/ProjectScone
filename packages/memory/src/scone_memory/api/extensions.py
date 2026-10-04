"""Application HTTP routes installed with the host's current authorization."""
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from fastapi import FastAPI, Request
from ..memory.engine import MemoryEngine

@dataclass(frozen=True)
class HttpExtensionContext:
    """Borrowed host services; extensions must not close the engine.

    Use ``space_for`` as the route dependency and call it again after
    asynchronous work before returning private content. The synchronous
    assertion checks credentials/roles/scope, but not storage tombstones.
    Extensions own their application resources through a lifespan wrapper.
    """
    app: FastAPI
    engine: MemoryEngine
    space_for: Callable[[Request], Awaitable[str]]
    assert_current_space: Callable[[Request, str], None]

HttpExtension = Callable[[HttpExtensionContext], None]
