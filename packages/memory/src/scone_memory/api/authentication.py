"""Optional host authentication adapter independent of account implementations."""
from collections.abc import Callable
from starlette.requests import HTTPConnection
from ..core.bearer_keys import KeyHolder
Authentication = Callable[[HTTPConnection], KeyHolder]

import asyncio
from collections.abc import Awaitable

async def run_authenticated(operation: Callable[[], Awaitable[None]],
                            assert_authorized: Callable[[], None],
                            close: Callable[[], Awaitable[None]], *, interval: float = 1) -> None:
    """Own a long operation and end it when host authorization is withdrawn."""
    async def run() -> None:
        await operation()
    assert_authorized()
    task=asyncio.create_task(run())
    async def watch() -> None:
        while not task.done():
            await asyncio.sleep(interval)
            try:
                assert_authorized()
            except Exception:
                task.cancel()
                await close()
                return
    watcher=asyncio.create_task(watch())
    try:
        await asyncio.shield(task)
    finally:
        if not task.done() and not task.cancelling():
            task.cancel()
        watcher.cancel()
        async def join() -> None:
            await asyncio.gather(task,watcher,return_exceptions=True)
        cleanup=asyncio.create_task(join())
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        cleanup.result()
