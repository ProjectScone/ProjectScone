"""Finish owned host cleanup before propagating caller cancellation."""
import asyncio
from collections.abc import Awaitable


async def finish_host_cleanup(cleanup: Awaitable[None]) -> None:
    task = asyncio.ensure_future(cleanup)
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
    task.result()
    if interrupted:
        raise asyncio.CancelledError
