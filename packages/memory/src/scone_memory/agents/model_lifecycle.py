"""Join optional cleanup of a factory-owned model before releasing its caller."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
import inspect
from typing import cast


async def _finish(close: Callable[[], object]) -> tuple[bool, bool]:
    async def invoke() -> bool:
        try:
            pending = close()
            if not inspect.isawaitable(pending):
                return True
            await pending
        except BaseException:
            # Keep provider exceptions out of shielded-future diagnostics.
            return True
        return False

    closing = asyncio.create_task(invoke())
    cancelled = False
    while not closing.done():
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            cancelled = cancelled or bool(current and current.cancelling())
    return (True if closing.cancelled() else closing.result()), cancelled


@asynccontextmanager
async def owned_model(model: object) -> AsyncIterator[bool]:
    """Yield whether cleanup requires a final evidence/deadline recheck.

    Adapters with owned resources implement nonblocking async aclose(). Cleanup
    must cooperate and finish; cancellation cannot abandon an owned close task.
    A primary failure or pause survives a secondary close error with a fixed
    diagnostic note. Provider cleanup exceptions are never chained or logged.
    """
    try:
        closer = getattr(model, 'aclose')
    except AttributeError:
        try:
            inspect.getattr_static(model, 'aclose')
        except AttributeError:
            closer = None
        else:
            raise RuntimeError('agent_model_cleanup_failed') from None
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current and current.cancelling():
            raise asyncio.CancelledError() from None
        raise RuntimeError('agent_model_cleanup_failed') from None
    except Exception:
        raise RuntimeError('agent_model_cleanup_failed') from None
    if closer is not None and not callable(closer):
        raise ValueError('invalid agent model cleanup')
    primary: BaseException | None = None
    try:
        yield closer is not None
    except BaseException as error:
        primary = error
        raise
    finally:
        if closer is not None:
            failed, cancelled = await _finish(cast(Callable[[], object], closer))
            if cancelled and not isinstance(primary, asyncio.CancelledError):
                primary = asyncio.CancelledError()
                if failed:
                    primary.add_note('agent_model_cleanup_failed')
                raise primary from None
            if failed:
                if primary is None:
                    raise RuntimeError('agent_model_cleanup_failed') from None
                primary.add_note('agent_model_cleanup_failed')
