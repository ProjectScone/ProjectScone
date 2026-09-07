"""Cancellation joins owned cleanup rather than repeatedly interrupting it."""

import asyncio


def cancel_once(task):
    if not task.done() and not task.cancelling():
        task.cancel()


async def settle(task):
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = cancelled or bool(asyncio.current_task().cancelling())
        except Exception:
            break
    try:
        return task.result(), cancelled
    except BaseException as exc:
        return exc, cancelled
