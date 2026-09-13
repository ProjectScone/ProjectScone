"""Read-only delivery of authorized native agent observations."""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
import json
import re

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Message, Send

from ..agents.history_models import AgentHistoryPage
from ..agents.run_service import AgentRunService
from ..agents.workflow import WorkflowError
from .agent_runs import _failure, _response

STREAM_SECONDS = 30.0
POLL_SECONDS = 0.5
VERIFY_SECONDS = 5.0
SEND_SECONDS = 5.0


def _selection(request: Request) -> tuple[int, str | None]:
    query = request.query_params
    if any(name not in ('limit', 'after') or len(query.getlist(name)) != 1 for name in query):
        raise WorkflowError('invalid_history_query')
    raw = query.get('limit', '50')
    if re.fullmatch(r'[1-9][0-9]{0,2}', raw) is None or int(raw) > 100:
        raise WorkflowError('invalid_history_query')
    after = query.get('after')
    if after is not None and (not after or len(after) > 4096):
        raise WorkflowError('invalid_history_cursor')
    return int(raw), after


def _stream_selection(request: Request) -> tuple[int, str | None]:
    limit, after = _selection(request)
    headers = request.headers.getlist('last-event-id')
    if len(headers) > 1:
        raise WorkflowError('invalid_history_cursor')
    previous = headers[0] if headers else None
    if previous is not None:
        if not previous or len(previous) > 4096 or (after is not None and after != previous):
            raise WorkflowError('invalid_history_cursor')
        after = previous
    return limit, after


def _payload(space: str, run_id: str, page: AgentHistoryPage) -> dict[str, object]:
    return {
        'space': space,
        'run_id': run_id,
        'available': page.available,
        'items': [entry.model_dump(mode='json') for entry in page.items],
        'next_after': page.next_after,
        'retained_from': page.retained_from,
        'omitted': page.omitted,
    }


def _frame(kind: str, data: dict[str, object], cursor: str | None = None) -> str:
    identifier = '' if cursor is None else 'id: ' + cursor + '\n'
    return 'event: ' + kind + '\n' + identifier + 'data: ' + json.dumps(
        data, ensure_ascii=False, allow_nan=False, separators=(',', ':')
    ) + '\n\n'


class _WindowOver(Exception):
    """The observation window ran out during a delivery that was otherwise
    within its verification bound. Not a failure of the source."""


async def _within(verify: float, window: float,
                  deliver: Callable[[], Awaitable[AgentHistoryPage]]) -> AgentHistoryPage:
    """Deliver within both bounds, or say which one was missed.

    ``verify`` is the claim the route makes about verification; missing it
    is a refusal. ``window`` is how much observation time is left; missing
    only that is the window ending, which is ordinary. The two were one
    ``TimeoutError`` at first, and near the end of a window a delivery
    costing a few milliseconds was refused as `history_unavailable` --
    telling a reader the source had failed when the clock had merely run
    down.

    ``asyncio.timeout`` cancels at an ``await``. Delivery does its
    ``stat``s, opens the journal and reads the tail synchronously, so a
    stalled disk cannot be interrupted: the call returns whenever the disk
    does, and the timeout's callback has not yet had a turn to fire. Left
    there, an answer that took four times the stated bound would go out
    as a healthy frame. So the clock is read again after the await, and
    an overrun is refused exactly as an interruption would have been.
    This does not shorten the stall -- nothing short of a thread can --
    but the bound the route claims is then true at the only place a
    reader can observe it.
    """
    loop = asyncio.get_running_loop()
    begun = loop.time()
    page: AgentHistoryPage | None = None
    try:
        async with asyncio.timeout(min(verify, window)):
            page = await deliver()
    except TimeoutError:
        pass
    elapsed = loop.time() - begun
    if elapsed > verify:
        raise TimeoutError
    if page is None:
        # The timeout fired at an await and it was the window's, not the
        # verification bound's. A synchronous stall that merely outlasts
        # the window needs nothing here: the loop's own clock ends the
        # observation before the next delivery. Checking `elapsed >
        # window` as well was provably redundant and is gone.
        raise _WindowOver
    return page


class _HistoryResponse(StreamingResponse):
    def __init__(self, output: AsyncGenerator[str, None]) -> None:
        self._history_output = output
        super().__init__(output, media_type='text/event-stream', headers={
            'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no', 'X-Content-Type-Options': 'nosniff',
        })

    async def stream_response(self, send: Send) -> None:
        async def bounded_send(message: Message) -> None:
            # A client that stops reading cannot retain an observer indefinitely.
            async with asyncio.timeout(SEND_SECONDS):
                await send(message)

        try:
            await super().stream_response(bounded_send)
        finally:
            # StreamingResponse does not close a suspended iterator after a send
            # failure. Ours owns no execution, but still releases it immediately.
            await self._history_output.aclose()


def mount_agent_history_routes(
    app: FastAPI,
    service: AgentRunService,
    space_for: Callable[..., Awaitable[str]],
    assert_current_space: Callable[[Request, str], None],
) -> None:
    @app.get('/v1/agent-runs/{run_id}/history')
    async def history(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            limit, after = _selection(request)
            page = await service.history_for_delivery(
                space, run_id, after=after, limit=limit,
                admission_guard=lambda: assert_current_space(request, space),
            )
            assert_current_space(request, space)
            return _response(_payload(space, run_id, page))
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.get('/v1/agent-runs/{run_id}/history/stream')
    async def stream(run_id: str, request: Request, space: str = Depends(space_for)) -> Response:
        try:
            limit, after = _stream_selection(request)
            # Admission has no window yet: only the verification bound applies.
            await _within(VERIFY_SECONDS, VERIFY_SECONDS, lambda: service.history_for_delivery(
                space, run_id, after=after, limit=limit,
                admission_guard=lambda: assert_current_space(request, space),
            ))
        except (WorkflowError, ValueError) as error:
            return _failure(error)
        except (TimeoutError, _WindowOver):
            return _failure(WorkflowError('history_unavailable'))

        async def output() -> AsyncGenerator[str, None]:
            cursor = after
            first = True
            deadline = asyncio.get_running_loop().time() + STREAM_SECONDS
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    yield _frame('end', {'reason': 'observation_window_ended'})
                    return
                try:
                    # Headers and every previous send may have suspended. Never
                    # reuse their earlier source/authority verification here.
                    page = await _within(VERIFY_SECONDS, remaining, lambda: service.history_for_delivery(
                        space, run_id, after=cursor, limit=limit,
                        admission_guard=lambda: assert_current_space(request, space),
                    ))
                    assert_current_space(request, space)
                    if first or page.items or page.omitted:
                        yield _frame('history', _payload(space, run_id, page), page.next_after)
                        first = False
                    else:
                        yield ': keep-alive\n\n'
                    cursor = page.next_after
                    if not page.available:
                        yield _frame('end', {'reason': 'history_unavailable'})
                        return
                except _WindowOver:
                    yield _frame('end', {'reason': 'observation_window_ended'})
                    return
                except Exception:
                    # Authority callbacks and storage backends may raise host
                    # exceptions. Once headers are sent, disclose only refusal;
                    # cancellation (BaseException) still propagates.
                    yield _frame('error', {'reason': 'history_unavailable'})
                    return
                if len(page.items) < limit:
                    await asyncio.sleep(min(POLL_SECONDS, max(0.0, deadline - asyncio.get_running_loop().time())))

        return _HistoryResponse(output())
