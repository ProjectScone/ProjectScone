"""Live delivery of the answer a running step is writing.

Metadata streams over ``/history/stream``; this is the text. The route
reads the process-local window the run service holds for a running step
and publishes it the way conversations publish a reply: ``text`` frames
with their sequence as the SSE id, ``withdraw`` when text streamed before a
tool turn was not the answer, ``gap`` when a reader fell behind the window,
``terminal`` once the run has a receipt, and ``end`` when the window is
gone or the observation window closed. Admission is the history route's:
the run must exist for this recipient with its committed and paused
sources verified, and the recipient is checked again before every frame;
a recipient that lost access after the headers gets a fixed ``error``
frame and nothing more. What streams is provisional -- the receipt is what
``/result`` returns -- and a window does not survive the process that
held it, so a reader after a restart is pointed at the receipt.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
import re

from fastapi import Depends, FastAPI, Request
from fastapi.responses import Response

from ..agents.run_service import AgentRunService
from ..agents.workflow import WorkflowError
from .text_stream import TextWindow
from .agent_history import VERIFY_SECONDS, _frame, _HistoryResponse, _within, _WindowOver
from .agent_runs import _failure

STREAM_SECONDS = 30.0
POLL_SECONDS = 0.5
_TERMINAL = frozenset({'completed', 'failed', 'cancelled', 'deadline', 'outcome_unknown', 'retry_not_allowed',
                       'sources_invalid', 'verification_unavailable', 'unavailable', 'paused', 'awaiting_input'})


def _cursor(request: Request) -> int:
    query = request.query_params
    if any(name != 'after' or len(query.getlist(name)) != 1 for name in query):
        raise WorkflowError('invalid_text_cursor')
    headers = request.headers.getlist('last-event-id')
    if len(headers) > 1:
        raise WorkflowError('invalid_text_cursor')
    raw = query.get('after')
    previous = headers[0] if headers else None
    if raw is not None and previous is not None and raw != previous:
        raise WorkflowError('invalid_text_cursor')
    chosen = raw if raw is not None else (previous if previous is not None else '0')
    if re.fullmatch(r'0|[1-9][0-9]{0,18}', chosen) is None or int(chosen) > 2**63 - 1:
        raise WorkflowError('invalid_text_cursor')
    return int(chosen)


def mount_agent_text_routes(
    app: FastAPI,
    service: AgentRunService,
    space_for: Callable[..., Awaitable[str]],
    assert_current_space: Callable[[Request, str], None],
) -> None:
    @app.get('/v1/agent-runs/{run_id}/steps/{step_id}/text/stream')
    async def stream(run_id: str, step_id: str, request: Request, space: str = Depends(space_for)) -> Response:
        try:
            if not service.public_text:
                raise WorkflowError('public_text_not_configured')
            cursor = _cursor(request)
            # The history route's admission: the run exists for this recipient
            # and its committed and paused sources verify. Metadata only is
            # read here; nothing executes.
            await _within(VERIFY_SECONDS, VERIFY_SECONDS, lambda: service.history_for_delivery(
                space, run_id, limit=1, admission_guard=lambda: assert_current_space(request, space),
            ))
            if not service.step_known(space, run_id, step_id):
                raise WorkflowError('step_not_found')
            window = service.text_window(space, run_id, step_id)
            if window is not None and not window.closed and cursor > window.last_sequence:
                raise WorkflowError('text_cursor_ahead')
        except (WorkflowError, ValueError) as error:
            return _failure(error)
        except (TimeoutError, _WindowOver):
            return _failure(WorkflowError('text_unavailable'))

        async def output() -> AsyncGenerator[str, None]:
            after = cursor
            loop = asyncio.get_running_loop()
            deadline = loop.time() + STREAM_SECONDS
            # A reader who arrives while the step writes is promised its text:
            # the window it attaches to keeps what lands until it has read
            # it, even when the step finishes a chunk ahead of it. One who
            # arrives after the end reads the receipt and no provisional text.
            listening: TextWindow | None = None

            async def follow():
                nonlocal after, listening
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        yield _frame('end', {'reason': 'observation_window_ended'})
                        return
                    # Headers and every previous send may have suspended; the
                    # recipient is checked again before anything is published.
                    try:
                        assert_current_space(request, space)
                        status = await service.status(space, run_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        yield _frame('error', {'reason': 'text_unavailable'})
                        return
                    if status is None:
                        yield _frame('end', {'reason': 'window_unavailable'})
                        return
                    current = service.text_window(space, run_id, step_id)
                    if listening is None and current is not None and not current.closed:
                        listening = current
                        listening.attach()
                    if listening is not None:
                        current = listening
                    if current is None or (current.closed and current.next_after(after) == (None, None)):
                        # No provisional text to read. The step's answer, if it has
                        # one, is in the receipt: say so once the run has one, and
                        # keep waiting for it -- bounded by the observation window --
                        # while other steps of the run are still running.
                        if status.status in _TERMINAL and not status.active_local:
                            yield _frame('terminal', {'status': status.status, 'read_receipt': True})
                            return
                        await asyncio.sleep(min(POLL_SECONDS, max(remaining, 0.0)))
                        continue
                    gap, chunk = current.next_after(after)
                    if gap is not None:
                        yield _frame('gap', {'after': after, 'next_sequence': gap})
                        after = gap - 1
                        continue
                    if chunk is not None:
                        sequence, text = chunk
                        if text is None:
                            yield _frame('withdraw', {'sequence': sequence}, str(sequence))
                        else:
                            yield _frame('text', {'sequence': sequence, 'text': text}, str(sequence))
                        after = sequence
                        continue
                    try:
                        await asyncio.wait_for(current.wait_after(after), timeout=min(POLL_SECONDS, max(remaining, 0.001)))
                    except TimeoutError:
                        pass

            try:
                async for frame in follow():
                    yield frame
            finally:
                if listening is not None:
                    listening.detach()

        return _HistoryResponse(output())
