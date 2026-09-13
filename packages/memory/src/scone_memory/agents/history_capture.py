"""Owned native observation, separate from model and approval execution authority."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
from uuid import uuid4
from typing import Literal

from .catalog import BoundAgent
from .workflow import StepContext, WorkflowError
from ..integrations.scoped_tools import ScopedMemoryTools
from .event_history import AgentEventHistoryStore
from .history_models import AgentCollectionEvent, AgentHistoryEntry
from .progress import AgentEventStream, AgentProgressEvent, AgentProgressGap, TerminalKind
from .run_store import AgentRunRequest
from .handoff_workflow import AgentHandoffPlan
from .task_workflow import AgentTaskPlan, _json


CollectionKind = Literal['collection_started', 'collection_finished', 'collection_failed']


def _check_deadline(deadline: float | None) -> None:
    owner = asyncio.current_task()
    if owner is not None and owner.cancelling():
        raise asyncio.CancelledError
    if deadline is not None and asyncio.get_running_loop().time() >= deadline:
        raise TimeoutError


class _Collection:
    def __init__(self, history: AgentRunHistory, step_id: str, selection_id: str) -> None:
        self.history, self.step_id, self.selection_id = history, step_id, selection_id
        self.collection_id = uuid4().hex
        self.generation: str | None = None
        self.invocation_id: str | None = None
        self.seen = self.lost = self.last = 0
        self.terminal: TerminalKind | None = None
        self.failed = self.interrupted = False
        self.disabled = False

    def marker(self, kind: CollectionKind) -> AgentCollectionEvent:
        return AgentCollectionEvent(
            kind,
            self.collection_id,
            datetime.now(timezone.utc).isoformat(),
            self.invocation_id,
            self.last,
            self.seen,
            self.lost,
            self.terminal,
            'history_unavailable' if self.failed else 'collection_interrupted' if self.interrupted else None,
        )

    def write(self, event: AgentProgressEvent | AgentProgressGap | AgentCollectionEvent) -> None:
        if self.disabled:
            return
        try:
            _, self.generation = self.history.store._append(
                self.history.request,
                step_id=self.step_id,
                selection_id=self.selection_id,
                event=event,
                collection_id=self.collection_id,
                activation_id=self.history.activation_id,
                expected_generation=self.generation,
            )
        except Exception:
            self.failed = True
            # With an uncertain first commit, there is no known generation to
            # pin. Do not retry and accidentally recreate a purged history.
            if self.generation is None:
                self.disabled = True

    async def consume(self, stream: AgentEventStream) -> None:
        try:
            async for event in stream:
                self.invocation_id = event.invocation_id
                if isinstance(event, AgentProgressGap):
                    self.lost += event.last_sequence - event.first_sequence + 1
                    self.last = event.last_sequence
                else:
                    self.seen += 1
                    self.last = event.sequence
                    if event.kind in ('turn_completed', 'turn_paused', 'turn_failed', 'turn_cancelled'):
                        self.terminal = event.kind
                if not self.failed:
                    self.write(event)
        except asyncio.CancelledError:
            self.interrupted = True
        except BaseException:
            # Never leave a failed observer task carrying private diagnostics.
            self.failed = True


class AgentRunHistory:
    """Bind a collector to one saved request; callers own authorization and storage."""

    def __init__(
        self,
        store: AgentEventHistoryStore,
        request: AgentRunRequest,
        *,
        activation_id: str | None = None,
        max_events: int = 128,
    ) -> None:
        _, _, self.request = store._identity(request)
        plan = self.request.plan.plan
        implementation = (
            'scone-agent-task-v1'
            if isinstance(plan, AgentTaskPlan)
            else (
                'scone-agent-handoff-v1'
                if isinstance(plan, AgentHandoffPlan)
                else 'scone-interactive-agent-v1'
            )
        )
        self.signature = hashlib.sha256(
            _json(
                {
                    'implementation': implementation,
                    'plan': plan.model_dump(mode='json'),
                    'bindings': self.request.plan.bindings,
                }
            ).encode()
        ).hexdigest()
        AgentEventStream(max_events=max_events)
        self.store, self.activation_id, self.capacity = store, activation_id, max_events

    @asynccontextmanager
    async def capture(
        self, *, step_id: str, selection_id: str, deadline: float | None = None
    ) -> AsyncIterator[AgentEventStream]:
        stream = AgentEventStream(max_events=self.capacity)
        collection = _Collection(self, step_id, selection_id)
        start = collection.marker('collection_started')
        entry = AgentHistoryEntry(
            position=1,
            step_id=step_id,
            selection_id=selection_id,
            event=start,
            collection_id=collection.collection_id,
            activation_id=self.activation_id,
        )
        self.store._selection(self.request, entry)
        _check_deadline(deadline)
        collection.write(start)
        reader = asyncio.create_task(collection.consume(stream))
        primary: BaseException | None = None
        try:
            _check_deadline(deadline)
            yield stream
        except BaseException as error:
            primary = error
            raise
        finally:
            if not stream._done:
                collection.interrupted = True
                reader.cancel()
            cancelled = False
            while not reader.done():
                try:
                    await asyncio.shield(reader)
                except asyncio.CancelledError:
                    owner = asyncio.current_task()
                    cancelled = cancelled or bool(owner and owner.cancelling())
            if collection.terminal is None:
                collection.interrupted = True
            kind: CollectionKind = (
                'collection_failed' if collection.failed or collection.interrupted else 'collection_finished'
            )
            collection.write(collection.marker(kind))
            owner = asyncio.current_task()
            cancelled = cancelled or bool(owner and owner.cancelling())
            if cancelled and not isinstance(primary, asyncio.CancelledError):
                raise asyncio.CancelledError() from None
            if primary is None:
                _check_deadline(deadline)


@asynccontextmanager
async def collect_agent(
    history: AgentRunHistory | None,
    *,
    agent: BoundAgent,
    context: StepContext,
    tools: ScopedMemoryTools,
    step_id: str,
    selection_id: str,
) -> AsyncIterator[AgentEventStream | None]:
    if history is None:
        yield None
        return
    request = history.request
    binding = tools.journal_binding()
    if (
        context.run_id != request.run_id
        or context.space != request.space
        or context.inputs != request.question
        or context.scope.get('plan') != history.signature
        or context.scope.get('recall') != request.scope
        or context.scope.get('exclude_session_id') != request.exclude_session_id
        or binding['space'] != request.space
        or binding['scope'] != request.recall_scope().kwargs()
        or binding['excluded_session'] != request.exclude_session_id
        or request.plan.bindings.get(selection_id) != agent.fingerprint
    ):
        raise WorkflowError('history_invocation_mismatch')
    async with history.capture(
        step_id=step_id, selection_id=selection_id, deadline=context.deadline
    ) as stream:
        yield stream
