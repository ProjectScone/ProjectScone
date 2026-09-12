"""Bounded local ownership of background agent runs and their durable receipts."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
import hashlib
import fcntl
import hmac
import os
from pathlib import Path
import stat

from .catalog import AgentCatalog
from .handoff_workflow import AgentHandoffPlan, AgentHandoffWorkflow, HandoffResult
from .input_store import AgentInputRecord, AgentInputStore
from .interactive_plan import InteractiveAgentPlan
from .interactive_workflow import InteractiveAgentWorkflow
from .plan_store import AgentPlanStore, PlanConflict
from .run_store import AgentRunRequest, AgentRunStore, RunConflict
from .task_workflow import AgentWorkflow
from .workflow import WorkflowError, WorkflowResult, WorkflowStatus, _integer, _name, _private_file
from ..core.validation import check_space
from ..memory.engine import MemoryEngine
from ..retrieval.recall_scope import RecallScope

AgentExecution = AgentWorkflow | AgentHandoffWorkflow | InteractiveAgentWorkflow


def _execution_progress(workflow: AgentExecution, request: AgentRunRequest) -> WorkflowStatus | None:
    if isinstance(workflow, AgentHandoffWorkflow):
        return workflow.progress(request.run_id, request.question)
    return workflow.status(request.run_id, request.question)


@dataclass(frozen=True)
class AgentRunStatus:
    run_id: str
    space: str
    created_at: str
    workflow_id: str
    plan_revision: int
    status: str
    active_local: bool
    completed_steps: tuple[str, ...]
    inflight: str | None
    outcome_unknown: bool
    error_class: str | None
    max_parallel: int = 1
    inflight_steps: tuple[str, ...] = ()
    waiting_steps: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentRunStatusPage:
    items: tuple[AgentRunStatus, ...]
    next_after: str | None


class AgentRunService:
    """One-process admission and cancellation; journals prevent duplicate execution.

    The caller owns authentication, the supplied memory engine and plan store.
    A started task outlives the request that admitted it. The host must await
    aclose on shutdown; cancellation and deadlines are cooperative. Other server
    processes cannot be cancelled through this instance. This is not a distributed
    worker queue. Registry and journal retention are owned by the host.
    """
    def __init__(self, directory: str | Path, *, key: bytes, catalog: AgentCatalog,
                 plans: AgentPlanStore, memory: MemoryEngine, scope_for: Callable[[str], RecallScope],
                 max_active: int = 4, max_runs: int = 4096, deadline_s: float = 120.0, max_parallel_tasks: int = 1) -> None:
        _integer(max_active, 1, 32)
        _integer(max_parallel_tasks, 1, 8)
        self._parallel = max_parallel_tasks
        if not callable(scope_for):
            raise ValueError('host scope resolver required')
        if type(key) is not bytes or len(key) != 32:
            raise WorkflowError('key_must_be_32_bytes')
        if type(deadline_s) not in (int, float) or not 0 < deadline_s <= 300:
            raise WorkflowError('invalid_budget')
        target = Path(directory).absolute()
        target.mkdir(mode=0o700, exist_ok=True)
        info = target.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise WorkflowError('private_directory_required')
        self._directory, self._key, self._catalog = target, key, catalog
        self._plans, self._memory, self._scope_for = plans, memory, scope_for
        self._maximum, self._deadline = max_active, deadline_s
        self._runs = AgentRunStore(target / 'requests.sqlite', key=key, max_runs=max_runs)
        self._inputs = AgentInputStore(self._runs)
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._workflows: dict[tuple[str, str], AgentExecution] = {}
        self._failures: dict[tuple[str, str], str] = {}
        self._owners: dict[tuple[str, str], int] = {}
        self._admissions: dict[tuple[str, str], asyncio.Task[object]] = {}
        self._closed = self._closing = False

    @property
    def max_parallel_tasks(self) -> int:
        return self._parallel

    async def policy(self, space: str) -> dict[str, str | int]:
        await self._space(space)
        return {'space': space, 'max_parallel_tasks': self._parallel, 'max_active_runs': self._maximum}

    def require_host(self, memory: MemoryEngine, plans: AgentPlanStore, catalog: AgentCatalog) -> None:
        if memory is not self._memory or plans is not self._plans or catalog is not self._catalog:
            raise ValueError('Agent run service must use this host memory, plan store and catalog')

    def _available(self) -> None:
        if self._closed or self._closing:
            raise WorkflowError('run_service_closed')

    async def _space(self, space: str) -> None:
        self._available()
        check_space(space)
        deleted = await self._memory.space_deleted(space)
        self._available()
        if deleted is not None:
            raise WorkflowError('space_deleted')

    def _scope(self, space: str) -> RecallScope:
        scope = self._scope_for(space)
        if not isinstance(scope, RecallScope):
            raise ValueError('host scope resolver must return a validated scope')
        return RecallScope.validated(**scope.kwargs())

    def _path(self, request: AgentRunRequest) -> Path:
        digest = hmac.new(self._key, b'agent-execution:' + request.space.encode() + b'\0' + request.run_id.encode(), hashlib.sha256).hexdigest()
        return self._directory / (digest + '.sqlite')

    def _claim(self, request: AgentRunRequest) -> int:
        descriptor = _private_file(Path(str(self._path(request)) + '.owner'))
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            raise WorkflowError('run_busy') from None
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def _open(self, request: AgentRunRequest) -> AgentExecution:
        plan = request.plan.checked_plan(self._catalog)
        scope = self._scope(request.space)
        if scope.as_dict() != request.scope:
            raise WorkflowError('run_scope_changed')
        if isinstance(plan, AgentHandoffPlan):
            return AgentHandoffWorkflow(self._path(request), key=self._key, catalog=self._catalog, plan=plan,
                memory=self._memory, space=request.space, scope=scope,
                exclude_session_id=request.exclude_session_id, deadline_s=self._deadline)
        if isinstance(plan, InteractiveAgentPlan):
            return InteractiveAgentWorkflow(self._path(request), key=self._key, catalog=self._catalog,
                request=request, memory=self._memory, inputs=self._inputs,
                activated=self._inputs.activated(request.space, request.run_id), deadline_s=self._deadline)
        return AgentWorkflow(self._path(request), key=self._key, catalog=self._catalog, plan=plan,
            memory=self._memory, space=request.space, scope=scope,
            exclude_session_id=request.exclude_session_id, deadline_s=self._deadline, max_parallel=request.max_parallel)

    def _progress(self, request: AgentRunRequest) -> WorkflowStatus | None:
        identity = (request.space, request.run_id)
        active = self._workflows.get(identity)
        if active is not None:
            return _execution_progress(active, request)
        if not self._path(request).exists():
            return None
        workflow = self._open(request)
        try:
            return _execution_progress(workflow, request)
        finally:
            workflow.close()

    def _status(self, request: AgentRunRequest) -> AgentRunStatus:
        progress = self._progress(request)
        identity = (request.space, request.run_id)
        active = identity in self._tasks or identity in self._admissions
        unknown = bool(progress and any(name not in progress.completed_steps for name in progress.attempts))
        return AgentRunStatus(run_id=request.run_id, space=request.space,
            created_at=request.created_at.isoformat(), workflow_id=request.plan.plan.workflow_id,
            plan_revision=request.plan.revision,
            status=('completed' if progress and progress.status == 'completed' else
                    'cancelled' if request.cancel_requested_at is not None and not active else
                    progress.status if progress else 'registered'), active_local=active,
            completed_steps=progress.completed_steps if progress else (),
            inflight=progress.inflight if progress else None, outcome_unknown=unknown and not active,
            error_class=progress.error_class if progress else self._failures.get(identity),
            max_parallel=request.max_parallel, inflight_steps=progress.inflight_steps if progress else (),
            waiting_steps=progress.waiting_steps if progress else ())

    async def status(self, space: str, run_id: str) -> AgentRunStatus | None:
        await self._space(space)
        request = self._runs.get(space, run_id)
        return None if request is None else self._status(request)

    async def list(self, space: str, *, limit: int = 20, after: str | None = None) -> AgentRunStatusPage:
        await self._space(space)
        page = self._runs.list(space, limit=limit, after=after)
        items = []
        for request in page.items:
            try:
                items.append(self._status(request))
            except WorkflowError as error:
                items.append(AgentRunStatus(run_id=request.run_id, space=request.space,
                    created_at=request.created_at.isoformat(), workflow_id=request.plan.plan.workflow_id,
                    plan_revision=request.plan.revision, status='unavailable', active_local=False,
                    completed_steps=(), inflight=None, outcome_unknown=True, error_class=error.code, max_parallel=request.max_parallel))
        return AgentRunStatusPage(tuple(items), page.next_after)

    async def request(self, space: str, run_id: str) -> AgentRunRequest | None:
        await self._space(space)
        return self._runs.get(space, run_id)

    async def start(self, space: str, run_id: str, *, workflow_id: str,
                    plan_revision: int, question: str, max_parallel: int = 1,
                    admission_guard: Callable[[], None] | None = None) -> AgentRunStatus:
        await self._space(space)
        if admission_guard is not None:
            admission_guard()
        self._available()
        _name(run_id)
        _name(workflow_id)
        _integer(plan_revision, 1, 2**63 - 1)
        _integer(max_parallel, 1, self._parallel)
        identity = (space, run_id)
        prior = self._runs.get(space, run_id)
        if prior is not None:
            if (prior.plan.plan.workflow_id != workflow_id or prior.plan.revision != plan_revision
                    or prior.question != question or prior.max_parallel != max_parallel or prior.scope != self._scope(space).as_dict()):
                raise RunConflict()
            prior.plan.checked_plan(self._catalog)
            current = self._status(prior)
            if current.active_local or current.status == 'completed':
                return current
            if current.outcome_unknown:
                raise WorkflowError('outcome_unknown')
            if prior.cancel_requested_at is not None:
                raise WorkflowError('run_cancelled')
            if current.status == 'sources_invalid':
                raise WorkflowError('sources_invalid')
            if isinstance(prior.plan.plan, InteractiveAgentPlan) and current.status == 'awaiting_input':
                return current
        if len(self._tasks) + len(self._admissions) >= self._maximum:
            raise WorkflowError('run_busy')
        if prior is None:
            saved = self._plans.get(space, workflow_id)
            if saved is None:
                raise WorkflowError('agent_plan_not_found')
            if saved.revision != plan_revision:
                raise PlanConflict()
            saved.checked_plan(self._catalog)
            prior = self._runs.register(space, run_id, plan=saved, question=question, scope=self._scope(space), max_parallel=max_parallel)
        descriptor = self._claim(prior)
        workflow = None
        try:
            current_request = self._runs.get(space, run_id)
            if current_request is None:
                raise WorkflowError('run_store_unavailable')
            if current_request.cancel_requested_at is not None:
                raise WorkflowError('run_cancelled')
            workflow = self._open(current_request)
            self._workflows[identity] = workflow
            admission = self._status(current_request)
        except BaseException:
            if workflow is not None:
                workflow.close()
            self._workflows.pop(identity, None)
            os.close(descriptor)
            raise
        self._owners[identity] = descriptor
        self._failures.pop(identity, None)
        task = asyncio.create_task(self._execute(current_request, workflow))
        self._tasks[identity] = task
        task.add_done_callback(lambda finished: self._finished(identity, workflow))
        return replace(admission, active_local=True)

    def _check_input_request(self, request: AgentRunRequest, guard: Callable[[], None] | None) -> None:
        self._available()
        if guard is not None:
            guard()
        request.plan.checked_plan(self._catalog)
        if request.scope != self._scope(request.space).as_dict():
            raise WorkflowError('run_scope_changed')
        if not isinstance(request.plan.plan, InteractiveAgentPlan):
            raise WorkflowError('interactive_plan_required')

    async def inputs(self, space: str, run_id: str, *,
                     admission_guard: Callable[[], None] | None = None) -> tuple[AgentInputRecord, ...]:
        await self._space(space)
        request = self._runs.get(space, run_id)
        if request is None:
            raise WorkflowError('run_not_found')
        self._check_input_request(request, admission_guard)
        # A separate read handle outlives an active owner's completion callback.
        # It never acquires the execution lease or runs a workflow step.
        workflow = self._open(request)
        assert isinstance(workflow, InteractiveAgentWorkflow)
        try:
            records = await workflow.inspect_inputs(run_id, request.question)
            self._check_input_request(request, admission_guard)
            return records
        finally:
            workflow.close()

    async def respond(self, space: str, run_id: str, task_id: str, *, response: str,
                      expected_revision: int, admission_guard: Callable[[], None] | None = None) -> AgentInputRecord:
        records = await self.inputs(space, run_id, admission_guard=admission_guard)
        if not any(record.task_id == task_id for record in records):
            raise WorkflowError('input_not_ready')
        # No await separates fresh source/scope verification from the transaction.
        return self._inputs.respond(space, run_id, task_id, response=response, expected_revision=expected_revision)

    async def continue_run(self, space: str, run_id: str, *, continuation_id: str,
                           responses: dict[str, int], admission_guard: Callable[[], None] | None = None) -> AgentRunStatus:
        await self._space(space)
        request = self._runs.get(space, run_id)
        if request is None:
            raise WorkflowError('run_not_found')
        self._check_input_request(request, admission_guard)
        _integer(request.max_parallel, 1, self._parallel)
        identity = (space, run_id)
        if identity in self._tasks or len(self._tasks) + len(self._admissions) >= self._maximum:
            raise WorkflowError('run_busy')
        owner = asyncio.current_task()
        if owner is None:
            raise WorkflowError('run_service_unavailable')
        descriptor = self._claim(request)
        self._admissions[identity] = owner
        self._owners[identity] = descriptor
        workflow: AgentExecution | None = None
        admitted = False
        try:
            current = self._status(request)
            if current.outcome_unknown:
                raise WorkflowError('outcome_unknown')
            if request.cancel_requested_at is not None:
                raise WorkflowError('run_cancelled')
            records = await self.inputs(space, run_id, admission_guard=admission_guard)
            if not set(responses) <= {record.task_id for record in records}:
                raise WorkflowError('input_not_ready')
            self._check_input_request(request, admission_guard)
            if len(self._tasks) + len(self._admissions) > self._maximum:
                raise WorkflowError('run_busy')
            self._inputs.activate(space, run_id, continuation_id, responses=responses)
            if current.status == 'completed':
                return current
            # Persisted activation can be reconciled by the same explicit request
            # after an admission failure. Startup and read paths never launch it.
            workflow = self._open(request)
            self._workflows[identity] = workflow
            admission = self._status(request)
            self._owners[identity] = descriptor
            self._failures.pop(identity, None)
            task = asyncio.create_task(self._execute(request, workflow))
            self._tasks[identity] = task
            task.add_done_callback(lambda finished: self._finished(identity, workflow))
            admitted = True
            return replace(admission, active_local=True)
        finally:
            self._admissions.pop(identity, None)
            if not admitted:
                self._owners.pop(identity, None)
                if workflow is not None:
                    workflow.close()
                    self._workflows.pop(identity, None)
                os.close(descriptor)

    async def _execute(self, request: AgentRunRequest, workflow: AgentExecution) -> None:
        identity = (request.space, request.run_id)
        try:
            await workflow.run(request.run_id, request.question)
        except asyncio.CancelledError:
            pass
        except WorkflowError as error:
            self._failures[identity] = error.code
        except Exception:
            self._failures[identity] = 'execution_failed'

    def _finished(self, identity: tuple[str, str], workflow: AgentExecution) -> None:
        try:
            workflow.close()
        except WorkflowError as error:
            self._failures[identity] = error.code
        finally:
            self._workflows.pop(identity, None)
            self._tasks.pop(identity, None)
            descriptor = self._owners.pop(identity, None)
            if descriptor is not None:
                os.close(descriptor)

    async def wait(self, space: str, run_id: str) -> AgentRunStatus | None:
        """Wait for this instance's task; cancelling the waiter leaves it owned."""
        await self._space(space)
        task = self._tasks.get((space, run_id))
        if task is not None:
            await asyncio.shield(task)
        return await self.status(space, run_id)

    async def cancel(self, space: str, run_id: str, *,
                     admission_guard: Callable[[], None] | None = None) -> AgentRunStatus | None:
        await self._space(space)
        if admission_guard is not None:
            admission_guard()
        self._available()
        request = self._runs.get(space, run_id)
        if request is None:
            return None
        task = self._tasks.get((space, run_id)) or self._admissions.get((space, run_id))
        descriptor = None
        if task is None:
            try:
                descriptor = self._claim(request)
            except WorkflowError as error:
                if error.code == 'run_busy':
                    raise WorkflowError('run_not_owned') from None
                raise
        try:
            progress = self._status(request)
            if progress.status == 'completed':
                return progress
            if task is None and progress.status == 'running':
                raise WorkflowError('run_not_owned')
            try:
                updated = self._runs.request_cancel(space, run_id)
                if updated is None:
                    raise WorkflowError('run_store_unavailable')
            finally:
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            return self._status(updated)
        finally:
            if descriptor is not None:
                os.close(descriptor)

    async def result(self, space: str, run_id: str) -> WorkflowResult | HandoffResult | None:
        await self._space(space)
        request = self._runs.get(space, run_id)
        if request is None:
            return None
        if (space, run_id) in self._tasks:
            raise WorkflowError('not_completed')
        workflow = self._open(request)
        try:
            result = await workflow.read_result(run_id, request.question)
            if self._scope(space).as_dict() != request.scope:
                raise WorkflowError('run_scope_changed')
            request.plan.checked_plan(self._catalog)
            return result
        finally:
            workflow.close()

    def close_idle(self) -> None:
        """Close before serving or after all work ends; never discard owned tasks."""
        if self._tasks or self._owners or self._admissions:
            raise WorkflowError('run_busy')
        if self._closed:
            return
        self._closing = True
        self._runs.close()
        self._closed = True

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closing = True
        tasks = (*self._tasks.values(), *self._admissions.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._runs.close()
        self._closed = True
