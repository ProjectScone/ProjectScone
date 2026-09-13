"""Read-only verification of a saved pending call and its retained sources."""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import cast

from .approval_context import _digest
from .approval_models import ToolApprovalRecord
from .catalog import BoundAgent
from .evidence_loop import ToolStep
from .turn_journal import TurnJournalError, TurnJournalInspection
from .workflow import JSONValue, WorkflowError, WorkflowPauseSnapshot, _encode
from ..integrations.scoped_tools import ScopedMemoryTools


async def inspect_tool_approval(snapshot: WorkflowPauseSnapshot, record: ToolApprovalRecord,
                                agent: BoundAgent, tools: ScopedMemoryTools) -> None:
    """Verify a current proposal; grant no activation, claim, or execution lease."""
    snapshot.check_current()
    call = record.call
    registration = next((tool for tool in agent.tools if tool.name == call.tool_name), None)
    if (record.revision == 4 or snapshot.checkpoint != 'agent-turn-v1'
            or snapshot.context.space != record.space or snapshot.context.run_id != record.run_id
            or snapshot.step_id != call.step_id or agent.fingerprint != call.binding
            or agent.definition.agent_id != call.agent_id or agent.model_id != call.model_id
            or registration is None or not registration.requires_approval
            or registration.revision != call.tool_revision or _digest(registration.info()) != call.tool_digest):
        raise WorkflowError('approval_not_pending')
    binding = _digest(tools.journal_binding())
    try:
        journal = TurnJournalInspection.read(snapshot.payload)
        operations = journal.operations()
        last_model = max(index for index, row in enumerate(operations) if row[0] == 'model')
        proposal = ToolStep.model_validate_json(json.dumps(operations[last_model][2]))
        executed = {row[1] for row in operations[last_model + 1:]}
        matched = False
        registered = {tool.name: tool for tool in agent.tools}
        for candidate in proposal.calls:
            request = cast(JSONValue, candidate.model_dump(mode='json'))
            digest = hashlib.sha256(_encode(request, 4 * 1024 * 1024)).hexdigest()
            if digest in executed or candidate.name not in registered:
                continue
            # Every attempted application call is journaled, even invalid ones.
            # A later member of this batch cannot jump over the first unfinished call.
            matched = (candidate.name == call.tool_name
                and registration.prepare_arguments(candidate.arguments) == call.arguments_json
                and journal.pending_identity('custom', request) == call.operation_digest)
            break
        if not matched:
            raise WorkflowError('approval_not_pending')
    except (TurnJournalError, ValueError, TypeError, RecursionError):
        raise WorkflowError('approval_key_or_integrity') from None
    try:
        async with asyncio.timeout(30) as timer:
            def check() -> None:
                active = asyncio.current_task()
                if active is not None and active.cancelling():
                    raise asyncio.CancelledError
                expires = timer.when()
                if timer.expired() or (expires is not None and asyncio.get_running_loop().time() >= expires):
                    raise WorkflowError('verification_unavailable')
                snapshot.check_current()
                if _digest(tools.journal_binding()) != binding:
                    raise WorkflowError('run_scope_changed')
            check()
            for kind, _, value in operations:
                if kind != 'memory':
                    continue
                if (not isinstance(value, dict) or set(value) != {'payload', 'source_digest'}
                        or not isinstance(value['payload'], str) or not isinstance(value['source_digest'], str)):
                    raise WorkflowError('approval_key_or_integrity')
                evidence = await tools.restore(value['payload'], value['source_digest'])
                if not await evidence.validate():
                    raise WorkflowError('sources_invalid')
                check()
            check()
    except (OSError, TimeoutError):
        raise WorkflowError('verification_unavailable') from None
    except ValueError:
        raise WorkflowError('sources_invalid') from None
    snapshot.check_current()
    if _digest(tools.journal_binding()) != binding:
        raise WorkflowError('run_scope_changed')
