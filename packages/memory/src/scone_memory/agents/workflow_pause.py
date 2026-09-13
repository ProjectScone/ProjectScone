"""Explicit durable pause contracts and authenticated checkpoint tickets."""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .workflow import JSONValue, StepCheckpoints, StepContext, WorkflowInputStep, WorkflowStep


@dataclass(frozen=True)
class WorkflowPaused:
    """Name a nonempty checkpoint; this is not a completed step or approval."""
    checkpoint: str


@dataclass(frozen=True)
class WorkflowPausableStep:
    """Resume only after an acknowledged pause, never after an unknown attempt.

    Trusted callbacks own their phase journal and external-operation semantics.
    They must validate checkpoint contents before reuse. A byte digest establishes
    checkpoint identity, not approval or completion of an external operation.
    """
    step_id: str
    version: str
    run: Callable[[StepContext], Awaitable[JSONValue | WorkflowPaused]]
    max_resumes: int = 32


@dataclass(frozen=True)
class PauseReceipt:
    checkpoint: str
    digest: str

    def payload(self) -> dict[str, str]:
        return {'checkpoint': self.checkpoint, 'digest': self.digest}


def pause_receipt(request: WorkflowPaused, checkpoints: StepCheckpoints) -> PauseReceipt:
    from .workflow import WorkflowError, _name
    if type(request) is not WorkflowPaused:
        raise WorkflowError('invalid_pause')
    _name(request.checkpoint)
    value = checkpoints.get(request.checkpoint)
    if not value:
        raise WorkflowError('pause_checkpoint_required')
    return PauseReceipt(request.checkpoint, hashlib.sha256(value).hexdigest())


def validate_pause(receipt: Mapping[str, str], checkpoints: StepCheckpoints) -> None:
    from .workflow import WorkflowError
    value = checkpoints.get(receipt['checkpoint'])
    if not value or not hmac.compare_digest(hashlib.sha256(value).hexdigest(), receipt['digest']):
        raise WorkflowError('pause_checkpoint_changed')


def paused_state(state: Mapping[str, JSONValue],
                 steps: Sequence[WorkflowStep | WorkflowInputStep | WorkflowPausableStep],
                 dependencies: Mapping[str, Sequence[str]] | None) -> dict[str, dict[str, str]]:
    """Validate every ticket before status, inspection or execution can use it."""
    from .workflow import WorkflowError, _name
    raw = state.get('pauses', {})
    if type(raw) is not dict or len(raw) > 32:
        raise WorkflowError('journal_key_or_integrity')
    results, attempts = state['results'], state['attempts']
    inflight = state.get('inflight_steps', [state['inflight']] if state['inflight'] else [])
    if not isinstance(results, dict) or not isinstance(attempts, dict) or not isinstance(inflight, list):
        raise WorkflowError('journal_key_or_integrity')
    declared = {step.step_id: step for step in steps}
    tickets: dict[str, dict[str, str]] = {}
    for name, value in raw.items():
        step = declared.get(name)
        count = attempts.get(name)
        if (not isinstance(step, WorkflowPausableStep) or type(count) is not int
                or not 1 <= count <= step.max_resumes + 1 or name in results or name in inflight
                or name == state.get('inflight') or type(value) is not dict
                or set(value) != {'checkpoint', 'digest'}):
            raise WorkflowError('journal_key_or_integrity')
        if any(type(item) is not str for item in value.values()):
            raise WorkflowError('journal_key_or_integrity')
        ticket = cast(dict[str, str], dict(value))
        try:
            _name(ticket['checkpoint'])
        except WorkflowError:
            raise WorkflowError('journal_key_or_integrity') from None
        if re.fullmatch(r'[0-9a-f]{64}', ticket['digest']) is None:
            raise WorkflowError('journal_key_or_integrity')
        required = (dependencies[name] if dependencies is not None
                    else [item.step_id for item in steps[:list(declared).index(name)]])
        if not set(required) <= results.keys():
            raise WorkflowError('journal_key_or_integrity')
        tickets[name] = ticket
    return tickets
