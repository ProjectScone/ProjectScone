"""Human-input receipts bind replies to their original task and run."""
from __future__ import annotations

import json

from dataclasses import dataclass, replace
from typing import Optional

from ._wire import identifier, integer, invalid, items, record, text, timestamp
from .agent_models import HumanInput, ModelTask, RunRequest, TaskPlan


@dataclass(frozen=True)
class InputRecord:
    space: str
    run_id: str
    task_id: str
    prompt: str
    context: str
    max_response_bytes: int
    revision: int
    response: Optional[str]
    activation_id: Optional[str]
    created_at: str
    responded_at: Optional[str]

    @classmethod
    def from_json(cls, value: object, *, expected_space: str, run_id: str) -> InputRecord:
        row = record(value)
        if row.get('space') != expected_space or row.get('run_id') != identifier(run_id):
            raise invalid('input identity')
        maximum = integer(row.get('max_response_bytes'), 1, 4000)
        revision = integer(row.get('revision'), 1, 3)
        response = text(row['response'], maximum) if row.get('response') is not None else None
        activation = identifier(row['activation_id']) if row.get('activation_id') is not None else None
        responded = timestamp(row['responded_at']) if row.get('responded_at') is not None else None
        if ((revision == 1) != (response is None) or (response is None) != (responded is None)
                or (revision == 3) != (activation is not None)):
            raise invalid('input revision state')
        context = row.get('context')
        # Context may be empty or whitespace; its encoded size is still bounded.
        if not isinstance(context, str):
            raise invalid('input context')
        text('x' + context, 32001, 'input context')
        return cls(expected_space, run_id, identifier(row.get('task_id')), text(row.get('prompt'), 2000),
                   context, maximum, revision, response, activation, timestamp(row.get('created_at')), responded)

    def match(self, request: RunRequest) -> None:
        if self.space != request.space or self.run_id != request.run_id or not isinstance(request.plan.plan, TaskPlan):
            raise invalid('input request binding')
        task = next((task for task in request.plan.plan.tasks if task.task_id == self.task_id), None)
        if (not isinstance(task, HumanInput) or task.prompt != self.prompt
                or task.max_response_bytes != self.max_response_bytes):
            raise invalid('input task binding')
        try:
            context = items(json.loads(self.context), 31)
        except (ValueError, RecursionError):
            raise invalid('input dependency context') from None
        rows = tuple(record(value) for value in context)
        if tuple(row.get('task_id') for row in rows) != task.depends_on:
            raise invalid('input dependency context')
        tasks = {value.task_id: value for value in request.plan.plan.tasks}
        for row, task_id in zip(rows, task.depends_on):
            dependency = tasks[task_id]
            output = row.get('text')
            if not isinstance(output, str):
                raise invalid('input dependency text')
            text('x' + output, 32001, 'input dependency text')
            if isinstance(dependency, HumanInput):
                if set(row) != {'task_id', 'kind', 'text'} or row.get('kind') != 'human_input':
                    raise invalid('input human dependency')
                text(output, dependency.max_response_bytes)
            elif isinstance(dependency, ModelTask):
                if (set(row) != {'task_id', 'agent_id', 'text', 'source_status', 'evidence_ids'}
                        or row.get('agent_id') != dependency.agent_id):
                    raise invalid('input model dependency')
                evidence = tuple(text(value, 32000, 'evidence identifier') for value in items(row.get('evidence_ids'), 2048))
                if len(set(evidence)) != len(evidence) or row.get('source_status') != ('retained' if evidence else 'none'):
                    raise invalid('input dependency evidence')

    def same_input(self, other: InputRecord) -> bool:
        return replace(self, revision=1, response=None, activation_id=None, responded_at=None) == replace(
            other, revision=1, response=None, activation_id=None, responded_at=None)

    def same_reply(self, other: InputRecord) -> bool:
        return replace(self, revision=2, activation_id=None) == replace(other, revision=2, activation_id=None)
