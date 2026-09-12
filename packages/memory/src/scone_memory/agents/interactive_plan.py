"""Explicit human-input task graphs, separate from legacy model-only plans."""
from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .catalog import Identifier
from .task_workflow import AgentTask
from .workflow import WorkflowError, _name


class HumanInputTask(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    kind: Literal['input']
    task_id: Identifier
    prompt: str = Field(min_length=1, max_length=2000)
    depends_on: tuple[Identifier, ...] = Field(default=(), max_length=31)
    max_response_bytes: int = Field(default=4000, ge=1, le=4000)

    @model_validator(mode='after')
    def valid(self) -> Self:
        try:
            for name in (self.task_id, *self.depends_on):
                _name(name)
        except WorkflowError:
            raise ValueError('invalid input task identifier') from None
        if (not self.prompt.strip() or len(self.prompt.encode()) > 2000
                or len(set(self.depends_on)) != len(self.depends_on) or self.task_id in self.depends_on):
            raise ValueError('invalid human input task')
        return self


class InteractiveAgentPlan(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    kind: Literal['interactive']
    workflow_id: Identifier
    tasks: tuple[AgentTask | HumanInputTask, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode='after')
    def valid(self) -> Self:
        ids = {task.task_id for task in self.tasks}
        if (len(ids) != len(self.tasks) or any(set(task.depends_on) - ids for task in self.tasks)
                or not any(isinstance(task, HumanInputTask) for task in self.tasks)):
            raise ValueError('interactive plan requires input tasks and unique known dependencies')
        self.ordered()
        return self

    def ordered(self) -> tuple[AgentTask | HumanInputTask, ...]:
        pending = list(self.tasks)
        ordered: list[AgentTask | HumanInputTask] = []
        done: set[str] = set()
        while pending:
            ready = [task for task in pending if set(task.depends_on) <= done]
            if not ready:
                raise ValueError('agent task dependencies contain a cycle')
            for task in ready:
                ordered.append(task)
                done.add(task.task_id)
                pending.remove(task)
        return tuple(ordered)
