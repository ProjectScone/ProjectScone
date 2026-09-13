"""Immutable identities and revisioned records for exact tool-call approvals."""
from __future__ import annotations

from datetime import datetime
import json
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .workflow import JSONValue, _encode, _name

Digest = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
Name = Annotated[str, Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9._:-]+$')]
Decision = Literal['approve', 'deny']


class ApprovalCall(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    step_id: Name
    selection_id: Name
    agent_id: Name
    model_id: Name
    binding: Digest
    tool_name: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    tool_revision: Name
    tool_digest: Digest
    arguments_json: str = Field(max_length=16000)
    operation_digest: Digest

    @model_validator(mode='after')
    def canonical_arguments(self) -> Self:
        value = json.loads(self.arguments_json)
        if not isinstance(value, dict) or _encode(cast(JSONValue, value), 16000).decode() != self.arguments_json:
            raise ValueError('approval arguments must be canonical JSON')
        return self

    def arguments(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self.arguments_json))


class ToolApprovalRecord(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    space: str = Field(min_length=1, max_length=64, pattern=r'^[a-z0-9_-]+$')
    run_id: Name
    request_id: Digest
    invocation_digest: Digest
    call: ApprovalCall
    revision: int = Field(ge=1, le=4)
    created_at: datetime
    decision: Decision | None = None
    decided_by: Name | None = None
    decided_at: datetime | None = None
    activation_id: Name | None = None
    activated_at: datetime | None = None
    consumed_at: datetime | None = None

    @model_validator(mode='after')
    def stage(self) -> Self:
        for value in (self.created_at, self.decided_at, self.activated_at, self.consumed_at):
            if value is not None and value.tzinfo is None:
                raise ValueError('approval timestamps require a timezone')
        if (any((item is not None) != (self.revision >= 2) for item in (self.decision, self.decided_by, self.decided_at))
                or any((item is not None) != (self.revision >= 3) for item in (self.activation_id, self.activated_at))
                or (self.consumed_at is not None) != (self.revision == 4)):
            raise ValueError('invalid approval revision')
        return self


class ToolApprovalActivation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    space: str = Field(min_length=1, max_length=64, pattern=r'^[a-z0-9_-]+$')
    run_id: Name
    activation_id: Name
    invocation_digest: Digest
    decisions: dict[str, int] = Field(min_length=1, max_length=32)
    decision_digests: dict[str, Digest]
    created_at: datetime

    @model_validator(mode='after')
    def valid(self) -> Self:
        if self.created_at.tzinfo is None or self.decisions.keys() != self.decision_digests.keys():
            raise ValueError('invalid approval activation')
        for request_id, revision in self.decisions.items():
            _name(request_id)
            if len(request_id) != 64 or any(c not in '0123456789abcdef' for c in request_id) or revision != 2:
                raise ValueError('activation requires decided revisions')
        return self
