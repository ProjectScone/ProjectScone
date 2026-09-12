"""Bounded sync control records and historical source outcomes."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..agents.catalog import Identifier
from ..core.validation import check_space
from ..core.errors import InvalidInput
from .document_source import DocumentSource

MAX_OUTCOMES = 300_000
TERMINAL_RESULTS = ('completed', 'partial')


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid',
                              revalidate_instances='always', hide_input_in_errors=True)


class SyncRunSpec(_Record):
    collection_id: Identifier
    configuration: str = Field(pattern=r'^[a-f0-9]{64}$')
    delete_missing: bool = False
    deadline_s: float = Field(default=300.0, gt=0, le=3600, allow_inf_nan=False)
    max_attempts: int = Field(default=3, ge=1, le=4)


class SyncRunRecord(_Record):
    space: str
    run_id: Identifier
    spec: SyncRunSpec
    created_at: datetime
    revision: int = Field(default=0, ge=0, le=2**31 - 1)
    attempt: int = Field(default=0, ge=0, le=4)
    status: Literal['registered', 'running', 'completed', 'partial', 'failed', 'cancelled'] = 'registered'
    last_started_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: Identifier | None = None
    collection_instance: str | None = Field(default=None, pattern=r'^[a-f0-9]{32}$')
    source_count: int = Field(default=0, ge=0, le=20_000)
    issue_count: int = Field(default=0, ge=0, le=MAX_OUTCOMES)
    outcome_count: int = Field(default=0, ge=0, le=MAX_OUTCOMES)
    skipped: int = Field(default=0, ge=0)

    @model_validator(mode='after')
    def consistent(self) -> Self:
        check_space(self.space)
        if self.created_at.tzinfo is None or self.revision < self.attempt or self.attempt > self.spec.max_attempts:
            raise ValueError('invalid sync control state')
        if (self.attempt == 0) != (self.last_started_at is None):
            raise ValueError('invalid sync attempt timestamp')
        for stamp in (self.last_started_at, self.cancel_requested_at, self.finished_at):
            if stamp is not None and (stamp.tzinfo is None or stamp < self.created_at):
                raise ValueError('invalid sync timestamp')
        if self.status == 'registered' and self.attempt:
            raise ValueError('registered sync has already started')
        if self.status != 'registered' and not self.attempt:
            raise ValueError('sync outcome has no attempt')
        terminal = self.status in TERMINAL_RESULTS
        if terminal != (self.collection_instance is not None):
            raise ValueError('sync result has no collection identity')
        if self.outcome_count != self.source_count + self.issue_count:
            raise ValueError('sync result counts disagree')
        if not terminal and (self.outcome_count or self.skipped):
            raise ValueError('unfinished sync has result counts')
        if (self.status in ('completed', 'partial', 'failed', 'cancelled')) != (self.finished_at is not None):
            raise ValueError('sync completion timestamp disagrees')
        if self.status == 'completed' and self.issue_count:
            raise ValueError('complete sync has scan issues')
        return self


class SyncSourceOutcome(_Record):
    path: str = Field(min_length=1, max_length=1024)
    status: Literal['added', 'updated', 'unchanged', 'deleted', 'absent', 'suppressed', 'failed']
    episode_id: int | None = Field(default=None, gt=0, lt=2**63)
    previous_episode_id: int | None = Field(default=None, gt=0, lt=2**63)
    code: Identifier | None = None

    @field_validator('path')
    @classmethod
    def canonical_path(cls, value: str) -> str:
        try:
            DocumentSource('0' * 32, value, 'outcome')
        except InvalidInput:
            raise ValueError('source outcome requires a canonical relative path') from None
        return value

    @model_validator(mode='after')
    def source_identity(self) -> Self:
        if self.status in ('added', 'updated', 'unchanged', 'absent') and self.episode_id is None:
            raise ValueError('source outcome requires its episode identity')
        if self.status in ('updated', 'deleted') and self.previous_episode_id is None:
            raise ValueError('source outcome requires its previous episode identity')
        if self.status == 'updated' and self.episode_id == self.previous_episode_id:
            raise ValueError('source replacement requires a distinct episode')
        if self.status not in ('updated', 'deleted') and self.previous_episode_id is not None:
            raise ValueError('source outcome has an unexpected previous episode')
        if self.status in ('deleted', 'failed') and self.episode_id is not None:
            raise ValueError('source outcome has an unexpected current episode')
        if (self.status == 'failed') != (self.code is not None):
            raise ValueError('source failure code disagrees with its outcome')
        return self


class SyncScanIssue(_Record):
    # Escaped filesystem diagnostics are display strings, never source locators.
    path: str = Field(max_length=49_154)
    path_escaped: bool = False
    code: Identifier


class SyncOutcome(_Record):
    index: int = Field(ge=0, lt=MAX_OUTCOMES)
    source: SyncSourceOutcome | None = None
    issue: SyncScanIssue | None = None

    @model_validator(mode='after')
    def one_outcome(self) -> Self:
        if (self.source is None) == (self.issue is None):
            raise ValueError('one source outcome or scan issue required')
        return self
