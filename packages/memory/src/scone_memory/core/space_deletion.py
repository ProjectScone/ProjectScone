"""Cleanup intents that outlive the space they erase; never source content."""
from __future__ import annotations

from typing import Protocol, TypeGuard, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .errors import InvalidInput
from .models import SpaceReceipt
from .retirement import Identity
from .timeutil import parse_rfc3339
from .validation import SPACE_NAME


def deletion_key(space: str) -> str:
    if not isinstance(space, str) or SPACE_NAME.fullmatch(space) is None:
        raise InvalidInput('invalid space deletion identity')
    return space


class SpaceDeletion(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', revalidate_instances='always')

    space: str
    requested_at: str
    chunk_ids: tuple[Identity, ...]
    receipt: SpaceReceipt
    retired_chunk_ids: tuple[Identity, ...] = ()

    @field_validator('space')
    @classmethod
    def valid_space(cls, value: str) -> str:
        return deletion_key(value)

    @field_validator('requested_at')
    @classmethod
    def valid_time(cls, value: str) -> str:
        parse_rfc3339(value)
        return value

    @field_validator('receipt', mode='before')
    @classmethod
    def detached_receipt(cls, value: object) -> object:
        raw = value.model_dump() if isinstance(value, SpaceReceipt) else value
        return SpaceReceipt.model_validate(raw, strict=True)

    @model_validator(mode='after')
    def consistent(self) -> SpaceDeletion:
        if len(set(self.retired_chunk_ids)) != len(self.retired_chunk_ids):
            raise ValueError('retired vector identities must be distinct')
        receipt = self.receipt
        if receipt.space != self.space or receipt.deleted_at is not None:
            raise ValueError('space deletion receipt must be its space impact preview')
        if len(set(self.chunk_ids)) != len(self.chunk_ids) or receipt.chunks != len(self.chunk_ids):
            raise ValueError('space deletion must retain distinct chunk identities matching its receipt')
        for name in ('episodes', 'chunks', 'facts', 'links', 'tombstones', 'events'):
            if getattr(receipt, name) < 0:
                raise ValueError('space deletion counts cannot be negative')
        return self


def encode_deletion(record: SpaceDeletion) -> str:
    return SpaceDeletion.model_validate(record).model_dump_json()


def validate_deletion(record: SpaceDeletion, space: str | None = None) -> SpaceDeletion:
    # Validate a detached representation before trusting even custom adapters.
    checked = SpaceDeletion.model_validate_json(encode_deletion(record))
    if space is not None and checked.space != space:
        raise InvalidInput('space deletion payload identity does not match its requested space')
    return checked


def decode_deletion(payload: str, space: str) -> SpaceDeletion:
    record = SpaceDeletion.model_validate_json(payload)
    if record.space != space:
        raise InvalidInput('space deletion payload identity does not match its catalog key')
    return record


def deletion_page(after: str | None, limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 101:
        raise InvalidInput('space deletion page limit must be 1..101')
    if after is not None:
        deletion_key(after)


@runtime_checkable
class SpaceDeletionStore(Protocol):
    """First value wins until acknowledged. Reads detach and validate payloads.

    Catalog rows survive delete_space. Pages sort by space, strictly after the
    cursor even if removed, with read-after-write visibility. Persistent stores
    commit before returning; memory stores last for their object lifetime.
    Callers serialize deletion and recovery with affected writers.
    """

    async def record_space_deletion(self, record: SpaceDeletion) -> SpaceDeletion: ...
    async def space_deletion(self, space: str) -> SpaceDeletion | None: ...
    async def page_space_deletions(self, after: str | None, limit: int) -> list[SpaceDeletion]: ...
    async def clear_space_deletion(self, space: str) -> None: ...


def supports_space_deletion(store: object) -> TypeGuard[SpaceDeletionStore]:
    return isinstance(store, SpaceDeletionStore) and all(callable(getattr(store, name, None)) for name in (
        'record_space_deletion', 'space_deletion', 'page_space_deletions', 'clear_space_deletion',
    ))
