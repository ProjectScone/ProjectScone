"""Durable source cleanup intents, kept separately from public tombstones.

An intent holds identities and a preview receipt, never source content. Stores
keep its first value until cleanup acknowledges it, and return detached copies.
The engine serializes source writes and recovery; this seam is not a distributed
lock or a transaction across document, vector, blob and event stores.
"""
from __future__ import annotations

from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import InvalidInput
from .models import ForgetReceipt
from .timeutil import parse_rfc3339
from .validation import SPACE_NAME

Identity = Annotated[int, Field(gt=0, lt=2**63)]
RetirementCursor = tuple[str, int]


class Retirement(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", revalidate_instances="always")

    space: str
    episode_id: Identity
    content_hash: str = Field(min_length=1, max_length=128)
    requested_at: str
    chunk_ids: tuple[Identity, ...]
    receipt: ForgetReceipt

    @field_validator("space")
    @classmethod
    def valid_space(cls, value: str) -> str:
        if SPACE_NAME.fullmatch(value) is None:
            raise ValueError("invalid retirement space")
        return value

    @field_validator("requested_at")
    @classmethod
    def valid_time(cls, value: str) -> str:
        parse_rfc3339(value)
        return value

    @field_validator("receipt", mode="before")
    @classmethod
    def detached_receipt(cls, value: object) -> object:
        # Model copies bypass validation, and frozen models can still contain
        # mutable lists. Revalidate a fresh representation at this boundary.
        return value.model_dump() if isinstance(value, ForgetReceipt) else value

    @model_validator(mode="after")
    def consistent(self) -> Retirement:
        if len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("retirement chunk identities must be distinct")
        if self.receipt.episode_id != self.episode_id or self.receipt.chunks != len(self.chunk_ids):
            raise ValueError("retirement receipt must describe its source and chunks")
        if self.receipt.forgotten_at is not None:
            raise ValueError("retirement must start with an impact preview")
        return self


def encode_retirement(record: Retirement) -> str:
    """Validate even unchecked model copies before retaining cleanup targets."""
    return Retirement.model_validate(record).model_dump_json()


def decode_retirement(payload: str, key: RetirementCursor | None = None) -> Retirement:
    record = Retirement.model_validate_json(payload)
    if key is not None and (record.space, record.episode_id) != key:
        raise InvalidInput("retirement payload identity does not match its catalog key")
    return record


def retirement_key(space: str, episode_id: int) -> RetirementCursor:
    if not isinstance(space, str) or SPACE_NAME.fullmatch(space) is None:
        raise InvalidInput("invalid retirement space")
    if type(episode_id) is not int or not 0 < episode_id < 2**63:
        raise InvalidInput("retirement episode_id must be a positive 64-bit integer")
    return space, episode_id


def retirement_page(after: RetirementCursor | None, limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 101:
        raise InvalidInput("retirement page limit must be 1..101")
    if after is not None:
        if not isinstance(after, tuple) or len(after) != 2:
            raise InvalidInput("retirement cursor must be (space, episode_id)")
        retirement_key(*after)


@runtime_checkable
class RetirementStore(Protocol):
    """Optional custom-store seam; every built-in document store implements it.

    Writes are durable before returning (in-memory stores last for their object
    lifetime). Pages are ordered by (space, episode_id), strictly after the
    cursor even if that record was cleared. Writes and clears must be visible
    to subsequent pages, including on backends with optional delayed refresh.
    """

    async def record_retirement(self, record: Retirement) -> Retirement: ...
    async def retirement(self, space: str, episode_id: int) -> Retirement | None: ...
    async def page_retirements(self, after: RetirementCursor | None, limit: int) -> list[Retirement]: ...
    async def clear_retirement(self, space: str, episode_id: int) -> None: ...
