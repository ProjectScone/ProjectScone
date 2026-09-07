"""Public generation events shared by text and voice; no reasoning channel."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TextDelta:
    """Provider-adapter public response text, not necessarily a token."""

    text: str


@dataclass(frozen=True)
class ReplyCompleted:
    """Adapter observed successful response end; not a persistence receipt."""


class TextModel(Protocol):
    def respond(self, messages: list[dict[str, str]]) -> AsyncIterator[TextDelta | ReplyCompleted]: ...
    async def aclose(self) -> None: ...
