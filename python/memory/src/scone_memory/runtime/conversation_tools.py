"""Explicit tool protocol and budgets for served self-hosted text sessions."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..agents.evidence_loop import ToolLoopLimits, ToolModel
from ..core.errors import InvalidInput
from .model_connections import ModelConnection, api_key

if TYPE_CHECKING:
    from .config import Settings


def _limits(settings: Settings) -> ToolLoopLimits:
    return ToolLoopLimits(max_tool_calls=settings.conversations_tool_max_calls,
        max_tool_rounds=settings.conversations_tool_max_rounds, timeout_s=settings.conversations_tool_timeout)


def validate_tool_settings(settings: Settings) -> None:
    if settings.conversations_tool_mode not in ('off', 'native', 'structured'):
        raise InvalidInput('SCONE_CONVERSATIONS_TOOL_MODE must be off, native, or structured')
    if type(settings.conversations_tool_initial_search) is not bool:
        raise InvalidInput('SCONE_CONVERSATIONS_TOOL_INITIAL_SEARCH must be a boolean')
    try:
        limits = _limits(settings)
    except ValueError:
        raise InvalidInput('conversation tool budgets are outside supported limits') from None
    if settings.conversations_tool_mode == 'off':
        if limits != ToolLoopLimits():
            raise InvalidInput('conversation tool budgets require SCONE_CONVERSATIONS_TOOL_MODE')
        return
    if not settings.conversations_journal or not settings.model_connections:
        raise InvalidInput('conversation tools require SCONE_CONVERSATIONS_JOURNAL and SCONE_MODEL_CONNECTIONS')
    if settings.conversations_model_factory or settings.conversations_personas or settings.conversations_registry:
        raise InvalidInput('conversation tool mode requires the saved self-hosted model runtime')
    if settings.adaptive_retrieval or settings.answer_review_policy != 'off':
        raise InvalidInput('conversation tools cannot combine with adaptive retrieval or answer review')


@dataclass(frozen=True)
class ConversationTools:
    mode: Literal['native', 'structured']
    limits: ToolLoopLimits
    initial_search: bool = True

    def __post_init__(self) -> None:
        if (self.mode not in ('native', 'structured') or not isinstance(self.limits, ToolLoopLimits)
                or type(self.initial_search) is not bool):
            raise ValueError('invalid conversation tool configuration')
        object.__setattr__(self, 'limits', ToolLoopLimits.model_validate(self.limits.model_dump()))

    def factory(self, connection: ModelConnection, *, think: bool | None = None) -> Callable[[], ToolModel]:
        connection = ModelConnection.model_validate(connection.model_dump())

        def create() -> ToolModel:
            from ..providers.tool_chat import SelfHostedToolChat
            from ..providers.structured_tool_chat import SelfHostedStructuredToolChat

            provider = SelfHostedStructuredToolChat if self.mode == 'structured' else SelfHostedToolChat
            return provider(connection.base_url, connection.model, api_key=api_key(connection),
                timeout_s=min(connection.timeout_s, self.limits.timeout_s), think=think)
        return create


def build_conversation_tools(settings: Settings) -> ConversationTools | None:
    validate_tool_settings(settings)
    if settings.conversations_tool_mode == 'off':
        return None
    mode: Literal['native', 'structured'] = 'native' if settings.conversations_tool_mode == 'native' else 'structured'
    return ConversationTools(mode, _limits(settings), settings.conversations_tool_initial_search)
