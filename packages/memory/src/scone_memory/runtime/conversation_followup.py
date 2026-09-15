"""Opt-in follow-up queries for served text conversations: carry, or rewrite with a self-hosted model."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.errors import InvalidInput
from ..providers.self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier
from ..retrieval.followup import MODES, REWRITE_TIMEOUT_S

if TYPE_CHECKING:
    from .config import Settings

#: Seconds a rewriting model may be given, at most.
MAX_FOLLOWUP_TIMEOUT_S = 60.0


def validate_followup_settings(settings: Settings) -> None:
    mode = settings.followup_queries
    if mode not in MODES:
        raise InvalidInput("SCONE_FOLLOWUP_QUERIES must be off, carry, or rewrite")
    timeout = settings.followup_timeout
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= MAX_FOLLOWUP_TIMEOUT_S:
        raise InvalidInput(f"SCONE_FOLLOWUP_TIMEOUT must be above 0 and at most {MAX_FOLLOWUP_TIMEOUT_S:g} seconds")
    if mode != "rewrite" and (settings.followup_url is not None or settings.followup_model is not None
                              or settings.followup_api_key is not None or timeout != REWRITE_TIMEOUT_S):
        raise InvalidInput("SCONE_FOLLOWUP_URL, _MODEL, _API_KEY and _TIMEOUT require SCONE_FOLLOWUP_QUERIES=rewrite")
    if mode == "off":
        return
    if not settings.conversations_journal:
        raise InvalidInput("follow-up queries require SCONE_CONVERSATIONS_JOURNAL")
    if settings.adaptive_retrieval:
        raise InvalidInput("SCONE_FOLLOWUP_QUERIES cannot be combined with SCONE_ADAPTIVE_RETRIEVAL, which plans its own queries")
    if settings.conversations_tool_mode != "off":
        raise InvalidInput("SCONE_FOLLOWUP_QUERIES is for ordinary search; tool mode chooses its own searches")
    if mode == "carry":
        return
    if not settings.followup_url or not settings.followup_model:
        raise InvalidInput("SCONE_FOLLOWUP_QUERIES=rewrite requires SCONE_FOLLOWUP_URL and SCONE_FOLLOWUP_MODEL")
    try:
        validate_self_hosted_endpoint(settings.followup_url)
        validate_self_hosted_identifier(settings.followup_model)
    except ValueError:
        raise InvalidInput("follow-up rewriting requires a valid self-hosted endpoint and model identifier") from None
    key = settings.followup_api_key
    if key is not None and (not key.strip() or any(ord(char) < 32 or ord(char) == 127 for char in key)):
        raise InvalidInput("SCONE_FOLLOWUP_API_KEY must be nonblank text without control characters")


def build_followup(settings: Settings) -> dict[str, object]:
    """The conversation options SCONE_FOLLOWUP_QUERIES asks for; empty when it is off."""
    validate_followup_settings(settings)
    if settings.followup_queries == "off":
        return {}
    options: dict[str, object] = {"followup_queries": settings.followup_queries}
    if settings.followup_queries == "rewrite":
        from ..providers.llm import OpenAICompatibleChat

        assert settings.followup_url is not None and settings.followup_model is not None
        # Its own endpoint and key: a rewriting model never borrows another setting's credentials.
        options["followup_model"] = OpenAICompatibleChat(validate_self_hosted_endpoint(str(settings.followup_url)),
            settings.followup_model, api_key=settings.followup_api_key, think=False, temperature=0,
            timeout=settings.followup_timeout, trust_env=False)
        options["followup_timeout"] = settings.followup_timeout
    return options
