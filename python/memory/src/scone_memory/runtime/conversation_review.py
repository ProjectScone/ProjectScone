"""Explicit operator configuration for reviewing native conversation answers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Literal, TypedDict

from ..core.errors import InvalidInput
from ..providers.self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier
from ..realtime.answer_review import AnswerReviewer, AnswerReviewLimits

if TYPE_CHECKING:
    from .config import Settings


class ReviewOptions(TypedDict):
    answer_reviewer: AnswerReviewer
    review_policy: Literal["report", "require_supported"]
    review_limits: AnswerReviewLimits


@dataclass(frozen=True)
class ConversationReview:
    reviewer: AnswerReviewer
    policy: Literal["report", "require_supported"]
    limits: AnswerReviewLimits

    def __post_init__(self) -> None:
        if not callable(getattr(self.reviewer, "review", None)):
            raise ValueError("answer reviewer must implement review")
        if self.policy not in ("report", "require_supported"):
            raise ValueError("answer review policy must be report or require_supported")
        if not isinstance(self.limits, AnswerReviewLimits):
            raise ValueError("answer review limits required")
        object.__setattr__(self, "limits", AnswerReviewLimits.model_validate(dict(vars(self.limits)), strict=True))

    def options(self) -> ReviewOptions:
        return {"answer_reviewer": self.reviewer, "review_policy": self.policy, "review_limits": self.limits}


def validate_review_settings(settings: Settings) -> None:
    if settings.answer_review_policy not in ("off", "report", "require_supported"):
        raise InvalidInput("SCONE_ANSWER_REVIEW_POLICY must be off, report, or require_supported")
    if settings.answer_review_quote_mode not in ('text', 'spans'):
        raise InvalidInput('SCONE_ANSWER_REVIEW_QUOTE_MODE must be text or spans')
    timeout = settings.answer_review_timeout
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 180:
        raise InvalidInput("SCONE_ANSWER_REVIEW_TIMEOUT must be finite in 1..180 seconds")
    if settings.answer_review_policy == "off":
        if (settings.answer_review_url is not None or settings.answer_review_model is not None
                or settings.answer_review_api_key is not None or timeout != 20.0 or settings.answer_review_quote_mode != 'text'):
            raise InvalidInput("answer review connection settings require SCONE_ANSWER_REVIEW_POLICY")
        return
    if not settings.conversations_journal:
        raise InvalidInput("answer review requires SCONE_CONVERSATIONS_JOURNAL")
    if not settings.answer_review_url or not settings.answer_review_model:
        raise InvalidInput("answer review requires SCONE_ANSWER_REVIEW_URL and SCONE_ANSWER_REVIEW_MODEL")
    try:
        validate_self_hosted_endpoint(settings.answer_review_url)
        validate_self_hosted_identifier(settings.answer_review_model)
    except ValueError:
        raise InvalidInput("answer review requires a valid self-hosted endpoint and model identifier") from None
    if settings.answer_review_api_key is not None and (
            type(settings.answer_review_api_key) is not str or not settings.answer_review_api_key.strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in settings.answer_review_api_key)):
        raise InvalidInput("SCONE_ANSWER_REVIEW_API_KEY must be nonblank text without control characters")


def build_conversation_review(settings: Settings) -> ConversationReview | None:
    validate_review_settings(settings)
    if settings.answer_review_policy == "off":
        return None
    from ..providers.answer_reviewer import SelfHostedAnswerReviewer

    assert settings.answer_review_url is not None and settings.answer_review_model is not None
    quote_mode: Literal['text', 'spans'] = 'spans' if settings.answer_review_quote_mode == 'spans' else 'text'
    reviewer = SelfHostedAnswerReviewer(settings.answer_review_url, settings.answer_review_model,
        api_key=settings.answer_review_api_key, timeout=settings.answer_review_timeout, quote_mode=quote_mode)
    policy: Literal["report", "require_supported"] = (
        "require_supported" if settings.answer_review_policy == "require_supported" else "report")
    return ConversationReview(reviewer, policy, AnswerReviewLimits(timeout_s=float(settings.answer_review_timeout)))
