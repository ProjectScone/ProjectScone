"""Bounded private assessment context; search observations are not evidence."""
from __future__ import annotations

import unicodedata

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.validation import MAX_QUERY


def query_key(query: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', query).split()).casefold()


class EvidenceSearchAttempt(BaseModel):
    """One completed search, measured before later retention checks.

    added_candidates counts additions to that bounded pool, not globally new
    passages, semantic novelty or currently retained evidence. Zero can reflect
    duplicates, filtering or capacity. Queries may contain untrusted text.
    """
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, revalidate_instances='always')
    query: str = Field(min_length=1, max_length=MAX_QUERY)
    round_number: int = Field(ge=1, le=4)
    added_candidates: int = Field(ge=0, le=100)
    degraded: bool

    @model_validator(mode='after')
    def nonblank_query(self) -> EvidenceSearchAttempt:
        if not query_key(self.query):
            raise ValueError('search query must be nonblank')
        return self


class EvidenceAssessmentContext(BaseModel):
    """Run-local history and budgets remaining after the current searches.

    No source text, selections, authorization settings or claimed answers are
    carried here. The current candidate snapshot remains the evidence boundary.
    """
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, revalidate_instances='always')
    searches: tuple[EvidenceSearchAttempt, ...] = Field(min_length=1, max_length=12)
    round_number: int = Field(ge=1, le=4)
    rounds_remaining: int = Field(ge=0, le=3)
    queries_remaining: int = Field(ge=0, le=11)

    @model_validator(mode='after')
    def consistent_history(self) -> EvidenceAssessmentContext:
        keys = [query_key(search.query) for search in self.searches]
        rounds = [search.round_number for search in self.searches]
        if len(keys) != len(set(keys)) or rounds != sorted(rounds) or max(rounds) > self.round_number:
            raise ValueError('search history must be unique and chronological')
        if self.round_number + self.rounds_remaining > 4 or len(keys) + self.queries_remaining > 12:
            raise ValueError('search history exceeds adaptive budgets')
        return self
