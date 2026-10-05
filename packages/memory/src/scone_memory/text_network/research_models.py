"""Strict values for temporary section coverage and source context."""
from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from .models import NetworkDocument, StrictValue, TextTerm, visible


class ResearchLimits(StrictValue):
    max_documents: int = Field(default=15, ge=1, le=15)
    max_terms: int = Field(default=60, ge=0, le=60)
    max_input_bytes: int = Field(default=512*1024, ge=1, le=512*1024)
    max_passages: int = Field(default=4000, ge=1, le=4000)
    max_sections: int = Field(default=128, ge=1, le=128)
    max_context_requests: int = Field(default=10, ge=1, le=10)
    max_context_characters: int = Field(default=12000, ge=600, le=12000)
    max_response_bytes: int = Field(default=512*1024, ge=1024, le=512*1024)


class TextSection(StrictValue):
    key: str = Field(min_length=1, max_length=128)
    document_key: str = Field(min_length=1, max_length=128)
    title: str = Field(max_length=200)
    start: int = Field(ge=0)
    end: int = Field(ge=1)

    @model_validator(mode='after')
    def valid(self) -> Self:
        if not visible(self.key) or not visible(self.document_key) or self.end <= self.start:
            raise ValueError('invalid_section')
        return self


class CoverageTerm(TextTerm):
    id: str


class SectionTermCount(StrictValue):
    node_id: str
    passage_count: int
    first_passage_start: int
    first_passage_end: int
    first_start: int
    first_end: int
    last_start: int
    last_end: int


class SectionStatistics(TextSection):
    passage_count: int
    complete: bool
    counts: list[SectionTermCount]


class ResearchCoverage(StrictValue):
    partial: bool
    reasons: list[str]
    scanned_passages: int


class SectionCoverage(StrictValue):
    version: Literal[1] = 1
    basis: Literal['literal_passage_counts'] = 'literal_passage_counts'
    persisted: Literal[False] = False
    offset_unit: Literal['unicode_codepoint'] = 'unicode_codepoint'
    digest: str
    network_digest: str
    documents: list[NetworkDocument]
    terms: list[CoverageTerm]
    sections: list[SectionStatistics]
    coverage: ResearchCoverage


class ContextRequest(StrictValue):
    key: str = Field(min_length=1, max_length=128)
    document_key: str = Field(min_length=1, max_length=128)
    content_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    before: int = Field(default=1, ge=0, le=3)
    after: int = Field(default=1, ge=0, le=3)

    @model_validator(mode='after')
    def valid(self) -> Self:
        if not visible(self.key) or not visible(self.document_key) or self.end <= self.start:
            raise ValueError('invalid_context_request')
        return self


class ContextWindow(StrictValue):
    key: str
    document_key: str
    content_sha256: str
    target_start: int
    target_end: int
    start: int
    end: int
    text: str
    truncated_before: bool
    truncated_after: bool
    target_truncated: bool


class ContextBatch(StrictValue):
    version: Literal[1] = 1
    persisted: Literal[False] = False
    offset_unit: Literal['unicode_codepoint'] = 'unicode_codepoint'
    digest: str
    windows: list[ContextWindow]
    coverage: ResearchCoverage
