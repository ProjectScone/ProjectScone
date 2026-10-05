"""Strict transient text-network values; no persistence or domain vocabulary."""
from __future__ import annotations

from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictValue(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra='forbid', hide_input_in_errors=True)


def visible(value: str) -> bool:
    return value == value.strip() and bool(value) and not any(ord(char) < 32 for char in value)


class TextDocument(StrictValue):
    key: str = Field(min_length=1, max_length=128)
    kind: str = Field(min_length=1, max_length=32)
    id: str = Field(min_length=1, max_length=128)
    title: str = Field(max_length=200)
    revision: int | None = Field(default=None, ge=1, le=2**53-1)
    content: str

    @model_validator(mode='after')
    def valid(self) -> Self:
        if not all(visible(value) for value in (self.key, self.kind, self.id)):
            raise ValueError('invalid_document')
        return self


class TextTerm(StrictValue):
    label: str = Field(min_length=1, max_length=80)
    kind: str = Field(min_length=1, max_length=32)
    aliases: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode='after')
    def valid(self) -> Self:
        if (not visible(self.label) or not visible(self.kind)
                or any(not visible(alias) or len(alias) > 80 for alias in self.aliases)):
            raise ValueError('invalid_term')
        return self


class NetworkLimits(StrictValue):
    max_documents: int = Field(default=15, ge=1, le=15)
    max_terms: int = Field(default=60, ge=0, le=60)
    max_input_bytes: int = Field(default=512*1024, ge=1, le=512*1024)
    max_passages: int = Field(default=4000, ge=1, le=4000)
    max_edges: int = Field(default=600, ge=1, le=600)
    max_cited_passages: int = Field(default=200, ge=1, le=200)
    max_candidates: int = Field(default=30, ge=0, le=30)
    max_variants: int = Field(default=30, ge=0, le=30)
    max_response_bytes: int = Field(default=512*1024, ge=1024, le=512*1024)


class NetworkDocument(StrictValue):
    key: str
    kind: str
    id: str
    title: str
    revision: int | None
    content_sha256: str


class NetworkNode(StrictValue):
    id: str
    label: str
    kind: str
    aliases: list[str]
    passage_count: int
    document_keys: list[str]
    community_id: str


class NetworkEdge(StrictValue):
    id: str
    source: str
    target: str
    kind: Literal['shared_passage'] = 'shared_passage'
    weight: int
    passage_ids: list[str]
    passages_truncated: bool


class NetworkMention(StrictValue):
    node_id: str
    start: int
    end: int


class NetworkPassage(StrictValue):
    id: str
    document_key: str
    start: int
    end: int
    excerpt: str
    excerpt_start: int
    mentions: list[NetworkMention]


class NetworkCandidate(StrictValue):
    label: str
    passage_count: int
    document_keys: list[str]


class NetworkVariant(StrictValue):
    left: str
    right: str
    reason: Literal['similar_spelling'] = 'similar_spelling'


class NetworkCommunity(StrictValue):
    id: str
    node_ids: list[str]
    label: str


class NetworkBridge(StrictValue):
    source: str
    target: str
    cut_edge: bool
    cross_community: bool


class NetworkAnalysis(StrictValue):
    communities: list[NetworkCommunity]
    bridges: list[NetworkBridge]
    isolated_node_ids: list[str]


class NetworkCoverage(StrictValue):
    partial: bool
    reasons: list[str]
    documents: int
    passages: int
    mentions: int
    edges: int


class TextNetwork(StrictValue):
    version: Literal[1] = 1
    basis: Literal['literal_shared_passage'] = 'literal_shared_passage'
    persisted: Literal[False] = False
    digest: str
    offset_unit: Literal['unicode_codepoint'] = 'unicode_codepoint'
    documents: list[NetworkDocument]
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]
    passages: list[NetworkPassage]
    candidates: list[NetworkCandidate]
    variants: list[NetworkVariant]
    analysis: NetworkAnalysis
    coverage: NetworkCoverage
