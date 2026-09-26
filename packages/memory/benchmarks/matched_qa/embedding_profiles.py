"""Explicit model configurations; never mix vector spaces in one collection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .pipeline import EMBED_MODEL, QUERY_PREFIX


@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    model: str
    dimensions: int
    query_prefix: str
    document_prefix: str
    collection: str
    batch_size: int
    minimum_request_interval: float
    rerank_provider: Literal['typesafe', 'openrouter']


def embedding_profile(name: str) -> EmbeddingProfile:
    if name == 'qwen':
        return EmbeddingProfile('qwen', EMBED_MODEL, 4096, QUERY_PREFIX, '', 'matched_scone', 32, 0., 'typesafe')
    if name == 'nemotron':
        return EmbeddingProfile('nemotron', 'nvidia/nemotron-3-embed-1b:free', 2048,
            'query: ', 'passage: ', 'matched_scone_nemotron_3_embed_1b', 128, 3.2, 'openrouter')
    raise ValueError('unknown embedding profile: ' + name)
