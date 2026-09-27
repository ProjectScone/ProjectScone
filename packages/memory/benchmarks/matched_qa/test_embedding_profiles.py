from __future__ import annotations

import pytest


def test_nemotron_uses_separate_dimensions_prefixes_and_index() -> None:
    from .embedding_profiles import embedding_profile
    qwen = embedding_profile('qwen')
    nemotron = embedding_profile('nemotron')
    assert nemotron.model == 'nvidia/nemotron-3-embed-1b:free'
    assert nemotron.dimensions == 2048
    assert nemotron.query_prefix == 'query: '
    assert nemotron.document_prefix == 'passage: '
    assert nemotron.collection != qwen.collection
    assert nemotron.batch_size == 128
    assert nemotron.minimum_request_interval > 3
    assert nemotron.rerank_provider == 'openrouter'
    assert qwen.model == 'qwen/qwen3-embedding-8b' and qwen.dimensions == 4096
    assert qwen.document_prefix == ''
    with pytest.raises(ValueError):
        embedding_profile('unknown')
