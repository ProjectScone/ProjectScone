import pytest
from scone_memory.deduplication import CandidateBatch, DeduplicationConfig, DocumentDuplicateDetector, DocumentRevision, SemanticCandidate
from scone_memory.embedders.hash import HashEmbedder


def document(text, source_id="incoming", scope="alpha", revision="1", **kwargs):
    return DocumentRevision(scope, source_id, revision, text, **kwargs)


class Provider:
    def __init__(self, documents=(), semantic_matches=(), truncated=False):
        self.documents, self.semantic_matches = tuple(documents), tuple(semantic_matches)
        self.truncated, self.calls = truncated, []

    async def candidates(self, doc, *, query_embeddings, embedder_id, limit):
        self.calls.append((doc.scope, query_embeddings, embedder_id, limit))
        return CandidateBatch(documents=self.documents if not query_embeddings else (), semantic_matches=self.semantic_matches if query_embeddings else (), truncated=self.truncated)


class Embedder:
    id, dim = "test-semantic-contract", 2

    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(tuple(texts))
        return [[1.0, 0.0] for _ in texts]


async def test_exact_text_and_covered_fraction_do_not_double_count_sources():
    text = "Calibration checks the optical sensor before every launch."
    result = await DocumentDuplicateDetector().inspect(document(text), Provider([document(text, "source-1"), document(text, "source-2")]))
    assert result.copied_fraction == 1 and result.requires_review
    assert {match.kind for match in result.matches} == {"exact"}
    assert {match.source_id for match in result.matches} == {"source-1", "source-2"}
    assert result.semantic_status == "unavailable"


async def test_partial_overlap_preserves_unicode_source_byte_spans_and_unique_text():
    shared = "Café instruments — 中文知识检索 calibrate together every morning."
    text = shared + " " + "Novel findings from the evening campaign. " * 4
    source = "Earlier: " + shared + " Later."
    result = await DocumentDuplicateDetector(DeduplicationConfig(semantic_enabled=False)).inspect(document(text), Provider([document(source, "old")]))
    assert 0.2 < result.copied_fraction < 0.35 and result.requires_review
    match = result.matches[0]
    assert text.encode()[match.start:match.end].decode() == shared
    assert source.encode()[match.source_start:match.source_end].decode() == shared
    assert result.semantic_status == "disabled"


async def test_normalization_matches_case_whitespace_and_composed_unicode():
    incoming = "CAFÉ  measurements verify the STRASSE temperature."
    old = "Cafe\u0301 measurements verify the Straße temperature."
    result = await DocumentDuplicateDetector().inspect(document(incoming), Provider([document(old, "old")]))
    assert result.copied_fraction == 1 and result.matches[0].kind == "normalized"
    assert old.encode()[result.matches[0].source_start:result.matches[0].source_end].decode() == old


async def test_threshold_is_covered_input_text_not_source_length_or_jaccard():
    shared = "A repeated passage with substantial original source detail."
    text, source = shared + " Unique. " * 20, shared + " Long source unrelated elsewhere. " * 100
    provider = Provider([document(source, "old")])
    lower = await DocumentDuplicateDetector(DeduplicationConfig(overlap_threshold=.2, semantic_enabled=False)).inspect(document(text), provider)
    upper = await DocumentDuplicateDetector(DeduplicationConfig(overlap_threshold=.4, semantic_enabled=False)).inspect(document(text), provider)
    assert lower.copied_fraction == upper.copied_fraction
    assert lower.requires_review and not upper.requires_review


async def test_paraphrase_default_on_uses_batched_query_only_and_separate_score():
    incoming = document("The river overflowed after heavy rain.\n\nResidents evacuated the area.")
    source = document("Flooding forced people to leave their homes.", "old", byte_offset=90, chunk_id="12")
    provider = Provider(semantic_matches=[SemanticCandidate(source, 0, .93, Embedder.id)])
    embedder, detector = Embedder(), None
    detector = DocumentDuplicateDetector(embedder=embedder)
    first, second = await detector.inspect(incoming, provider), await detector.inspect(incoming, provider)
    assert len(embedder.calls) == 1 and len(embedder.calls[0]) == 2
    assert first.copied_fraction == 0 and first.requires_review and first.semantic_status == "complete"
    assert first.semantic_matches[0].score == .93 and first.semantic_matches[0].source_start == 90
    assert first.semantic_matches[0].chunk_id == "12"
    assert second.metrics.embedding_cache_hits == 2 and second.metrics.embedded_passages == 0
    assert first.metrics.embedded_passages == 2
    assert first.metrics.total_ms >= first.metrics.embedding_ms >= 0


async def test_semantics_disabled_makes_no_embed_or_semantic_search_calls():
    embedder, provider = Embedder(), Provider()
    result = await DocumentDuplicateDetector(DeduplicationConfig(semantic_enabled=False), embedder=embedder).inspect(document("Unique document content."), provider)
    assert not embedder.calls and len(provider.calls) == 1 and result.semantic_status == "disabled"


async def test_hash_embedder_is_not_misrepresented_as_paraphrase_detection():
    result = await DocumentDuplicateDetector(embedder=HashEmbedder()).inspect(document("Unique content."), Provider())
    assert result.semantic_status == "unavailable" and "semantic_embedder_unavailable" in result.limitations


async def test_embedding_cache_separates_scope_revision_content_and_model():
    embedder = Embedder()
    detector = DocumentDuplicateDetector(embedder=embedder)
    for doc in (document("First."), document("First.", scope="beta"), document("First.", revision="2"), document("Changed.")):
        await detector.inspect(doc, Provider())
    embedder.id = "test-next-model"
    await detector.inspect(document("First."), Provider())
    assert len(embedder.calls) == 5


async def test_scope_mismatch_rejected_before_returning_source_evidence():
    with pytest.raises(ValueError, match="scope"):
        await DocumentDuplicateDetector().inspect(document("Scoped content."), Provider([document("Scoped content.", "old", scope="beta")]))


async def test_candidate_limits_and_truncation_cannot_claim_complete_coverage():
    provider = Provider([document("Source text " + str(i), str(i)) for i in range(4)], truncated=True)
    result = await DocumentDuplicateDetector(DeduplicationConfig(max_candidates=2, semantic_enabled=False)).inspect(document("Source text 1"), provider)
    assert result.metrics.candidates_checked == 2 and not result.complete and "candidate_limit" in result.limitations


async def test_semantic_failure_preserves_lexical_evidence_with_visible_status():
    class Broken(Embedder):
        async def embed(self, texts):
            raise RuntimeError("sensitive provider error")
    text = "This paragraph is definitely copied from the original document."
    result = await DocumentDuplicateDetector(embedder=Broken()).inspect(document(text), Provider([document(text, "old")]))
    assert result.copied_fraction == 1 and result.semantic_status == "failed"
    assert "semantic_processing_failed" in result.limitations and "sensitive" not in repr(result)


@pytest.mark.parametrize("kwargs", [{"overlap_threshold": 0}, {"overlap_threshold": float("nan")}, {"max_candidates": 0}, {"min_match_chars": 0}])
def test_configuration_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        DeduplicationConfig(**kwargs)


async def test_empty_and_input_limits_are_explicit():
    detector = DocumentDuplicateDetector(DeduplicationConfig(max_document_bytes=32))
    result = await detector.inspect(document(""), Provider())
    assert result.copied_fraction == 0 and not result.requires_review
    with pytest.raises(ValueError, match="byte limit"):
        await detector.inspect(document("中" * 12), Provider())


async def test_whitespace_is_not_duplicate_evidence():
    result = await DocumentDuplicateDetector(DeduplicationConfig(semantic_enabled=False)).inspect(document(" \n\t "), Provider([document(" \n\t ", "old")]))
    assert result.copied_fraction == 0 and not result.requires_review


async def test_common_repetitive_text_is_bounded_and_reports_incomplete_analysis():
    config = DeduplicationConfig(semantic_enabled=False, max_match_operations=100)
    result = await DocumentDuplicateDetector(config).inspect(document("abc " * 200), Provider([document("abc " * 200 + "changed", "old")]))
    assert not result.complete and "lexical_work_limit" in result.limitations
    assert 0 <= result.copied_fraction <= 1


async def test_semantic_scores_do_not_inflate_copied_fraction_or_bypass_model_identity():
    old = document("Different language about similar subjects.", "old")
    provider = Provider(semantic_matches=[SemanticCandidate(old, 0, .2, Embedder.id)])
    result = await DocumentDuplicateDetector(embedder=Embedder()).inspect(document("A novel premise."), provider)
    assert not result.requires_review and result.copied_fraction == 0
    provider.semantic_matches = (SemanticCandidate(old, 0, .99, "other-model"),)
    with pytest.raises(ValueError, match="embedder"):
        await DocumentDuplicateDetector(embedder=Embedder()).inspect(document("A novel premise."), provider)


async def test_concurrent_queries_share_inflight_embedding_and_cache_can_be_cleared():
    import asyncio
    embedder, provider = Embedder(), Provider()
    detector = DocumentDuplicateDetector(embedder=embedder)
    incoming = document("A new document to encode.")
    reports = await asyncio.gather(*(detector.inspect(incoming, provider) for _ in range(5)))
    assert len(embedder.calls) == 1
    assert sum(report.metrics.embedded_passages for report in reports) == 1
    detector.clear_cache(scope="alpha")
    await detector.inspect(incoming, provider)
    assert len(embedder.calls) == 2
    detector.clear_cache()
    await detector.inspect(incoming, provider)
    assert len(embedder.calls) == 3


async def test_embedding_cache_zero_disables_retention():
    embedder = Embedder()
    detector = DocumentDuplicateDetector(DeduplicationConfig(embedding_cache_size=0), embedder=embedder)
    await detector.inspect(document("A new document."), Provider())
    await detector.inspect(document("A new document."), Provider())
    assert len(embedder.calls) == 2


async def test_query_passage_truncation_is_visible():
    detector = DocumentDuplicateDetector(DeduplicationConfig(semantic_passage_chars=10, max_semantic_passages=2), embedder=Embedder())
    result = await detector.inspect(document("A long paragraph describing several novel technical discoveries."), Provider())
    assert result.semantic_status == "partial" and not result.complete
    assert "semantic_passage_limit" in result.limitations


async def test_candidate_chunk_offset_and_disjoint_sources_preserve_all_evidence():
    first = "First shared passage with different source grounding."
    second = "Another duplicated passage from an independent document."
    incoming = document(first + " Unique middle. " + second)
    provider = Provider([document(first, "one", byte_offset=100, chunk_id="c1"), document(second, "two", byte_offset=500, chunk_id="c2")])
    result = await DocumentDuplicateDetector(DeduplicationConfig(semantic_enabled=False)).inspect(incoming, provider)
    assert result.copied_bytes == len((first + second).encode())
    assert {span.source_start for span in result.matches} == {100, 500}
    assert {span.chunk_id for span in result.matches} == {"c1", "c2"}


async def test_semantic_provider_cancellation_is_not_swallowed():
    import asyncio
    class Cancelled(Embedder):
        async def embed(self, texts):
            raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await DocumentDuplicateDetector(embedder=Cancelled()).inspect(document("Some text."), Provider())


async def test_invalid_embedder_vectors_report_failure():
    class Invalid(Embedder):
        async def embed(self, texts):
            return [[float("nan"), 0] for _ in texts]
    result = await DocumentDuplicateDetector(embedder=Invalid()).inspect(document("Some text."), Provider())
    assert result.semantic_status == "failed" and not result.complete


async def test_ascii_whitespace_normalization_preserves_original_offsets():
    text = "\tAlpha  measurements\n\nverify\tcalibration every morning. "
    old = " Alpha measurements verify calibration every morning. "
    result = await DocumentDuplicateDetector(DeduplicationConfig(semantic_enabled=False)).inspect(document(text), Provider([document(old, "old")]))
    assert result.copied_fraction == 1
    assert result.matches[0].end == len(text.encode())


async def test_copy_match_does_not_split_expanded_unicode_cluster():
    source = "ß" + "x" * 30
    incoming = "s" + "x" * 30
    result = await DocumentDuplicateDetector(DeduplicationConfig(semantic_enabled=False)).inspect(document(incoming), Provider([document(source, "old")]))
    match = result.matches[0]
    assert incoming.encode()[match.start:match.end].decode() == "x" * 30
    assert source.encode()[match.source_start:match.source_end].decode() == "x" * 30


async def test_unrelated_documents_do_not_block_each_others_embedding_batch():
    import asyncio
    class Concurrent(Embedder):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.both_started = asyncio.Event()
            self.active = 0
        async def embed(self, texts):
            self.active += 1
            self.started.set()
            if self.active == 2:
                self.both_started.set()
            await self.both_started.wait()
            self.active -= 1
            return await super().embed(texts)
    embedder = Concurrent()
    detector = DocumentDuplicateDetector(embedder=embedder)
    first = asyncio.create_task(detector.inspect(document("First document.", "first"), Provider()))
    await embedder.started.wait()
    second = asyncio.create_task(detector.inspect(document("Second document.", "second"), Provider()))
    try:
        await asyncio.wait_for(embedder.both_started.wait(), .25)
        results = await asyncio.gather(first, second)
        assert all(result.semantic_status == "complete" for result in results)
    finally:
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


async def test_copied_passages_can_overlap_in_the_original_source():
    # Both new passages must be covered even when their source spans overlap.
    source = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    incoming = source[:38] + " unique separator " + source[20:35]
    result = await DocumentDuplicateDetector(DeduplicationConfig(semantic_enabled=False, min_match_chars=12)).inspect(document(incoming), Provider([document(source, "old")]))
    assert result.copied_bytes == 38 + len(source[20:35])


async def test_semantic_candidate_bytes_have_an_aggregate_bound():
    first = document("First candidate text.", "one")
    second = document("Other candidate text.", "two")
    provider = Provider(semantic_matches=[SemanticCandidate(first, 0, .95, Embedder.id), SemanticCandidate(second, 0, .94, Embedder.id)])
    result = await DocumentDuplicateDetector(DeduplicationConfig(max_candidate_bytes=30), embedder=Embedder()).inspect(document("Incoming."), provider)
    assert len(result.semantic_matches) == 1
    assert result.semantic_status == "partial" and "semantic_candidate_byte_limit" in result.limitations


async def test_repeated_semantic_source_consumes_candidate_byte_budget_once():
    source = document("Shared candidate text.", "one")
    provider = Provider(semantic_matches=[SemanticCandidate(source, 0, .95, Embedder.id), SemanticCandidate(source, 1, .94, Embedder.id)])
    result = await DocumentDuplicateDetector(DeduplicationConfig(max_candidate_bytes=30), embedder=Embedder()).inspect(document("Incoming one.\n\nIncoming two."), provider)
    assert len(result.semantic_matches) == 2 and result.semantic_status == "complete"


@pytest.mark.parametrize("alphabet", ["abcdef", "aβ中文éø"])
def test_optimized_span_matcher_agrees_with_exhaustive_coverage(alphabet):
    import random
    from scone_memory.deduplication.lexical import CopiedSpanMatcher, covered_bytes
    randomizer = random.Random(72)
    for _ in range(100):
        query = "".join(randomizer.choices(alphabet, k=70))
        source = "".join(randomizer.choices(alphabet, k=80))
        expected_positions = set()
        for start in range(len(query) - 2):
            if query[start:start + 3] in source:
                expected_positions.update(range(start, start + 3))
        matcher = CopiedSpanMatcher(document(query), 3, 1_000_000)
        matches = matcher.compare(document(source, "old"))
        assert not matcher.truncated
        assert covered_bytes(matches) == sum(len(query[index].encode()) for index in expected_positions)
