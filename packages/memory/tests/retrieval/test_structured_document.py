from __future__ import annotations

import pytest
from scone_memory.retrieval.section_routing import ChoiceBatch, FetchDecision, RouteMenu, SectionRouter, SectionSnapshot
from scone_memory.retrieval.structured_document import StructuredDocumentIndex


class Embeddings:
    id = 'test-vectors'
    dim = 2
    async def embed(self, texts):
        return [[1., .1] if 'cats' in t.lower() else [.1, 1.] for t in texts]
    async def embed_queries(self, texts):
        return await self.embed(texts)


class Decisions:
    definition = 'test'
    def __init__(self, mode='original'):
        self.mode = mode
    async def choose(self, query: str, menus: tuple[RouteMenu, ...]) -> ChoiceBatch:
        rows = []
        for menu in menus:
            scores = [0.] * (len(menu.options) + 1)
            target = next((i for i, o in enumerate(menu.options) if o.title == 'Cats'), 0)
            scores[target] = 1.
            rows.append(tuple(scores))
        return ChoiceBatch(tuple(rows), 'test')
    async def choose_fetch(self, query, sections, max_bytes):
        return FetchDecision(self.mode, 'test', tuple(1. if m == self.mode else 0.
                             for m in ('original', 'section_vector', 'flat_vector')))


def book(text='# Cats\nCats purr.\n# Dogs\nDogs bark.'):
    return SectionSnapshot.from_markdown('authorized', 'animals', text)


async def test_auto_keeps_vectors_central_and_fetches_original_bytes() -> None:
    snapshot = book()
    decisions = Decisions()
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    result = await index.retrieve('Tell me about cats', snapshot, SectionRouter(decisions), decisions)
    assert result.mode == 'original'
    assert result.baseline
    assert result.evidence[0].text == '# Cats\nCats purr.\n'
    for item in result.evidence:
        assert snapshot.content.encode()[item.start:item.end].decode() == item.text
    assert result.query_embedding_ms >= 0


async def test_direct_fetch_cannot_exceed_budget_and_scoped_search_keeps_broad_lane() -> None:
    snapshot = book('# Cats\n' + 'Cats purr. ' * 100 + '\n# Dogs\nDogs bark.')
    decisions = Decisions()
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions, max_bytes=40)
    assert result.mode == 'section_vector'
    assert result.reason == 'original_exceeds_budget'
    assert sum(len(p.text.encode()) for p in result.evidence) <= 40
    assert result.baseline


async def test_changed_document_is_rejected_before_using_old_vectors() -> None:
    snapshot = book()
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    decisions = Decisions()
    with pytest.raises(ValueError, match='snapshot'):
        await index.retrieve('cats', book('# Cats\nNew facts.'), SectionRouter(decisions), decisions)


async def test_no_match_returns_same_flat_evidence() -> None:
    class NoMatch(Decisions):
        async def choose(self, query, menus):
            return ChoiceBatch(tuple(tuple(0. for _ in m.options) + (1.,) for m in menus), 'test')
    snapshot = book()
    decisions = NoMatch()
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions)
    assert result.mode == 'flat_vector'
    assert result.reason == 'no_match'
    assert result.evidence == result.baseline


async def test_flat_mode_skips_jev_and_parent_original_includes_descendants() -> None:
    snapshot = book('# Cats\nIntroduction.\n## Food\nCats eat fish.')
    decisions = Decisions()
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions, mode='flat_vector')
    assert result.route is None
    parent = next(n for n in snapshot.nodes if n.title == 'Cats')
    assert 'Cats eat fish.' in snapshot.original(parent.id)


async def test_scoped_backend_failure_preserves_successful_baseline() -> None:
    from scone_memory.backends.memory import InMemoryVectorIndex
    class FailedScopedIndex(InMemoryVectorIndex):
        async def search(self, space, vector, limit, as_of=None, tags=(), where=None, conditions=None):
            if where:
                raise RuntimeError('scoped backend failed')
            return await super().search(space, vector, limit, as_of, tags, where, conditions)
    snapshot = book()
    decisions = Decisions('section_vector')
    index = await StructuredDocumentIndex.build(snapshot, Embeddings(), vectors=FailedScopedIndex())
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions, mode='section_vector')
    assert result.mode == 'flat_vector'
    assert result.reason == 'scoped_search_failed'
    assert result.evidence == result.baseline


async def test_heading_only_sources_are_retrievable() -> None:
    snapshot = book('# Cats purr\n# Dogs bark\n')
    decisions = Decisions()
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions, mode='original')
    assert result.evidence
    assert 'Cats purr' in result.evidence[0].text
    assert result.route is not None


async def test_malformed_fetch_decision_falls_back() -> None:
    class InvalidFetch(Decisions):
        async def choose_fetch(self, query, sections, max_bytes):
            return FetchDecision('original', 'test', (float('nan'),), -4, -8)
    snapshot = book()
    decisions = InvalidFetch()
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions)
    assert result.reason == 'fetch_decision_failed'
    assert result.evidence == result.baseline


async def test_weaker_ancestor_does_not_replace_the_best_section_with_whole_book() -> None:
    from scone_memory.retrieval.section_routing import RouteResult
    snapshot = book('# Animals\nOverview.\n## Cats\nCats purr.\n## Dogs\nDogs bark.')
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    cat = next(n for n in snapshot.nodes if n.title == 'Cats')
    parent = next(n for n in snapshot.nodes if n.title == 'Animals')
    route = RouteResult(snapshot.version, (cat.id, parent.id), (.9, .2), 'routed')
    originals = index._originals(route)
    assert [item.section_id for item in originals] == [cat.id]


async def test_vector_guided_routing_reaches_deep_original_with_one_address_call() -> None:
    snapshot = book('# Animals\nOverview.\n## Cats\nCats purr.\n## Dogs\nDogs bark.')

    class PathDecisions(Decisions):
        async def choose(self, query: str, menus: tuple[RouteMenu, ...]) -> ChoiceBatch:
            rows = []
            for menu in menus:
                scores = [0.] * (len(menu.options) + 1)
                target = next((i for i, option in enumerate(menu.options)
                               if option.title in ('Animals / Cats', 'Cats')), 0)
                scores[target] = 1.
                rows.append(tuple(scores))
            return ChoiceBatch(tuple(rows), 'test')

    decisions = PathDecisions()
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    old = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions)
    guided = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions,
                                  routing='vector_candidates')
    assert old.route is not None and old.route.requests == 2
    assert guided.route is not None and guided.route.requests == 1
    assert guided.mode == 'original'
    assert [p.text for p in guided.evidence] == ['## Cats\nCats purr.\n']
    assert guided.evidence == old.evidence
    assert guided.baseline == old.baseline


@pytest.mark.parametrize('outcome', ['none', 'invalid', 'failed'])
async def test_vector_guided_route_failures_preserve_broad_evidence(outcome: str) -> None:
    class FailedRoute(Decisions):
        async def choose(self, query: str, menus: tuple[RouteMenu, ...]) -> ChoiceBatch:
            if outcome == 'failed':
                raise RuntimeError('unavailable')
            if outcome == 'invalid':
                return ChoiceBatch(((float('nan'),),), 'test')
            return ChoiceBatch(tuple((0.,) * len(m.options) + (1.,) for m in menus), 'test')

    snapshot = book()
    decisions = FailedRoute()
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions,
                                  routing='vector_candidates')
    assert result.mode == 'flat_vector'
    assert result.reason == {'none': 'no_match', 'invalid': 'invalid_response',
                             'failed': 'provider_failed'}[outcome]
    assert result.evidence == result.baseline
    assert result.baseline


async def test_vector_guided_flat_mode_still_skips_address_routing() -> None:
    snapshot = book()
    decisions = Decisions()
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions,
                                  mode='flat_vector', routing='vector_candidates')
    assert result.route is None
    assert result.evidence == result.baseline


async def test_empty_vector_results_do_not_misreport_an_empty_document() -> None:
    from scone_memory.backends.memory import InMemoryVectorIndex

    class NoHits(InMemoryVectorIndex):
        async def search(self, space, vector, limit, as_of=None, tags=(), where=None, conditions=None):
            return []

    snapshot = book()
    decisions = Decisions()
    index = await StructuredDocumentIndex.build(snapshot, Embeddings(), vectors=NoHits())
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions,
                                  routing='vector_candidates')
    assert result.reason == 'no_vector_candidates'
    assert result.mode == 'flat_vector'
    assert result.route is None
    assert result.evidence == result.baseline == ()


class UnusedDecisions(Decisions):
    async def choose(self, query, menus):
        pytest.fail('local structure retrieval must not call address routing')

    async def choose_fetch(self, query, sections, max_bytes):
        pytest.fail('local structure retrieval must not call fetch selection')


async def test_local_structure_restores_a_split_paragraph_without_model_decisions() -> None:
    paragraph = 'Cats need fresh water and balanced food. ' * 24
    snapshot = book('# Cats\n' + paragraph + '\n\n' + 'Unrelated background. ' * 100 + '\n# Dogs\nDogs bark.')
    index = await StructuredDocumentIndex.build(snapshot, Embeddings(), chunk_size=120)
    decisions = UnusedDecisions()
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions,
                                  mode='local_structure', limit=5)
    assert paragraph not in result.baseline[0].text
    assert paragraph in result.evidence[0].text
    assert 'Dogs bark.' not in result.evidence[0].text
    assert result.route is result.fetch is None
    assert result.fetch_decision_ms == 0
    raw = snapshot.content.encode()
    assert all(raw[p.start:p.end].decode() == p.text for p in result.evidence)
    assert len({(p.start, p.end) for p in result.evidence}) == len(result.evidence)


async def test_local_structure_can_read_a_short_original_section() -> None:
    snapshot = book('# Cats\nCats purr.\n\nThey also sleep.\n# Dogs\nDogs bark.')
    index = await StructuredDocumentIndex.build(snapshot, Embeddings(), chunk_size=120)
    decisions = UnusedDecisions()
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions,
                                  mode='local_structure', limit=1)
    assert result.evidence[0].text == '# Cats\nCats purr.\n\nThey also sleep.\n'
    assert result.route is result.fetch is None


@pytest.mark.parametrize('budget', [1, 17, 300])
async def test_local_structure_respects_utf8_and_small_budgets(budget: int) -> None:
    snapshot = book('# Cats\n' + 'Cats café 🐈. ' * 100 + '\n# Dogs\nOther facts.')
    index = await StructuredDocumentIndex.build(snapshot, Embeddings(), chunk_size=120)
    decisions = UnusedDecisions()
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions,
                                  mode='local_structure', limit=2, max_bytes=budget)
    assert sum(len(p.text.encode()) for p in result.evidence) <= budget
    assert len(result.evidence) <= 2
    assert all(snapshot.content.encode()[p.start:p.end].decode() == p.text for p in result.evidence)


async def test_local_structure_does_not_expand_a_huge_section_or_paragraph() -> None:
    snapshot = book('# Cats\n' + 'Cats facts. ' * 1000)
    index = await StructuredDocumentIndex.build(snapshot, Embeddings(), chunk_size=120)
    decisions = UnusedDecisions()
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions,
                                  mode='local_structure', limit=1, max_bytes=8000)
    assert result.evidence == result.baseline
    assert len(result.evidence[0].text.encode()) < 2000


async def test_local_structure_does_not_spend_two_slots_on_child_and_parent() -> None:
    snapshot = book('# Parent\nIntroduction.\n## Child\nCats purr.\n# Other\nDogs bark.')
    index = await StructuredDocumentIndex.build(snapshot, Embeddings(), chunk_size=120)
    decisions = UnusedDecisions()
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions,
                                  mode='local_structure', limit=2)
    assert [p.text for p in result.evidence] == ['## Child\nCats purr.\n', '# Other\nDogs bark.']


async def test_local_hybrid_selects_relevant_paragraph_without_section_padding() -> None:
    snapshot = book('# Animals\nDogs bark.\n\nCats need taurine.\n\nBirds fly.')
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    decisions = UnusedDecisions()
    result = await index.retrieve('taurine', snapshot, SectionRouter(decisions), decisions,
                                  mode='local_hybrid', limit=1)
    assert result.evidence[0].text.strip() == 'Cats need taurine.'
    assert result.route is result.fetch is None
    assert result.fetch_decision_ms == 0


async def test_local_hybrid_lexical_lane_recovers_evidence_outside_vector_shortlist() -> None:
    snapshot = book(''.join(f'# Section {i}\nOrdinary background {i}.\n' for i in range(40))
                    + '# Appendix\nZirconium stabilizes the sensor.')
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    decisions = UnusedDecisions()
    result = await index.retrieve('zirconium sensor', snapshot, SectionRouter(decisions), decisions,
                                  mode='local_hybrid', limit=5)
    assert not any('Zirconium' in p.text for p in result.baseline)
    assert any(p.text.strip() == 'Zirconium stabilizes the sensor.' for p in result.evidence)


async def test_local_hybrid_recovers_long_paragraph_once_with_exact_source_offsets() -> None:
    paragraph = 'Cats need fresh water and balanced food. ' * 24
    snapshot = book('# Cats\n' + paragraph + '\n\nDogs bark.\n')
    index = await StructuredDocumentIndex.build(snapshot, Embeddings(), chunk_size=120)
    decisions = UnusedDecisions()
    result = await index.retrieve('cats water', snapshot, SectionRouter(decisions), decisions,
                                  mode='local_hybrid', limit=5)
    assert sum(paragraph in p.text for p in result.evidence) == 1
    raw = snapshot.content.encode()
    assert all(raw[p.start:p.end].decode() == p.text for p in result.evidence)
    assert all(a.end <= b.start or b.end <= a.start
               for i, a in enumerate(result.evidence) for b in result.evidence[i + 1:])


@pytest.mark.parametrize('budget', [1, 17, 300])
async def test_local_hybrid_preserves_utf8_bounds_for_oversized_paragraphs(budget: int) -> None:
    snapshot = book('# Cats\n' + 'Cats café 🐈. ' * 100)
    index = await StructuredDocumentIndex.build(snapshot, Embeddings(), chunk_size=120)
    decisions = UnusedDecisions()
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions,
                                  mode='local_hybrid', limit=2, max_bytes=budget)
    assert result.evidence
    assert sum(len(p.text.encode()) for p in result.evidence) <= budget
    assert len(result.evidence) <= 2
    assert all(snapshot.content.encode()[p.start:p.end].decode() == p.text for p in result.evidence)


async def test_local_hybrid_keeps_heading_only_evidence() -> None:
    snapshot = book('# Cats purr\n# Dogs bark\n')
    index = await StructuredDocumentIndex.build(snapshot, Embeddings())
    decisions = UnusedDecisions()
    result = await index.retrieve('cats', snapshot, SectionRouter(decisions), decisions,
                                  mode='local_hybrid', limit=1)
    assert result.evidence[0].text == '# Cats purr\n'


@pytest.mark.parametrize('content', [
    '# Cats\n' + 'Cats need taurine. ' * 50 + '\n\nDogs bark.',
    '# Cats purr\n# Dogs\nDogs bark.',
])
async def test_local_hybrid_does_not_replace_unprojectable_relevant_hit_with_unrelated_paragraph(content: str) -> None:
    snapshot = book(content)
    index = await StructuredDocumentIndex.build(snapshot, Embeddings(), chunk_size=120)
    decisions = UnusedDecisions()
    result = await index.retrieve('cats taurine', snapshot, SectionRouter(decisions), decisions,
                                  mode='local_hybrid', limit=1, max_bytes=100)
    assert 'Cats' in result.evidence[0].text
    assert 'Dogs' not in result.evidence[0].text
