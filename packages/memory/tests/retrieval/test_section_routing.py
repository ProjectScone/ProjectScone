from __future__ import annotations

import asyncio
import pytest

from scone_memory.retrieval.section_routing import (
    ChoiceBatch, RouteMenu, SectionRouter, SectionSnapshot,
)


class Chooser:
    definition = 'test-v1'
    calls = 0

    async def choose(self, query: str, menus: tuple[RouteMenu, ...]) -> ChoiceBatch:
        self.calls += 1
        distributions = []
        for menu in menus:
            probabilities = tuple(.9 if i == 0 else 0.0 for i in range(len(menu.options))) + (.1,)
            distributions.append(probabilities)
        return ChoiceBatch(tuple(distributions), 'test', 10, 2)


def snapshot(text: str = '# Animals\nintro é\n## Cats\nCats purr.\n## Dogs\nDogs bark.',
             scope: str = 'one') -> SectionSnapshot:
    return SectionSnapshot.from_markdown(scope, 'book', text)


def test_sections_preserve_parent_prose_and_unicode() -> None:
    book = snapshot()
    assert [n.text for n in book.nodes if n.text] == ['intro é\n', 'Cats purr.\n', 'Dogs bark.']
    cats = next(n for n in book.nodes if n.title == 'Cats')
    assert book.path(cats.id) == ('Animals', 'Cats')
    assert book.version != snapshot('# Animals\nchanged').version


@pytest.mark.asyncio
async def test_routing_cache_is_bound_to_source_scope_and_definition() -> None:
    chooser = Chooser()
    router = SectionRouter(chooser)
    result = await router.route('question', snapshot())
    assert result.section_ids
    assert not result.cache_hit
    assert (await router.route('question', snapshot())).cache_hit
    assert not (await router.route('question', snapshot(scope='two'))).cache_hit
    assert not (await router.route('question', snapshot('# Animals\nnew'))).cache_hit
    chooser.definition = 'test-v2'
    assert not (await router.route('question', snapshot())).cache_hit


@pytest.mark.asyncio
async def test_no_match_and_invalid_probabilities_fall_back_without_caching() -> None:
    class NoneChooser(Chooser):
        async def choose(self, query: str, menus: tuple[RouteMenu, ...]) -> ChoiceBatch:
            return ChoiceBatch(tuple(tuple(0.0 for _ in m.options) + (1.0,) for m in menus), 'test')
    router = SectionRouter(NoneChooser())
    result = await router.route('unrelated', snapshot())
    assert result.section_ids == ()
    assert result.reason == 'no_match'

    class InvalidChooser(Chooser):
        async def choose(self, query: str, menus: tuple[RouteMenu, ...]) -> ChoiceBatch:
            return ChoiceBatch(((float('nan'),),), 'test')
    router = SectionRouter(InvalidChooser())
    result = await router.route('question', snapshot())
    assert result.reason == 'invalid_response'
    assert not (await router.route('question', snapshot())).cache_hit


@pytest.mark.asyncio
async def test_budget_and_provider_failure_preserve_flat_fallback() -> None:
    deep = snapshot('# One\n## Two\n### Three\nvalue')
    result = await SectionRouter(Chooser(), max_rounds=1).route('question', deep)
    assert result.reason == 'budget_exhausted'
    assert result.section_ids == ()

    class Failed(Chooser):
        async def choose(self, query: str, menus: tuple[RouteMenu, ...]) -> ChoiceBatch:
            raise RuntimeError('provider unavailable')
    assert (await SectionRouter(Failed()).route('question', deep)).reason == 'provider_failed'


@pytest.mark.asyncio
async def test_cancellation_propagates() -> None:
    class Cancelled(Chooser):
        async def choose(self, query: str, menus: tuple[RouteMenu, ...]) -> ChoiceBatch:
            raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await SectionRouter(Cancelled()).route('question', snapshot())


def test_empty_document_and_invalid_bounds() -> None:
    assert len(snapshot('').nodes) == 1
    with pytest.raises(ValueError):
        SectionRouter(Chooser(), beam_width=0)


@pytest.mark.asyncio
async def test_ancestor_menu_exposes_descendant_addresses() -> None:
    class Inspect(Chooser):
        async def choose(self, query, menus):
            if menus[0].path == ():
                animal = next(o for o in menus[0].options if o.title == 'Animals')
                assert 'Cats' in animal.outline
                assert 'Dogs' in animal.outline
            return await super().choose(query, menus)
    result = await SectionRouter(Inspect()).route('cats', snapshot())
    assert result.reason == 'routed'


def test_oversized_menu_skips_descendant_outline_construction() -> None:
    from scone_memory.retrieval.section_routing import _Path, _menu
    book = snapshot(''.join(f'# Topic {i}\n## Details {i}\nbody\n' for i in range(255)))
    menu = _menu(book, _Path(book.nodes[0].id))
    assert len(menu.options) > 254
    assert all(not option.outline for option in menu.options)


async def test_vector_addresses_skip_ancestors_and_keep_complete_paths() -> None:
    book = snapshot()
    cats = next(n.id for n in book.nodes if n.title == 'Cats')

    class Inspect(Chooser):
        async def choose(self, query: str, menus: tuple[RouteMenu, ...]) -> ChoiceBatch:
            assert len(menus) == 1
            assert [(o.section_id, o.title, o.terminal) for o in menus[0].options] == [
                (cats, 'Animals / Cats', True)]
            return await super().choose(query, menus)

    result = await SectionRouter(Inspect(), max_rounds=1).route(
        'cats', book, candidate_section_ids=(cats, cats))
    assert result.reason == 'routed'
    assert result.section_ids == (cats,)
    assert result.requests == 1


async def test_vector_address_cache_cannot_reuse_a_different_candidate_set() -> None:
    book = snapshot()
    cats = next(n.id for n in book.nodes if n.title == 'Cats')
    dogs = next(n.id for n in book.nodes if n.title == 'Dogs')
    router = SectionRouter(Chooser(), max_rounds=1)
    first = await router.route('pets', book, candidate_section_ids=(cats,))
    repeated = await router.route('pets', book, candidate_section_ids=(cats,))
    changed = await router.route('pets', book, candidate_section_ids=(dogs,))
    hierarchical = await router.route('pets', book)
    assert first.section_ids == (cats,)
    assert repeated.cache_hit and repeated.requests == 0
    assert changed.section_ids == (dogs,) and not changed.cache_hit
    assert not hierarchical.cache_hit
    assert hierarchical.reason == 'budget_exhausted'


@pytest.mark.parametrize('kind', ['foreign', 'oversized'])
async def test_foreign_or_unbounded_vector_addresses_are_rejected(kind: str) -> None:
    book = snapshot()
    identifiers = (('foreign-document-section',) if kind == 'foreign'
                   else (book.nodes[0].id,) * 255)
    with pytest.raises(ValueError, match='candidate'):
        await SectionRouter(Chooser()).route('cats', book, candidate_section_ids=identifiers)


async def test_empty_vector_addresses_do_not_invoke_the_provider() -> None:
    result = await SectionRouter(Chooser()).route('cats', snapshot(), candidate_section_ids=())
    assert result.reason == 'no_match'
    assert result.section_ids == ()
    assert result.requests == 0
