"""Checked complete source inventories, independent of ordinary listing caps."""
from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel, ValidationError
from typing import TypeGuard, TypeVar

from ..core.errors import InvalidInput
from ..core.graph_read import LedgerPager, checked_ledger_page
from ..core.models import Episode, Fact, FactLink
from ..core.affirmations import Affirmation, affirmation_store
from ..core.ports import ArchiveLinkInventory, DocumentStore, EpisodeInventory


async def prepare(documents: DocumentStore, space: str) -> None:
    if (not isinstance(documents, EpisodeInventory) or not callable(documents.page_episodes)
            or not isinstance(documents, LedgerPager) or not callable(documents.page_facts)
            or not isinstance(documents, ArchiveLinkInventory) or not callable(documents.space_fact_links)):
        raise InvalidInput('archive export requires complete episode, fact and space-link inventories')
    flush = getattr(documents, 'prepare_archive_read', None)
    if callable(flush):
        await flush(space)


def _positive(value: object) -> TypeGuard[int]:
    return type(value) is int and 0 < value < 2**63


async def episodes(documents: DocumentStore, space: str, expected: int) -> AsyncIterator[Episode]:
    if not isinstance(documents, EpisodeInventory) or not callable(documents.page_episodes):
        raise InvalidInput('archive export requires a paged episode inventory')
    if type(expected) is not int or expected < 0:
        raise InvalidInput('archive episode count is invalid')
    before = None
    seen = 0
    while True:
        page = await documents.page_episodes(space, before, 100, None)
        if not isinstance(page, list) or len(page) > 100:
            raise InvalidInput('archive episode page exceeds its shape or limit')
        if not page:
            if seen != expected:
                raise InvalidInput('archive episode inventory does not match its source count')
            return
        detached: list[Episode] = []
        for raw in page:
            if (not isinstance(raw, Episode) or raw.space != space or not _positive(raw.episode_id)
                    or (before is not None and raw.episode_id >= before)):
                raise InvalidInput('archive episode page has invalid scope, identity or order')
            try:
                episode = Episode.model_validate(raw.model_dump(), strict=True)
            except ValidationError as error:
                raise InvalidInput('archive episode page contains an invalid record') from error
            detached.append(episode)
            before = episode.episode_id
        seen += len(detached)
        if seen > expected:
            raise InvalidInput('archive episode inventory changed beyond its source count')
        for episode in detached:
            yield episode


async def facts(documents: DocumentStore, space: str) -> list[Fact]:
    if not isinstance(documents, LedgerPager) or not callable(documents.page_facts):
        raise InvalidInput('archive export requires a paged fact inventory')
    before = None
    inventory: list[Fact] = []
    while True:
        page = await documents.page_facts(space, before, 1000)
        problem = checked_ledger_page(page, space=space, before_id=before, limit=1000)
        if problem:
            raise InvalidInput(f'archive fact page is invalid: {problem}')
        if not page:
            inventory.reverse()
            return inventory
        for raw in page:
            if not _positive(raw.fact_id):
                raise InvalidInput('archive fact page contains an invalid identity')
            try:
                inventory.append(Fact.model_validate(raw.model_dump(), strict=True))
            except ValidationError as error:
                raise InvalidInput('archive fact page contains an invalid record') from error
        before = inventory[-1].fact_id


Row = TypeVar('Row', bound=BaseModel)


def _ledger_rows(raw_rows: object, space: str, model: type[Row], identity: str) -> list[Row]:
    if not isinstance(raw_rows, list):
        raise InvalidInput('archive ledger inventory must be a list')
    checked: list[Row] = []
    previous = 0
    for raw in raw_rows:
        value = getattr(raw, identity, None)
        if (not isinstance(raw, model) or getattr(raw, 'space', None) != space
                or not _positive(value) or value <= previous):
            raise InvalidInput('archive ledger inventory has invalid scope, identity or order')
        try:
            checked.append(model.model_validate(raw.model_dump(), strict=True))
        except ValidationError as error:
            raise InvalidInput('archive ledger inventory contains an invalid record') from error
        previous = value
    return checked


async def links(documents: DocumentStore, space: str) -> list[FactLink]:
    if not isinstance(documents, ArchiveLinkInventory) or not callable(documents.space_fact_links):
        raise InvalidInput('archive export requires a complete space-link inventory')
    return _ledger_rows(await documents.space_fact_links(space), space, FactLink, 'link_id')


async def affirmations(documents: DocumentStore, space: str) -> list[Affirmation]:
    store = affirmation_store(documents)
    if store is None:
        return []
    return _ledger_rows(await store.space_affirmations(space), space, Affirmation, 'affirmation_id')
