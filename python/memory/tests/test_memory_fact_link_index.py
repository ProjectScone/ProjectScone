"""Bounded pair lookup does not scan unrelated stored links."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from scone_memory.backends.memory import InMemoryDocumentStore
from scone_memory.core.models import FactLink, LINK_KINDS
from scone_memory.core.ports import NewFactLink


STAMP = "2026-09-07T00:00:00Z"


def new_link(space="alpha", left=1, right=2, kind="supports", **kwargs):
    return NewFactLink(space=space, from_fact=left, to_fact=right, kind=kind, created_at=STAMP, **kwargs)


class NoScans(dict[int, FactLink]):
    def values(self):
        raise AssertionError("bounded operations must not scan global links")

    def items(self):
        raise AssertionError("bounded operations must not scan global links")

    def __iter__(self):
        raise AssertionError("bounded operations must not scan global links")


async def test_lookup_and_duplicate_insert_never_scan_global_link_records():
    store = InMemoryDocumentStore()
    wanted = await store.insert_fact_link(new_link(quote="original"))
    for number in range(100):
        await store.insert_fact_link(new_link(left=100 + number, right=1000 + number))
    store._links = NoScans(store._links)
    assert await store.fact_links_between("alpha", [2, 1], 49) == [wanted]
    assert await store.insert_fact_link(new_link(quote="replacement")) == wanted
    assert (await store.insert_fact_link(new_link(quote="replacement"))).quote == "original"
    added = await store.insert_fact_link(new_link(left=2, right=3))
    assert added.link_id > wanted.link_id


async def test_parallel_links_directions_tenant_and_insertion_order_are_preserved():
    store = InMemoryDocumentStore()
    first = await store.insert_fact_link(new_link(left=3, right=1, kind="contradicts"))
    hidden = await store.insert_fact_link(new_link(space="beta", left=3, right=1))
    second = await store.insert_fact_link(new_link(left=1, right=3, kind="supports"))
    third = await store.insert_fact_link(new_link(left=3, right=1, kind="supports"))
    fourth = await store.insert_fact_link(new_link(left=1, right=3, kind="derived_from"))
    assert await store.fact_links_between("alpha", [3, 1, 3], 49) == [first, second, third, fourth]
    assert await store.fact_links_between("beta", [3, 1], 49) == [hidden]
    assert await store.fact_links_between("alpha", [1], 49) == []
    assert await store.fact_links_between("alpha", [1, 3], 2) == [first, second]


async def test_lookup_caps_supported_pairs_and_return_count():
    store = InMemoryDocumentStore()
    for left in range(1, 18):
        for right in range(1, 18):
            for kind in LINK_KINDS:
                await store.insert_fact_link(new_link(left=left, right=right, kind=kind))
    found = await store.fact_links_between("alpha", list(range(1, 18)), 100000)
    assert len(found) == 49
    assert all(link.from_fact <= 16 and link.to_fact <= 16 for link in found)
    assert [link.link_id for link in found] == sorted(link.link_id for link in found)
    assert await store.fact_links_between("alpha", [], 49) == []
    assert await store.fact_links_between("alpha", [1, 2], 0) == []
    assert await store.fact_links_between("alpha", [1, 2], -1) == []


async def test_pair_probe_count_is_bounded_by_selected_ids_and_supported_kinds():
    store = InMemoryDocumentStore()
    await store.insert_fact_link(new_link())

    class CountReads(dict):
        reads = 0

        def get(self, key, default=None):
            self.reads += 1
            return super().get(key, default)

    index = CountReads(store._link_keys)
    store._link_keys = index
    await store.fact_links_between("alpha", list(range(1, 25)), 49)
    assert 0 < index.reads <= 16 * 16 * len(LINK_KINDS)


async def test_deleting_a_space_removes_index_entries_and_allows_fresh_identity():
    store = InMemoryDocumentStore()
    first = await store.insert_fact_link(new_link())
    other = await store.insert_fact_link(new_link(space="beta"))
    deleted = await store.delete_space("alpha", STAMP)
    assert deleted.links == 1
    assert not any(key[0] == "alpha" for key in store._link_keys)
    assert await store.fact_links_between("alpha", [1, 2], 49) == []
    assert await store.fact_links_between("beta", [1, 2], 49) == [other]
    replacement = await store.insert_fact_link(new_link())
    assert replacement.link_id > first.link_id
    assert await store.fact_links_between("alpha", [1, 2], 49) == [replacement]
    assert await store.insert_fact_link(new_link()) == replacement


async def test_unsupported_kinds_do_not_create_unbounded_pair_buckets():
    store = InMemoryDocumentStore()
    for number in range(10):
        with pytest.raises(ValidationError):
            await store.insert_fact_link(new_link(kind=f"unknown-{number}"))
    assert store._links == {}
    assert store._link_keys == {}
    assert await store.fact_links_between("alpha", [1, 2], 49) == []
