"""Real stores verify structure expansion from exact retained source bytes."""
from __future__ import annotations

from ..paths import TESTS_ROOT

import hashlib
from pathlib import Path

import pytest

from scone_memory.backends.memory import InMemoryDocumentStore
from scone_memory.backends.sqlite import SqliteDocumentStore
from scone_memory.core.models import RecallItem, RecallResult
from scone_memory.core.ports import NewChunk, NewEpisode, TextFilter
from scone_memory.retrieval.filters import parse_filter

STAMP = "2026-09-07T00:00:00.000Z"
FIXTURE = TESTS_ROOT / "fixtures/structure/handbook.md"


def test_structure_capability_exists():
    from scone_memory.ingestion import structure
    assert callable(getattr(structure, "parse_structure", None)), "structure parsing is missing"


def test_parser_preserves_unicode_nested_sections_code_and_table_boundaries():
    from scone_memory.ingestion.structure import parse_structure
    content = FIXTURE.read_bytes().decode()
    parsed = parse_structure(content)
    assert parsed == parse_structure(content)
    assert parsed.content_hash == hashlib.sha256(content.encode()).hexdigest()
    sections = {section.title: section for section in parsed.sections}
    assert "This is code, not a heading" not in sections
    assert sections["Limits"].parent_section_id == sections["Service"].section_id
    assert sections["Service"].end == sections["Recovery"].start
    blocks = parsed.blocks
    assert b"".join(content.encode()[b.start:b.end] for b in blocks) == content.encode()
    table, = (block for block in blocks if block.kind == "table")
    assert content.encode()[table.header.start:table.header.end].decode() == "| Tier | Requests | Region |\n"
    assert len(table.rows) == 2
    assert content.encode()[table.rows[1].start:table.rows[1].end].decode() == "| team | 42 | 東京 |\n"
    assert sum(block.kind == "fenced_code" for block in blocks) == 1


@pytest.fixture(params=["memory", "sqlite"])
async def documents(request, tmp_path):
    store = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(tmp_path / "structure.sqlite")
    yield store
    if isinstance(store, SqliteDocumentStore):
        await store.close()


async def seed(documents, text=None, needle="42", space="alpha", digest=None):
    content = text if text is not None else FIXTURE.read_bytes().decode()
    episode = await documents.insert_episode(NewEpisode(
        space=space, kind="file", content=content,
        content_hash=digest or hashlib.sha256(content.encode()).hexdigest(),
        created_at=STAMP, ingested_at=STAMP, source="manual/handbook.md",
        tags=("operations",), metadata={"team": "support", "priority": "5", "session_id": "source-session"}))
    start = content.encode().index(needle.encode())
    chunk, = await documents.insert_chunks([NewChunk(
        episode_id=episode.episode_id, space=space, ordinal=0,
        start=start, end=start + len(needle.encode()), text=needle, created_at=STAMP)])
    return episode, RecallResult(items=[RecallItem(chunk_id=chunk.chunk_id, episode_id=episode.episode_id,
                                                   text=needle, score=1.0, created_at=STAMP)])


async def test_expansion_adds_table_headers_with_exact_provenance(documents):
    from scone_memory.retrieval.structural import expand_structural_context
    episode, recall = await seed(documents)
    before = recall.model_dump()
    expanded = await expand_structural_context(documents, "alpha", recall, scope=TextFilter())
    section, = expanded.sections
    assert section.title == "Limits"
    assert "| Tier | Requests | Region |" in section.text
    assert section.text.encode() == episode.content.encode()[section.start:section.end]
    assert section.episode_id == episode.episode_id
    assert section.content_hash == episode.content_hash
    assert section.matched_chunk_ids == (recall.items[0].chunk_id,)
    assert expanded.returned_bytes == len(section.text.encode())
    assert expanded.returned_bytes > len(recall.items[0].text.encode())
    assert recall.model_dump() == before


@pytest.mark.parametrize("scope", [
    TextFilter(tags=("other",)), TextFilter(where={"team": "other"}),
    TextFilter(conditions=parse_filter({"field": "priority", "above": 9})),
    TextFilter(kind="note"), TextFilter(source_prefix="foreign/"),
    TextFilter(since="2026-09-08T00:00:00.000Z"),
    TextFilter(until="2026-09-06T00:00:00.000Z"),
    TextFilter(as_of="2026-09-06T00:00:00.000Z"),
])
async def test_every_scope_filter_is_revalidated(documents, scope):
    from scone_memory.retrieval.structural import expand_structural_context
    _, recall = await seed(documents)
    expanded = await expand_structural_context(documents, "alpha", recall, scope=scope)
    assert expanded.sections == ()
    assert expanded.out_of_scope == 1


async def test_missing_foreign_deleted_and_tampered_sources_never_surface(documents):
    from scone_memory.retrieval.structural import expand_structural_context
    episode, recall = await seed(documents)
    foreign = await expand_structural_context(documents, "beta", recall, scope=TextFilter())
    assert not foreign.sections and foreign.provenance_missing == 1
    await documents.delete_episode("alpha", episode.episode_id)
    deleted = await expand_structural_context(documents, "alpha", recall, scope=TextFilter())
    assert not deleted.sections and deleted.provenance_missing == 1
    _, corrupt = await seed(documents)
    corrupt.items[0].text = "unretained text"
    invalid = await expand_structural_context(documents, "alpha", corrupt, scope=TextFilter())
    assert not invalid.sections and invalid.invalid_provenance == 1


async def test_whole_sections_are_omitted_when_the_byte_budget_cannot_preserve_blocks(documents):
    from scone_memory.retrieval.structural import StructuralLimits, expand_structural_context
    _, recall = await seed(documents, text="# Large\n" + "résumé " * 1000, needle="résumé")
    expanded = await expand_structural_context(documents, "alpha", recall, scope=TextFilter(),
                                               limits=StructuralLimits(max_bytes=100))
    assert expanded.sections == () and expanded.returned_bytes == 0
    assert expanded.truncated and expanded.omitted_sections == 1


async def test_sources_sections_chunks_and_input_bytes_are_bounded(documents):
    from scone_memory.retrieval.structural import StructuralLimits, expand_structural_context
    _, first = await seed(documents, text="# One\n42", needle="42")
    _, second = await seed(documents, text="# Two\n42", needle="42")
    recall = RecallResult(items=first.items + second.items)
    source_limited = await expand_structural_context(documents, "alpha", recall, scope=TextFilter(), limits=StructuralLimits(max_sources=1))
    assert len(source_limited.sections) == 1 and source_limited.omitted_sources == 1
    section_limited = await expand_structural_context(documents, "alpha", recall, scope=TextFilter(), limits=StructuralLimits(max_sections=1))
    assert len(section_limited.sections) == 1 and section_limited.omitted_sections == 1
    chunk_limited = await expand_structural_context(documents, "alpha", recall, scope=TextFilter(), limits=StructuralLimits(max_chunks=1))
    assert chunk_limited.chunks_considered == 1 and chunk_limited.omitted_chunks == 1
    byte_limited = await expand_structural_context(documents, "alpha", recall, scope=TextFilter(), limits=StructuralLimits(max_source_bytes=3))
    assert not byte_limited.sections and byte_limited.omitted_sources == 2


async def test_excluded_session_and_stale_recall_text_are_rejected(documents):
    from scone_memory.retrieval.structural import expand_structural_context
    _, recall = await seed(documents)
    excluded = await expand_structural_context(documents, "alpha", recall, scope=TextFilter(), exclude_session_id="source-session")
    assert not excluded.sections and excluded.out_of_scope == 1
    recall.items[0].text = "fabricated"
    stale = await expand_structural_context(documents, "alpha", recall, scope=TextFilter())
    assert not stale.sections and stale.invalid_provenance == 1


def test_parser_is_immutable_and_ids_bind_to_all_source_bytes():
    from pydantic import ValidationError
    from scone_memory.ingestion.structure import parse_structure
    text = "# Same\r\nα\r\n## Same\r\n~~~md\r\n# fake\r\n~~~\r\n"
    parsed = parse_structure(text)
    assert len({section.section_id for section in parsed.sections}) == 3
    assert len(parsed.sections) == 3
    assert parsed.sections[-1].parent_section_id == parsed.sections[-2].section_id
    assert parsed.sections[-1].end == len(text.encode())
    assert {section.section_id for section in parsed.sections}.isdisjoint(
        section.section_id for section in parse_structure(text + "x").sections)
    with pytest.raises(ValidationError):
        parsed.sections[0].start = 12
    assert b"".join(text.encode()[block.start:block.end] for block in parsed.blocks) == text.encode()


def test_pathological_heading_count_is_bounded():
    from scone_memory.ingestion.structure import parse_structure
    with pytest.raises(ValueError, match="structure node limit"):
        parse_structure("# x\n" * 9000)


async def test_pathological_structure_is_explicitly_omitted(documents):
    from scone_memory.retrieval.structural import expand_structural_context
    _, recall = await seed(documents, text="# x\n" * 9000, needle="x")
    expanded = await expand_structural_context(documents, "alpha", recall, scope=TextFilter())
    assert not expanded.sections and expanded.omitted_sources == 1 and expanded.truncated


async def test_positive_scope_and_time_offsets(documents):
    from scone_memory.retrieval.structural import expand_structural_context
    _, recall = await seed(documents)
    scope = TextFilter(tags=("operations",), where={"team": "support"},
                       conditions=parse_filter({"field": "priority", "at_least": 5}),
                       kind="file", source_prefix="manual/", since="2026-09-06T19:00:00-05:00",
                       until=STAMP, as_of=STAMP)
    expanded = await expand_structural_context(documents, "alpha", recall, scope=scope)
    assert len(expanded.sections) == 1 and expanded.out_of_scope == 0


async def test_bad_chunk_offsets_do_not_expand_even_when_text_is_recalled(documents):
    from scone_memory.retrieval.structural import expand_structural_context
    episode, recall = await seed(documents)
    bad, = await documents.insert_chunks([NewChunk(episode_id=episode.episode_id, space="alpha", ordinal=9,
                                                   start=-1, end=2, text="42", created_at=STAMP)])
    recall.items[0].chunk_id = bad.chunk_id
    expanded = await expand_structural_context(documents, "alpha", recall, scope=TextFilter())
    assert not expanded.sections and expanded.invalid_provenance == 1


async def test_only_bounded_recall_ids_and_sources_are_read(documents):
    from scone_memory.retrieval.structural import StructuralLimits, expand_structural_context
    episode, recall = await seed(documents)
    class ReadOnlyIds:
        def __init__(self):
            self.chunk_requests = []
            self.source_requests = []
        async def get_chunks(self, space, ids):
            self.chunk_requests.append(tuple(ids))
            return await documents.get_chunks(space, ids)
        async def get_episode(self, space, episode_id):
            self.source_requests.append(episode_id)
            return await documents.get_episode(space, episode_id)
    reader = ReadOnlyIds()
    recall.items *= 100
    expanded = await expand_structural_context(reader, "alpha", recall, scope=TextFilter(), limits=StructuralLimits(max_chunks=2))
    assert len(reader.chunk_requests) == 2 and len(reader.chunk_requests[0]) == 2
    assert len(reader.chunk_requests[1]) == 1
    assert reader.source_requests == [episode.episode_id, episode.episode_id]
    assert len(expanded.sections) == 1 and expanded.omitted_chunks == 98


async def test_foreign_or_missing_episode_from_adapter_is_rejected(documents):
    from scone_memory.retrieval.structural import expand_structural_context
    episode, recall = await seed(documents)
    class BadEpisodeAdapter:
        async def get_chunks(self, space, ids):
            return await documents.get_chunks(space, ids)
        async def get_episode(self, space, episode_id):
            return episode.model_copy(update={"space": "foreign"})
    expanded = await expand_structural_context(BadEpisodeAdapter(), "alpha", recall, scope=TextFilter())
    assert not expanded.sections and expanded.provenance_missing == 1


@pytest.mark.parametrize("identity", ["default", "dedup_key", "custom"])
async def test_real_engine_episode_identity_is_not_a_raw_byte_hash(documents, identity):
    from scone_memory import HashEmbedder, InMemoryVectorIndex, MemoryEngine
    from scone_memory.memory.engine import Record
    from scone_memory.retrieval.structural import expand_structural_context
    engine = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder(), clock=lambda: STAMP).open()
    try:
        content = "# Unicode handbook\n\nThe café launch budget is 42 tokens.\n"
        if identity == "custom":
            added, = await engine.remember_many("alpha", [Record(content, content_hash="opaque-import-identity")])
        else:
            added = await engine.remember("alpha", content, dedup_key="stable-key" if identity == "dedup_key" else None)
        episode = await documents.get_episode("alpha", added.episode_id)
        assert episode.content_hash != hashlib.sha256(episode.content.encode()).hexdigest()
        recalled = await engine.recall("alpha", "café launch budget", limit=1)
        expanded = await expand_structural_context(documents, "alpha", recalled, scope=TextFilter())
        section, = expanded.sections
        assert section.content_hash == episode.content_hash
        assert section.source_sha256 == hashlib.sha256(episode.content.encode()).hexdigest()
        assert expanded.invalid_provenance == 0
    finally:
        await engine.close()


async def test_final_revalidation_drops_source_deleted_during_later_read(documents):
    from scone_memory.retrieval.structural import expand_structural_context
    first, first_recall = await seed(documents, text="# First\n42")
    second, second_recall = await seed(documents, text="# Second\n42")
    class DeleteEarlier:
        async def get_chunks(self, space, ids):
            return await documents.get_chunks(space, ids)
        async def get_episode(self, space, episode_id):
            if episode_id == second.episode_id:
                await documents.delete_episode(space, first.episode_id)
            return await documents.get_episode(space, episode_id)
    expanded = await expand_structural_context(DeleteEarlier(), "alpha", RecallResult(items=first_recall.items + second_recall.items), scope=TextFilter())
    assert [section.episode_id for section in expanded.sections] == [second.episode_id]
    assert expanded.provenance_missing >= 1
