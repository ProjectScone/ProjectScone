"""Owned file revisions must not conflate separate paths or lose provenance."""
from dataclasses import replace

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.document_source import DocumentSource, source_revision_key
from scone_memory.ingestion.files import DocumentManifest, digest, document_provenance, encode_manifest, store_document
from scone_memory.ingestion.formats.types import DocumentSegment, ParsedDocument


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return await super().embed(texts)


@pytest.fixture(params=['memory', 'sqlite'])
async def engine(request, tmp_path):
    if request.param == 'sqlite':
        from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
        documents, vectors = SqliteDocumentStore(tmp_path / 'source.db'), SqliteVectorIndex(tmp_path / 'source.db')
    else:
        documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    memory = await MemoryEngine(documents, vectors, CountingEmbedder()).open()
    yield memory
    await memory.close()


async def prepared(engine):
    raw = b'Observatory calibration report'
    original = await engine.attach('alpha', raw, 'text/plain', 'report.txt')
    manifest = DocumentManifest(original_sha256=digest(raw), filename='report.txt',
        parsed=ParsedDocument(format='text', parser='fixture', segments=(DocumentSegment(text=raw.decode(), locator='line:1'),)))
    return original, manifest


async def test_distinct_source_paths_keep_independent_revisions_and_shared_original(engine):
    original, manifest = await prepared(engine)
    owner = DocumentSource('a' * 32, 'one/report.txt', 'parser-v1')
    first = await store_document(engine, 'alpha', original, manifest, source=owner)
    second = await store_document(engine, 'alpha', original, manifest, source=replace(owner, path='two/report.txt'))
    assert first.added.episode_id != second.added.episode_id
    assert first.original == second.original
    assert (await document_provenance(engine, 'alpha', first.added.episode_id)).segments[0].text == 'Observatory calibration report'
    episode = await engine.episode('alpha', first.added.episode_id)
    assert episode.metadata['source_collection'] == owner.collection_id
    assert episode.metadata['source_path_hash'] == digest(owner.path.encode())
    assert episode.metadata['source_parser_revision'] == owner.parser_revision
    await engine.forget('alpha', first.added.episode_id)
    assert (await engine.attachment('alpha', original.attachment_id))[1] == b'Observatory calibration report'
    assert (await document_provenance(engine, 'alpha', second.added.episode_id)).original == original


async def test_owned_retry_reuses_embedding_and_can_be_looked_up(engine):
    original, manifest = await prepared(engine)
    owner = DocumentSource('a' * 32, 'report.txt', 'parser-v1')
    first = await store_document(engine, 'alpha', original, manifest, source=owner)
    calls = engine.embedder.calls
    again = await store_document(engine, 'alpha', original, manifest, source=owner)
    assert again.added.deduplicated and again.added.episode_id == first.added.episode_id
    assert engine.embedder.calls == calls
    key = source_revision_key(owner, original.attachment_id, digest(encode_manifest(manifest)))
    assert (await engine.episode_by_key('alpha', key)).episode_id == first.added.episode_id


async def test_parser_revision_and_collection_are_part_of_owned_identity(engine):
    original, manifest = await prepared(engine)
    owner = DocumentSource('a' * 32, 'report.txt', 'parser-v1')
    variants = (owner, replace(owner, parser_revision='parser-v2'), replace(owner, collection_id='b' * 32))
    results = [await store_document(engine, 'alpha', original, manifest, source=value) for value in variants]
    assert len({result.added.episode_id for result in results}) == 3


async def test_default_document_identity_and_metadata_are_unchanged(engine):
    original, manifest = await prepared(engine)
    result = await store_document(engine, 'alpha', original, manifest)
    key = f'document-v1:{original.attachment_id}:{result.manifest.attachment_id}'
    episode = await engine.episode_by_key('alpha', key)
    assert episode.source == f'attachment:{original.attachment_id}'
    assert set(episode.metadata) == {'document_format', 'document_original', 'document_manifest', 'evidence_origin'}
    owned = await store_document(engine, 'alpha', original, manifest, source=DocumentSource('a' * 32, 'report.txt', 'v1'))
    assert owned.added.episode_id != result.added.episode_id


@pytest.mark.parametrize('changes', [
    {'collection_id': ''}, {'collection_id': 'not-a-collection-id'}, {'path': ''}, {'path': '/absolute.txt'},
    {'path': '../escape.txt'}, {'path': 'one/../report.txt'}, {'path': './report.txt'}, {'path': 'one//report.txt'},
    {'path': 'one\\report.txt'}, {'path': '\ud800'}, {'path': 'bad\x00name'}, {'path': 'x' * 1025},
    {'parser_revision': ''}, {'parser_revision': 'x' * 129}, {'parser_revision': '\ud800'},
])
def test_invalid_source_identity_is_refused(changes):
    values = {'collection_id': 'a' * 32, 'path': 'report.txt', 'parser_revision': 'v1'} | changes
    with pytest.raises(InvalidInput):
        DocumentSource(**values)


def test_revision_key_is_bounded_and_sensitive_to_exact_path_and_both_content_digests():
    owner = DocumentSource('a' * 32, 'notes/Report 🥐.txt', 'v1')
    key = source_revision_key(owner, 'b' * 64, 'c' * 64)
    assert len(key) <= 256
    assert source_revision_key(replace(owner, path='notes/report 🥐.txt'), 'b' * 64, 'c' * 64) != key
    assert source_revision_key(owner, 'd' * 64, 'c' * 64) != key
    assert source_revision_key(owner, 'b' * 64, 'd' * 64) != key
    for original, manifest in [('bad', 'c' * 64), ('b' * 64, 'not-a-digest')]:
        with pytest.raises(InvalidInput):
            source_revision_key(owner, original, manifest)


async def test_a_managed_source_can_return_to_earlier_bytes_after_retirement(engine):
    original, manifest = await prepared(engine)
    owner = DocumentSource('a' * 32, 'report.txt', 'v1')
    first = await store_document(engine, 'alpha', original, manifest, source=owner)
    await engine.forget('alpha', first.added.episode_id)
    original, manifest = await prepared(engine)
    next_generation = replace(owner, generation=2)
    returned = await store_document(engine, 'alpha', original, manifest, source=next_generation)
    assert returned.added.episode_id != first.added.episode_id
    assert (await document_provenance(engine, 'alpha', returned.added.episode_id)).filename == 'report.txt'


@pytest.mark.parametrize('generation', [-1, True, 1.5, 2**63])
def test_source_generation_is_a_bounded_nonnegative_integer(generation):
    with pytest.raises(InvalidInput):
        DocumentSource('a' * 32, 'report.txt', 'v1', generation=generation)
