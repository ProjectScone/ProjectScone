from io import BytesIO

import pytest
from PIL import Image
from pydantic import ValidationError

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput, NotFound


def picture(color='yellow'):
    out = BytesIO()
    Image.new('RGB', (16, 16), color).save(out, format='PNG')
    return out.getvalue()


@pytest.fixture(params=['memory', 'sqlite'])
async def memory(request, tmp_path):
    if request.param == 'sqlite':
        from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
        from scone_memory.backends.blobs import FileBlobStore
        path = tmp_path / 'images.db'
        documents, vectors = SqliteDocumentStore(path), SqliteVectorIndex(path)
        blobs = FileBlobStore(tmp_path / 'blobs')
    else:
        documents, vectors, blobs = InMemoryDocumentStore(), InMemoryVectorIndex(), None
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs).open()
    yield engine
    await engine.close()


def context(source='catalog/pikachu', caption='Pikachu, the electric mouse Pokémon.'):
    from scone_memory.ingestion.images import ImageAttribute, ImageContext, ImageEntity
    return ImageContext(source=source,
        attributes=(ImageAttribute(kind='alt', value=caption, origin='html'),),
        entities=(ImageEntity(entity_id='pokemon:25', name='Pikachu', aliases=('ピカチュウ',),
            relationship='depicts', attribute_indexes=(0,)),))


async def test_external_description_and_entity_alias_recall_the_actual_image(memory):
    import scone_memory.ingestion as ingestion
    assert hasattr(ingestion, 'ingest_image'), 'image context ingestion is missing'
    result = await ingestion.ingest_image(memory, 'alpha', picture(), media_type='image/png', context=context())
    found = await ingestion.recall_images(memory, 'alpha', 'Who is Pikachu?', entity_id='pokemon:25')
    assert len(found.matches) == 1
    match = found.matches[0]
    assert match.episode_id == result.added.episode_id
    assert match.image.attachment_id == result.image.attachment_id
    assert match.context.attributes[0].origin == 'html'
    assert match.context.entities[0].relationship == 'depicts'
    assert (await memory.attachment('alpha', match.image.attachment_id))[1] == picture()
    alias = await ingestion.recall_images(memory, 'alpha', 'ピカチュウ', entity_id='pokemon:25')
    assert alias.matches[0].image == match.image
    assert (await memory.recall('alpha', 'Pikachu')).items


async def test_same_image_keeps_distinct_source_occurrences_and_deduplicates_retries(memory):
    from scone_memory.ingestion.images import ingest_image, image_provenance
    first = await ingest_image(memory, 'alpha', picture(), media_type='image/png', context=context())
    other = await ingest_image(memory, 'alpha', picture(), media_type='image/png',
        context=context('museum/pikachu', 'Pikachu on a museum poster.'))
    again = await ingest_image(memory, 'alpha', picture(), media_type='image/png', context=context())
    assert first.image.attachment_id == other.image.attachment_id
    assert first.added.episode_id != other.added.episode_id
    assert first.added.episode_id == again.added.episode_id and again.added.deduplicated
    assert (await image_provenance(memory, 'alpha', other.added.episode_id)).context.source == 'museum/pikachu'


async def test_entity_filter_and_scope_exclude_wrong_images(memory):
    from scone_memory.ingestion.images import ImageContext, ImageEntity, ingest_image, recall_images
    first = context()
    second = ImageContext(source='other', attributes=first.attributes,
        entities=(ImageEntity(entity_id='other:pikachu', name='Pikachu', relationship='mentions', attribute_indexes=(0,)),))
    await ingest_image(memory, 'alpha', picture(), media_type='image/png', context=first)
    await ingest_image(memory, 'alpha', picture('blue'), media_type='image/png', context=second)
    matches = await recall_images(memory, 'alpha', 'Pikachu', entity_id='pokemon:25')
    assert len(matches.matches) == 1 and matches.matches[0].context.source == first.source
    assert not (await recall_images(memory, 'beta', 'Pikachu')).matches
    assert not (await recall_images(memory, 'alpha', 'Pikachu', entity_id='missing')).matches


async def test_modified_or_unlinked_provenance_is_refused(memory):
    from scone_memory.ingestion.images import ingest_image, image_provenance
    result = await ingest_image(memory, 'alpha', picture(), media_type='image/png', context=context())
    episode = await memory.episode('alpha', result.added.episode_id)
    forged = await memory.remember('alpha', 'forged image description', metadata=episode.metadata,
        attachment_ids=[attachment.attachment_id for attachment in episode.attachments])
    with pytest.raises(InvalidInput):
        await image_provenance(memory, 'alpha', forged.episode_id)
    with pytest.raises(NotFound):
        await image_provenance(memory, 'beta', result.added.episode_id)
    await memory.forget('alpha', result.added.episode_id)
    from scone_memory.ingestion.images import recall_images
    found = await recall_images(memory, 'alpha', 'Pikachu')
    assert not found.matches
    assert forged.episode_id in found.unresolved_episode_ids


def test_entity_links_require_existing_attribute_evidence():
    from scone_memory.ingestion.images import ImageContext, ImageEntity
    with pytest.raises(ValidationError):
        ImageContext(source='catalog', attributes=context().attributes,
            entities=(ImageEntity(entity_id='pokemon:25', name='Pikachu', relationship='depicts', attribute_indexes=(7,)),))


async def test_invalid_bytes_or_mismatched_mime_do_not_create_episodes(memory):
    from scone_memory.ingestion.images import ingest_image
    for raw, mime in ((b'fake', 'image/png'), (picture(), 'image/jpeg')):
        with pytest.raises(InvalidInput):
            await ingest_image(memory, 'alpha', raw, media_type=mime, context=context())
    assert (await memory.documents.counts('alpha')).episodes == 0
