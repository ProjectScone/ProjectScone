"""The same image/entity behavior across each configured framework backend pair."""
from scone_memory.ingestion import ingest_image, recall_images
from test_image_context import context, picture


async def test_image_entity_context_uses_the_shared_backend_contract(engine):
    saved = await ingest_image(engine, 'alpha', picture(), media_type='image/png', context=context())
    await ingest_image(engine, 'beta', picture('blue'), media_type='image/png', context=context())
    result = await recall_images(engine, 'alpha', 'Pikachu', entity_id='pokemon:25')
    assert len(result.matches) == 1
    assert result.matches[0].episode_id == saved.added.episode_id
    assert result.matches[0].image.attachment_id == saved.image.attachment_id
    assert not result.unresolved_episode_ids
    assert not (await recall_images(engine, 'alpha', 'Pikachu', entity_id='unrelated')).matches
    await engine.forget('alpha', saved.added.episode_id)
    assert not (await recall_images(engine, 'alpha', 'Pikachu', entity_id='pokemon:25')).matches
