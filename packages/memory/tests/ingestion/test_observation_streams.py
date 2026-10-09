"""Observation streams keep the frames that show something new, dated when they were seen.

These tests use the deterministic ``HashImageEmbedder``: identical bytes are one vector and different bytes are
unrelated ones. They prove the gate, the dating, the bounds and the search plumbing, not how well a real model
sees.
"""
from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.embedders import HashImageEmbedder
from scone_memory.ingestion.images import ImageAttribute, ImageContext, ingest_image
from scone_memory.ingestion.observations import ObservationStream, sightings

PHRASES = {'red valve': 'red', 'blue pump': 'blue', 'green lamp': 'green'}


def picture(color: str) -> bytes:
    out = BytesIO()
    Image.new('RGB', (16, 16), color).save(out, format='PNG')
    return out.getvalue()


class CountingEmbedder(HashImageEmbedder):
    """The hash embedder, counting the images it was asked to embed."""

    def __init__(self) -> None:
        super().__init__(phrases={phrase: picture(color) for phrase, color in PHRASES.items()})
        self.images_embedded = 0

    async def embed_images(self, images):  # type: ignore[no-untyped-def]
        self.images_embedded += len(images)
        return await super().embed_images(images)


async def engine_with_lane() -> tuple[MemoryEngine, CountingEmbedder]:
    embedder = CountingEmbedder()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                image_embedder=embedder, image_vectors=InMemoryVectorIndex()).open()
    return engine, embedder


def at(second: int) -> str:
    return f'2026-10-08T12:{second // 60:02d}:{second % 60:02d}Z'


async def offer(stream: ObservationStream, color: str, second: int, place: str | None = 'line-1'):
    return await stream.observe(picture(color), media_type='image/png', observed_at=at(second), place=place)


async def test_the_gate_keeps_the_first_frame_and_new_sights_and_drops_repeats():
    engine, _ = await engine_with_lane()
    stream = ObservationStream(engine, 'plant', 'camera-1')
    first = await offer(stream, 'red', 0)
    repeat = await offer(stream, 'red', 1)
    new = await offer(stream, 'blue', 2)
    assert (first.kept, first.reason, first.novelty) == (True, 'first', None)
    assert (repeat.kept, repeat.reason, repeat.ingested) == (False, 'seen_recently', None)
    assert repeat.novelty == pytest.approx(0.0, abs=1e-9)
    assert (new.kept, new.reason) == (True, 'novel') and new.novelty is not None and new.novelty >= stream.novelty
    assert (stream.seen, stream.kept, stream.dropped) == (3, 2, 1)
    assert first.ingested is not None and first.ingested.image_lane == 'indexed'


async def test_a_familiar_sight_at_another_place_is_kept():
    engine, _ = await engine_with_lane()
    stream = ObservationStream(engine, 'plant', 'cart-camera')
    await offer(stream, 'red', 0, place='bay-a')
    same_place = await offer(stream, 'red', 1, place='bay-a')
    moved = await offer(stream, 'red', 2, place='bay-b')
    assert same_place.kept is False
    assert (moved.kept, moved.reason) == (True, 'moved')


async def test_a_still_scene_is_kept_again_when_the_heartbeat_is_due():
    engine, _ = await engine_with_lane()
    stream = ObservationStream(engine, 'plant', 'camera-1', heartbeat_seconds=30)
    await offer(stream, 'red', 0)
    early = await offer(stream, 'red', 29)
    due = await offer(stream, 'red', 30)
    after = await offer(stream, 'red', 31)
    assert early.kept is False
    assert (due.kept, due.reason) == (True, 'heartbeat')
    assert after.kept is False, 'the heartbeat is measured from the last kept frame'


async def test_the_gate_only_remembers_its_window_of_kept_frames():
    engine, _ = await engine_with_lane()
    stream = ObservationStream(engine, 'plant', 'camera-1', window=1)
    await offer(stream, 'red', 0)
    await offer(stream, 'blue', 1)
    back = await offer(stream, 'red', 2)
    assert (back.kept, back.reason) == (True, 'novel'), 'red left the one-frame window when blue was kept'
    wide = ObservationStream(engine, 'plant', 'camera-2', window=2)
    await offer(wide, 'red', 0)
    await offer(wide, 'blue', 1)
    assert (await offer(wide, 'red', 2)).kept is False


async def test_a_kept_frame_is_dated_when_it_was_seen_and_embedded_once():
    engine, embedder = await engine_with_lane()
    stream = ObservationStream(engine, 'plant', 'camera-1')
    kept = await offer(stream, 'red', 5)
    assert kept.ingested is not None
    episode = await engine.documents.get_episode('plant', kept.ingested.added.episode_id)
    assert episode is not None and episode.created_at == '2026-10-08T12:00:05.000Z'
    assert episode.source == 'observation:camera-1'
    assert embedder.images_embedded == 1, 'the gate embedded the frame; storing it did not embed it again'
    await offer(stream, 'red', 6)
    assert embedder.images_embedded == 2, 'a dropped frame is embedded once and stored nowhere'


async def test_sightings_are_newest_first_with_place_and_what_was_left_out():
    engine, _ = await engine_with_lane()
    stream = ObservationStream(engine, 'plant', 'cart-camera')
    await offer(stream, 'red', 0, place='bay-a')
    await offer(stream, 'blue', 10, place='bay-a')
    await offer(stream, 'red', 20, place='bay-b')
    found = await sightings(engine, 'plant', 'where is the red valve', min_similarity=0.5)
    assert [(s.place, s.observed_at) for s in found.items] == [
        ('bay-b', '2026-10-08T12:00:20.000Z'), ('bay-a', '2026-10-08T12:00:00.000Z')]
    assert found.latest is found.items[0] and found.earliest is found.items[1]
    assert all(s.stream == 'cart-camera' and s.similarity >= 0.5 for s in found.items)
    assert (found.searched, found.below_threshold, found.unresolved, found.window_full) == (3, 1, 0, False)
    assert found.latest.image.media_type == 'image/png'
    nothing = await sightings(engine, 'plant', 'green lamp', min_similarity=0.5)
    assert nothing.items == () and nothing.latest is None and nothing.below_threshold == 3


async def test_sightings_narrow_by_place_and_by_stream():
    engine, _ = await engine_with_lane()
    cart = ObservationStream(engine, 'plant', 'cart-camera')
    fixed = ObservationStream(engine, 'plant', 'dock-camera')
    await offer(cart, 'red', 0, place='bay-a')
    await offer(cart, 'red', 10, place='bay-b')
    await offer(fixed, 'red', 20, place='dock')
    by_place = await sightings(engine, 'plant', 'red valve', min_similarity=0.5, place='bay-a')
    by_stream = await sightings(engine, 'plant', 'red valve', min_similarity=0.5, stream='dock-camera')
    everywhere = await sightings(engine, 'plant', 'red valve', min_similarity=0.5)
    assert [s.place for s in by_place.items] == ['bay-a']
    assert [(s.stream, s.place) for s in by_stream.items] == [('dock-camera', 'dock')]
    assert [s.place for s in everywhere.items] == ['dock', 'bay-b', 'bay-a']


async def test_a_full_window_says_a_newer_sighting_may_exist():
    engine, _ = await engine_with_lane()
    stream = ObservationStream(engine, 'plant', 'cart-camera')
    await offer(stream, 'red', 0, place='bay-a')
    await offer(stream, 'red', 10, place='bay-b')
    narrow = await sightings(engine, 'plant', 'red valve', min_similarity=0.5, window=1)
    assert len(narrow.items) == 1 and narrow.window_full is True
    wide = await sightings(engine, 'plant', 'red valve', min_similarity=0.5, window=5)
    assert len(wide.items) == 2 and wide.window_full is False


async def test_images_stored_another_way_are_not_sightings():
    engine, _ = await engine_with_lane()
    context = ImageContext(source='catalogue.pdf', attributes=(ImageAttribute(kind='alt', value='a part photo'),))
    await ingest_image(engine, 'plant', picture('red'), media_type='image/png', context=context)
    found = await sightings(engine, 'plant', 'red valve', min_similarity=0.5)
    assert (found.items, found.searched) == ((), 0)


async def test_a_forgotten_frame_is_not_a_sighting():
    engine, _ = await engine_with_lane()
    stream = ObservationStream(engine, 'plant', 'cart-camera')
    old = await offer(stream, 'red', 0, place='bay-a')
    await offer(stream, 'red', 10, place='bay-b')
    assert old.ingested is not None
    await engine.forget('plant', old.ingested.added.episode_id)
    found = await sightings(engine, 'plant', 'red valve', min_similarity=0.5)
    assert [s.place for s in found.items] == ['bay-b']


async def test_frames_out_of_order_bad_times_and_bad_settings_are_refused():
    engine, _ = await engine_with_lane()
    stream = ObservationStream(engine, 'plant', 'camera-1')
    await offer(stream, 'red', 10)
    with pytest.raises(InvalidInput, match='time order'):
        await offer(stream, 'blue', 9)
    with pytest.raises(InvalidInput, match='RFC 3339'):
        await stream.observe(picture('blue'), media_type='image/png', observed_at='yesterday')
    assert (stream.seen, stream.kept) == (1, 1), 'a refused frame is not counted'
    for bad in ({'novelty': 0}, {'window': 0}, {'window': 65}, {'heartbeat_seconds': 0}):
        with pytest.raises(InvalidInput):
            ObservationStream(engine, 'plant', 'camera-1', **bad)  # type: ignore[arg-type]
    with pytest.raises(InvalidInput, match='min_similarity'):
        await sightings(engine, 'plant', 'red valve', min_similarity=2)
    with pytest.raises(InvalidInput, match='window'):
        await sightings(engine, 'plant', 'red valve', min_similarity=0.5, window=201)


async def test_streams_and_sightings_need_the_image_lane():
    plain = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    with pytest.raises(InvalidInput, match='image lane'):
        ObservationStream(plain, 'plant', 'camera-1')
    with pytest.raises(InvalidInput, match='image lane'):
        await sightings(plain, 'plant', 'red valve', min_similarity=0.5)


async def test_a_supplied_image_vector_must_be_the_embedders_and_is_refused_before_anything_is_stored():
    engine, embedder = await engine_with_lane()
    context = ImageContext(source='camera', attributes=(ImageAttribute(kind='alt', value='frame'),))
    for bad in ([0.5] * (embedder.dim - 1), [float('nan')] * embedder.dim, 'not a vector'):
        with pytest.raises(InvalidInput, match='image vector'):
            await ingest_image(engine, 'plant', picture('red'), media_type='image/png', context=context,
                               image_vector=bad)  # type: ignore[arg-type]
    assert await engine.documents.recent_episodes('plant', 10) == [], 'nothing was stored for a refused vector'
    plain = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    with pytest.raises(InvalidInput, match='image lane'):
        await ingest_image(plain, 'plant', picture('red'), media_type='image/png', context=context,
                           image_vector=[0.0] * embedder.dim)
