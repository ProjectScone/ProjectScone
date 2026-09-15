"""The image lane: stored images found by a text query in an image embedder's space.

Captions find an image by the words said about it. The image lane keeps one
vector per stored image, made from the image's bytes by an image embedder
that puts text queries in the same space, in an index of its own, and fuses
what it finds with the other lanes by rank. It is off unless a recall asks
for it; an engine without it says so. These tests use the deterministic
``HashImageEmbedder``: they prove the plumbing (the separate index, the
fusion, the provenance, forgetting), not how well any real model sees.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.core.ports import ImageEmbedder
from scone_memory.embedders import HashImageEmbedder
from scone_memory.ingestion.images import ImageAttribute, ImageContext, image_provenance, ingest_image, recall_images
from scone_memory.retrieval.image_lane import IMAGE_WEIGHT
from scone_memory.retrieval.recall import LANE_DEPTH, UNFILTERED_DEPTH


def picture(color: str) -> bytes:
    out = BytesIO()
    Image.new("RGB", (16, 16), color).save(out, format="PNG")
    return out.getvalue()


def context(caption: str, source: str) -> ImageContext:
    return ImageContext(source=source, attributes=(ImageAttribute(kind="alt", value=caption, origin="html"),))


#: Five images whose captions share no word with the questions asked of them.
#: Only the image embedder's text side knows what each one shows.
SHOWS = {"red": "red bicycle", "green": "green parrot", "blue": "blue kettle",
         "yellow": "yellow raincoat", "purple": "purple tulips"}
QUESTIONS = {"red": "Where is the photo of the red bicycle?", "green": "Find the green parrot picture.",
             "blue": "Show me the blue kettle.", "yellow": "Which one has the yellow raincoat?",
             "purple": "Photo with purple tulips please."}
#: Passages that say the questions' words, so the text lanes have an answer of their own.
SAYING = ["The red bicycle was sold last spring.", "A green parrot visited the garden twice.",
          "Our blue kettle whistles too loudly.", "Nobody found the yellow raincoat after the storm.",
          "Purple tulips came up early this year."]
FILLER = [f"Meeting note {n}: budget review moved to Thursday." for n in range(12)]
#: Captions unlike each other as well as unlike the questions. Captions made
#: from one template ("Scan 0000, shelf A", "Scan 0004, shelf E") read to
#: recall's restatement rule as one claim restated, and it puts the newest
#: first whatever the lanes said; that rule is not this lane's to change.
CAPTIONS = ["Holiday album, first page", "Inventory card, shed shelf", "Survey sheet, north border",
            "Logbook cover, winter volume", "Catalogue insert, autumn edition"]


def embedder(dim: int = 64) -> HashImageEmbedder:
    return HashImageEmbedder(dim=dim, phrases={phrase: picture(color) for color, phrase in SHOWS.items()})


@pytest.fixture(params=["memory", "sqlite"])
async def stores(request, tmp_path):
    if request.param == "sqlite":
        from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
        from scone_memory.backends.blobs import FileBlobStore
        documents = SqliteDocumentStore(tmp_path / "memory.db")
        vectors = SqliteVectorIndex(tmp_path / "memory.db")
        # The image lane's namespace in SQLite is a file of its own.
        images = SqliteVectorIndex(tmp_path / "image-vectors.db")
        blobs = FileBlobStore(tmp_path / "blobs")
    else:
        documents, vectors, images, blobs = InMemoryDocumentStore(), InMemoryVectorIndex(), InMemoryVectorIndex(), None
    return documents, vectors, images, blobs


async def lane_engine(stores, image_embedder: ImageEmbedder | None = None) -> MemoryEngine:
    documents, vectors, images, blobs = stores
    return await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs,
                              image_embedder=image_embedder or embedder(), image_vectors=images).open()


async def fill(engine: MemoryEngine) -> dict[str, str]:
    for text in FILLER:
        await engine.remember("s", text)
    stored = {}
    for n, color in enumerate(SHOWS):
        saved = await ingest_image(engine, "s", picture(color), media_type="image/png",
                                   context=context(CAPTIONS[n], f"album/{n}"))

        stored[color] = saved.image.attachment_id
    for text in SAYING:
        await engine.remember("s", text)
    return stored


def test_the_hash_image_embedder_is_deterministic_and_puts_a_phrase_on_its_image():
    one, two = embedder(), embedder()
    assert isinstance(one, ImageEmbedder)
    assert one.dim == 64 and one.id == two.id and one.id.startswith("hash-image-64")
    import asyncio

    async def check() -> None:
        [red, red_again, blue] = await one.embed_images([picture("red"), picture("red"), picture("blue")])
        assert red == red_again == (await two.embed_images([picture("red")]))[0]
        assert red != blue and len(red) == 64
        assert abs(sum(x * x for x in red) - 1.0) < 1e-9
        [asked, other, stop] = await one.embed_texts(["Where is the RED  bicycle?", "a red car", "the of and"])
        assert asked == red, "a text naming a phrase lands on that phrase's image"
        assert other != red and max(abs(a - b) for a, b in zip(other, red)) > 0.1
        assert any(stop), "a text naming nothing still gets a vector that is not zero"
        # A phrase matches whole words only.
        [partial] = await one.embed_texts(["unred bicycles"])
        assert partial != red
        # A text naming two phrases lands between their images.
        [both] = await one.embed_texts(["red bicycle and blue kettle"])
        assert both != red and both != blue
        cos = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731 - unit vectors
        assert cos(both, red) > 0.5 and cos(both, (await one.embed_images([picture("blue")]))[0]) > 0.5
        salted = HashImageEmbedder(dim=64, salt="other")
        assert salted.id == "hash-image-64-other" and (await salted.embed_images([picture("red")]))[0] != red

    asyncio.run(check())
    with pytest.raises(ValueError, match="dim"):
        HashImageEmbedder(dim=1)
    with pytest.raises(ValueError, match="needs a word"):
        HashImageEmbedder(phrases={"?!": picture("red")})


async def test_an_engine_refuses_half_an_image_lane_and_a_shared_namespace(tmp_path):
    with pytest.raises(InvalidInput, match="image_vectors is missing"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), image_embedder=embedder())
    with pytest.raises(InvalidInput, match="image_embedder is missing"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), image_vectors=InMemoryVectorIndex())
    shared = InMemoryVectorIndex()
    with pytest.raises(InvalidInput, match="index of its own"):
        MemoryEngine(InMemoryDocumentStore(), shared, HashEmbedder(), image_embedder=embedder(), image_vectors=shared)
    from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
    path = tmp_path / "one.db"
    with pytest.raises(InvalidInput, match="index of its own"):
        MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder(),
                     image_embedder=embedder(), image_vectors=SqliteVectorIndex(path))
    with pytest.raises(InvalidInput, match="image_embedder must have"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                     image_embedder=object(), image_vectors=InMemoryVectorIndex())  # type: ignore[arg-type]
    # Two in-memory SQLite handles are two databases, not one.
    MemoryEngine(SqliteDocumentStore(":memory:"), SqliteVectorIndex(":memory:"), HashEmbedder(),
                 image_embedder=embedder(), image_vectors=SqliteVectorIndex(":memory:"))


async def test_each_stored_image_gets_one_vector_in_the_image_index_only(stores):
    engine = await lane_engine(stores)
    documents, vectors, images, _ = stores
    try:
        assert images.dim == 64 and vectors.dim == 256, "each index is sized by its own embedder"
        saved = await ingest_image(engine, "s", picture("red"), media_type="image/png",
                                   context=context("Scan 0001", "album/1"))
        assert saved.image_lane == "indexed"
        [chunk] = await documents.chunks_of("s", saved.added.episode_id)
        assert await images.ids("s") == [chunk.chunk_id]
        assert await vectors.ids("s") == [chunk.chunk_id], "the caption's text vector stays where it was"
        [stored] = (await images.vectors_of("s", [chunk.chunk_id])).values()
        [expected] = await embedder().embed_images([picture("red")])
        assert max(abs(a - b) for a, b in zip(stored, expected)) < 1e-6, "the vector is the image's, not the caption's"
        again = await ingest_image(engine, "s", picture("red"), media_type="image/png",
                                   context=context("Scan 0001", "album/1"))
        assert again.added.episode_id == saved.added.episode_id and len(await images.ids("s")) == 1
        # The same bytes in another occurrence are another stored image, with a vector of their own.
        await ingest_image(engine, "s", picture("red"), media_type="image/png", context=context("Scan 0002", "album/2"))
        assert len(await images.ids("s")) == 2
    finally:
        await engine.close()


async def test_the_lane_finds_the_right_image_for_text_queries(stores):
    """The measurement quoted in docs/image-embedding-lane.md. It is exact on
    purpose: the embedders are deterministic, and a change that moves these
    numbers must move the documented ones with it."""
    engine = await lane_engine(stores)
    try:
        stored = await fill(engine)
        lane_first, off_top3, fused_ranks, deep_ranks = 0, 0, [], []
        for color, question in QUESTIONS.items():
            off = await engine.recall("s", question, limit=3)
            assert all("image" not in item.lanes for item in off.items)
            off_top3 += any(item.metadata.get("image_original") == stored[color] for item in off.items)
            # Deep enough to hold every image, so the lane's own first place is seen.
            deep = await engine.recall("s", question, limit=10, image_lane=True)
            [first] = [item for item in deep.items if item.lanes.get("image") == 1]
            lane_first += first.metadata["image_original"] == stored[color]
            deep_ranks.append(next(n for n, item in enumerate(deep.items, 1)
                                   if item.metadata.get("image_original") == stored[color]))
            on = await engine.recall("s", question, limit=3, image_lane=True)
            assert not [note for note in on.degraded if note.startswith("image lane")]
            ranks = [n for n, item in enumerate(on.items, 1) if item.metadata.get("image_original") == stored[color]]
            fused_ranks.append(ranks[0] if ranks else None)
            if ranks:
                # The item carries the attachment and the caption's provenance, and both resolve.
                item = on.items[ranks[0] - 1]
                provenance = await image_provenance(engine, "s", item.episode_id)
                assert item.metadata["image_manifest"] == provenance.manifest.attachment_id
                assert item.metadata["evidence_origin"] == "image_context" and item.source == provenance.context.source
        assert off_top3 == 0, "without the lane the captions cannot find these images"
        assert lane_first == 5, "the lane itself ranks the right image first for every question"
        # Fused, the passage saying the question's words comes first, and the
        # right image second. For "purple tulips" the hashed text vector lane
        # happens to rank other captions near the top, and those, with their
        # image-lane places, outrank the lane's first choice at IMAGE_WEIGHT
        # 1.0: the image falls out of the top three.
        assert fused_ranks == [2, 2, 2, 2, None]
        # Deeper lanes change the fused order: in a recall of ten it comes third.
        assert deep_ranks == [2, 2, 2, 2, 3]

    finally:
        await engine.close()


async def test_the_lane_is_off_unless_a_recall_asks(stores):
    class Counting(HashImageEmbedder):
        texts = 0

        async def embed_texts(self, texts):
            Counting.texts += len(texts)
            return await super().embed_texts(texts)

    engine = await lane_engine(stores, Counting(dim=64, phrases={"red bicycle": picture("red")}))
    try:
        await fill(engine)
        result = await engine.recall("s", QUESTIONS["red"], limit=10)
        assert Counting.texts == 0 and all("image" not in item.lanes for item in result.items)
        await engine.recall("s", QUESTIONS["red"], limit=10, image_lane=True)
        assert Counting.texts == 1
    finally:
        await engine.close()


async def test_an_engine_without_the_lane_says_so():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    saved = await ingest_image(engine, "s", picture("red"), media_type="image/png", context=context("Scan 0001", "album/1"))
    assert saved.image_lane == "not_configured"
    result = await engine.recall("s", "Scan", image_lane=True)
    assert result.items, "the other lanes still answer"
    assert "image lane: this engine has no image embedder and image index" in result.degraded
    quiet = await engine.recall("s", "Scan")
    assert not [note for note in quiet.degraded if note.startswith("image lane")]


async def test_a_failing_image_embedder_degrades_the_lane_not_the_recall(stores):
    class Broken(HashImageEmbedder):
        async def embed_texts(self, texts):
            raise RuntimeError("model went away")

    engine = await lane_engine(stores, Broken(dim=64))
    try:
        await fill(engine)
        result = await engine.recall("s", QUESTIONS["red"], image_lane=True)
        assert result.items and "image lane: RuntimeError: model went away" in result.degraded
    finally:
        await engine.close()


async def test_forgetting_an_image_removes_its_vector_and_deleting_the_space_removes_all(stores):
    engine = await lane_engine(stores)
    _, _, images, _ = stores
    try:
        first = await ingest_image(engine, "s", picture("red"), media_type="image/png", context=context("Scan 0001", "album/1"))
        second = await ingest_image(engine, "s", picture("blue"), media_type="image/png", context=context("Scan 0002", "album/2"))
        [kept] = await engine.documents.chunks_of("s", second.added.episode_id)
        assert len(await images.ids("s")) == 2
        await engine.forget("s", first.added.episode_id)
        assert await images.ids("s") == [kept.chunk_id]
        found = await engine.recall("s", "red bicycle", limit=5, image_lane=True)
        assert all(item.episode_id != first.added.episode_id for item in found.items)
        await engine.delete_space("s")
        assert await images.ids("s") == []
    finally:
        await engine.close()


async def test_the_lane_keeps_to_the_recall_filters(stores):
    engine = await lane_engine(stores)
    try:
        await fill(engine)
        with pytest.raises(InvalidInput, match="image_lane must be a boolean"):
            await engine.recall("s", QUESTIONS["red"], image_lane="yes")  # type: ignore[arg-type]
        from scone_memory.ingestion.images import ImageEntity
        await ingest_image(engine, "s", picture("orange"), media_type="image/png", context=ImageContext(
            source="album/tagged", attributes=(ImageAttribute(kind="alt", value="Card, drawer seven", origin="html"),),
            entities=(ImageEntity(entity_id="card:7", name="Card", relationship="depicts", attribute_indexes=(0,)),)))
        # Under an entity's tag only the tagged image is left for the lane, far as it is from the question.
        tagged = await recall_images(engine, "s", QUESTIONS["red"], entity_id="card:7", image_lane=True)
        assert [match.context.source for match in tagged.matches] == ["album/tagged"]
        assert tagged.recall.items[0].lanes.get("image") == 1
        scoped = await recall_images(engine, "s", QUESTIONS["red"], image_lane=True)


        assert scoped.matches and scoped.matches[0].context.source == "album/0"
        assert "image" in scoped.recall.items[0].lanes
        # An entity tag no image carries: the image lane finds nothing either.
        assert not (await recall_images(engine, "s", QUESTIONS["red"], entity_id="missing", image_lane=True)).matches
        later = await engine.recall("s", QUESTIONS["red"], limit=10, image_lane=True, as_of="2000-01-01T00:00:00Z")
        assert all("image" not in item.lanes for item in later.items)
    finally:
        await engine.close()


async def test_vectors_another_image_embedder_wrote_are_not_compared(stores):
    documents, vectors, images, blobs = stores
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs,
                                image_embedder=embedder(), image_vectors=images).open()
    await fill(engine)
    other = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs,
                               image_embedder=HashImageEmbedder(dim=64, salt="other"), image_vectors=images).open()
    try:
        result = await other.recall("s", QUESTIONS["red"], image_lane=True)
        assert result.items and all("image" not in item.lanes for item in result.items)
        assert any(note.startswith("image lane: VectorsNotComparable") for note in result.degraded)
    finally:
        await engine.close()


async def test_the_image_lane_weight_is_reported_on_the_recall_event():
    from scone_memory import InMemoryEventLog

    events = InMemoryEventLog()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=events,
                                image_embedder=embedder(), image_vectors=InMemoryVectorIndex()).open()
    await fill(engine)
    await engine.recall("s", QUESTIONS["red"], image_lane=True)
    await engine.recall("s", QUESTIONS["red"])
    on, off = [event.payload for event in await events.query("s", kind="recall")][::-1]
    assert on["fusion_weights"]["image"] == IMAGE_WEIGHT and on["image_lane"] is True
    assert on["lane_candidates"]["image"] >= 1 and "image" in on["latency_ms"]

    assert "image" not in off["fusion_weights"] and off["image_lane"] is False


class _Spy(InMemoryVectorIndex):
    """An image index that records the conditions each search was given."""

    def __init__(self, narrows: bool) -> None:
        super().__init__()
        self.narrows_conditions = narrows
        self.given: list[object] = []
        self.closed = False

    async def search(self, space, vector, limit, as_of=None, tags=(), where=None, conditions=None):
        self.given.append(conditions)
        return await super().search(space, vector, limit, as_of, tags, where, conditions)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("narrows", [True, False])
async def test_a_condition_reaches_an_image_index_only_when_it_narrows_by_conditions(narrows):
    images = _Spy(narrows)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                image_embedder=embedder(), image_vectors=images).open()
    stored = await fill(engine)
    found = await engine.recall("s", QUESTIONS["red"], limit=10, image_lane=True,
                                conditions={"field": "document_format", "is": "image"})
    assert images.given and (images.given[-1] is not None) is narrows
    # Either way the condition holds: in the index, or in recall's post-filter.
    assert all(item.metadata.get("document_format") == "image" for item in found.items)
    assert any(item.lanes.get("image") == 1 and item.metadata["image_original"] == stored["red"]
               for item in found.items)
    await engine.close()
    assert images.closed, "closing the engine closes the image index it holds"


async def test_a_forget_that_lands_while_the_image_is_embedded_leaves_no_vector(stores):
    """Forgetting removes image vectors by chunk id; a vector written after
    that, for a chunk already gone, would outlive the forgotten image."""
    import asyncio

    from scone_memory.core.errors import Gone

    entered, release = asyncio.Event(), asyncio.Event()

    class Held(HashImageEmbedder):
        async def embed_images(self, images):
            entered.set()
            await release.wait()
            return await super().embed_images(images)

    engine = await lane_engine(stores, Held(dim=64))
    documents, _, images, _ = stores
    try:
        note = await engine.remember("s", FILLER[0])
        saving = asyncio.create_task(ingest_image(engine, "s", picture("red"), media_type="image/png",
                                                  context=context(CAPTIONS[0], "album/0")))
        await entered.wait()
        [held] = [e for e in await documents.page_episodes("s", None, 10, None) if e.episode_id != note.episode_id]
        await engine.forget("s", held.episode_id)
        release.set()
        with pytest.raises(Gone, match="forgotten while its image was indexed"):
            await saving
        assert await images.ids("s") == [], "no vector is left for the forgotten image"
    finally:
        await engine.close()


async def test_a_forgotten_episode_is_not_indexed():
    from scone_memory.core.errors import Gone
    from scone_memory.retrieval.image_lane import index_image

    images = InMemoryVectorIndex()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                image_embedder=embedder(), image_vectors=images).open()
    saved = await ingest_image(engine, "s", picture("red"), media_type="image/png", context=context(CAPTIONS[0], "album/0"))
    episode = await engine.episode("s", saved.added.episode_id)
    await engine.forget("s", saved.added.episode_id)
    with pytest.raises(Gone, match="forgotten while its image was indexed") as gone:
        await index_image(engine.documents, embedder(), images, "s", episode, picture("red"))
    stone = await engine.tombstone("s", episode.episode_id)
    assert stone is not None and gone.value.forgotten_at == stone.forgotten_at
    assert await images.ids("s") == []

    class Forgetting:
        """A store caught mid-forget: the chunks are gone, the tombstone not yet written."""

        async def chunks_of(self, space, episode_id):
            return []

        async def tombstone(self, space, episode_id):
            return None

    from scone_memory.core.errors import NotFound
    with pytest.raises(NotFound, match="being forgotten") as missing:
        await index_image(Forgetting(), embedder(), images, "s", episode, picture("red"))  # type: ignore[arg-type]
    assert not isinstance(missing.value, Gone)


def _conditioned_engine(narrows: bool, reds: int):
    async def build() -> tuple[MemoryEngine, str]:
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                    image_embedder=embedder(), image_vectors=_Spy(narrows)).open()
        for n in range(reds):
            await ingest_image(engine, "s", picture("red"), media_type="image/png",
                               context=context(f"{CAPTIONS[n % len(CAPTIONS)]} {n}", f"album/{n}"))
        target = await ingest_image(engine, "s", picture("blue"), media_type="image/png",
                                    context=context("Ticket stub, drawer", "album/blue"))
        return engine, target.image.attachment_id
    return build()


@pytest.mark.parametrize("narrows", [True, False])
async def test_an_image_index_that_cannot_narrow_is_searched_deeper(narrows):
    """The text index narrowing by conditions does not make the image index
    do so: an image lane post-filtered looks as deep as a post-filtered vector lane."""
    engine, blue = await _conditioned_engine(narrows, reds=6)
    try:
        found = await engine.recall("s", "red bicycle", limit=1, image_lane=True,
                                    conditions={"field": "image_original", "is": blue})
        assert [item.metadata["image_original"] for item in found.items] == [blue]
        assert "image" in found.items[0].lanes, "the lane still finds the image the condition names"
        report = found.narrowing
        assert report is not None and report.image_lane == ("in_store" if narrows else "postfiltered")
        assert report.image_window == (LANE_DEPTH if narrows else LANE_DEPTH * UNFILTERED_DEPTH)
        assert report.image_returned == (1 if narrows else 7) and report.window_exhausted is False
        # A source bound lives on the episode, which no image vector carries: post-filtered either way.
        bound = await engine.recall("s", "red bicycle", limit=1, image_lane=True, source_prefix="album/blue")
        assert bound.narrowing is not None and bound.narrowing.image_lane == "postfiltered"
        assert bound.narrowing.image_window == LANE_DEPTH * UNFILTERED_DEPTH
    finally:
        await engine.close()


async def test_a_full_post_filtered_image_window_says_the_bound_bit():
    # candidate_limit sets every lane's window, so five red images fill the image lane's.
    engine, blue = await _conditioned_engine(False, reds=5)
    try:
        found = await engine.recall("s", "red bicycle", limit=1, candidate_limit=5, image_lane=True,
                                    conditions={"field": "image_original", "is": blue})
        report = found.narrowing
        assert report is not None and report.image_lane == "postfiltered"
        assert report.image_returned == report.image_window == 5
        assert report.vector_lane == "in_store", "only the image lane's window can have bitten"
        assert report.text_lane == "in_store"
        assert report.window_exhausted is True
        quiet = await engine.recall("s", "red bicycle", limit=1, candidate_limit=5,
                                    conditions={"field": "image_original", "is": blue})
        assert quiet.narrowing is not None and quiet.narrowing.image_lane == "off"
        assert quiet.narrowing.image_window == 0 and quiet.narrowing.window_exhausted is False
    finally:
        await engine.close()


async def test_a_full_image_window_makes_a_phrase_short_answer_say_so():
    engine, _ = await _conditioned_engine(False, reds=5)
    try:
        # Only the image lane and the text lane run; the text lane finds no caption saying "red bicycle".
        found = await engine.recall("s", "red bicycle", limit=2, candidate_limit=5, lanes=["text"], image_lane=True,
                                    require=["Ticket stub"])
        assert found.phrases is not None and found.phrases.dropped_required == 5
        assert found.phrases.short is True, "the image lane's full window left passages unchecked"
    finally:
        await engine.close()


async def test_a_full_image_window_narrowed_in_the_index_is_not_a_bound_that_bit():
    """Every row the image index holds was eligible, so its full window hides
    nothing, even when another lane's post-filter removed candidates."""
    engine = await MemoryEngine(InMemoryDocumentStore(), _Spy(False), HashEmbedder(),
                                image_embedder=embedder(), image_vectors=_Spy(True)).open()
    try:
        await engine.remember("s", FILLER[0])
        for n in range(LANE_DEPTH):
            await ingest_image(engine, "s", picture("red"), media_type="image/png", context=context(CAPTIONS[n], f"album/{n}"))
        found = await engine.recall("s", "red bicycle budget", limit=1, image_lane=True,
                                    conditions={"field": "document_format", "is": "image"})
        report = found.narrowing
        assert report is not None and report.image_lane == "in_store" and report.vector_lane == "postfiltered"
        assert report.image_returned == report.image_window == LANE_DEPTH
        assert report.vector_returned < report.vector_window and report.postfiltered_out >= 1
        assert report.window_exhausted is False
    finally:
        await engine.close()
