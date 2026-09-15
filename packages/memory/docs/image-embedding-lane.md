# Image embedding lane

[Image context](image-context.md) · [Recall semantics](retrieval-and-storage.md)

An image ingested with `ingest_image` is found through the words said about it:
its alt text, caption and other context become an episode that the text and
vector lanes search. The image lane adds a second way in. An image embedder
that puts images and text queries into one space (a CLIP-style model) embeds
each stored image's bytes; a recall that asks for the lane embeds the query
with the same model and fuses the nearest images with the other lanes by rank.

```python
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.embedders import ClipImageEmbedder  # optional, see below
from scone_memory.ingestion import ingest_image

engine = await MemoryEngine(
    InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
    image_embedder=ClipImageEmbedder(),      # images and queries into one space
    image_vectors=InMemoryVectorIndex(),     # an index of its own
).open()
saved = await ingest_image(engine, "catalog", png_bytes, media_type="image/png", context=context)
assert saved.image_lane == "indexed"

result = await engine.recall("catalog", "a red bicycle against a wall", image_lane=True)
for item in result.items:
    if "image" in item.lanes:
        print(item.lanes["image"], item.metadata["image_original"], item.metadata["image_manifest"])
```

## What the lane holds

- **One vector per stored image occurrence**, made from the image bytes, keyed by
  the first chunk of the episode that carries the image. The same bytes in two
  occurrences (two pages, two captions) are two episodes and two vectors. An
  exact retry of `ingest_image` writes the same key again, so it repairs a
  vector a failed write left out and never adds a second one.
- **An index of its own.** `image_vectors` must not be the text vectors' index:
  a cosine between an image and a query means something only when one model
  made both, and an index holds one width. The engine refuses the same object,
  and two SQLite handles on one database file. With SQLite, give the image lane
  its own file: `SqliteVectorIndex("~/.scone-memory/image-vectors.db")`. The
  index records which image embedder wrote it, as the text index does; a recall
  through a different image embedder is refused by the lane and named in
  `degraded` (`image lane: VectorsNotComparable: ...`), and the other lanes answer.
- **The episode's tags, metadata and time** on every vector, so `tags`, `where`
  and `as_of` narrow the lane in the index. `conditions` narrow it in the index
  when the index evaluates conditions itself (the in-memory and SQLite indexes
  do); otherwise, and for `kind`, `source_prefix`, `since` and `until`, recall
  post-filters what the lane found, over the vector lane's window, as it does for
  the vector lane. The narrowing report describes the text and vector lanes only.


## What a recall gets

`image_lane=True` runs the lane; it is off otherwise, and a recall that does not
ask never calls the image embedder. Items the lane placed have `lanes["image"]`,
their 1-based rank in the lane. The chunk is the image's caption, so the item's
`metadata` carries `image_original` (the image's attachment id, for
`engine.attachment`), `image_manifest` (the retained context the caption was made
from) and `evidence_origin: image_context`; `image_provenance` verifies both.
The lane's cosines are on the image model's scale, so they never become an item's
`similarity`, `top_similarity` or `low_confidence`. The recall event records
`image_lane`, `fusion_weights.image` and `lane_candidates.image`.

`recall_images(..., image_lane=True)` passes the lane through, so an image-only
search can use both the context and the pixels.

An engine without an image embedder and index still answers a recall that asks
for the lane, from the other lanes, with `image lane: this engine has no image
embedder and image index` in `degraded`; `ingest_image` on it returns
`image_lane="not_configured"`. A failing image embedder degrades the lane the same
way, with its error.

The lane's weight in fusion is `IMAGE_WEIGHT = 1.0`, the text lane's own. It is
not tuned: there is no image retrieval benchmark here and no model to run one.

## Forgetting

Forgetting an episode deletes its image vector with its text vectors, by the same
chunk ids, including a forget resumed by `recover()`, an expiry, `forget_matching`
and a keyed `replace`. Deleting a space sweeps the image index too.

## Embedders

`ImageEmbedder` (`scone_memory.core.ports`) is `id`, `dim`,
`embed_images(bytes...)` and `embed_texts(str...)`.

- `HashImageEmbedder(dim=64, phrases={"red bicycle": red_png})` is deterministic and
  needs no model. It does not look at pixels: an image's vector is spread from a
  hash of its bytes; a text naming a phrase as whole words lands on that phrase's
  image (between images when it names several), and any other text on a hash of
  itself. Use it for tests and for trying the plumbing.
- `ClipImageEmbedder(model="clip-ViT-B-32", device=None)` runs a CLIP checkpoint
  through `sentence-transformers`. That package, torch and the model are **not**
  dependencies: nothing imports them until the adapter is built, and building it
  without them raises `InvalidInput` naming `pip install sentence-transformers`.
  The model downloads on first use. Its test against the real model is skipped
  where the package is not installed, which includes the machine this was built on.

## What was measured, and what was not

The only measurement is on a small in-test fixture
(`tests/retrieval/test_image_lane.py::test_the_lane_finds_the_right_image_for_text_queries`),
with `HashImageEmbedder`: five images whose captions share no word with the five
questions asked of them, five text passages that do say the questions' words, and
twelve unrelated notes, on the in-memory and SQLite stores (identical on both).
With the lane off, the right image was in the top three for 0 of 5 questions.
With it on, the lane itself ranked the right image first for 5 of 5, and in the
fused top three it came second (behind the passage saying the question's words)
for 4 of 5. For the fifth the hashed text vector lane happened to rank other captions
near its top, and those, with their places in the image lane, outranked the
lane's first choice at weight 1.0, so it fell out of the top three
(in a recall of ten, where every lane looks deeper, it came third).

Because the embedder's text side is told which phrase shows which image, this
proves the lane's plumbing (the separate index, fusion, provenance, filters and
forgetting), not retrieval quality. **How well a real image model retrieves images
here is unmeasured.**

Known limits: the image index is not covered by `doctor`, `check_vectors` or
`reembed_vectors`; a space merge or an archive import stores the image episodes
again but does not write their image vectors; a crash between storing an image's
episode and writing its vector leaves the image without a vector until
`ingest_image` is retried. Captions made from one template ("Scan 0001, shelf A",
"Scan 0002, shelf B") look to recall's restatement rule (`demote_restated`) like
one claim restated, and it orders the newest first after fusion, whatever rank the
image lane gave them.
