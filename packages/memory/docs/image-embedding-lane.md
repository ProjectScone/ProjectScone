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
  made both, and an index holds one width. Every first-party index defaults to
  one namespace, so a second handle built with the same arguments is the same
  storage: its image vectors would overwrite the captions' text vectors, which
  have the same chunk ids. Each index names where its rows live (`location`),
  and the engine refuses an image index equal to the text index there. Give it
  its own namespace:

  | Index | The image lane's own storage |
  | --- | --- |
  | `SqliteVectorIndex` | another file: `SqliteVectorIndex("~/.scone-memory/image-vectors.db")` |
  | `PostgresVectorIndex` | another schema: `PostgresVectorIndex(url, schema="scone_images")` |
  | `ElasticsearchVectorIndex` | another prefix: `prefix="scone_images"` |
  | `OpenSearchVectorIndex` | another index: `index="scone_image_vectors"` |
  | `QdrantVectorIndex`, `MilvusVectorIndex` | another collection: `collection="scone_images"` |
  | `ChromaVectorIndex` | another collection; in-process clients on one path, or in memory, share a store |
  | `LanceDBVectorIndex` | another table: `table="scone_images"` |
  | `RedisVectorIndex`, `ElastiCacheVectorIndex` | another prefix: `prefix="scone_images"` |

  A local file or directory (SQLite, Chroma, LanceDB) is compared as the file it
  is, by device and inode, so another letter case on a case-insensitive
  filesystem (macOS's default), a hard or symbolic link, or `..` is recognised
  as the same storage. A server's configuration is compared as written (a URL, a
  schema) or by an injected client's identity. Two spellings of one server
  (`localhost` and `127.0.0.1`), or two client objects on one server, are not
  recognised; neither is an index that names no location (the in-memory index,
  a custom one), except as the same object, nor a `LangChainVectorIndex` whose
  store is bound after the engine is built.
- **The image embedder that made each vector.** Every image vector carries the
  embedder's id and width in its metadata (`image_embedder`, `image_embedder_dim`).
  How the lane keeps another model's vectors out depends on the index, and
  `engine.image_writer_check` says which:

  - `recorded` (`InMemoryVectorIndex`, `SqliteVectorIndex`): the index records its
    writer, as the text index does, and refuses any other image embedder (below).
  - `tagged` (every other index: Postgres, Elasticsearch, OpenSearch, Qdrant,
    Milvus, Chroma, LanceDB, Redis, ElastiCache, LangChain, a custom one): the index
    cannot record a writer, so the lane searches only vectors whose
    `image_embedder` is its own embedder's id, as a `where` condition in the index,
    and ignores the rest. A vector written before vectors were tagged (by an
    engine older than this) is ignored until its image is written again: rebuild
    the lane once after upgrading (`reembed_images`, see
    [Rebuilding the lane](#rebuilding-the-lane)); an exact retry of `ingest_image`
    also writes it.

    When that search leaves the lane's window short, the lane searches the index
    once more under the same filters without its tag and counts the live images
    it finds beyond its own. `degraded` then says `image lane: ignored N image
    vectors under this recall's filters that <embedder id> did not tag (another
    image model's, or written before image vectors were tagged); rebuild them with
    reembed_images()`, with `at least N` when that second search filled its window
    too. A recall whose own window is full makes no second search and says
    nothing about vectors it did not tag. The count is exact on an index that
    searches exactly; an approximate index can miss vectors in either search.

    **One tagged index serves one image model.** An image has one vector, keyed by
    its caption's chunk id, whichever model wrote it last: another model's
    `ingest_image` or `reembed_images` replaces the first model's vector for that
    image, and the first model's lane loses it (a recall of its whose window is
    short then says it ignored it, as above). Two models writing to one tagged index take its images from
    each other. Give each model an image index of its own.
    The width is recorded, not searched on: an index holds one width.

  On a `recorded` index, an engine whose image embedder is not the recorded one
  (another model, or the same width under another id) is told so when it opens:
  `engine.image_block` names both embedders. It writes nothing there:
  `ingest_image` stores the episode and its attachments, reads the record again
  just before it would write, and returns `image_lane="blocked"` with the reason
  in `image_lane_blocked`, so one ingest cannot record the index as mixed and
  leave it unusable by every embedder. Its recalls are refused by the lane and
  named in `degraded` (`image lane: VectorsNotComparable: ...; rebuild them with
  reembed_images()`), and the other lanes answer. To move to another image model,
  rebuild the lane with it ([Rebuilding the lane](#rebuilding-the-lane)), or give
  it an image index of its own. The record is read, then the vector written, in
  two steps, so a write by another embedder between them in another process can
  still record the index as mixed; a rebuild settles that too.
- **The episode's tags, metadata and time** on every vector, so `tags`, `where`
  and `as_of` narrow the lane in the index. `conditions` narrow it in the index
  when the index evaluates conditions itself (the in-memory and SQLite indexes
  do); otherwise, and for `kind`, `source_prefix`, `since` and `until`, recall
  post-filters what the lane found. Whether a condition narrows in the index is
  asked of the image index itself, not of the text index: a post-filtered image
  lane looks as deep as a post-filtered vector lane (twenty-five times an
  unnarrowed recall's window). The narrowing report says what the lane did
  (`image_lane`: `in_store`, `postfiltered` or `off`; `image_window`;
  `image_returned`), and a full post-filtered image window sets
  `window_exhausted`, as a full image window sets `phrases.short`.


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

## When an image's vector cannot be made

`ingest_image` stores the image, its context and the caption episode before it
asks the image embedder for a vector. When the embedder raises (a model that is
down, an image it refuses) or the image index will not take the write, the
receipt is `image_lane="failed"` with `image_lane_error` naming the error's type
and message (`ConnectionError: image model unreachable`); `POST /v1/images`
answers 200 with the same fields. The image is stored and linked, and the text
lanes find it by its caption; only the image lane lacks it. An exact retry of
`ingest_image` writes the vector. A forget that lands while the image is embedded
still raises `Gone` (or `NotFound`), and an index recording another image embedder
still answers `blocked`.

The lane's weight in fusion is `IMAGE_WEIGHT = 1.0`, the text lane's own. It is
not tuned: there is no image retrieval benchmark here and no model to run one.

## Rebuilding the lane

`engine.reembed_images(space, limit=100, before=None)` embeds a space's stored
images again with the engine's image embedder, from each image's original bytes,
one bounded pass at a time. Use it to move the lane to another image model, to
tag vectors written before vectors were tagged, or to write the vectors of
images whose indexing `failed`.

```python
before = None
while True:
    report = await engine.reembed_images("catalog", limit=200, before=before)
    if report.scan_complete:
        break
    before = report.resume_before
print(report.writer, report.spaces_pending)
```

The same pass is `scone-memory --space catalog reembed-images --limit 200
[--before ID]` (add `--json` for the report) and `POST /v1/images/reembed
{"limit": 200, "before": null}`, which a key that may write the space can call;
`GET /v1/capabilities` has `images.reembed` true when the engine can serve it.
The image index is shared by every space, but a key reaches only its own, so
the route's `spaces_pending` names at most the key's own space, and
`pending_elsewhere` is true when some other space is still pending; it never
names that space. The CLI and Python report name every pending space.
The route takes one ingest slot for the whole pass, as an ingest does, so a
server whose slots are all embedding answers 429 `ingest_busy`.
Both refuse (`InvalidInput`, HTTP 422) on an engine without the image lane, and
the `scone-memory` command and server build their engines without it, so they
reach it only when a Python program builds the engine with the lane and hands it
to `create_app(engine)` or `runtime.cli.run(args, engine, ...)`.

A pass reads at most `limit` (1 to 1000) file episodes, newest id first, from
`before`. The report says:

| Field | Meaning |
| --- | --- |
| `scanned`, `reembedded` | file episodes read; images given a new vector (a file episode that carries no image is read and skipped) |
| `forgotten` | images forgotten while they were re-embedded: no vector is kept |
| `failed`, `error` | images whose bytes could not be read, whose embedding failed, or whose vector the index would not take, and the first error; each keeps whatever vector it had |
| `scan_complete`, `resume_before` | false and the episode to walk on from when the pass stopped at `limit`; a full page is told from the end of the space by looking for one more |
| `orphans_removed` | on the pass that completes the walk, the space's image vectors whose chunk was gone and are now removed; None when the index cannot list its vectors |
| `writer` | what the image index holds after the pass, below |
| `spaces_pending` | on the pass that completes the walk under a rebuild marker, every space still holding a vector this embedder did not tag |

What `writer` says depends on the index:

- **An index that records its writer, which already records this embedder**
  (`recorded`): the pass rewrites the vectors, tagged, and the lane stays on.
- **An index that records another writer, a mix, or none over vectors it cannot
  vouch for**: the first pass takes this embedder's rebuild marker (`rebuilding`).
  From then the lane is refused for every engine (the rebuilding model's
  `engine.image_block` says `image lane rebuild in progress`; any other model's
  names the marker as the recorded writer), `ingest_image` through this embedder still
  writes (under its own marker) and through any other is `blocked`. The pass
  that completes a space's walk records this embedder as the writer only when no
  space holds a vector it did not tag, read with one search per space as deep as
  the space holds vectors. A rebuild of one space of several, or one that left
  an image `failed` over the old model's vector, stays `rebuilding` and names the
  spaces in `spaces_pending`: rebuild those (or run the failed space again) and
  the pass that completes last records the writer. A write by another image
  embedder between that check and the record raises `VectorWriterChanged`
  (HTTP 409 with `code: writer_changed`); run the pass again. `refused` means another image embedder wrote to the index
  while the pass ran: the record is mixed, any image the pass reached after that
  write is `failed`, and the next pass takes the marker again.
- **An index that cannot record a writer** (`tagged`): nothing is recorded or
  refused. Each rewritten vector carries this embedder's tag, so the lane sees it
  at once and ignores whatever another model tagged; `orphans_removed` is None
  unless the index can list its vectors.

No backend was changed for this: every index already takes `upsert`, `delete` and
`where` on metadata. A document store that cannot list episodes (`page_episodes`)
is refused by name. The Postgres, Elasticsearch, OpenSearch, Mongo and AWS
backends run only in CI; the rebuild's tests run on the in-memory and SQLite
stores and on an in-memory index that hides its writer record.

## Forgetting

Forgetting an episode through an engine with the lane deletes its image vector
with its text vectors, by the same chunk ids, including a forget resumed by
`recover()`, an expiry, `forget_matching` and a keyed `replace`. Deleting a space
through it sweeps the image index too.

Only an engine with the lane can reach its index. The HTTP server, the CLI,
directory sync and any engine built without `image_vectors` forget the episode,
its chunks, text vectors and attachments, but leave the image's vector and its
metadata (`image_original`, `image_manifest`) in the image index. The forget's
receipt says which happened: `image_vector` is `removed` from an engine with the
lane, `not_reached` from one without it for an episode `ingest_image` stored, and
`none` for any other episode. An engine with the lane removes what those forgets
and space deletions left:

- **When it opens**, every image vector whose chunk is gone, in every space;
  `engine.image_vectors_removed` counts them (None when the image index cannot
  list what it holds, as a custom index may not). This reads every id the image
  index holds, one page of chunks at a time.
- **When a recall meets one.** A forget through another engine while this one is
  open leaves vectors that would still rank in the lane and fill its window. A
  hit whose chunk is gone is removed and the index searched again, up to
  `IMAGE_SEARCHES = 2` times; `degraded` says `image lane: removed N vectors of
  images already forgotten`, and adds that forgotten images still filled the
  window when the last search met some too, so the lane ranked fewer images than
  it looks for. Removed vectors are never candidates, but they count in
  `image_returned`, and a window they filled sets `window_exhausted` and
  `phrases.short` as any full window does: images deeper than it went unseen.

The image vector is written after the episode is stored, so a forget can land
while `ingest_image` is still embedding the image. Forgetting deletes the chunks
before the vectors; after writing the vector, `ingest_image` reads the chunk
again, and when it is gone it deletes the vector it just wrote and raises `Gone`
(or `NotFound` while that forget has not yet written its tombstone). No vector
outlives the forgotten image, and no result says `indexed` for it.

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
for 5 of 5, in a recall of ten as well. This is at the hashed vector lane's
default voice of 0.01 (`HASHED_VECTOR_WEIGHT`). At its previous default of 0.25
it was 4 of 5: for the fifth question the hashed text vector lane ranked other
captions near its top, and those, with their places in the image lane,
outranked the lane's first choice at weight 1.0, so it fell out of the top three
(in a recall of ten it came third).

Because the embedder's text side is told which phrase shows which image, this
proves the lane's plumbing (the separate index, fusion, provenance, filters and
forgetting), not retrieval quality. **How well a real image model retrieves images
here is unmeasured.**

Known limits: the lane is configured and asked for from Python only; no `SCONE_*`
setting builds it, so the `reembed-images` command and route answer only for a
Python-built engine, and forgets through the CLI and server
leave image vectors until an engine with the lane opens or recalls (see
Forgetting). A forget's `image_vector` is decided by the engine that starts it;
when an engine of the other kind finishes it through `recover()`, the receipt and
`forget_status` still show the first engine's answer, and an engine with the lane
still removes the vector when it next opens. The image index is not covered by `doctor`, `check_vectors` or
`reembed_vectors` (its own rebuild is `reembed_images`); a space merge or an archive import stores the image episodes
again but does not write their image vectors, and a crash between storing an image's
episode and writing its vector leaves the image without a vector, until
`ingest_image` is retried or the space is rebuilt. On an index that cannot
record its writer, the lane counts the vectors it ignores only when its own window
is short, and one image model's writes replace another's (see What the lane holds). Captions made from one template ("Scan 0001, shelf A",
"Scan 0002, shelf B") look to recall's restatement rule (`demote_restated`) like
one claim restated, and it orders the newest first after fusion, whatever rank the
image lane gave them.
