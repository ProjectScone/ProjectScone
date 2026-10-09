# Observation streams

[Image embedding lane](image-embedding-lane.md) · [Image context](image-context.md) · [Recall semantics](retrieval-and-storage.md)

A camera on a production line, a cart, a robot or a wearable produces frames far faster than the scene changes.
An observation stream filters them where they are produced: each frame is embedded once, and it is stored only
when it shows something new. What is stored is an ordinary image episode, dated at the instant it was seen, so
the image lane, recency, `since`, `as_of` and forgetting all apply to it unchanged.

```python
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.embedders import ClipImageEmbedder
from scone_memory.ingestion import ObservationStream, sightings

engine = await MemoryEngine(
    InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
    image_embedder=ClipImageEmbedder(), image_vectors=InMemoryVectorIndex(),
).open()

camera = ObservationStream(engine, "plant", "line-1-camera")
receipt = await camera.observe(jpeg_bytes, media_type="image/jpeg",
                               observed_at="2026-10-08T12:00:05Z", place="station-4")
receipt.kept, receipt.reason        # True, "novel"

found = await sightings(engine, "plant", "a pallet blocking the conveyor", min_similarity=0.24)
if found.latest is not None:
    found.latest.observed_at, found.latest.place, found.latest.image.attachment_id
```

Nothing here calls a network service. With a local image embedder and local stores, a stream runs offline.

## What the gate keeps

A frame is compared with the last `window` kept frames (8 by default) in the image embedder's space. Its
novelty is 1 minus the highest cosine among them. The receipt says which rule applied:

| `reason` | Kept | When |
| --- | --- | --- |
| `first` | yes | Nothing has been kept by this stream object yet. |
| `novel` | yes | Novelty is at least `novelty` (0.1 by default). |
| `moved` | yes | The frame looks familiar, but `place` differs from the last kept frame's. The same sight somewhere else is a new fact about where things are. |
| `heartbeat` | yes | Familiar, same place, but `heartbeat_seconds` (60 by default) have passed since the last kept frame, so a still scene leaves a trace. |
| `seen_recently` | no | Familiar, same place, heartbeat not due. |

`stream.seen`, `stream.kept` and `stream.dropped` count the frames offered since the stream object was made.

- **A kept frame is embedded once.** The vector the gate computed is the one the image lane stores
  (`ingest_image(..., image_vector=...)`). A dropped frame is embedded once and stored nowhere.
- **Frames arrive in time order.** An `observed_at` earlier than the last one is refused, because the heartbeat
  is measured from the last kept frame.
- **The gate's memory lives in the process.** A new `ObservationStream` keeps its first frame whatever an
  earlier one saw. After a restart, expect one extra kept frame per stream.
- **Frames can expire.** `forget_after` on the stream schedules each kept frame's forgetting, as it does for
  any other record, which bounds the disk a stream uses.

## When and where was this last seen?

`sightings` embeds the query with the image embedder, takes the `window` stored frames nearest to it (50 by
default, at most 200), keeps those at or above `min_similarity`, and returns them newest first. `stream` and
`place` narrow the search to one source or one place.

| Field | Meaning |
| --- | --- |
| `items` | Matching frames, newest first: `observed_at`, `place`, `stream`, `similarity` and the retained `image`. |
| `latest`, `earliest` | The first and last of `items`, or `None`. |
| `searched` | Frames the search looked at. |
| `below_threshold` | Of those, the frames under `min_similarity`. |
| `unresolved` | Matching frames left out because their stored image was forgotten during the search. |
| `window_full` | The search filled its window, so stored frames beyond it were not compared. A newer sighting than `latest` may exist: narrow by stream or place, or raise `window`. |

Images stored some other way (`ingest_image` from a document, say) are never sightings.

## Choosing the thresholds: measured, not assumed

Both thresholds are cosines in the image embedder's space, so they belong to the model. These numbers are from
`ClipImageEmbedder()` (CLIP ViT-B/32) on *Elephants Dream* (Blender Foundation, Creative Commons Attribution):
654 frames, one per second, from a film with about 86 scene cuts in 11 minutes.

**Novelty.** Between consecutive frames it was 0.076 at the median and 0.23 at the 90th percentile. Within a
single continuous shot sampled five times a second it stayed between 0.003 and 0.034.

| `novelty` | Frames kept of 654 |
| ---: | ---: |
| 0.05 | 471 (72%) |
| 0.10 (default) | 259 (40%) |
| 0.15 | 132 (20%) |
| 0.20 | 62 (9%) |
| 0.30 | 18 (3%) |

A film cuts every few seconds. A fixed camera on a line changes far less, so it keeps far fewer frames at the
same setting. Measure on your own footage before fixing a value.

**`min_similarity` has no default, and a score alone cannot say "never seen".** On the same film, at
`novelty=0.1`:

| Query | In the film? | Best cosine | Median cosine |
| --- | --- | ---: | ---: |
| "film credits text on a black screen" | yes | 0.331 | 0.224 |
| "a telephone" | yes | 0.259 | 0.215 |
| "a man looking through a telescope" | yes | 0.256 | 0.215 |
| "a red sports car" | **no** | 0.223 | 0.171 |

Viewed by eye, the best frames for the telephone, the telescope-like instrument and the credits were
the right frames. The query for a car that
is not in the film still found a red-lit scene at 0.223, only about 0.02 below real matches. With this model, a
threshold separates present from absent by a narrow margin. Treat a sighting as a candidate with evidence
attached (the frame is returned so it can be checked), choose `min_similarity` from your own footage and
queries, and prefer a stronger image embedder where one fits the device.

## What this does not do

- It does not detect or track objects. A frame matches a query as a whole image.
- It does not caption frames. Pass `caption=` when something else has described the frame; the text lanes then
  find it by those words. Say `caption_origin="model_generated"` when a model wrote it.
- It does not read video files. Decode frames with `ingestion.video_frames` or your camera's own SDK and offer
  them one at a time.
