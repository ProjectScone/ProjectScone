# Image context and entity retrieval

An image's alt text, caption, title, description, sidecar metadata, and explicit
entity associations can explain its purpose without running a vision model.
Scone indexes these source assertions and links search results to the retained
image bytes. It does not treat an external description as visual verification.

```python
from pathlib import Path
from scone_memory.ingestion import (
    ImageAttribute, ImageContext, ImageEntity, ingest_image, recall_images,
)

context = ImageContext(
    source="catalog/characters.html",
    locator="#pikachu-portrait",
    attributes=(
        ImageAttribute(kind="alt", value="Pikachu, the electric mouse Pokémon", origin="html"),
        ImageAttribute(kind="caption", value="Pikachu using Thunderbolt", origin="html"),
        ImageAttribute(kind="metadata", name="artist", value="Example artist", origin="sidecar"),
    ),
    entities=(ImageEntity(
        entity_id="pokemon:25", name="Pikachu", aliases=("ピカチュウ",),
        relationship="depicts", attribute_indexes=(0, 1),
    ),),
)
saved = await ingest_image(memory, "catalog", Path("pikachu.png").read_bytes(),
    media_type="image/png", context=context, filename="pikachu.png")

found = await recall_images(memory, "catalog", "Who is Pikachu?", entity_id="pokemon:25")
for match in found.matches:
    # Fetch only when needed; ranking does not download each image.
    attachment, image_bytes = await memory.attachment("catalog", match.image.attachment_id)
    print(match.context.attributes, match.context.entities)
```

`entity_id` is an exact caller-managed identity filter, distinct from names and
aliases used for text retrieval. It prevents conflating different entities with
the same name. Relations are explicitly `depicts`, `mentions`, or `associated_with`.
Each relation names the attribute indexes supporting that assertion. This does
not automatically assert or approve a fact in the knowledge ledger, run an entity
resolver, or infer identity from pixels. Callers control entity linking.

## Occurrences and provenance

The same image reused on two pages shares its content-addressed blob but retains
two separate context manifests and searchable episodes. A source/locator and the
actual attributes participate in identity. Exact retries reuse an occurrence and
repair missing attachment links; changed context creates a new occurrence.
Replacing old descriptions requires explicitly retiring the previous episode.

Context manifests retain structured attributes, origins, entity IDs and evidence
references. Small scalar episode metadata and hashed entity tags use existing
backend filters; no nested metadata support is required from the vector store.
PNG, JPEG and WebP still images are validated in a bounded worker before indexing.
Install the optional `images` extra for Pillow. No OCR or LLM is invoked, and no
model is downloaded. EXIF/GPS or other embedded metadata is not harvested
implicitly; a caller may explicitly supply selected `embedded` attributes.

`image_provenance` verifies the retained manifest hash, original reference,
attachment links, source and indexed text. `recall_images` retains the underlying
recall diagnostics and reports unresolved episode IDs when stale or invalid
provenance is refused. Returned scores are ranking values, not identity or factual
confidence. Recall fetches small manifests and authorized blob descriptors; it
leaves large image downloads to the consumer. Blob retrieval retains the configured
store's integrity checks.

## Supplied HTML

```python
from scone_memory.ingestion import image_contexts_from_html
occurrences = image_contexts_from_html(html_text, source="catalog/characters.html")
for occurrence in occurrences:
    print(occurrence.src, occurrence.context)
    # Match src to bytes your ingestion source supplied, then call ingest_image.
```

This helper parses supplied HTML, without a browser, network request, URL fetch,
script execution or automatic entity inference. It captures alt/title/ARIA labels,
unambiguous `aria-describedby` references, the nearest figure's direct captions,
and `data-*` metadata. Scripts/styles/templates are excluded. Each occurrence
has a deterministic node locator and the supplied HTML's SHA-256. The HTML itself
is not automatically retained; callers retain it through their source pipeline.

This is explicit markup extraction, not full browser DOM/CSS/accessibility
semantics. Dynamic images, CSS captions, `srcset` selection and cross-document
references need a source-specific integration. Arbitrary surrounding text can be
provided explicitly with kind `surrounding_text`. Unknown metadata names use kind
`metadata`; values remain plain text and must be escaped by rendering clients.

Limits: HTML 1 MB, 10,000 nodes, depth below 256, 1,000 attributed images, and 4 MB
of output context. Each occurrence allows 64 attributes, 32 entities and 128 KB of
structured context; attribute values allow 16,000 characters. Images allow 10 MB,
20 million pixels and 15 seconds of worker processing. These are processing bounds,
not an OS sandbox or a native-library memory quota.

## HTTP

1. `POST /v1/attachments` uploads image bytes using the existing authenticated API.
2. `POST /v1/images` accepts `{"attachment_id": "...", "context": {...}}` and indexes
   that space's retained image. It requires write permission and shares ingestion
   admission control. Context bodies are capped at 132 KB.
3. `GET /v1/images/search?query=Who%20is%20Pikachu%3F&entity_id=pokemon%3A25` returns
   matches with attributes, entities, original attachment descriptors and an
   authenticated `download_path`. The optional limit is 1–25.

The caller's key determines the space. Downloads require authorization separately;
no credentials are embedded in returned URLs. The framework returns image references
and associated information; the separate web application decides how to display
an image beside a generated answer. This change does not automatically alter chat
output or insert image bytes into an LLM's prompt.

## Evaluation

Tests cover image identity, same-image occurrences, entity disambiguation,
HTML caption association, Unicode, forged provenance, tenant boundaries,
forgetting, HTTP permissions and actual image-byte downloads. The shared backend
contract test runs on every available configured backend pair, including a real
Qdrant server when `SCONE_TEST_QDRANT_URL` is supplied. Skipped services are not
considered validated. Generated colored-image fixtures test metadata retrieval;
they do not measure visual entity recognition or prove an industry-first claim.
