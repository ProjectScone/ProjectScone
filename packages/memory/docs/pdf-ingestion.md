# PDF text ingestion and page provenance

Install the optional parser in the same environment as the framework:

```sh
pip install 'scone-memory[pdf]'
```

The native API extracts a PDF's existing text layer, indexes that derived text,
and retains both the original PDF and a JSON provenance manifest. It uses your
configured document, vector and blob stores. Parsing does not invoke a model,
fetch a URL, download OCR weights, or contact a service.

```python
from pathlib import Path
from scone_memory.ingestion import PdfLimits, ingest_pdf, pdf_provenance

result = await ingest_pdf(
    memory,
    "research",
    Path("report.pdf").read_bytes(),
    filename="report.pdf",
    limits=PdfLimits(max_pages=100, timeout_seconds=30.0),
)

recall = await memory.recall("research", "What did the report conclude?")
for item in recall.items:
    if item.episode_id == result.added.episode_id:
        pages = await pdf_provenance(
            memory, "research", item.episode_id, chunk_id=item.chunk_id
        )
        print([page.number for page in pages.pages])

# The attachment service still returns the unchanged original bytes.
attachment, original_pdf = await memory.attachment(
    "research", result.original.attachment_id
)
```

For an exact retrieved chunk, pass `chunk_id` as above. The resolver validates
that the chunk belongs to that episode and matches its retained text. You can
alternatively pass explicit `start` and `end` offsets to `pdf_provenance`. They are half-open **UTF-8 byte offsets into extracted text**,
not byte positions in the PDF file. A UTF-8 character cannot be split by a span.
The returned pages carry one-based page numbers, text spans, unrotated PDF media
box dimensions in physical points (including `/UserUnit` scaling), and clockwise
page rotation. Page separators are two
newlines and do not belong to either page's text span.

The parser reads pypdf's layout mode, which places text on a character grid
by its position on the page, and then reads that grid in reading order (next
section). Page dimensions are not glyph or paragraph bounding boxes. This does
not detect table cells or headings. Page crop boxes and complex transformations
are not resolved into highlight rectangles. PDF text extraction can misorder or
omit content; inspect the retained source when assessing evidence.

## Reading order from the text layer

A two-column page comes out of the text layer as a grid holding both columns
side by side, so reading it row by row puts half a sentence from each column
on every line, and a chunk cut anywhere in that text quotes two paragraphs at
once. The parser therefore lays each page out before the text is stored, from
the grid's own geometry and without a layout model:

- Every run of characters on a row (runs are separated by four or more
  spaces) becomes a region whose box is its grid position. The same whitespace
  partitioning that orders OCR words (`ocr.layout.order_columns`) finds the
  gutters, with a gutter a few characters wide, and the page is written out
  column by column: what spans the columns above them first, then each column
  top to bottom, then what spans them below.
- A cut is kept only where every column reads as prose: at least three rows,
  fifteen characters wide at its widest, most rows a single run, and the
  typical row at least half the widest. A table page fails those, stays as it
  was extracted, and its receipt says `tabular_kept_whole`. A page with more
  than `MAX_REGIONS_PER_PAGE` runs also stays as extracted, with `region_limit`.
- A page number on a line of its own is left out of the page's text on every
  page. A line whose words recur at the top or bottom of three or more pages is
  a running header or footer: it stays where it first appears (that may be the
  page where it is the title) and is left out after. Every line left out is
  recorded in the page's `running` tuple, so nothing extracted is lost.

Each laid-out page carries a `reading_order` receipt with
`strategy: grid-columns-v1`, the number of columns found (0 when none), and
notes; a page that was cut also carries its regions, with `region_geometry:
normalized_text_grid` to say that their boxes are estimated from grid
positions rather than measured on a raster, `provider_index` (the run's
grid order) and `reading_column`. The manifest is schema version 3, as with
OCR reading order, and `pdf_provenance` returns the same receipts and regions,
so a caller can see which column a chunk came from. The parser string ends in
`+grid-columns-v1`.

A single-column page comes out as it did before, apart from the running
lines. To keep the text exactly as pypdf writes it, row by row across the
page, pass `PypdfParser(layout='as_extracted')` to `ingest_pdf` or the
document registry; such pages carry no receipt and the manifest stays at
schema version 1.

Measured on the thirteen PDFs under `reference/` (a two-column ICML paper,
three narrative reports, five SEC filings full of financial tables, a slide
deck and a table page): see the pull request that introduced the pass for the
per-document counts of pages cut, pages kept whole, and lines left out.

## Coverage and failures

- A PDF with no extractable text fails explicitly and suggests OCR. A scanned
  image is not silently converted into a successful empty document.
- A mixed document retains page numbering and reports `empty_pages`. Its episode
  has `pdf_coverage=partial`; empty pages may be blank or may require OCR.
- `pdf_coverage=text_layer` means every page yielded text. It does **not** certify
  that every visible word, figure or table was understood.
- Encrypted PDFs, malformed files, absent parser dependencies and exceeded limits
  raise `InvalidInput` before creating an episode. Password handling is not
  implemented; provide a separately decrypted input when appropriate.
- Parser failures do not indicate that the memory database is unavailable.

The defaults are 25 MiB of PDF input, 100 pages, 2,000,000 extracted UTF-8 bytes
and 30 seconds. Callers may lower these bounds; maximum supported settings are
25 MiB, 1,000 pages, 2,000,000 text bytes and 120 seconds. The framework's existing
attachment byte limit also applies.

Parsing runs in a separate process. Timeout and caller cancellation kill and reap
that process. Page and extracted-text limits refuse oversized results; they are
**not a peak memory quota**. A PDF content stream may decompress into much more
memory before a text bound can be checked. Use deployment-level process/container
memory and concurrency limits for untrusted or concurrent ingestion. The native
API does not provide a shared admission queue or sandbox the parser's OS access.

## Source identity and durability

The original and manifest are content-addressed attachments linked to the derived
file episode. Episode metadata contains small references and coverage labels;
page arrays live in the manifest. A deduplication identity includes original PDF
bytes, parser version and the actual extraction manifest. Identical text in two
different originals therefore keeps separate provenance. Repeating the same
original and extraction reuses its episode.

`pdf_provenance` checks the requesting space, both attachment links and media
types, content hashes, schema, extracted text hash and complete page-span mapping.
Unlinked, modified, foreign-scope and forgotten evidence is refused. This is
verification of retained records, not a signed parser attestation or independent
proof that extracted text accurately represents the PDF. Trusted integrations may
supply a `PdfParser`; they remain responsible for its extraction quality.

Writes use existing attachment and memory primitives and are not one atomic
transaction. A storage failure can leave unlinked attachments or a partially
linked episode. The resolver refuses incomplete links. Retrying identical input
and parser output reuses the episode and retries its links. Durable parsing jobs,
work ownership, automatic retries and cleanup of abandoned attachments remain
follow-on work.

## HTTP ingestion and page evidence

With the `api,pdf` extras installed, `scone-memory serve` exposes:

1. `POST /v1/attachments` with raw PDF bytes and `Content-Type: application/pdf`.
2. `POST /v1/documents/pdf` with `{"attachment_id":"<returned SHA-256>"}`.
3. `GET /v1/recall?q=...` to search the extracted text.
4. `GET /v1/episodes/{episode_id}/pdf?chunk_id={chunk_id}` to resolve a returned
   chunk to its source pages. Omit `chunk_id` for all nonempty pages.

All requests require bearer authentication. Ingestion requires a write/full key;
page evidence and the returned relative `download_path` allow read keys in the
same space. Forgotten sources stop resolving. The ingest response contains
`added`, `original`, `manifest`, and `empty_pages`, matching the Python result.
Retrying identical bytes and parser output reuses the episode.

`documents.pdf` in `/v1/capabilities` reflects parser dependency availability.
Missing dependencies return 501; malformed PDFs or files needing OCR return 422,
without declaring the memory service unavailable. `documents.pdf.provenance`
exposes retained evidence inspection independently of parser installation.

The JSON request is limited to 4 KiB and accepts only `attachment_id`. HTTP uses
the fixed default `PdfLimits` above and shares the server's ingestion admission
limit with episode and image writes. Saturated admission returns 429 with
`Retry-After: 1`. Uploading alone does not parse or index a PDF. Failed parsing
can leave the previously uploaded attachment, but creates no searchable episode.

This HTTP slice extracts text layers only. For explicitly configured scanned-page
rendering and OCR with source regions through Python, see
[scanned PDF ingestion](pdf-ocr.md). Document layout models and Webapp integration
remain separate capabilities.

## Validation and upstream dependency

```sh
pip install -e '.[pdf-test]'
pytest -q tests/ingestion/test_pdf_ingestion.py tests/memory/test_attachments.py tests/ingestion/test_ingestion_component.py
```

Tests construct real PDFs with text, multiple pages, rotation, encryption and
raster-only content. They cover recall and provenance on in-memory and SQLite
stores, source-specific deduplication, failure/retry, scope checks and real parser
process termination. Fixtures contain synthetic content and no private records.

The implementation uses the optional BSD-3-Clause licensed
[pypdf package](https://github.com/py-pdf/pypdf); its upstream license remains with
that dependency. Its [text extraction documentation](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)
explains layout mode and the distinction between PDF text layers and OCR.
