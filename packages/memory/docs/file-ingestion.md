# File ingestion and restart recovery

Install `scone-memory[documents]` for the built-in text, structured-data,
Office and PDF readers. Add `document-workflows` for encrypted workflow
checkpoints, or `document-converters` for optional binary Office/message
readers. Dependencies and recognition models are explicitly provisioned
by the application operator.

```python
from scone_memory.ingestion import ingest_document, document_provenance

result = await ingest_document(
    memory, "research", b'{"release":{"codename":"Polaris"}}',
    filename="release.json",
)
evidence = await document_provenance(memory, "research", result.added.episode_id)
for segment in evidence.segments:
    print(segment.locator, segment.text)
```

The original bytes and extraction manifest are retained as linked attachments.
Search indexes the extracted text; `document_provenance(..., chunk_id=...)`
checks the retained source, manifest and chunk before returning overlapping
source segments. Repeated identical originals and extraction outputs reuse
their identity. This does not make arbitrary parser output authoritative.

Attachments are identified by their bytes and keep the first upload's filename
and media type. Each extraction separately records the filename used to select
its parser. `result.filename` and `evidence.filename` are that extraction label;
`result.original.filename` is the original upload label and may differ or be
absent. Identical bytes can have distinct CSV and plain-text interpretations,
each with its own extraction manifest and deduplication identity.

Configured PDF OCR and image readers retain typed `DocumentTextRegion` values
on each segment. Each region includes its recognized text, normalized box,
recognizer score, block/line identifiers and half-open `start`/`end` offsets
in the **segment's UTF-8 bytes**. `coordinate_space` identifies the displayed
page or image frame with a top-left origin. PDF dimensions in segment metadata
still describe the unrotated media box; apply the recorded rotation when
displaying it. Recognition scores are not factual confidence.

Chunk citations return whole overlapping source segments and only regions
that overlap the chunk. Region spans remain relative to the full segment,
including after empty PDF pages are omitted. New manifests containing regions
use schema version 2, or version 3 when a PDF's optional inferred column order
is recorded. Those regions also retain `provider_index` and `reading_column`;
the segment's `metadata.ocr_reading_order` describes the whole page strategy,
column count and limitations even when chunk citations return fewer regions.
See [column reading order](pdf-ocr.md#estimate-column-reading-order).
Documents without regions or table cells retain version 1 and their
existing serialized attachment identities; existing version 1 evidence stays
readable. Re-extract an old OCR document to obtain typed regions. When using a
workflow, change its `parser_revision` and use a new run for that re-extraction.

## Coverage

| Reader | Evidence retained | Limits |
|---|---|---|
| Text, Markdown and code files | Line locators | Source text only; no AST or semantic code graph |
| JSON/JSONL/NDJSON, CSV/TSV, XML | JSON paths, rows/cells or XML locators | No schema-specific semantic interpretation |
| IPYNB v4 | Cell sources and saved text outputs with JSON Pointer locators | No code execution, image-output analysis, or legacy v3 conversion |
| HTML | Visible text, table cells, spans and source-linked headers | Bounded parser; no browser execution, stylesheets or remote resource fetching |
| DOCX, XLSX, PPTX | Paragraphs/tables, sheet cell references, slides and notes | No rendered Office layout or macro execution |
| ODT, ODS, ODP, EPUB | Format-local segment locators | Text extraction; no rendered layout |
| EML | Message-part locators | No recursive attachment ingestion |
| RTF, XLS/XLSB, MSG | Converter/reader locators | Optional dependencies; message attachments are not extracted |
| DOC, PPT | Converted text locators | Explicit offline converter; macOS textutil also supports DOC; page/slide structure may be lost |
| PDF | Page locators, extraction method, configured OCR regions and engine | Native text by default; OCR requires an explicit parser |
| Images | Frame/region locators and typed OCR geometry | Explicit `ImageDocumentParser` and OCR engine required |
| Audio/video | Audio-stream timestamps | Explicit `MediaDocumentParser` and transcription provider required; video frames are not analyzed |

OpenDocument extraction uses current content: `text:tracked-changes` revision
history and `office:change-info` metadata are omitted. Current text, including
tracked insertions, remains in its document order. Revision history is not
emitted as a separate searchable view.

OpenDocument comments, footnotes and endnotes are separate searchable segments.
Their `content_role` and `parent_locator` identify their relationship to a
paragraph, table row or spreadsheet cell. Comments retain available author,
date and name metadata; notes retain their ID and citation label. Those labels
and author details are not inserted into the document's body text or cell values.
Annotations outside a paragraph use `body/comment:N` locators (with a slide
prefix in presentations). Nested annotations carry their own metadata. These
segments consume the same text and segment budgets as ordinary content.

ODP speaker notes use `slide:N/notes/...` locators, with `content_role` set to
`speaker_notes` and `parent_locator` set to their slide. Visible slide paragraphs
are numbered separately. Comments within speaker notes retain their own comment
role and point to the corresponding note paragraph. Notes remain searchable and
consume the same extraction budgets as slide content.

DOCX extraction likewise omits deleted content and old move locations before
numbering paragraphs and tables. Current insertions and move destinations remain.
Directly hidden runs (`w:vanish`) are omitted; inherited style visibility is not
resolved. Formatting properties do not supply text or tabs. Word ruby and Excel
phonetic hints are omitted while their base text remains. Word nonbreaking
hyphens and position tabs remain in extracted text. This is a text view, not a
rendered preview; headers, footers and text-box geometry remain open coverage gaps.

Word text boxes retain their own paragraphs/table rows, with `content_role=textbox`,
the source archive `member`, and the anchor paragraph or table row as `parent_locator`.
They are queued after body text instead of being concatenated into the anchor.
Nested boxes receive nested locators; extraction order is not page layout order.
For DOCX main/note/comment XML parts, alternate content selects the first choice
whose required namespace URIs are supported for text extraction (Word main,
Word 2010 wordprocessingShape, and VML), otherwise its fallback. Prefix aliases
and local namespace shadowing are honored. Missing/invalid requirements, malformed
branch ordering, or an unsupported choice without fallback are explicit errors.
Unused alternatives still count against XML construction limits. This is text
extraction support, not full drawing rendering or general markup-compatibility
processing for all Office formats.

Referenced DOCX footnotes, endnotes and comments are extracted after the main
body, in first-reference order. Each part is emitted once, with its `member`,
`content_role`, note/comment ID and first current `parent_locator`; this does not
enumerate every cross-reference or the full range covered by a comment. Comments
also retain available author, date and initials. Deleted and hidden references,
unreferenced annotations and separator notes do not become searchable content.
Current-text filtering also applies inside notes. Dangling, ambiguous and invalid
part references are rejected. All extracted parts share the document's text and
segment budgets, and nested reference locators are bounded.

DOCX, XLSX and PPTX locate their main document through `_rels/.rels` and resolve
child relationships relative to that selected part. Nonstandard main-part paths
are supported. An unreferenced conventional filename does not supply document
text. Missing, external or ambiguous main-document declarations are rejected;
older minimal ZIP containers without package relationships must be repaired or
re-exported before extraction.

Use `document_formats()` or authenticated `GET /v1/documents/formats` to
inspect default-reader dependencies on the running installation. Availability
does not guarantee that every valid variant of a format is supported.
Media readers must be registered explicitly on a `BuiltinDocumentParser`;
the default HTTP route does not configure OCR or transcription providers.

The local LlamaIndex reference also advertises HWP, PPTM and MBOX
readers, which remain gaps. Table understanding, semantic chunking, layout
reconstruction, directory synchronization and general connector ingestion
also remain open. The [PDF OCR guide](pdf-ocr.md) describes separate OCR
geometry, recognition limits and model-quality caveats.

Notebook segments distinguish `cell_source` from `saved_output` in metadata.
Code, Markdown and raw cells keep their cell index and any valid cell id.
Saved stream/error outputs and plain-text display results are extracted;
visible HTML is a fallback when plain text is absent or empty. Alternative
representations are not indexed twice. `outputs_without_text` counts outputs
that supply no extractable text, including image-only results. Notebook image
attachments and interactive widgets remain in the retained original. Saved
outputs are observations from the file, not results executed or verified by Scone.

Text and MIME line locators count LF, CRLF and bare CR terminators; Unicode
separators and form feeds remain inside their source line. JSONL records split
only at LF (including CRLF), so Unicode separators inside strings remain data.
CSV supports quoted multiline fields. TSV uses literal quotes and tab delimiters;
it does not use the Excel quoted-tab dialect. Empty delimited records are skipped
without renumbering later row locators or their physical line ranges. HTML `pre`
content preserves source indentation, tabs and newlines. Normal HTML flow
collapses ASCII whitespace and preserves nonbreaking spaces; external CSS is not
interpreted.

### HTML table evidence

HTML and HTML MIME bodies retain typed `DocumentTableCell` evidence in
`segment.table_cells`. Cells record a table locator, source cell locator,
zero-based grid row/column, row/column spans, header status and exact value text.
`start` and `end` are half-open offsets in the segment's UTF-8 bytes, not HTML
source offsets. Cell locators count source cells, including hidden cells; they
remain stable when footer rows are placed after the body.

Each header reference carries the header cell's locator, text and association
(`explicit`, `row`, `column`, `rowgroup` or `colgroup`). The reader resolves
`headers` IDs within the table, scoped headers, and automatic row/column headers,
including multiple header levels and spanning cells. An explicit empty `headers`
attribute disables inference. Hidden headers never supply text. Unresolved IDs
are disclosed through `metadata.table_notes=unresolved_headers`.
Tracking is limited to 20,000 source IDs of at most 4,096 characters. If that
budget prevents resolving a later explicit reference, `header_id_limit` is also
reported; unrelated visible text remains extractable.

Data-cell text includes its associated labels before the value. For example,
`Europe / 2026 / Sales: €20` indexes the declared row and column context together.
The cell's byte span covers only `€20`; its header references identify the
original source cells. Header-only rows, captions, empty cells within nonempty
rows, PRE whitespace and ordinary inline whitespace are retained. Wholly empty
rows produce no text segment; later grid coordinates are not renumbered.

```python
evidence = await document_provenance(memory, "team", result.added.episode_id)
for segment in evidence.segments:
    for cell in segment.table_cells:
        print(cell.locator, cell.text, [(h.text, h.locator) for h in cell.headers])
```

These documents use manifest version 4 and parser `scone-text-tables-v1`.
Versions 1–3 remain readable, and documents without table evidence keep their
existing serialization. Chunk-filtered citations return overlapping value cells
with their full header references; header source rows can lie outside the chunk.
The complete retained manifest validates those references before filtering.
Change a durable workflow's `parser_revision` and use a new run to re-extract
previously flattened tables.

Nested tables, overlapping cells, spans outside a row group and malformed
table placement retain the previous visible text with `table_status=text_fallback`
and a specific `table_notes` reason. No structured cells are claimed for those
tables. Resource exhaustion fails explicitly: at most 20,000 cells per table,
1,000 columns, 100,000 occupied slots, 128 headers per cell, one million table
operations and 8 MB of serialized cell evidence, within the existing text,
segment and wall-time limits. The slot and evidence limits also apply across
the complete document. This does not detect tables in OCR geometry or add typed
table evidence to Office and delimited readers yet.

## Durable extraction checkpoints

`DocumentIngestionWorkflow` reuses the shared encrypted workflow journal:

```python
from scone_memory.ingestion import DocumentIngestionWorkflow

# memory uses persistent document/vector/blob stores; checkpoint_key is a
# persistent 32-byte key supplied by the application, not regenerated per run.
job = DocumentIngestionWorkflow(
    memory, "document-workflow.db", key=checkpoint_key,
    parser_revision="application-reader-v1",
)
try:
    receipt = await job.run(
        "import-42", space="research", attachment_id=original.attachment_id,
    )
finally:
    job.close()
```

The source must already be retained. Supply `filename="release.json"` to both
`run()` and `status()` to choose an extraction label explicitly; omitting it
uses the retained filename. The label is bound to the run, so changing or
removing an explicit label requires a new run id. Existing runs that omitted
the argument keep their original checkpoint binding. Extraction saves its
manifest before indexing. Reopening the same journal with the same key,
parser revision, limits and source identity resumes indexing without
repeating completed extraction. Changed parser/model options need a changed
`parser_revision`. Source validation rejects missing or deleted evidence.
An `OSError` (other than confirmed `FileNotFoundError`) or SQLite operational
failure during verification reports `verification_unavailable`. It preserves
completed receipts and prevents execution or result return until an explicit
retry verifies the evidence successfully. Callback error messages are not
exposed. A missing source, changed evidence, or an explicit verifier rejection
still permanently invalidates that run.

This is one caller-owned active document per journal. It does not provide a
background queue, distributed worker leases, or a persisted chunking plan.
Cancellation interrupts the active call; an explicit
retry resumes eligible stages. For retained PDFs, the separate `PdfOcrWorkflow`
provides encrypted per-page OCR checkpoints and indexing recovery; see the
[PDF OCR guide](pdf-ocr.md#resume-completed-pages-after-interruption).

Indexing saves each complete, validated embedding batch inside the encrypted
journal before calling the next batch. After cancellation or process failure
during embedding, a retry rebuilds the chunk plan and reuses matching vectors.
Receipts bind the space, source content identities, ordered UTF-8 chunk spans,
exact embedding input texts, batch boundaries, and embedder `id` and dimension.
The configured `id` must identify the model revision and its embedding options;
changing model behavior without changing that identity cannot be detected.
Changed chunking or contextual input also prevents reuse. No episode is written
until every vector validates. Malformed provider batches are never saved.

`job.status(...).checkpoint_count` reports retained intermediate batches,
separately from completed extraction/index steps. Each receipt is limited to
16 MiB; one run permits at most 4,096 receipts and 128 MiB of encrypted receipt
data. Missing/deleted evidence invalidates the run and removes its receipts on
the next verification; successful completion removes them too. Temporary
verification outages preserve them. This is logical deletion from an encrypted,
caller-owned journal, not secure erasure of SQLite pages or backups.

This recovery covers unfinished embedding work. If the process dies after
episode/chunk writes begin, the engine's existing inflight-write recovery may
still re-embed those chunks before the document workflow resumes. Replaying an
already completed workflow verifies its recorded source; it does not migrate
an existing vector index to a new model. HTTP ingestion does not automatically
use these caller-owned journals.

## HTTP and execution boundaries

Retain the original through the existing attachment upload route with its
filename, then post `{"attachment_id":"<retained SHA-256>"}` to
`POST /v1/documents`. Include `"filename":"report.csv"` to choose the parser
independently of the original upload label, including for nameless attachments.
The response's `filename` reports this choice without rewriting upload metadata.
Read provenance through
`GET /v1/episodes/{episode_id}/document?chunk_id=...`. The API uses its
authenticated space, write-role authorization and ingestion backpressure.
HTTP indexing is synchronous and does not automatically create a durable
workflow journal.

Parsers enforce input, extracted-text, segment, archive and execution limits.
Office/ODF/EPUB ZIP members must use stored or deflated compression. Standalone
XML and each XML archive member use the same construction limits: 16 MiB of XML,
200,000 elements and nesting depth 128; the tree
builder also bounds expanded names, attributes and text to 32 MiB. A raw-name
pass limits names and namespace URIs to 1,024 bytes and attributes to 256 per
element before namespace expansion. EPUB chapter doctypes are
accepted without fetching external DTDs. Standard HTML entity names resolve
locally; internal DTD subsets, custom entity declarations and external entities
remain forbidden.
JSON source paths are checked before descending into nested values, so oversized
keys cannot accumulate every longer path prefix before the final locator check.
Manifest byte size is checked before direct ingestion retains an original.
Storage failure can still leave retained attachments; this is not an atomic
transaction across independent stores. Python workers use isolated startup
and this installation's package root. Child environments include only
execution/locale/temp settings and explicitly supported converter/Tesseract
settings. These controls are not an OS sandbox or native-memory quota.

The shared indexing path checks each embedding response before writing new
episodes: it must contain exactly one vector per requested chunk, with the
configured dimension and finite numeric values. A malformed provider response
raises `ValueError` and cannot produce a searchable receipt. Recovery performs
the same checks and keeps the interrupted-write marker until indexing succeeds,
so a corrected provider can retry. These structural checks cannot detect a
provider that returns the right number of valid vectors in the wrong order.
