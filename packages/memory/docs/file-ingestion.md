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

## Inspect a source after a keyed update

When a source was stored with `dedup_key`, read its current episode with the
same exact key. This is useful after an uncertain replacement error: the
new source may have been stored even if returning its receipt failed.

```python
from scone_memory.core.errors import Gone, NotFound

try:
    current = await memory.episode_by_key("research", "doc:observatory")
except Gone as error:
    print("The source was forgotten at", error.forgotten_at)
except NotFound:
    print("No source is recorded under this key")
else:
    print(current.episode_id, current.content, current.attachments)
```

`SyncMemoryEngine.episode_by_key(space, key)` provides the same blocking
operation. In a shell, use
`scone-memory source-key doc:observatory --space research --json`.
Successful CLI JSON contains the native Episode fields. Missing and forgotten
sources exit with code 2 and a diagnostic on stderr.

Authenticated `GET /v1/episodes/by-key?dedup_key=...` returns the same source
shape as `GET /v1/episodes/{episode_id}`. Use your HTTP client's query-parameter
encoding for keys containing spaces, `#`, `?` or other reserved characters.
The bearer key determines the space. HTTP 404 means no recorded source under
that key; HTTP 410 includes `forgotten_at`. The operation is advertised as
`episodes.by_key` in `/v1/capabilities`.

Keys are exact UTF-8 strings of 1–256 characters; whitespace and case remain
significant. A key addresses the current source, not a history of replacements
or a source stored only by content hash. The read includes attachment metadata,
does not embed or write anything, and retries observed identity changes at most
three times before raising `Conflict` (HTTP 409). It is not a lock or a transaction
with a later write; coordinate competing writers before deciding to retry an
update. Rust and the lightweight `scone-client` package do not yet expose this
native operation.

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
| DOCX | Paragraphs, typed table cells/merges, declared header rows and referenced notes | Direct source properties; no rendered layout, inherited style resolution or macros |
| XLSX | Sheet cell references, declared table headers, ranges and totals roles | Stored values; no formula execution or rendered layout |
| PPTX | Slides, table text and notes | No rendered Office layout or macro execution |
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
table evidence to presentation or delimited readers. XLSX declarations are covered below.

### Word table merges and context

DOCX tables retain the same cell evidence, including `gridSpan`, legacy horizontal
merges, vertical merges and skipped leading/trailing grid columns. Contiguous
rows marked with the direct `tblHeader` property supply column headers; bold text,
first-row styling and late header markers do not establish a header relationship.
`metadata.header_basis=word_repeating_rows` identifies this interpretation.
Both transitional and strict WordprocessingML namespaces are supported. Unknown
table, row or cell namespaces retain text with an explicit fallback.

A merged cell keeps the first cell's locator and records the additional source
cells in `merged_locators`. Its text retains the source cells' nonempty content,
separated by newlines. Spans describe the combined grid area. On later rows,
values carry `context` references to non-header cells that span into their row.
For example, `West / Revenue: €20` distinguishes the spanning data cell `West`
(`association=row_span`) from the declared column header `Revenue`. These
references survive chunk filtering even when their source cells lie in an
earlier segment. They do not reclassify data as headers.

Tables with merge-source or row-context evidence use manifest version 5;
unmerged tables use version 4. Empty new fields are omitted, preserving existing
HTML table manifest bytes. The parser identifies this extraction as
`native-xml-word-tables-v1`. Use a new durable run and parser revision when
re-extracting older flattened Word tables. Referenced notes, comments, text boxes
and relocated package members retain their existing source roles and locators.

Deleted cells are excluded. Historical property snapshots do not override
current grid properties. A tracked row-deletion marker triggers
`tracked_row_structure` text fallback: row and cell-content revision states are
independent, so the reader does not discard independently live text or claim a
resolved current grid. Nested tables and inconsistent merge continuations also
use explicit text fallback. Inherited table styles and full tracked-layout
reconciliation remain open.

Word extraction applies the same cell, column, occupied-slot, reference and
evidence bounds. It checks cumulative text and segment limits while constructing
contextual rows, including preceding document content, before retaining a result.

### Declared spreadsheet tables

XLSX worksheets resolve their `tableParts` relationships to source table
ranges and column declarations. Each retained cell in a valid declared table
carries `DocumentTableCell` evidence. Grid coordinates are relative to the table
range, while the locator retains the actual sheet and A1 reference, including
ranges near the bottom or right edge of a worksheet.

The declared header row supplies column references to its retained cells. A
value such as `Revenue: €20` keeps the label and value searchable together;
its evidence span covers only `€20`. `table_range`, `table_name`, `table_member`,
`member`, `header_basis=xlsx_table_declaration`, and `table_role` (`header`,
`data`, or `totals`) preserve the source interpretation. Cached formula values
keep `formula=cached-value`; formulas are never recalculated. A formula without
a saved result produces no invented value and adds `missing_cached_formula`
to the table's extraction notes.

`headerRowCount=0` leaves the first data row as data and records
`header_row_absent`. Missing header cells add `missing_header_cell`; differing
column-declaration names add `column_name_mismatch`. Header text always comes
from the retained worksheet cell, not a replacement label from the declaration.
Ordinary worksheet cells outside declared tables keep their existing text and
locators. First-row styling alone does not declare a header.

Malformed declarations, overlapping table ranges, merges intersecting a table,
contradictory cell coordinates, duplicate cells (including empty cells), or
unsupported worksheet/shared-string markup retain plain extracted text with
`table_status=text_fallback` and a reason. Every character used as an inline or
shared-string header must belong to its supported source string structure.
Unsafe or missing package relationships remain explicit parser errors.

The declaration reader permits at most 1,000 tables, 100,000 inspected worksheet
cells and 100,000 declared grid slots per worksheet, within document-wide cell
and evidence limits. Oversized table ranges and excessive merge comparisons
fall back explicitly; extracted-text, segment and wall-time limits still apply.
Header expansion is checked against the cumulative text limit during emission.

Structured XLSX tables use manifest version 4 and parser
`native-xml-xlsx-tables-v1`. Durable extraction checkpoints and filtered citations
retain the same header evidence. Use a new run and parser revision to re-extract
older flattened workbooks. General worksheet header inference, merged layouts
outside declared tables, number-format rendering, XLS/XLSB table structure and
spreadsheet image/chart interpretation remain separate gaps.

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
New source episodes also retain extraction labels of at most 256 characters in
`metadata.document_filename`, within the existing metadata value limit;
the source inventory exposes it as an optional `document_filename` display label.
The label does not change the original/manifest deduplication key or grant access
to a file. Replaying an older source preserves its recorded metadata; its exact
extraction filename remains available through document provenance. Longer labels
also remain complete in the manifest and provenance without an inventory label.
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

## Preparing keyed source updates

For caller-managed source text, `MemoryEngine.replace(space, Record(...,
dedup_key=...))` and `remember(..., dedup_key=..., replace=True)` prepare the new
record before removing the old one. The HTTP `/v1/episodes` replacement option
and CLI `remember --key ... --replace` use this same path.

Validation, chunking and all embedding batches finish first. Invalid input,
provider failure, invalid embedding vectors or cancellation during preparation
leave the old episode, its chunks, vectors and linked originals available.
Even a same-text duplicate validates its input. A replacement key determines
its identity; the import-only `Record.content_hash` override is refused here.
Source strings, tags and metadata must be valid UTF-8 on every backend.

```python
from scone_memory import Record

result = await memory.replace(
    "research",
    Record("The observatory moved to Porto.", kind="file",
           source="observatory.txt", dedup_key="document:observatory"),
)
print(result.outcome)  # accepted, duplicate, or updated
```

An update checks the key's current episode and latest forget receipt again after
embedding. A source changed or forgotten during preparation causes refusal,
including an initially absent key that another caller created and forgot in the
meantime. Changing the engine's embedding/chunk configuration or store references
during preparation also causes refusal. A new intentional attempt after a forget
is allowed. Mutable caller tags and metadata are copied during validation.

The final forget/store phase is **not an atomic swap**. Serialize competing
commits for a key; storage failure after forgetting can leave no current record
or an incompletely reported new record. The error names the removed episode and
asks the caller to inspect the current key before retrying. It does not claim a
key is empty when only its revision receipt failed. Claims citing the old source
continue to stand under the existing forget policy.

This improves the existing keyed text-update API. For automatic local file
revision tracking and explicit missing-file deletion, use the separate
[directory synchronization workflow](directory-sync.md). Retaining inspectable
revision history, metadata-only updates and transactional attachment transfer
remain separate gaps; ordinary content-addressed document ingestion does not
itself manage an external source's current revision.

### Inspecting source removal

Servers with `episodes.forget: true` in `/v1/capabilities` expose the complete
source-removal workflow. This flag describes implementation support; the key's
role still determines whether DELETE is allowed.

- `GET /v1/episodes/{id}/impact` previews chunks and attachments removed and
  citing facts and links retained. It does not reserve a snapshot against writes.
- `DELETE /v1/episodes/{id}` removes the source through durable cleanup and
  returns a receipt. Repeating DELETE can finish an interrupted cleanup.
- `GET /v1/episodes/{id}/forget-status` reads `present`, `pending`, or `forgotten`
  without starting or resuming cleanup. The engine equivalent is
  `await memory.forget_status(space, episode_id)`.

A pending status includes the original `requested_at` and `impact`. It takes
precedence over an existing tombstone until all cleanup steps acknowledge
completion. A forgotten status includes `forgotten_at` but no full receipt:
the tombstone does not retain the original attachment or citing-claim inventory.
Unknown IDs and IDs belonging to another space return 404. Valid read keys may
inspect status and impact; keys without write permission receive 403 on DELETE.
Custom document stores must implement the callable durable-retirement protocol
before this workflow is advertised.

After an interrupted response, read status before deciding whether to resume.
Status is a point-in-time observation, not a transaction or a lock against later
writes. Browser and transport stacks may replay idempotent DELETE requests after
connection failures; the durable retirement identity binds retries to the same
source. Clients should not automatically initiate another removal attempt.

Source removal leaves citing claims, links and their stored quotes in the ledger.
Shared attachments remain where another source in the space carries them.
Downloaded copies and backups are outside this action; this is not a promise of
complete erasure from every storage location.

## Select local OCR when importing PDFs

The ordinary `POST /v1/documents` route extracts embedded PDF text by default.
An operator can enable the existing local OCR parser in both `scone serve`
compositions by configuring an installed Tesseract executable:

```sh
export SCONE_DOCUMENT_OCR_EXECUTABLE=/absolute/path/to/tesseract
export SCONE_DOCUMENT_OCR_LANGUAGE=eng
export SCONE_DOCUMENT_OCR_PSM=3
export SCONE_DOCUMENT_OCR_DPI=150
scone serve
```

The `pdf-ocr` Python extra and the selected Tesseract language data must already
be installed. No model, language pack or executable is downloaded or started at
server startup. The executable must be an absolute path to an executable file.
Language defaults to `eng`; PSM defaults to 3 and accepts 3, 6, 11 or 12; DPI
defaults to 150 and is bounded to 72–300. Unsupported settings or missing PDF
rendering dependencies refuse startup. Installed dependencies and configuration
do not promise that language data, a particular file or recognition will work.

Authenticated `GET /v1/documents/formats` includes `pdf_ocr.available`, `modes`
and `reading_orders`. A native host may instead supply
`document_ocr=DocumentOcr(my_recognizer, dpi=150)` to `create_app` or
`create_conversation_app`, using `scone_memory.ingestion.document_ocr.DocumentOcr`
and an existing `OcrEngine` implementation. The caller owns that recognizer.

After uploading the original, select OCR explicitly in the indexing request:

```json
{
  "attachment_id": "<original SHA-256>",
  "filename": "scan.pdf",
  "pdf_ocr": {"mode": "missing_text", "reading_order": "columns_ltr"}
}
```

`missing_text` preserves readable embedded text and recognizes pages lacking it
or whose text extraction fails. `all_pages` recognizes every page, including
those with embedded text. Reading order is `provider`, `columns_ltr` or
`columns_rtl`; the latter two infer columns geometrically, not semantically.
The browser Documents import queue exposes these choices per PDF when available.

The response echoes the selected `pdf_ocr`; the retained manifest records mode,
reading order and actual DPI in `parsed.metadata.pdf_ocr`. Verified document
provenance exposes that same metadata beside actual page extraction methods,
recognizer names, region geometry and UTF-8 spans. Different selections have
distinct manifest identities even if the resulting text is identical. Existing
imports that omit OCR keep their original content identities. OCR does not
establish the correctness of recognized text; inspect source evidence.

Selecting OCR on another file type or on an unconfigured server fails before
extraction. Recognition errors do not fall back to another provider or silently
accept a partial extraction. The existing bounded ingestion lane, 30-second
extraction deadline, pixel/region limits and original-backed indexing apply.
These HTTP imports remain synchronous: after an uncertain write, inspect the
source library instead of automatically repeating the import. Native per-page
checkpoint recovery remains the separate `PdfOcrWorkflow` interface.


## Durable local document jobs

The standard memory and conversation hosts can own document imports independently
of an HTTP connection. Enable this with `SCONE_DOCUMENT_JOBS_CONFIG` pointing to
an owned, regular **0600 JSON file**, containing:

```json
{
  "schema_version": 1,
  "state_dir": "document-jobs",
  "key_env": "SCONE_DOCUMENT_JOBS_KEY",
  "parser_revision": "installed-parser-v1",
  "max_active": 2,
  "max_imports": 4096,
  "max_attempts": 3,
  "deadline_s": 120.0
}
```

Set the named environment variable to a separately generated 32-byte encryption
key encoded as 64 hexadecimal characters. Keep it outside configuration and source
control; retain it to reopen saved requests and journals. The state directory is
local, owned and private; relative paths resolve beside the configuration file.
Startup opens encrypted state but does not replay imports or call models. No
cloud queue or external worker is required.

`parser_revision` is an operator promise: change it when OCR executables, trained
language data or custom extraction behavior change. Installed Python dependency
versions, OCR settings and the operator revision bind the parser used by each
request. An incompatible parser refuses to resume an old job. Optional `limits`
uses `DocumentLimits`; requests freeze those limits, deadline and attempt budget.
Changing the host budget does not grant old requests more attempts.

Upload an original to `/v1/attachments`, then use a unique import ID:

```http
POST /v1/document-jobs
Content-Type: application/json
Authorization: Bearer <write key>

{"import_id":"report-2026-09","attachment_id":"<SHA-256>","filename":"report.pdf"}
```

Add the same optional `pdf_ocr` selection as synchronous document imports. The
202 response acknowledges ownership, not completed indexing. Repeating an ID
with identical input only reads its current state; changed input is a conflict.
An admitted task continues after a browser disconnect. Unuploaded local files
are not durable jobs.

- `GET /v1/document-jobs?limit=20&after=<cursor>` pages saved job statuses within
  the authenticated space. Cursor order is stable opaque identity order, not a
  completion ranking or a frozen snapshot.
- `GET /v1/document-jobs/{id}` reads stage progress; `/request` reads the immutable
  original filename, OCR choices, parser identity and execution limits.
- `GET /v1/document-jobs/{id}/result` verifies retained original and indexed source
  evidence before returning a receipt. It never extracts, indexes or resumes.
- `POST /v1/document-jobs/{id}/resume` and `/cancel` require JSON
  `{"expected_revision":1}` using the current control revision. Read-only keys
  cannot mutate jobs, and stale controls cannot affect newer attempts.

After a restart, incomplete jobs remain passive until an explicit resume. A
resume reuses completed extraction/indexing stages and saved embedding batches;
these generic document jobs do **not** promise per-page OCR recovery. Each failed
stage waits for an explicit resume, within the saved attempt budget (1–4 total
admissions). Async cancellation and deadlines are cooperative. Cancellation may
follow a partial write; inspect status and retained source evidence before
assuming no work occurred. Cancellation stops owned work even if writing its
intent fails, while reporting that storage failure.

A per-job local file lock prevents simultaneous execution by another process.
Capacity is bounded per service; excess admission returns 429 with Retry-After.
This is local task ownership, not a distributed queue. Request/result reads do
not start jobs. A confirmed forgotten source invalidates completed evidence;
a temporary verification outage refuses the result without replaying work.

The capability `documents.jobs` is advertised only when configured. The existing
synchronous `/v1/documents` endpoint remains available.
