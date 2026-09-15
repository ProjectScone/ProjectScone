# Scanned PDF ingestion with source regions

Scone's opt-in OCR pipeline renders PDF pages, invokes an explicitly configured
recognizer, and indexes recognized text with page and region provenance. It uses
no generative LLM, PaddleOCR runtime, PaddleOCR weights, network endpoint, or
implicit model download. The included baseline invokes an installed Tesseract
executable and its installed language data. This is a Scone-owned processing
pipeline, not a Scone-trained recognition model.

```sh
pip install 'scone-memory[pdf-ocr]'
# Separately provision Tesseract and the language data you trust through your
# deployment's package-management process. The Python package never installs it.
```

```python
from pathlib import Path
from scone_memory.ingestion import OcrPdfOptions, OcrPdfParser, PdfLimits, ingest_pdf, pdf_provenance
from scone_memory.ocr import TesseractOcr

parser = OcrPdfParser(
    TesseractOcr(language="eng", page_segmentation=3),
    options=OcrPdfOptions(mode="missing_text", dpi=150),
)
result = await ingest_pdf(
    memory, "research", Path("scan.pdf").read_bytes(),
    parser=parser, limits=PdfLimits(timeout_seconds=60.0),
)
recall = await memory.recall("research", "What calibrates Polaris?")
for item in recall.items:
    if item.episode_id != result.added.episode_id:
        continue
    evidence = await pdf_provenance(memory, "research", item.episode_id, chunk_id=item.chunk_id)
    for page in evidence.pages:
        for region in page.regions:
            print(page.number, region.text, region.box, region.start, region.end)
```

## Pages scanned the wrong way up

A page scanned upside down or on its side is read as noise: measured on a
rendered page turned 180 degrees, Tesseract returned "aun Ul 8dJJO INOqUeY".
Page segmentation mode 1 turns a sideways page but not an upside-down one.
`TesseractOcr(orientation=True)` (`SCONE_DOCUMENT_OCR_ORIENTATION=1` for the
hosted document OCR) runs Tesseract's orientation detection (`--psm 0`, which
needs the `osd` language data) before reading a page. When the detection is
at least `min_orientation_confidence` sure (2.0 by default) that the page is
turned a quarter, half or three quarters, the page is turned in a bounded
child process, read, and every box is given back in the page as it was given,
so regions still match the page's own geometry. The engine name says what
happened: `:rotated90`, `:rotated180` or `:rotated270`; `:osd-upright`;
`:osd-unsure` when a turn was detected below the confidence; `:osd-unknown`
when the detection could not judge, as on a page with too few characters.
Without the `osd` data installed the detection always fails, so a failed
detection checks the installed languages once and, when `osd` is missing,
refuses the page with that reason instead of reading every page as one it
could not judge. A language list too long to fit the engine name with the
orientation suffix (96 characters) is refused when the engine is built.
Each checked page costs one more Tesseract run. With the setting on, the
hosted OCR identity used by document jobs and directory sync ends
`:orientation`, so documents extracted before are extracted again.

## Coverage and geometry

The default `missing_text` mode preserves native text and only recognizes pages
without extractable text, or whose text layer is unreadable: mostly private-use
code points, `(cid:N)` runs, replacement characters or control bytes, the way a
font without a usable Unicode map extracts. A page with a short text layer over a scan is therefore
not automatically recognized. Choose `all_pages` explicitly to replace all native
text with OCR; this mode reads page metadata without extracting discarded text.
The default text-only `PypdfParser` still requires an existing text layer.
Pages with no content stream are valid empty pages. In OCR mode, a page-level
text extraction failure also falls back to recognizing that page; successful
native pages keep their text. The parser identity records when this fallback
was needed. Encrypted/corrupt document structure, invalid page geometry and
resource-limit violations still fail; OCR does not bypass those checks.

By default OCR regions retain the recognizer's block and line order. Their half-open `start`
and `end` offsets index UTF-8 bytes in the searchable document, including preceding
pages. Boxes are `(left, top, right, bottom)`, normalized to `[0, 1]` in the
**displayed, rendered page with a top-left origin**. This coordinate frame includes
PDF page rotation and cropping. It differs from `width_points` and `height_points`,
which describe the unrotated PDF media box. Do not apply those physical dimensions
to OCR boxes without the appropriate page transform.

The provenance resolver returns complete overlapping pages and their regions.
Filter regions against a retrieved span when highlighting only that passage.
Tesseract regions are words; other implementations may return lines. Recognizer
scores are retained on a `[0, 1]` scale without dropping low-score words. They do
not express the probability that a claim is true or that transcription is exact.

Coverage labels are `text_layer`, `ocr`, `mixed`, or `partial` when pages remain
empty. Blank pages and failed recognition cannot be distinguished solely by empty
output. An entirely empty result fails before storage. Original PDF bytes remain
unchanged. Provider-order OCR uses manifest schema v2; inferred column order uses
v3. Native-text v1 manifest serialization and
deduplication identity remain unchanged. Existing v1 manifests stay readable.

## Runtime boundaries

- PDF rendering and Tesseract run in separate child processes; there is no shell
  interpolation. Tesseract's language selection cannot inject paths or flags.
- Python workers use isolated startup and this installation's package root,
  excluding the working directory and inherited `PYTHONPATH`. Child processes
  receive an allowlist of execution, locale, temporary-directory and configured
  Tesseract/converter settings, rather than the server's entire environment.
  Rendered PNGs record their actual DPI for the recognizer.
- The direct `OcrPdfParser` has one whole-document wall deadline. Timeout or caller cancellation stops
  processing before ingestion writes. POSIX workers run in dedicated process
  groups, so wrapper descendants are terminated too. On Windows only direct child
  termination is supported; configure a direct executable there.
- Defaults are 150 DPI, at most 20 million rendered pixels per page and 10,000
  recognized regions per page. DPI is bounded to 72–300; the region ceiling is
  50,000. Rendering checks dimensions before allocating the bitmap. Input, output,
  extracted text, page counts and subprocess output have explicit bounds.
  The serialized provenance manifest must fit the attachment byte limit; direct
  ingestion checks it before retaining the original. Recognition-provider
  timeouts and invalid results are classified separately from document expiry.
- Pages run sequentially, limiting simultaneous raster allocations. These bounds
  are not a native-library memory quota or OS sandbox. Compressed PDF/image inputs
  still require patched dependencies and deployment-level memory/concurrency
  limits. A trusted custom `OcrEngine` is responsible for cancelling its own work.
- Debug logging records page number, configured engine, region count and elapsed
  recognition time; it does not log document text or images. Process failures are
  ingestion errors, not memory-store outage signals.

## Estimate column reading order

Select `OcrPdfOptions(reading_order="columns_ltr")` for left-to-right columns or
`"columns_rtl"` for right-to-left columns. The default `"provider"` preserves
the recognizer's order and existing manifest/checkpoint identities.

The original whitespace algorithm groups adjacent words on observed lines,
then looks for column gutters at least six percent of the displayed page width.
Both sides must contain at least three non-overlapping text rows and overlap
vertically. Spanning text may precede or follow the columns; a crossing region
inside their body prevents that split. Within each recovered column, words
retain provider order. Column direction does not reverse letters or words.

The source manifest records `reading_order` with the strategy, direction,
column count and notes. Each output region keeps its unchanged text, box and
recognizer score, plus `provider_index` (zero-based original position) and
`reading_column` (one-based inferred column, or zero for spanning/unassigned
text). UTF-8 offsets describe the reordered searchable text. PDF manifests and
generic PDF-document manifests use version three for these observations;
existing versions one and two remain readable and byte-stable by default.
Generic document citations carry the receipt in `metadata.ocr_reading_order`;
filtered citations preserve original provider indices, without renumbering.

`geometry_inferred` marks an estimate. `no_separating_gutter` means the provider
order was retained. The search considers up to 32 candidate gaps per partition
and at most eight columns; `candidate_limit` and `column_limit` disclose those
bounds. This does not identify tables, captions, sidebars or semantic headings.
Tables and mixed layouts can resemble columns, so select this strategy only
for appropriate documents and inspect the recorded geometry. Dense pages,
rotated text, narrow gutters and mixed column bands may need another layout
implementation. No models are downloaded or invoked by this ordering step.

The selected strategy is part of OCR workflow binding. Changing it requires a
new run id; extraction/indexing retries preserve the chosen order and source
mapping. Page OCR checkpoints retain the original recognizer observations,
then deterministically rebuild the ordered text on resume.

## Label the page's regions

Every recognized page's regions, and every text-layer page laid out in
reading order or kept whole with a table in it, carry a `label` saying
what the region is, from one
vocabulary: `title`, `heading`, `paragraph`, `list`, `table`, `figure`,
`caption`, `header`, `footer`, `page_number`, `footnote`, `formula`,
`code`, `sidebar`, `reference`. The page's `labels` receipt (and the
segment's `layout_labels` metadata) says where they came from.

Without a layout engine the labels are inferred (`labels-v1`) from what
geometry and text can tell, and the receipt names the rules that fired:
a page number by its shape in the page's top or bottom strip; a running
`header` or `footer` by a line recurring at the edge of three pages or
more, every page but the first it appears on; a `table` by the aligned
grid the table inference finds; a `list` by lines opening with a bullet,
a number or a letter and its mark, two or more of them; a `footnote` by
its mark at the bottom of the page; a `title` on the first page, in its
top third, by a line 1.4 times the page's median line height (or, on a
text layer's grid, which has no sizes, by its isolation); a `heading` by
1.25 times the median height or, when short and unpunctuated, by the
space above and below it. Everything else is a `paragraph`. A figure, a
formula, a caption or a sidebar is only an engine's to say. A page past
the line bound is left unlabelled and says so.

With a layout engine, each region takes the label of the engine box it
lies in most (`engine-boxes-v1`); a region outside every box is
unlabelled and counted. The seam is `ocr.types.LayoutEngine`: `analyse(
image) -> LayoutResult` of labelled, normalized boxes. The host runs one
as an executable that reads a PNG on standard input and answers in JSON
lines -- the page's size, then one region per line with `label`, a
pixel `box`, and optional `score` and `order` -- configured by
`SCONE_DOCUMENT_LAYOUT_EXECUTABLE`. Labels may be this vocabulary's or
PP-Structure's and deepdoc's words (`doc_title`, `paragraph_title`,
`text`, `figure caption`, `vision_footnote`, `equation`, …); a label
with no name here is dropped and counted. A size that does not match the
page, a box off the page or a line that is not JSON refuses the page.
The engine runs under the page's deadline; its answer is kept in the
page's OCR checkpoint and in the checkpoint binding, so pages recognized
without an engine are not reused with one. `ocr_choices` says `layout:
engine` or `layout: inferred`.

## Inspect possible tables without repeating OCR

The `aligned-rows-v2` strategy groups retained OCR rectangles into
candidate cells and checks for common column gaps across at least three
consecutive rows. A currency sign on its own, set apart from the number
to its right as a statement sets them, is that number's cell, and the
gutter before the cell runs to the number. A row of fewer cells beside
the full rows -- a title across the table, a "Total" beside two numbers
-- is read as cells that span the grid's columns, each over the
contiguous columns its own width covers (`column_span`: a column inside
the cell or the cell inside it, or the two overlapping by half of the
narrower and a tenth of the wider), judged against the full rows'
column extents; nothing narrower is widened, and a row that fits no
column stays unassigned. The rows of fewer cells just above the first
full row are the grid's own too -- a title across it, a caption over
its value columns ("Three Months Ended March 31," with the years below,
placed loosely: a caption centred over two columns falls short of half
of the last, so a tenth of its own width claims one), the years over a
blank label column -- placed the nearest first, each just above the one
below, up to three (`MAX_HEADER_ROWS`); a run of two rows too short to
be a grid is read that way as well, so a row of years displaced by the
three-column row beneath it heads that grid. Prose is kept out by its
shape: a sentence above the grid (a citation like `[28]` closing it
counts) is not its title, nor is a wrapped word over a later column
(a lone cell over one column short of the first); at the grid's foot a
row opening with
a footnote's mark or holding a line of prose (a sentence, six words or
forty characters) is its note or the prose below it, not its last row,
while a total, a subtotal across its numbers or a wrapped word is; three rows of
fewer cells in a row are prose beside the grid, which ends before them;
and two columns of lines about as wide as each other are prose side by
side, not a table of two columns (a notation list's first column is
narrow and ragged); a list aligns as a grid of two as well, and is not
one: a column of one bullet glyph on every row beside its items, or a
column of enumerators with their closing punctuation (`1.`, `(a)`,
`(ii)`, `2.1.`, `32.1*`), or of dashes or asterisks, beside lines of
prose, is a list, while enumerators, dashes (none) or asterisks (a
note) beside short labels, or plain numbers, are a table's first
column. All stay unassigned, and a bulleted or enumerated list is then
labelled `list` by the label rules, which read the same marks. The layout's `notes` say `header_rows_limit`
when the rows above a grid were read up to `MAX_HEADER_ROWS` and a row
above them was not. It returns `geometry_inferred` results; it does not identify
semantic headers, multi-line cells or missing values. Aligned prose can
resemble a table, and irregular tables can remain unassigned. A layout
stored under `aligned-rows-v1` is one read before spans were.

A page's candidates also reach the document: the segment of a
recognized page, or of a text-layer page whose runs the rules read as
a table (laid out in columns or kept whole), carries them as
`table_cells` the way the DOCX and HTML readers give
theirs (`DocumentTableCell` with row, column, `column_span`, and a byte
span of the segment's text holding exactly the cell's words), located
`page:N/table:T/cell:R,C`, and validated the same way. Only the regions
labelled `table` are read for cells; a page whose labels name no table
carries none, whatever a grid over its lines would propose. The cells
come in the order the page's text reads them, each keeping its row and
column, so a table a recognizer read column by column cites its columns
in turn; a table whose cells do not sit together in the page's text is
left out rather than cited wrongly, and the segment's `tables` and
`tables_unreadable` metadata count both. A cell's text is the page's
bytes, spaces and all, so a statement's `$` stands apart from its number
as on the page (the table query reads `$        2,903` as it reads
`$2,903`). A table's header is read from its shape: the first row
filling every column, or every column but the first (a statement's
years over its blank label column), is the header when none of its
cells is a value -- a number, a loss in parentheses, a percentage or a
dash, while a bare year is a label -- and a column below it is mostly
values; its cells say `is_header`, every cell below carries a
`column` reference to the header over it, and the segment says
`header_basis: pdf_first_row` and counts `tables_headed`, so the table
query names the columns. A table of words alone gets no header, as
nothing tells one from a first row. No row span or empty cell is
invented.

```python
from scone_memory.ocr import infer_tables

layout = infer_tables(page.regions)
for table in layout.tables:
    for cell in table.cells:
        print(cell.row, cell.column, cell.text, cell.regions)
print("Unassigned source regions:", layout.unassigned)
```

`cell.regions` and `unassigned` are zero-based indices into the supplied page's
region sequence. Every region occurs exactly once across those two groups.
Cell text joins its referenced observations with spaces; the original region
text, UTF-8 offsets and displayed-page boxes remain the evidence. Rows and
columns are zero-based. Candidate cells are returned in row-major order.

The HTTP API advertises `documents.ocr.tables` and provides an authenticated,
read-only `GET /v1/episodes/{episode_id}/document/ocr-tables?page=1` for retained
PDF documents with OCR provenance. The response binds the layout to the space,
episode, page, original and manifest SHA-256 identifiers, and page text SHA-256.
It validates retained provenance and rechecks the episode and current key scope
before returning. Source deletion or changed provenance prevents publication.
This route does not invoke OCR, update memory, or save an analysis receipt.

The console offers explicit analysis, paged candidate rows, source-region
selection, an unassigned-text inventory, and a JSON download with source
references. It verifies response binding and region coverage before displaying
results. Users should inspect the scan and referenced regions before relying
on the inferred cells.

Analysis accepts at most 5,000 regions and 2,000,000 UTF-8 text bytes per page,
64 candidates, 1,000 rows per candidate and 2–12 columns. Exceeding an analysis
limit fails explicitly rather than publishing a partial inventory. The function
requires no OCR executable, model, network service or optional rendering package.

## Resume completed pages after interruption

The generic `DocumentIngestionWorkflow` and configured HTTP document jobs now
retain completed OCR page observations inside their existing extraction-step
journal; see [generic document recovery](file-ingestion.md#durable-extraction-checkpoints).
They keep the whole-extraction deadline per attempt and the generic document
manifest. The separate interface below provides independently timed page steps
and PDF-specific receipts.

Install `scone-memory[pdf-ocr,agents]` and use `PdfOcrWorkflow` for scans that
need durable page progress. Retain the source first and supply a persistent
32-byte journal key and an application revision covering the recognizer's
model, language and settings. No models or services are downloaded or started.

```python
from scone_memory.ingestion import PdfOcrWorkflow, OcrPdfOptions, PdfLimits

original = await memory.attach(
    "research", Path("scan.pdf").read_bytes(), "application/pdf", filename="scan.pdf",
)
job = PdfOcrWorkflow(
    memory, "scan-journal.db", key=checkpoint_key,
    engine=TesseractOcr(language="eng", page_segmentation=6),
    recognizer_revision="tesseract-eng-psm6-deployment-v1",
    options=OcrPdfOptions(mode="missing_text"),
    limits=PdfLimits(timeout_seconds=60.0),
    index_timeout_seconds=120.0,
)
try:
    result = await job.run("scan-42", space="research", attachment_id=original.attachment_id)
    print(result.added.episode_id, result.reused_pages, result.reused_index)
finally:
    job.close()
```

Reopen the same journal and persistent memory stores, using the same key and
configuration, then call `run` with the same identifiers to resume. Completed
pages reuse their full text and region geometry without rendering or recognizing
them again. Native text stays native in `missing_text` mode. Assembly rebuilds
absolute UTF-8 spans and validates the final PDF manifest before indexing.

`page_status(run_id, space=..., attachment_id=..., page=1)` reports committed
page state, attempt count and error class without exposing source text. It
returns `None` for a page that has no OCR receipt, including native-text pages.
The index step also retains validated embedding batches in its encrypted
journal. Retrying after interruption during embedding skips batches whose
source, chunk spans, embedding inputs and model identity still match. This does
not persist a chunk plan or change engine recovery after episode writes begin;
see [embedding recovery and limits](file-ingestion.md#durable-extraction-checkpoints).

`reused_pages` lists reused OCR page numbers; `reused_index` reports whether the
final indexing receipt was reused. A reused receipt retains its original `Added`
fields; it does not issue another `remember` call.

Source binding is committed before page inspection, even if no pages need OCR.
The binding includes the source identity, space, render options, limits,
recognizer revision, and installed PDF/parser renderer versions. Changing those
under the same run ID fails. Use a new run ID for a deliberate new extraction.
Source checks run before and after page/index steps, including reused receipts.
Observed source loss permanently invalidates that run; reattaching bytes does
not reactivate it. Replaying a completed run also validates the retained episode
and provenance, so forgetting its indexed evidence cannot silently restore it.

Each page has its own wall deadline; indexing has a separate deadline. There is
no shared whole-document deadline in this workflow. The caller may still cancel
the coroutine or impose its own total deadline. A killed process releases the
journal lock; retry reuses committed pages and repeats the interrupted page or
index step. `max_retries` bounds repeat attempts per stage (default one retry).
Async providers must propagate cancellation.

Page results are encrypted inside the local journal and are limited by
`max_checkpoint_bytes` (default 1,000,000 bytes, including serialization and
binding overhead). A larger result fails without indexing a partial document.
The journal is application-owned retention: forgetting memory blocks reuse but
does not erase every encrypted page receipt. Delete the journal and its key when
its retention period ends. This workflow does not provide a background queue,
distributed leases, automatic HTTP integration, or separate durable chunking
and embedding stages. Rendering still starts one isolated worker per new OCR
page; page checkpoints improve recovery, not recognizer accuracy or fresh-run
rendering throughput.

## Extending and evaluating recognition

Implement the `OcrEngine` protocol to supply a different approved recognizer. It
accepts PNG bytes plus pixel, region and time limits and returns an `OcrResult`.
It must preserve image dimensions and return finite normalized boxes and scores.
Scone validates the result and builds the searchable text and source spans.
No model provider is selected or downloaded on the caller's behalf.

The implementation draws architectural inspiration from separate OCR processing
stages and geometry preservation in the read-only PaddleOCR reference. No source
code, runtime, model, or weight from that project is included. Its leaderboard
results do not apply to Scone. This milestone does not implement neural detection,
learned document layout, semantic table structure recovery, or generative document parsing.

Generated PDF/image fixtures exercise real Tesseract recognition and source
resolution, mixed native/scanned pages, UTF-8 offsets, invalid output, deadlines,
wrapper cleanup, source identity and resource bounds. They are integration tests,
not a representative accuracy benchmark or evidence of improvement over Tesseract.
Compare fixed real documents and reference transcriptions before claiming gains.

```sh
pytest -q tests/ingestion/test_ocr_contract.py tests/ingestion/test_ocr_runtime.py tests/ingestion/test_pdf_ocr.py tests/ingestion/test_pdf_ingestion.py
```

PDF rasterization uses the optional [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)
package. Recognition uses [Tesseract](https://github.com/tesseract-ocr/tesseract).
Their licenses and dependency notices remain with their respective distributions.
