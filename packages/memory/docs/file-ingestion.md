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
recognizer score, block/line identifiers (and a `paragraph` number when the
engine reports one) and half-open `start`/`end` offsets in the **segment's
UTF-8 bytes**.

Recognized words are kept as the lines and paragraphs they were read in:
words on a line are joined by a space, lines by a line break and paragraphs
by a blank line. An image read by an engine that reports its layout, as
Tesseract does, is one segment per paragraph (`frame:N/paragraph:M`, with the
paragraph's box, block and word count in its metadata), each word a region
spanning its own bytes. Before this, every Tesseract word was its own
segment, so two short paragraphs were stored as 31 one-word paragraphs. An
engine that reports no block, paragraph or line is read as before, one
segment per region (`frame:N/region:M`). A PDF page recognized by OCR puts
the same breaks in its page text, and its parser identity ends
`scone-ocr-v2`. `coordinate_space` identifies the displayed
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

## Claims from stored source files

A source file, or a manifest the text reader keeps line by line
(`pyproject.toml`, `Cargo.toml`, `requirements*.txt`), ingested as a document
says what it defines, imports, calls and depends on, recorded as claims cited
to the document's episode and quoted from its lines (at most 2,000 characters
of a line), under the filename it was stored with; the receipt's `claims`
counts them, and it is zero for every other document. A `package.json` is
walked as JSON, not kept as lines, so its dependencies are read through `map`
rather than from a stored document; so is an MCP configuration in JSON
(`.mcp.json` and its kin), while Codex's `.codex/config.toml` is kept as
lines and read from the stored document like any manifest in TOML.
The reader sees the document's segments one per line, so every quote is a
line the episode holds. A name given bare (`utils.py`) names a bare module;
give the path from the project root, as the directory sync does, for the
module the graph's other files name.

## Note front matter

A Markdown note that opens with a `---` line and closes it (`---` or `...`)
within 200 lines (Obsidian, Jekyll, Hugo, Zettlr) has its front matter read
into the document's metadata and kept out of the note's lines, whose numbers
stay those of the file; a note that is only front matter keeps the block as
its text. Scalars (a trailing ` #` comment cut, quotes removed), inline lists
(`[a, b]`) and item lists (`- item`, indented or not) are read; `title`,
`tags` and `aliases` keep their names, lists join with commas, and every
other key is `frontmatter_<key>` (lowercased, `-` as `_`), so a note's
`source:` cannot pass for the engine's. Sixteen keys are kept. What is data
the reader cannot keep is counted in `frontmatter_skipped`, never guessed
at: a nested mapping, a list of mappings, a folded or literal block scalar,
a value over 256 characters, a key past the bound or repeated, and a key
that names one of the reader's own counters; a blank line, a comment and an
empty value are not data and are not counted. `frontmatter_keys` says how
many landed. Only `.md`, `.markdown` and `.mdx` are read this way.

## Coverage

| Reader | Evidence retained | Limits |
|---|---|---|
| Text, Markdown and code files | Line locators; a Markdown note's YAML front matter as metadata (`title`, `tags`, `aliases`, `frontmatter_<key>`) | Source text only; no AST or semantic code graph; front matter read without a YAML parser: scalars and lists, nested mappings counted as skipped |
| JSON/JSONL/NDJSON, CSV/TSV, XML | JSON paths, rows/cells or XML locators | No schema-specific semantic interpretation |
| IPYNB v4 | Cell sources and saved text outputs with JSON Pointer locators | No code execution, image-output analysis, or legacy v3 conversion |
| HTML | Visible text, table cells, spans and source-linked headers; declared headings, list items, captions and preformatted blocks | Bounded parser; no browser execution, stylesheets or remote resource fetching |
| DOCX, and DOCM, DOTX, DOTM | Paragraphs with declared heading, list and caption roles, typed table cells/merges, declared header rows, referenced notes and charts' cached series | Direct run properties; styles and numbering are followed only to name a paragraph's role and heading level; no rendered layout or macros |
| XLSX, and XLSM, XLTX, XLTM | Sheet cell references, declared table headers, ranges and totals roles | Stored values; no formula execution or rendered layout |
| PPTX, and PPTM, POTX, POTM, PPSX, PPSM | Slides, table text, notes and charts' cached series | No rendered Office layout or macro execution |
| ODT, ODS, ODP, EPUB | Format-local segment locators | Text extraction; no rendered layout |
| HWPX | Sections and paragraphs in order; a table's cells, and the paragraphs of a header, footer, note, caption or text box, after the paragraph that holds them (`content_role`, `parent_locator`) | Hangul's XML package (KS X 6101) only; the binary HWP 5 container is not read; pictures, charts and fields are not read; no rendered layout |
| EML | Message-part locators | No recursive attachment ingestion |
| RTF, XLS/XLSB, MSG | Converter/reader locators | Optional dependencies; message attachments are not extracted |
| DOC, PPT | Converted text locators | Explicit offline converter; macOS textutil also supports DOC; page/slide structure may be lost |
| PDF | Page locators, extraction method, configured OCR regions and engine, the section its bookmarks put each page in | Native text by default; OCR requires an explicit parser |
| Images | Frame/region locators and typed OCR geometry | Explicit `ImageDocumentParser` and OCR engine required |
| Audio/video | Audio-stream timestamps | Explicit `MediaDocumentParser` and transcription provider required; video frames are not analyzed |

The macro-enabled, template and slideshow variants of Word, Excel and
PowerPoint files are the same package as the plain one with another content
type on its main part, and are read alike, under their own extension
(`document_format` says `docm`). Macros a package carries are neither run nor
read; a document that carried them says so in its metadata (`macros:
present, not read`), so a search over a folder of macro-enabled files can
tell which ones held code.

A paragraph a document marks as a heading carries `heading_level` in its
segment's metadata, as a decimal string, with its text unchanged. For DOCX the
level is 1 to 9 and comes from the paragraph's own outline level, else from its
style's (followed through the styles it is based on) or from a built-in style
named `heading N` or `Title`, whatever the style's id is in the document's
language (see Declared blocks below, which also gives `block_role` and
`heading_basis`). For ODT it is 1 to 10, from `text:h` and its outline level (1
when none is given). For HTML it is 1 to 6, from `h1` to `h6`. A DOCX without a
readable styles part still reads; its paragraphs' own outline levels count, and
a style id it cannot look up is read by its spelling (`Heading2`, `Title`). A
style is followed through at most 32 styles it is based on. A paragraph whose
style's chain runs longer, or round in a circle, before any style in it says
whether it is a heading carries `heading_level_unresolved: style_chain` instead
of a level. Headings inside
text boxes keep their level; headings inside table cells become part of their
row's text and carry none. Structure chunking cuts at these headings, and the heading path embeds
each chunk under them (see retrieval-and-storage.md).

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
Word 2010 wordprocessingShape, VML, and the Office 2016 chart namespace, so a
newer chart is read rather than its picture fallback), otherwise its fallback. Prefix aliases
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

A link in a DOCX paragraph (body, textbox, note or comment) or in a slide or
notes paragraph keeps its words in the text, and the segment's `links`
metadata says where they point: a JSON list of `text`, `target` and the
link's `start`/`end` in the segment's UTF-8 bytes, with the link's own
surrounding spaces left out of the span. An internal bookmark is `#name`. A
target lives in the part's relationships, which are external for a web or
mail address; they are read only as addresses and never followed. A link
whose relationship is missing, is not a hyperlink, is blank or is longer
than 2,048 characters is counted in `links_unresolved` and not recorded, and
so is every link of a part whose relationships cannot be read (the part's
text is read as before). Links are recorded while the list fits one metadata
value of 4,096 bytes, and at most 200 per segment; `links_cut` counts the
rest, so two links to long presigned addresses never fail the document. Links
in table cells and in field codes (`HYPERLINK` fields) are not read yet.
Targets are recorded data: a page showing them must not make them live
without the reader choosing to follow one.

A chart in a Word or PowerPoint file (DOCX or PPTX, or another member of their
families) becomes a segment of its own after the text it sits in (`paragraph:3/chart:1`, `slide:2/chart:1`; `content_role` `chart`,
`parent_locator` its paragraph or slide). Its text is the chart's title and
kind, then one line per series of category and value pairs, for example
`Revenue (bar chart)` then `2025: Q1 10; Q2 12.5`. The chart kinds Office 2016
added (waterfall, histogram, treemap, sunburst, box and whisker, funnel) keep
their data apart from their series; each series is read from the data it names,
its kind is its layout (`waterfall`, `treemap`), and of nested category levels
the first is read and `chart_category_levels_cut` counts the rest. Only the values cached in the
chart part are read: nothing is recalculated from the embedded workbook and
nothing is rendered. A series without categories is read by point number. At
most 64 series per chart and 1,000 points per series are read; past those,
`chart_series_cut` and `chart_points_cut` say how many were left out. The
segment also carries `chart_type`, `chart_series` and `chart_points`. A chart
whose relationship or part cannot be read is not a reason to refuse the file:
it is left out and counted in the document's `charts_unreadable`. A file with a
chart read says `charts` and its parser id ends `+charts-v1`; a file without
charts is read exactly as before. A package that also carries macros keeps its
`macros` note beside the chart counts.

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

The local LlamaIndex reference also advertises an HWP reader; HWPX, the
XML package, is read here, and the binary HWP 5 container remains a gap. Table understanding, semantic chunking, layout
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

### PDF table evidence

A PDF page's tables, inferred from the geometry of the recognized or
text-layer regions labelled `table` -- a text-layer page kept whole
carries its runs as regions when a table is among them (see
[pdf-ingestion.md](pdf-ingestion.md)) -- (see [pdf-ocr.md](pdf-ocr.md#inspect-possible-tables-without-repeating-ocr)),
reach `segment.table_cells` with the same record: row, column,
`column_span` where a cell reaches across the grid's columns, and the
cell's exact byte span of the page's text. No header or row span is
inferred, so `is_header` is false and `headers` empty; the segment's
`tables` and `tables_unreadable` metadata count the grids proposed and
those left out because their cells did not read together. The cells
come in the page's text order, each with its row and column.

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

## Rebuild a document as Markdown

```bash
scone doc-markdown report.docx > report.md          # the receipt goes to stderr
scone doc-markdown report.docx --json               # the whole record, spans and all
curl -H "Authorization: Bearer $KEY" \
  "http://127.0.0.1:7437/v1/episodes/42/document/markdown?max_bytes=200000"
```

```python
from scone_memory.ingestion import BuiltinDocumentParser
from scone_memory.ingestion.formats.markdown_assembly import assemble_markdown

parsed = await BuiltinDocumentParser().parse(data, "report.docx")
result = assemble_markdown(parsed)
print(result.markdown)
```

`assemble_markdown` writes a parsed document as Markdown: headings at their
level, paragraphs in reading order, lists as nested lists, captions beside what
they caption, preformatted text as fenced code, and tables as pipe tables. It
runs no model and adds no structure the parsed document does not carry, so the
same parsed document always gives the same bytes. The command reads a file and
opens no store; the route rebuilds a stored document from its retained manifest.

### Declared blocks

The readers record what the source declares, in segment metadata:

| Key | Values | Set by |
|---|---|---|
| `block_role` | `heading`, `list_item`, `caption`, `code` | Word paragraphs; HTML `h1`-`h6`, `li`, `figcaption`, `pre` |
| `heading_level` | `1`-`9` (Word outline levels reach 9) | Both |
| `heading_basis` | `outline_level`, `style`, `style_id` | Word |
| `list_level` | Nesting depth from `0` | Both |
| `list_kind` | `bullet`, `ordered`; absent when the source does not say | Both |
| `list_id` | Word `numId`; HTML ordinal of the outermost list | Both |
| `list_item_id` | Ordinal of the `li` the text sits in | HTML |
| `list_start` | The number an `<ol start>` declares, when it is 0-999999999 | HTML |
| `caption_target` | The table a `<caption>` belongs to | HTML |

In an email or mailbox, each HTML part is read on its own, so its `list_id` and
`list_item_id` carry the part's locator prefix (`message:2/mime:1/1`) and two
parts' lists never share an id. A heading, `pre`, `figcaption` or table inside
an `li` keeps its own role and also carries that item's `list_level`,
`list_item_id`, `list_id` and `list_kind`.

A Word paragraph is a heading because its own outline level, or the outline
level or name (`heading 2`, `Title`) of the style it is based on, says so --
never because of its font or length. Style IDs are localized (`Berschrift1`),
so the style's name decides; only an ID the styles part does not define is read
by its spelling, and `heading_basis=style_id` says so. Outline level 9 is body
text. A list item has numbering attached, directly or through its style, that
is not `numId 0`; its kind is read from the numbering part. The document's
default paragraph style is not consulted. A styles or numbering part that
cannot be read, or a `basedOn` chain longer than 32 styles, is named in the
document's `structure_notes` (`styles_unreadable`, `numbering_unreadable`,
`style_chain_cut`) and the text is read regardless.

A document that declares none of this parses byte-identically to before, so
its manifest and deduplication identity are unchanged. A document that does
gets a new manifest digest; use a new durable run and parser revision to
re-extract it.

EPUB, PPTX, spreadsheets and PDF record no roles: their text becomes
paragraphs (and declared tables stay tables). An ODT heading carries
`heading_level` alone, and a level with no role is written as a heading. A PDF
text layer declares no headings, and none are inferred from it; the record says
`structure_declared: false`.

### What the Markdown says and where it came from

Text is escaped so it cannot read as structure: `# not a heading` becomes
`\# not a heading`, and inline markup characters (`*`, `_` at a word edge,
`[`, `<`, backticks, `|`) are escaped. A line break inside a block stays a line
break; a blank line starts a new paragraph.

A pipe table has one header row and no spans, and the writer does not pretend
otherwise. A declared header row is the header. A table with no header cells
uses column names its reader recorded (`table_columns`, for CSV and JSON), or
else an empty header row -- its first row is never promoted. Any other header
row, a second one at the top or one between body rows, is written as a body row
and counted in `extra_header_rows`. A cell spanning rows or columns is written in
its first slot with the slots it covers left empty, and a quoted note before the
table says how many cells spanned and how many header rows were demoted. A line
break in a cell or a column name is written as `<br>` and counted. A table its
reader could only read as text (`table_status=text_fallback`) keeps its lines as
paragraphs after a note naming the reason.

A table is written whole where its first segment is. A spreadsheet reader emits
cells row by row, so a cell beside a declared table sits between the table's
rows, and an HTML `<caption>` can come after the first row. Such a caption is
written before the table; any other segment between its rows is written after
it, counted in `interleaved_segments`, and named in the note.

Referenced Word notes, comments and text boxes, which the reader queues after
the body, are quoted with their role: `> footnote 2: ...`. A Word heading below
level 6 is written at 6 and counted in `headings_clamped`; a list item nested
more than one level below its predecessor is nested one level and counted in
`list_levels_clamped`; a list item whose kind is unsaid is written as a bullet
and counted in `list_kinds_unsaid`; role metadata no reader writes (a heading
level `0`, an unknown role) is written as a paragraph and counted in
`roles_unreadable`. An ordered list the document interrupts carries on
counting, and an `<ol start>` is where it starts; `<li value>` and a Word
numbering's start value are not read, so those lists count from 1. A paragraph,
code block, heading, caption or table inside an HTML list item stays inside it,
indented under the item, and a block that is the first thing in its item is
written after the item's marker (`- ## Title`). `blocks.list_item` counts items,
whatever opens them. Two separate lists that would touch are written with
different markers (`-` then `*`, `1.` then `1)`), which is how CommonMark tells
them apart; no text changes.

A code block's line ends are written as `\n`, and the closing fence follows the
code's own final line break, so a Markdown reader sees the same code the source
held.

`spans` maps the Markdown back to the source. Each span has its byte range
(`markdown_start`, `markdown_end`), its 1-based `first_line` and `last_line`,
and `sources`: segment locators with `start` and `end` in
`extracted_text_utf8_bytes` -- the segments joined by a blank line, which is a
stored document's episode text, so a span and a retrieved chunk can be compared
directly. Every non-blank line is in exactly one span. Lines the writer made up
(a table's delimiter row, a generated header row, a note) are `generated: true`
and name the rows they describe.

`max_bytes` (default and maximum 8,000,000) bounds the Markdown. No block is
written past it; a table's note, header and delimiter row are written together
or not at all, and its rows are cut between rows. `bound.cut` says whether it
cut, `bound.cut_at` is the first extracted-text byte not written (a table is
written where its first segment is, so a segment between its rows can be
unwritten though it comes before the row the bound cut at), and
`bound.segments_omitted` counts segments not wholly written. The command repeats
this on stderr.

Inline formatting (bold, code spans) is not kept by the readers, so it is not
in the Markdown, and images are not emitted. Some of what a segment carries is
not written, and the record counts it among the segments written:

- A Word or PowerPoint chart is quoted like other side content (`> chart:
  Revenue (bar chart)`, then one line per series); its cached values are not
  made a table.
- A link's text is written as text; its target (the `links` metadata of a Word
  or slide paragraph) is not, and `link_targets_unwritten` counts them.
- A PDF page's bookmark `section` is not written as a heading, because the
  titles are not in the text and a page can open mid-section;
  `sections_unwritten` counts the pages that carried one.
- A page whose text layer the PDF reader named `unreadable` is written as
  extracted and counted in `unreadable_segments`.
- A Word paragraph whose style chain ran short (`heading_level_unresolved`) is
  written as a paragraph and counted in `headings_unresolved`.

OCR paragraphs of an image (`frame:N/paragraph:M`) are paragraphs, each traced
to its own locator. A PowerPoint deck's sections name no text; its slides are
written in the presentation's order.

Measured on this repository's `packages/memory/docs` (45 files): each file was
rendered to HTML by an independent CommonMark renderer (markdown-it-py with
tables), read by the HTML reader and rebuilt. Parsed back with the same
renderer, the rebuilt Markdown holds 310 of 310 headings with the same level and
text in order, 414 of 414 list items (61 nested) with the same text, 41 of 41
tables with 279 of 279 rows and 806 of 806 cells equal, 156 of 156 code blocks
whose content is exactly the source's, and 1,529 of 1,529 paragraphs. (Before
the closing fence stopped adding a line, none of the 156 code blocks matched
exactly: each gained a trailing blank line.) The extracted text as stored today,
read as Markdown, holds none of those tables or list items, 2 code blocks, and
95 "headings" that are all `#` comment lines from code blocks (none matches a
source heading). The corpus has no code block, heading or table inside a list
item, and no spreadsheet or mail, so those cases are covered by constructed
fixtures only; one of them, a numbered step holding a code block, rebuilt with 3
list items instead of 2 before list items kept their blocks. No DOCX corpus is
in the repository; Word reconstruction is covered by constructed fixtures only.

## Import a page by URL

```bash
SCONE_URL_IMPORT=1 scone import-url https://example.org/report.html
curl -X POST http://127.0.0.1:7437/v1/documents/from-url -H "Authorization: Bearer $KEY" \
  -d '{"url": "https://example.org/report.html"}'
```

The document lane reads what a caller hands it; this fetches the page
itself and reads it as the document its media type says it is (HTML,
plain text, Markdown, JSON, CSV or PDF, by the same readers as an
uploaded file). The bytes fetched are retained as the original, the text
is indexed, and the episode says where it came from: `document_url`,
`document_final_url` after redirects, `document_media_type` and
`document_fetched_at`. The same bytes read the same way are the same
document, so importing a page again does not duplicate it.

It is off unless the server is started with `SCONE_URL_IMPORT=1`, because
a server that fetches whatever URL it is told to will fetch its own
metadata service, its database, or the neighbour on its subnet. When on,
three rules hold, each with a test: every address a hostname resolves to
must be on the public internet, at the first URL and at every redirect,
and the connection is made to the address that was checked rather than to
the name again, so a name that changes its answer between the check and
the connection gains nothing (`SCONE_URL_IMPORT_PRIVATE=1`, or
`WebLimits(allow_private=True)` in code, is for a lab and says so); a page
is read up to `max_bytes`
(10 MiB by default) and refused past it rather than cut, a redirect chain
past `max_redirects` (5) is refused, and the fetch has a deadline; a media
type the document lane does not read is refused, not guessed at. Only
`http` and `https` are fetched, and a URL carrying credentials is not
sent. `scone import-url --json` prints the record: episode, URLs, media
type, bytes, redirects, format and segment count.

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
retry resumes eligible stages. With `DocumentOcr(...).parser(selection)`, or a
`BuiltinDocumentParser` configured with `OcrPdfParser`, the extraction step also
saves completed OCR pages in its existing encrypted journal. Resume re-inspects
the PDF, then reuses those observations without rendering or recognizing the
completed pages again. Native-text pages retain their embedded text. The final
manifest is identical to an uninterrupted extraction using the same observations.

Page receipts bind original bytes, PDF inspection, OCR options, extraction limits
and installed parser/renderer versions; the outer workflow additionally binds
space, run and the caller's parser revision. Change the revision when the
recognizer, trained data or settings change. A changed binding or corrupt receipt
fails explicitly. One binding receipt is retained even when no page needs OCR.
The existing whole-extraction deadline still applies on each attempt; it is not
a new per-page time allowance. A completed page does not mean the whole source
is searchable. Stage completion and final evidence verification remain required.

Custom parsers can implement `CheckpointedDocumentParser.parse_checkpointed`
using the optional `ExtractionCheckpoints` protocol. A subclass overriding
`parse` must also explicitly override `parse_checkpointed` to enable recovery;
an inherited recovery method never bypasses custom extraction or redaction.
No optional agents/cryptography import is needed to use the parser contracts.
For independently executed page steps with separate page deadlines, use
[PdfOcrWorkflow](pdf-ocr.md#resume-completed-pages-after-interruption).

Indexing saves each complete, validated embedding batch inside the encrypted
journal before calling the next batch. After cancellation or process failure
during embedding, a retry rebuilds the chunk plan and reuses matching vectors.
Receipts bind the space, source content identities, ordered UTF-8 chunk spans,
exact embedding input texts, batch boundaries, and embedder `id` and dimension.
The configured `id` must identify the model revision and its embedding options;
changing model behavior without changing that identity cannot be detected.
Changed chunking or contextual input also prevents reuse. No episode is written
until every vector validates. Malformed provider batches are never saved.

`job.status(...).checkpoint_count` reports retained intermediate receipts (OCR binding/pages and embedding batches),
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
workflow journal. A mailbox (`.mbox`) is read as one document of many messages: each message as an `.eml` is, its headers and text parts under `message:N/`, every segment carrying the message's number, date and sender so a passage recalled from a mailbox says which mail it came from; attachments are counted, not read, mbox `>From ` quoting is undone, a failure names the message it was in, and past 1,000 messages the rest are counted (`messages_unread`); the document's own limits (20,000 segments, 2 MB of text) refuse a mailbox whole before that, as they do an `.eml`. A message is opened only by an envelope line (`From sender Www Mmm dd hh:mm:ss yyyy`); a file not opened by one is read whole as one message.

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

### Reusing embeddings across updates

A file that changes on one line is stored again whole, and every chunk of
it is embedded again though all but one are the same text as before. With
an embedding cache the embedder sees only the chunks whose text is new:
vectors are kept by the embedder's id and width and the exact text it was
given (a contextual prefix included), so a hit is the vector the embedder
would have returned, and a different embedder, width or prefix is a
different key. `SCONE_EMBEDDING_CACHE=memory` keeps vectors for the
process; a path keeps them in a file every process that opens it shares,
so tomorrow's `scone sync` reuses what today's embedded. Unset (or
`none`), nothing is cached. A file cache holds at most 20,000 vectors
(`max_entries`; about 120 MB on disk at 768 doubles each) and the
in-memory one 5,000 (a Python list of floats is about four times the
packed size); both drop the least recently used past that, evicting as
they write, and their record says how many they dropped. A vector read
back from the file is checked for width and finiteness, and a row that
fails is removed rather than served. A cache is not evidence and cannot
refuse a write: one that fails (a full disk, a damaged or read-only
file, a locked database) is a miss, counted as `failures` in its record
with the last failure named, and the embedder answers instead. A path
that cannot be opened is refused by name before any store is opened.
`reembed_vectors` clears the cache first, since a model can change
behind an id that did not; the engine closes the cache with its stores.

Every receipt says what it did not pay for: `Added.embeddings_reused` on
the record, `embeddings_reused` in `map`'s and `sync`'s receipts.
Interrupted preparation keeps nothing partial: a batch is kept in the
cache only after every vector in it was validated. One interaction to
know: contextual embeddings prefix each chunk with the record's day and
source, so a file re-stored on a later day is a new key and is embedded
again unless the record carries its own `created_at`.

Measured on this framework's own source (409 files, 7,679 chunks, the
hash embedder): a second `map` after a one-line edit to `engine.py`
embedded 1 chunk and reused 140; a third with nothing changed embedded
none. Without the cache the second pass embeds all 141 chunks of the
edited file.

```bash
SCONE_EMBEDDING_CACHE=~/.scone/embeddings.sqlite scone map src/ --graph
# 312 file(s) read, 1 updated, 311 already here, 4 chunk embedding(s) reused, unchanged since last stored, …
```

### More than one repository in a space

A map names every file by its path below the root, so two repositories
mapped into one space share every name they have in common (both have
a `src/main.py`) and the graph folds them into one file. `scone map
ROOT --repo NAME` (and `scone sync --repo NAME`) holds every file as
`NAME/path` instead: the repositories keep their files apart, and the
graph explorer shows each as a top directory. Name every repository in
a shared space or none; a name is one path segment, refused otherwise
(and `requirements` is refused, since files under a directory of that
name are read as manifests), and a directory mapped again under a new
name is read again under it. Without a name a map has no other
repository to link to: its own package still resolves within it, and
nothing the space holds of it -- possibly stale -- is read as another's.

What a repository publishes is written in its manifest -- the package a
`pyproject.toml` names (`libpkg`), a `package.json` (`@acme/ui`), a
`Cargo.toml` (`acme-core`) or a `go.mod` (`example.com/svc`) -- and an
import of it in another repository is the link between the two. Before
reading, `map --graph` asks the space what its other repositories
publish and which files they mapped, reads this tree's own manifests
first, and follows such an import to the file by the language's own
rule: `libpkg.util` under `src/libpkg`, `libpkg` or `lib/libpkg`;
`@acme/ui/button` under the package's directory, `src`, `lib` or
`dist`, and `@acme/ui` alone to its `index`; `acme_core::store` to
`src/store.rs` or `src/store/mod.rs` and the crate alone to `lib.rs`; a
Go import path to the directory of Go files below the module's path.
The import then reaches the file, so a call through it reaches the
function (`app/src/main.py:run calls lib/src/libpkg/util.py:tidy`). A
package nobody here publishes, or a module whose file was never
mapped, stays a name; nothing is guessed. The receipt counts the
imports that reached another repository and names what the space
publishes (`cross_repository_imports`, `published`); a map that changed
nothing records nothing and counts 0.

```bash
scone map ~/code/lib --graph --repo lib
scone map ~/code/app --graph --repo app
# map: 3 file(s) read, 9 claim(s), 2 import(s) reach another repository's file
```

### Inspecting source removal

Servers with `episodes.forget: true` in `/v1/capabilities` expose the complete
source-removal workflow. This flag describes implementation support; the key's
role still determines whether DELETE is allowed.

- `GET /v1/episodes/{id}/impact` previews chunks and attachments removed and
  citing facts and links retained. It does not reserve a snapshot against writes.
- `DELETE /v1/episodes/{id}` removes the source through durable cleanup and
  returns a receipt. Repeating DELETE can finish an interrupted cleanup.
  The claims and links that cited the source stand and are listed. With
  `?with_claims=exclude` (`forget --with-claims exclude`;
  `await memory.forget(space, id, with_claims="exclude")`) each ledger
  claim that cited or affirmed the source and that no retained source still
  supports is excluded from recall with a reason naming the episode --
  a reversible policy (`include` undoes it), never an erasure: interval,
  history and quote stay. A claim another retained source affirms, one
  stated on its own authority, a proposed or declined one, and one already
  excluded stay and are listed under `claims_kept_other_support`,
  `claims_kept_not_in_ledger` and `claims_already_excluded`; `claims_policy`
  says which path ran. Nothing is excluded before the source is gone, so an
  interrupted cleanup leaves the default and a repeat finishes the job. Any
  other value of `with_claims` is refused before the engine is asked.
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

A host can also offer languages a request may choose for its own scan:
`SCONE_DOCUMENT_OCR_LANGUAGES=deu,jpn+eng` (comma-separated, Tesseract names).
Each is checked at startup the way `SCONE_DOCUMENT_OCR_LANGUAGE` is, and
installing its language data is still the operator's job. A request then names
one in `pdf_ocr.language` (`{"mode": "all_pages", "reading_order": "provider",
"language": "deu"}`). A language the host did not list, including the host's own
default named explicitly, is refused before the original is read. A request that
names no language is read with the host's default, and its selection, its
retained `pdf_ocr` metadata and its job identity serialize exactly as before.

Authenticated `GET /v1/documents/formats` includes `pdf_ocr.available`, `modes`
and `reading_orders`, and `languages` when the host offers any. A native host may instead supply
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

`missing_text` preserves readable embedded text and recognizes pages lacking it,
whose text extraction fails, or whose text layer is unreadable (mostly private-use
code points, `(cid:N)` runs, replacement characters or control bytes). Without OCR
such a page keeps its text, its segment carries `unreadable: true` and the
document metadata lists it in `unreadable_pages`. `all_pages` recognizes every page, including
those with embedded text. Every page's regions carry a `label` (title,
heading, paragraph, list, table, footnote, header, footer, page number,
…), inferred from the page or given by a layout engine; see
[region labels](pdf-ocr.md#label-the-pages-regions). Reading order is `provider`, `columns_ltr` or
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
source library instead of automatically repeating the import. Use the durable
document-job service below for explicit restart recovery, including OCR pages.


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
resume reuses completed extraction/indexing stages, saved embedding batches and
completed OCR pages when using the host's configured document OCR parser. Each failed
stage waits for an explicit resume, within the saved attempt budget (1–4 total
admissions). Async cancellation and deadlines are cooperative. Cancellation may
follow a partial write; inspect status and retained source evidence before
assuming no work occurred. Cancellation stops owned work even if writing its
intent fails, while reporting that storage failure.

A per-job local file lock prevents simultaneous execution by another process.
Capacity is bounded per service, including pending original reads before execution;
excess admission returns 429 with Retry-After.
This is local task ownership, not a distributed queue. Request/result reads do
not start jobs. A confirmed forgotten source invalidates completed evidence;
a temporary verification outage refuses the result without replaying work.

The capability `documents.jobs` is advertised only when configured. The existing
synchronous `/v1/documents` endpoint remains available.

### Languages read from their syntax tree

The Python reader walks Python's own tree and the brace reader finds
the headers a brace family shares (JavaScript, TypeScript, Go, Rust,
Java, C, C#, Kotlin, Swift, PHP and their kin — TypeScript and
JavaScript also through a grammar when `scone-memory[code-graph]` is
installed). Ruby, Lua, Perl, fish, shell and languages like them were
neither, and a file in one of them was prose that happened to contain
code: no declaration names on its chunks, no cuts at its definitions.
With the optional `scone-memory[code-languages]` extra (one grammar
pack) a file with a suffix the reader knows — `.rb`, `.rake`, `.lua`,
`.sh`, `.bash`, `.zsh`, `.pl`, `.pm`, `.fish` — is read from
its syntax tree by the one convention the grammars share: a definition
node carries its name in a field called `name`. Declarations are named
by everything that holds them (`Cart.total`), carry their byte and line
spans, cut the chunks as the other readers' do, and name a recalled
chunk in `declaration`. What the grammar does not name is not a
declaration here; a language whose grammar names things another way
(Kotlin's and Elixir's do) keeps the reader it had, and a brace-family
file (PHP, Swift, Scala) keeps the brace reader. Without the extra
nothing changes. `scone map` and `scone sync` read these suffixes as
they read the Python and brace families, so a repository's Ruby, Lua,
shell, Perl and fish files reach the graph by the same walk. The same
pack gives the brace family beyond TypeScript -- Go, Rust, Java, C#,
Swift, C, C++, Scala, PHP -- its
declarations and its bound calls from a syntax tree (see the code
graph in retrieval-and-storage.md); without it those files keep the
line reader's declarations and no calls.

The same tree speaks to the graph: every definition is a
`defines` claim held by what encloses it (`app/cart.rb:Shop.Cart defines
app/cart.rb:Shop.Cart.add`), what the file loads by a literal name is an
`imports` claim (`require`/`require_relative`/`load`, Lua's `require`,
`source` and `.` in shell and fish, Perl's `use` and `require` without the
lowercase pragmas), and `WHY:`/`NOTE:`/`TODO:`/`ADR-12` comments become
`notes`, `flags` and `cites` on the declaration they sit in. A load whose
target is not a literal (`require name`, `source "$HOME/x.sh"`) is not
claimed, and nor is one inside a function body, which runs when the
function is called rather than when the file loads, or one nested more
than five levels under a top-level statement; a file that shows no
`imports` may still load something one of those ways. Calls are not
claimed: binding one needs a receiver's type or a name resolved across
files, which these grammars do not supply.
