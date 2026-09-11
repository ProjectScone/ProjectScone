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

## Coverage

| Reader | Evidence retained | Limits |
|---|---|---|
| Text, Markdown and code files | Line locators | Source text only; no AST or semantic code graph |
| JSON/JSONL/NDJSON, CSV/TSV, XML | JSON paths, rows/cells or XML locators | No schema-specific semantic interpretation |
| HTML | Visible extracted text | Bounded parser; no browser execution, stylesheets or remote resource fetching |
| DOCX, XLSX, PPTX | Paragraphs/tables, sheet cell references, slides and notes | No rendered Office layout or macro execution |
| ODT, ODS, ODP, EPUB | Format-local segment locators | Text extraction; no rendered layout |
| EML | Message-part locators | No recursive attachment ingestion |
| RTF, XLS/XLSB, MSG | Converter/reader locators | Optional dependencies; message attachments are not extracted |
| DOC, PPT | Converted text locators | Explicit offline converter; macOS textutil also supports DOC; page/slide structure may be lost |
| PDF | Page locators and extraction method | Native text by default; dedicated PDF API retains richer OCR region geometry |
| Images | Frame and OCR-region locators | Explicit `ImageDocumentParser` and OCR engine required |
| Audio/video | Audio-stream timestamps | Explicit `MediaDocumentParser` and transcription provider required; video frames are not analyzed |

Use `document_formats()` or authenticated `GET /v1/documents/formats` to
inspect default-reader dependencies on the running installation. Availability
does not guarantee that every valid variant of a format is supported.
Media readers must be registered explicitly on a `BuiltinDocumentParser`;
the default HTTP route does not configure OCR or transcription providers.

The local LlamaIndex reference also advertises HWP, PPTM, MBOX and IPYNB
readers, which remain gaps. Table understanding, semantic chunking, layout
reconstruction, directory synchronization and general connector ingestion
also remain open. The [PDF OCR guide](pdf-ocr.md) describes separate OCR
geometry, recognition limits and model-quality caveats.

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

The source must already be retained with a filename. Extraction saves its
manifest before indexing. Reopening the same journal with the same key,
parser revision, limits and source identity resumes indexing without
repeating completed extraction. Changed parser/model options need a changed
`parser_revision`. Source validation rejects missing or deleted evidence.

This is one caller-owned active document per journal. It does not provide a
background queue, distributed worker leases, per-page OCR checkpoints, or
separate durable chunking/embedding stages. Cancellation interrupts the active
call; an explicit retry resumes eligible stages.

## HTTP and execution boundaries

Retain the original through the existing attachment upload route with its
filename, then post `{"attachment_id":"<retained SHA-256>"}` to
`POST /v1/documents`. Read provenance through
`GET /v1/episodes/{episode_id}/document?chunk_id=...`. The API uses its
authenticated space, write-role authorization and ingestion backpressure.
HTTP indexing is synchronous and does not automatically create a durable
workflow journal.

Parsers enforce input, extracted-text, segment, archive and execution limits.
Manifest byte size is checked before direct ingestion retains an original.
Storage failure can still leave retained attachments; this is not an atomic
transaction across independent stores. Python workers use isolated startup
and this installation's package root. Child environments include only
execution/locale/temp settings and explicitly supported converter/Tesseract
settings. These controls are not an OS sandbox or native-memory quota.
