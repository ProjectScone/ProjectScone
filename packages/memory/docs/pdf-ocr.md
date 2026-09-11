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

## Coverage and geometry

The default `missing_text` mode preserves native text and only recognizes pages
without extractable text. A page with a short text layer over a scan is therefore
not automatically recognized. Choose `all_pages` explicitly to replace all native
text with OCR; this mode reads page metadata without extracting discarded text.
The default text-only `PypdfParser` still requires an existing text layer.
Pages with no content stream are valid empty pages. In OCR mode, a page-level
text extraction failure also falls back to recognizing that page; successful
native pages keep their text. The parser identity records when this fallback
was needed. Encrypted/corrupt document structure, invalid page geometry and
resource-limit violations still fail; OCR does not bypass those checks.

OCR regions retain the recognizer's block and line order. Their half-open `start`
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
unchanged. OCR uses manifest schema v2; native-text v1 manifest serialization and
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

## Resume completed pages after interruption

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
learned document layout, table structure recovery, or generative document parsing.

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
