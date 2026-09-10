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
- The whole document has one wall deadline. Timeout or caller cancellation stops
  processing before ingestion writes. POSIX workers run in dedicated process
  groups, so wrapper descendants are terminated too. On Windows only direct child
  termination is supported; configure a direct executable there.
- Defaults are 150 DPI, at most 20 million rendered pixels per page and 10,000
  recognized regions per page. DPI is bounded to 72–300; the region ceiling is
  50,000. Rendering checks dimensions before allocating the bitmap. Input, output,
  extracted text, page counts and subprocess output have explicit bounds.
- Pages run sequentially, limiting simultaneous raster allocations. These bounds
  are not a native-library memory quota or OS sandbox. Compressed PDF/image inputs
  still require patched dependencies and deployment-level memory/concurrency
  limits. A trusted custom `OcrEngine` is responsible for cancelling its own work.
- Debug logging records page number, configured engine, region count and elapsed
  recognition time; it does not log document text or images. Process failures are
  ingestion errors, not memory-store outage signals.

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
pytest -q tests/test_ocr_contract.py tests/test_ocr_runtime.py tests/test_pdf_ocr.py tests/test_pdf_ingestion.py
```

PDF rasterization uses the optional [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)
package. Recognition uses [Tesseract](https://github.com/tesseract-ocr/tesseract).
Their licenses and dependency notices remain with their respective distributions.
