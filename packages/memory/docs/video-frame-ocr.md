# Video frame OCR

`VideoDocumentParser` is an explicit library parser for visible text in video.
It uses locally configured ffmpeg/ffprobe and a caller-owned `OcrEngine`; it
performs no generated vision interpretation and starts no model service.

```python
from scone_memory.ingestion.files import ingest_document
from scone_memory.ingestion.formats.types import DocumentLimits
from scone_memory.ingestion.video_frames import VideoFrameDecoder, VideoFramePolicy
from scone_memory.ingestion.video_ocr import VideoDocumentParser
from scone_memory.ocr.tesseract import TesseractOcr

parser = VideoDocumentParser(
    VideoFrameDecoder(ffmpeg_path="/absolute/path/ffmpeg",
                      ffprobe_path="/absolute/path/ffprobe"),
    TesseractOcr(executable="/absolute/path/tesseract", language="eng",
                 page_segmentation=6),
    model_revision="tesseract-eng-weights-and-settings-v1",
    policy=VideoFramePolicy(interval_seconds=5),
)
result = await ingest_document(memory, "my-space", video_bytes,
                               filename="slides.mp4", parser=parser,
                               limits=DocumentLimits(timeout_seconds=120.0))
```

Change `model_revision` when OCR weights, language data, settings, dependencies,
or behavior change. A custom OCR engine can replace Tesseract without changing
the parser. No OCR packages or language data are downloaded by this module.

## Evidence and limits

The parser samples the first non-attached video stream and recognizes exact PNG
pixels. Source locators identify the stream and decoded frame ordinal. Regions
use displayed-frame normalized coordinates and exact segment-relative UTF-8
byte spans. Manifest version 6 retains the original-video hash, decoder/model
and sampling revisions, rational time base, original presentation timestamps,
requested times, frame PNG hashes/dimensions, and every sampled frame's empty
OCR status. Empty means no recognized text, not verified absence of visible text.

There is one text segment per nonempty frame. Sampling does not prove coverage
of unsampled moments. Requests beyond the final decoded frame are counted.
Entirely empty OCR is refused because common document ingestion requires text;
the parser does not manufacture text to represent a visual-only source.

The default selection limit is 64 frames, configurable up to 256; input and
execution obey `DocumentLimits`. Each OCR result is limited to 10,000 regions,
with a shared 20,000-region limit across the video. Pixel, PNG byte and duration
limits come from `VideoFramePolicy`; extracted text and segment limits come
from `DocumentLimits`. Budgets are checked again on copied model instances.
Built-in decoder and Tesseract subprocess deadlines own their process groups. Synchronous caller
checkpoint writes are checked after returning and cannot produce a successful
result after the deadline; they are not forcibly interruptible transactions.

## Recovery

Pass this parser and a stable `parser_revision` to
`DocumentIngestionWorkflow` for encrypted, source/space/run-scoped recovery.
That workflow supplies `ExtractionCheckpoints`; direct `parse_checkpointed`
callers own authenticated storage and scoping themselves.

Recovery decodes and checks sampled pixels again, then validates retained OCR
receipts before making new OCR calls. Each receipt binds both its frame position
and exact pixels to the full source/configuration/frame inventory. Empty results
are reusable. Missing, changed or inconsistent completed receipts refuse.
Interrupted work resumes only the unfinished frames. A completed workflow read
validates retained evidence without decoding or calling OCR.

An expired final checkpoint write may have persisted a completed receipt even
though the request reports timeout. Explicit retry reuses that completed work.
Older receipt revisions are refused when their extraction binding differs.

Standard launcher configuration, checked frame-serving routes, generated vision
interpretations and video citation UI are subsequent integration work. Existing
standard media configuration continues to transcribe audio from video. This
library parser does not automatically replace that behavior.
