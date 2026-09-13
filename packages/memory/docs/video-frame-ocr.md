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

## Standard hosts and explicit imports

Set `SCONE_DOCUMENT_VIDEO_CONFIG` to an owned, mode-0600 regular JSON file
(no symbolic or additional hard links, maximum 16 KiB):

```json
{
  "schema_version": 1,
  "ffmpeg_executable": "/absolute/path/ffmpeg",
  "ffprobe_executable": "/absolute/path/ffprobe",
  "ocr_executable": "/absolute/path/tesseract",
  "model_revision": "installed-language-data-v1",
  "language": "eng",
  "page_segmentation": 6,
  "policy": {"interval_seconds": 5}
}
```

Both the memory host and conversation host load this configuration without
executing tools or downloading models. Install `scone-memory[images]` for frame
validation, along with the chosen local executables and OCR language data.
Executable bytes, OCR settings, sampling
policy and the operator revision bind the extraction revision. Replace installed
executables only with an explicit configuration reload; bump `model_revision`
for changed trained data or shared-library behavior. The standard loader uses
installed Tesseract. Applications can instead inject
`DocumentVideo(VideoDocumentParser(...), revision="host-v1")` into `build_app`
or `create_app` with any caller-owned `OcrEngine`. The host owns its resources.

`GET /v1/documents/formats` advertises a separate `video_ocr` capability, including
supported extensions and `includes_audio: false`. After uploading the original,
set `"video_ocr": true` in `POST /v1/documents` or `POST /v1/document-jobs`.
It must be a JSON boolean and cannot be combined with `pdf_ocr` for one document.
Without the flag, the existing parser selection is unchanged: configured media
transcription still extracts audio from video. Frame OCR never implies a speech
transcript. `.ts` remains TypeScript; transport-stream video uses `.mpegts`.
To retain both sources of evidence, import the original separately with and
without the frame-OCR flag when media transcription is also configured.

Encrypted jobs retain the selection and host revision. Retrying an interrupted
video job after restart reuses completed frame receipts; reading a completed
result makes no new OCR calls. A changed host revision refuses to resume the old
job. Legacy requests without `video_ocr` continue to select the existing parser.

```python
jobs = client.document_jobs(expected_space="my-space")
if jobs.formats().video_ocr_available:
    original = jobs.upload(video_bytes, media_type="application/octet-stream")
    status = jobs.start("slides-v1", attachment_id=original.attachment_id,
                        filename="slides.mp4", video_ocr=True)
```

The SDK checks the retained request and result against the selected extraction
mode; a server that silently ignores the flag does not produce a successful
acknowledgement. Python 3.9 remains supported.

A directory-sync collection can independently set `"video_ocr": true`. This
selects frame OCR for video extensions and keeps PDF/text/audio dispatch for the
other files. Video extensions join the default scan set only for an opted-in
collection (or existing media-transcription configuration). An explicit
`extensions` list remains authoritative. Changing the video revision changes
that collection's parser binding. Directory sync retains whole-document results;
per-frame interruption receipts are provided by document jobs, not directory sync.

Checked frame-serving routes, generated vision interpretations and video citation
UI remain subsequent integration work.
