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
When every sampled frame returns no recognized text, manifest version 7 retains
that inventory with zero segments. The source episode has empty content and zero
text chunks or vectors. No caption, filename text or visual interpretation is
manufactured, and neither embedding nor text-distillation models are invoked.
Nonempty video documents retain version 6 and their existing serialized identity.

Visual-only retention verifies both retained attachments before writing. An
interruption leaves the source identity marked until retry or engine recovery
verifies the evidence and repairs both attachment links. Directory synchronization
accepts zero chunks only for verified visual-only evidence with no unfinished
write. Ordinary empty notes and caller-supplied metadata cannot select this path.

Archive profile `scone.archive/1` omits attachment bytes. Import and space merge
therefore refuse visual-only source records explicitly before mutations; they do
not create an unverifiable empty source or delete its original space. Use [attachment archives](attachment-archives.md) with
`export(..., include_attachments=True)` to transfer retained video evidence, or
keep storage backups that carry document and attachment stores together.

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

## Verified frame viewing

A configured video host serves retained frame pixels at:

```text
GET /v1/episodes/{episode_id}/document/video/frames/{ordinal}
Authorization: Bearer <space-scoped-read-key>
```

Obtain the ordinal from `GET /v1/episodes/{episode_id}/document` under
`video.frames`. It is the decoded frame ordinal, not the sampling-list index or
an approximate playback time. Only retained frames can be requested. The response
is `image/png` with `Cache-Control: no-store` and `nosniff`; the
`X-Scone-Video-Frame-SHA256`, `X-Scone-Video-Frame-Ordinal`, `X-Scone-Video-PTS` and
`X-Scone-Video-Time-Base` headers bind the returned pixels to the stored evidence.
The timestamp is an integer in that rational time base, including the original
stream offset. It is not a floating-point time guessed from nominal frame rate.

The decoder reproduces the retained sampling plan and checks the source hash,
decoder revision, complete frame inventory, dimensions and PNG bytes. Reading a
frame performs no OCR or generated vision inference. Current OCR language or
model choices do not reinterpret stored observations. A changed decoder or
non-reproducible frame refuses instead of serving different pixels under an old
citation. The library equivalent is `VideoFrameDecoder.read_frame(data,
filename, evidence, ordinal, limits=...)`; `VideoDocumentParser.read_frame`
delegates to the same verification.

Frame reads require current source access. After decoding, the host rechecks
original and manifest bytes, current links, the raw source row and authorization
before sending the response. Observed forgetting, unlinking or access changes
prevent frame delivery. These are separate storage observations, not a distributed
transaction; a change after its final observation is not guaranteed to be caught. Requests share bounded host ingestion capacity and return
429 when it is full; cancelling a read releases its slot.

This implementation redecodes the sampled inventory for each request and does
not retain a frame cache. The work remains bounded by the recorded sampling
policy and the host document limits. Explicit sampled-frame interpretations are
described below. Semantic retrieval over visual-only sources remains subsequent
integration work.

## Browser frame catalogue

`GET /v1/episodes/{episode_id}/document/video/catalogue` returns an independently
versioned representation for clients whose JSON numbers cannot retain int64
timestamps. The response has `schema_version: 1`,
`timestamp_encoding: "decimal-string"`, the current `space`, `episode_id` as
decimal text, and `evidence` with the full document provenance. Inside
`evidence.video`, `start_timestamp`, `duration_ticks`, and every frame's
`presentation_timestamp` are canonical signed decimal strings (`duration_ticks`
is positive). The rational `time_base` remains a string. Use integer arithmetic
for `(presentation_timestamp - start_timestamp) * time_base`; do not first
convert these timestamp strings to floating-point numbers.

The catalogue retains original/manifest identities, UTF-8 text regions and all
sampled frames, including those with empty OCR. It performs no decoding or OCR
and works without current decoder configuration. Existing stored manifests and
the ordinary document endpoint retain their previous numeric representation.
Catalogue reads share bounded ingestion capacity, set `no-store` and `nosniff`,
and recheck attachment bytes, links, the source row and access before returning.
As with frame reads, this is observed-state validation rather than an atomic
transaction across stores. A valid catalogue is evidence about retained frames;
it does not guarantee that the current decoder can reproduce their pixels.


## Explicit sampled-frame interpretation

Hosts with a configured decoder and selected vision model advertise
`documents.video.understand`. Request an unsaved model description with:

```text
POST /v1/episodes/{episode_id}/document/video/frames/{ordinal}/understand
Authorization: Bearer <space-scoped-write-key>
Content-Type: application/json

{"prompt": "Describe the visible scene and state uncertainty."}
```

The host selects the current `vision` connection from Models for each request.
Only the verified PNG of the selected retained frame is sent to that model.
The response binds the original and manifest hashes, PNG hash, frame ordinal,
integer presentation timestamp as decimal text, rational time base, dimensions,
and actual model identifier. `understanding.origin` is `model_generated` and
`persisted` is `false`. This performs no OCR, embedding, retrieval indexing or
source mutation. Viewing a source never starts interpretation automatically.

Prompts accept at most 16,000 Unicode codepoints within a 200,000-byte JSON body;
model descriptions accept at most 64,000 codepoints. Inference shares bounded
ingestion capacity (429 when full). Decoding, source checks and inference share
a 120-second deadline (504); explicit clock checks also refuse late results when
provider work blocked the event loop or swallowed cancellation. Invalid or
mismatched model responses return 502 without exposing provider output.

The host checks retained bytes, current source links and authorization before
inference and again before returning. Observed source removal or access changes
refuse the result. These remain separate storage observations, not an atomic
transaction across stores. The browser independently checks the catalogue before
and after the request and discards results on cancellation or changed source,
frame or prompt. A browser cancellation stops display and transport; it cannot
guarantee that a self-hosted provider stops work already received.

This describes one sampled frame, not an entire video or activity between frames.
Descriptions remain unsaved and do not make visual-only sources semantically
retrievable. Archive transfer of retained video attachments remains separate work.
