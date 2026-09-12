# Audio and video document ingestion

A host can enable timestamped media imports on the existing document HTTP surface.
`DocumentMedia` combines an explicitly configured `MediaDocumentParser` with an
operator revision. The transcriber is supplied by the host; no model is downloaded,
endpoint selected, or provider started by enabling the routes.

```python
from scone_memory.api.__main__ import build_app
from scone_memory.ingestion.document_media import DocumentMedia
from scone_memory.ingestion.formats.media import MediaDocumentParser, MediaTranscriber


def media_app(settings, engine, transcriber: MediaTranscriber, ffmpeg_path: str):
    media = DocumentMedia(
        MediaDocumentParser(
            transcriber,
            ffmpeg_executable=ffmpeg_path,
            max_duration_seconds=60.0,
        ),
        revision="local-transcriber-model-v1",
    )
    return build_app(settings, engine, document_media=media)
```

`ffmpeg_path` must name an existing absolute executable. The host owns the
transcriber and its resources. Its `transcribe(audio_wav: bytes)` coroutine receives
mono 16 kHz signed 16-bit PCM WAV and returns a nonempty bounded tuple of
`TranscriptionSegment` values with text and observed start/end times in seconds.
Configure a locally managed implementation with a timestamp-producing model.
A text-only speech response is insufficient: do not invent word or segment times.

`build_app` forwards the same media configuration to the memory API, the composed
conversation host, and durable imports when `SCONE_DOCUMENT_JOBS_CONFIG` is set.
For a custom host, `create_app(..., document_media=media)` and
`load_document_imports(..., document_media=media)` expose the same composition
points. Pass the same configuration to both when composing them yourself.

Change `revision` whenever the model, native decoder, duration limit, or provider
behavior changes. It is included in the durable parser fingerprint; result reads
and resume refuse a changed configuration. Loading the service and reading a
completed result do not transcribe again. Interrupted extraction may need a new
transcription call on explicit resume; there are no intermediate audio checkpoints.

Use the normal workflow:

1. Read authenticated `GET /v1/documents/formats`. Media suffixes appear only when
   the host configured them. Availability describes configuration, not a provider
   health check or an accuracy measurement.
2. Upload the original bytes to `POST /v1/attachments`.
3. Pass its attachment ID and explicit filename to `POST /v1/documents`, or to
   `POST /v1/document-jobs` with a caller-chosen import ID for durable execution.
4. Read the saved episode's `/document` evidence. Each transcription segment keeps
   source times, audio-stream identity and exact retained text. Manifest metadata records
   the host's transcriber revision. The original
   attachment remains downloadable and its digest is unchanged.

Video processing transcribes the first audio stream only; it does not inspect or
summarize video frames. Inputs must be native media bytes, never URLs or playlists.
Duration, input bytes, text bytes, segment count and elapsed execution are bounded.
Negative, non-finite, reversed, out-of-audio, and out-of-order timestamps are refused
before the transcript is stored. Overlapping segments may be retained when their
start times are ordered; overlap does not establish speaker identity.

`.ts` continues to mean TypeScript in document imports. Name MPEG transport streams
with `.mpegts`; `.mp4`, `.mkv`, `.webm` and the other supported video suffixes remain
unambiguous. A per-import `pdf_ocr` selection is only valid for PDFs.

Read roles can inspect retained evidence but cannot start imports. Another memory
space cannot read the source. A changed key scope or write role during synchronous
transcription is checked again before storage. Forgetting the episode removes its
readable provenance; evidence reads never re-run recognition.

The included integration tests use locally generated audio/video and a scripted
transcript to verify decoding, timestamps, authentication, retention and restart
behavior. They do not measure transcription accuracy. Timeline playback, built-in
model adapters for timestamped document transcription, and video-frame analysis
remain separate capabilities.
