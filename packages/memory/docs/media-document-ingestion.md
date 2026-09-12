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

For an already running local service that returns observed segment timestamps,
Scone includes `LocalDocumentTranscriber`:

```python
from scone_memory.providers.transcription import LocalDocumentTranscriber

transcriber = LocalDocumentTranscriber(
    base_url="http://127.0.0.1:8000/v1",
    model="operator-selected-timestamp-model",
    timeout=120,
)
app = media_app(settings, engine, transcriber, ffmpeg_path)
```

Replace the model identifier with the exact model served locally. The adapter
accepts loopback addresses or `localhost`, with an optional explicit `api_key`;
it does not select a service, download weights or start inference at construction.
The service must support multipart `audio/transcriptions` with
`response_format=verbose_json` and `timestamp_granularities[]=segment`, returning
numeric `start`, `end` and `text` for every observed segment. This contract is
documented by [vLLM's transcription protocol](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/speech_to_text/transcription/protocol/);
support depends on the served model. A text-only response is explicitly refused.
The adapter retains segment text and times; it does not infer timing from the
top-level transcript, and it makes no speaker-identification claim.

Each call owns its HTTP client and closes it on success, failure or cancellation.
It disables redirects and environment proxies, rejects encoded responses, bounds
the entire request as well as response bytes and segment count, and never retries
an uncertain transcription. The media parser's remaining extraction deadline
still applies, even when shorter than the provider timeout. Change the host
revision when changing the selected model or its configuration.

`build_app` forwards the same media configuration to the memory API, the composed
conversation host, and durable imports when `SCONE_DOCUMENT_JOBS_CONFIG` is set.
For a custom host, `create_app(..., document_media=media)` and
`load_document_imports(..., document_media=media)` expose the same composition
points. Pass the same configuration to both when composing them yourself.

Change `revision` whenever the model, native decoder, duration limit, or provider
behavior changes. It is included in the durable parser fingerprint; result reads
and resume refuse a changed configuration. Loading the service and reading a
completed result do not transcribe again.

Durable native media imports save completed, validated transcript observations in
the encrypted extraction journal before uploading the document manifest. If that
upload fails or is interrupted, explicit resume re-decodes the retained original,
compares the normalized WAV's SHA-256, byte count and duration, then reuses the
saved observations without another transcription call. The resulting text,
timestamps, locators and manifest identity stay unchanged. Completed-result reads
do not perform this decoding; it is only needed to resume incomplete extraction.

Receipts bind the source bytes, exact filename, document limits, decoder settings
and host transcriber revision. Missing host bindings, changed decoded audio,
corrupt receipts and journal read failures refuse reuse without falling back to
the model. Only compact timestamped observations and WAV identity are journaled,
within the existing 16-MiB receipt limit; decoded WAV bytes are not copied into the
journal. This supports the native parser's existing text and segment budgets.

The recovery guarantee begins when the receipt write commits. A crash during the
model call, or between its response and that commit, can still require another
call on explicit resume. A failed receipt write fails extraction rather than
claiming completed work. With whole-file transcription this recovers a completed response. Opt-in audio
windows, described below, preserve progress between smaller model calls. Custom parsers
that override `parse` keep their ordinary behavior unless they also opt into
`parse_checkpointed`; standalone callers must bind model revisions and memory
scope through the checkpoint-owner contract.

Use the normal workflow:

1. Read authenticated `GET /v1/documents/formats`. Media suffixes appear only when
   the host configured them. Availability rechecks that the configured local decoder
   is still an executable file. It is not a transcriber health check or an accuracy
   measurement; the `requires` field names the decoder and timestamped provider.
2. Upload the original bytes to `POST /v1/attachments`.
3. Pass its attachment ID and explicit filename to `POST /v1/documents`, or to
   `POST /v1/document-jobs` with a caller-chosen import ID for durable execution.
4. Read the saved episode's `/document` evidence. Each transcription segment keeps
   source times, audio-stream identity and exact retained text. Manifest metadata records
   the host's transcriber revision and the normalized WAV's SHA-256 and byte count. The original
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

For checked playback, `GET /v1/episodes/{episode_id}/document/audio` returns the
full normalized mono 16 kHz PCM WAV used as the transcription source. It decodes the retained
original again without calling the transcription model, checks the current host
revision and recorded WAV digest/length, and rechecks source and caller scope
before returning `audio/wav`. Decoder work shares the document ingestion slot
limit. Responses use `Cache-Control: no-store`; clients should also verify the
manifest's digest and length before creating a temporary playback URL.

Older extractions without the normalized WAV identity remain readable but cannot
use this checked playback route. A changed decoder output, revision, forgotten
source or revoked access refuses playback instead of serving unverified audio.

The included integration tests use locally generated audio/video and a scripted
transcript to verify decoding, timestamps, authentication, retention and restart
behavior. They do not measure transcription accuracy. Browser timeline controls,
additional native model adapters, and video-frame analysis remain separate
capabilities from this HTTP surface. The loopback integration test runs a real
multipart HTTP service with scripted segments, normalizes stereo audio, and checks
retained evidence after reopening SQLite without another transcription request.

## Configure the standard server

The standard `scone-memory serve` launcher can enable media without a custom Python
host. Create an owned regular file with mode `0600`, for example
`/absolute/path/to/document-media.json`:

```json
{
  "schema_version": 1,
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "operator-selected-timestamp-model",
  "model_revision": "local-weights-v1",
  "ffmpeg_executable": "/absolute/path/to/ffmpeg",
  "max_duration_seconds": 60,
  "timeout_seconds": 120
}
```

Set `SCONE_DOCUMENT_MEDIA_CONFIG` to this path alongside the normal server settings.
The same configuration applies to the memory-only host, the composed conversation
host and durable document jobs. An explicit `build_app(..., document_media=...)`
argument takes precedence over the file for custom host composition.

The endpoint must be loopback or `localhost`, and ffmpeg must already exist and be
executable. If the service requires authentication, add `"api_key_env":
"LOCAL_TRANSCRIPTION_KEY"` and supply that environment variable privately. Do not
put the credential value in JSON. Missing credentials, unknown fields, duplicate
keys and invalid or nonprivate files refuse startup. Loading config does not
contact the endpoint, start a model, or resume an interrupted import.

The derived transcriber revision binds the nonsecret settings, the explicit
`model_revision` and the SHA-256 of the decoder executable's contents. Loading the
configuration reads the decoder in bounded blocks without executing it; empty,
nonregular, changing or larger-than-512-MiB executables refuse startup. Installer
symlinks are supported and bind the target's contents. Changing the model, endpoint,
decoder bytes, decoder path or limits changes that identity and prevents a saved
job from resuming or returning a result under the new configuration.

Restart the host after changing the decoder; its fingerprint is captured at config
load, not monitored continuously. The hash covers the executable itself, not dynamic
libraries or programs called by a wrapper. Bump `model_revision` when changing those
dependencies, model weights or server behavior at unchanged paths. Credential
rotation alone preserves extraction identity. Startup errors identify file,
configuration-content, provider/credential or decoder-fingerprint failures without
printing configuration values or secrets.

Provider timeouts are ceilings, not an extension of extraction budgets. Synchronous
imports and checked audio decoding each use the default 30-second document budget.
Durable imports use their job configuration's `limits.timeout_seconds` (up to 120
seconds); a slower successful import may still exceed the separate playback read
budget. A decode timeout returns a time-limit error before any audio digest check.

## Recover completed audio windows

Set `"chunk_seconds": 30` in the private media configuration, or pass
`chunk_seconds=30` to `MediaDocumentParser`, to transcribe bounded windows.
The integer maximum is configurable from 1 to 120 seconds. Omission or `null`
keeps whole-recording transcription and its existing extraction identity.
Enabling or changing chunking changes the host revision; use a new import after
changing that configuration. Custom injected hosts must also change their revision.

The decoder still prepares the complete normalized audio within the existing
source-byte and duration limits. Windows cover its samples exactly once. Near
an interior boundary, Scone searches the final 20% (at most two seconds) for at
least 160 ms of quiet audio and can cut inside that pause. Otherwise it uses the
maximum window length. This deterministic heuristic does not identify speech,
skip silence, overlap audio, deduplicate text, or establish transcription accuracy.
A model can still misrecognize speech at a boundary; evaluate window length against
your local model and recordings before selecting it.

Each provider receives a mono 16 kHz PCM WAV window and returns times relative to
that window. Scone validates those times before translating them into the original
recording's timeline. An explicit empty tuple means the provider observed no text
in that window. With the standard local adapter, chunking enables `allow_empty`:
an empty `segments` list is accepted only alongside an empty string `text`.
Text without observed timestamps is still refused. An entirely empty transcription
cannot become a document.

Durable extraction commits each validated window to its encrypted journal before
starting the next model call. After interruption, explicit resume re-decodes and
checks the whole audio, validates the completed prefix, then processes unfinished
windows. A partially executed model call must run again; completed windows do not.
Corrupt receipts, holes in the prefix, changed inputs/settings, and changed decoder
output fail without silently replacing saved observations. Total text and segment
budgets apply across windows, and all work shares the original extraction deadline.
Decoded audio remains outside the journal.

Evidence keeps source-global timestamps and the full normalized audio digest.
Checked playback serves that full recording, so citations retain their original
positions even though inference used smaller WAV inputs. The manifest records the
window implementation and configured maximum. Completed receipt replay preserves
the same manifest as uninterrupted extraction with the same model observations.
