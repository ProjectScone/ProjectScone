# Voice sessions, personas and speech providers

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

## Scone Voice (native sessions)

Scone owns the audio runtime in `scone_memory.realtime.voice.VoiceSession`: its event types,
bounded input queue, turn lifecycle, interruption, response/speech sequencing and
memory capture. It uses standard Python `asyncio` and the native Scone engine.
Python 3.11+ is needed for this runtime's structured deadlines; the
base package and independent Rust/Python CLIs retain their existing requirements.

The native provider interfaces live in `scone_memory.realtime.audio`:

| Interface | Host adapter implements |
| --- | --- |
| `AudioTransport` | `receive()` audio stream, `send(chunk, turn_id)`, `clear(turn_id)`, `aclose()` |
| `SpeechRecognizer` | `transcribe(audio)` yielding speech-start and transcript events, `aclose()` |
| `VoiceModel` | `respond(messages)` yielding public text deltas and explicit completion, `aclose()` |
| `SpeechSynthesizer` | `synthesize(text)` yielding PCM chunks, `aclose()` |
| `SpeechActivityDetector` | optional `detect(chunk)` returning a strict boolean, `aclose()` |

Stream methods return async iterators supporting `aclose()` (async generators
work). Transport audio uses Scone's frozen `AudioChunk(pcm, sample_rate, channels)`:
signed 16-bit little-endian interleaved PCM, explicit 8–192 kHz sample rate, mono
or stereo. Adapters handle required format conversion; Scone does not silently
resample. Methods and factories must cooperate asynchronously and never block
the event loop. Resources must be fresh and distinct per session.

```python
from scone_memory.realtime.voice import VoiceSession

# These factories are configured by your trusted host, not supplied as code or
# credentials by a browser. They implement the Scone interfaces above.
session = VoiceSession(
    memory, "authorized-space", "voice-session-1",
    transport_factory=audio_transport_factory,
    stt_factory=speech_recognizer_factory,
    model_factory=language_model_factory,
    tts_factory=speech_synthesizer_factory,
    capture=True,
    where={"collection": "manuals"},
    session_timeout=1800,
    turn_timeout=30,
)
try:
    await session.run()
finally:
    await session.close()
```

The host obtains participant consent and authenticates the connection before
starting. `capture=True` authorizes public transcript writes; the flag alone does
not prove consent. No microphone, provider, credential, model download or endpoint
is selected implicitly. This is a native session—not a browser signaling server.

`SpeechStarted()` interrupts current output. A new final `Transcript` also
supersedes any active reply. Interim/empty transcripts do not trigger memory
writes or model calls. Scone cancels and drains the old response, asks transport
to clear its output by turn ID, and rejects late output from the old generation.
That clear request is not proof previously played audio was unheard. Input EOF
must be consumed by the recognizer and drains the final reply.

An optional `activity_factory` supplies a local speech detector independently of
the recognizer. A false→true speech transition queues an interruption before the
same PCM chunk reaches transcription. Activity and recognizer events share a
serialized controller, including duplex recognizers that consume PCM in another
task. Input remains bounded; no
samples are removed or resampled. The detector is closed with the other session
resources, including on failure or cancellation. Without it, the recognizer's
speech-start events continue to own early interruption.

## Personas and independent provider selection

`realtime.persona.Persona` is a frozen, versioned configuration containing a name,
instructions and independent reply, transcription, speech and optional activity
choices. It chooses an existing model/voice; it does not train or clone a voice.
Serialize with `model_dump_json()` and load with `model_validate_json()`. Unknown
fields, unsupported schema versions and blank instructions are rejected. Changing
the speech selection does not change instructions, other models or memory scope.

```python
from scone_memory.realtime.persona import Persona
from scone_memory.realtime.providers import ProviderRegistry

# These are operator-defined IDs, not installed provider defaults.
persona = Persona.model_validate({
    "schema_version": 1, "id": "juniper", "name": "Juniper",
    "instructions": "Be concise. Explain the source behind each answer.",
    "reply": {"provider": "local", "model": "reply-v1"},
    "transcription": {"provider": "transcriber", "model": "speech-v1"},
    "speech": {"provider": "voice-a", "model": "tts-v1", "voice": "alto"},
    "activity": None,
})

# Host-created factories close over credentials. Nothing in a persona imports
# Python code, selects network endpoints, or grants a memory space/recall scope.
registry = ProviderRegistry(
    reply={("local", "reply-v1"): language_model_factory},
    transcription={("transcriber", "speech-v1"): speech_recognizer_factory},
    speech={("voice-a", "tts-v1", "alto"): speech_synthesizer_factory},
)
bound = registry.resolve(persona)  # checks EVERY choice; creates no resources
session = bound.voice(memory, "authorized-space", "persona-session-1",
                      transport_factory=audio_transport_factory, capture=True,
                      where={"collection": "manuals"})
# Or bound.text(memory, "authorized-space", "text-session-1", where=...).
```

The registry admits exact `(provider, model)` pairs and, for speech, exact
`(provider, model, voice)` triples. There is no fallback to another provider.
Each host should expose only the choices that user may use. A bound text session
constructs no audio resources and obtains a fresh selected model per turn.
Credentials and PCM compatibility remain adapter responsibilities; successful
binding is not a remote availability or compatibility check.

Direct Deepgram, OpenAI, Cartesia, ElevenLabs and Silero adapters, an HTTP persona
catalog and browser voice selection are separate pending integrations. This
native composition API does not advertise browser voice as available.

Only public `TextDelta` and `ReplyCompleted` events are accepted from the model.
Scone groups text into speech segments; synthesis and output are awaited before
requesting further model output, providing backpressure at those interfaces.
Reply-end and audio-end are different: a reply is retained only after explicit
completion, stream closure, successful synthesis and output acceptance. A missing
completion or empty synthesis fails the turn. `send()` acceptance is **not** proof
a person heard the audio. Tool and hidden-reasoning events have no accepted type;
adapters must never relabel private reasoning as public text.

Final user text is saved before response generation. Successful assistant text
becomes a `conversation` episode with capture/session/turn IDs, speaker/role and
`representation=aggregated_text`. Assistant metadata identifies
`completion_evidence=adapter_end_and_output_accepted` and `playback=unverified`.
Interrupted partial replies are not retained as completed. A cancelled or timed-out
write is unconfirmed and stops the session; inspect storage before retrying.
There is no automatic uncertain-write replay.

Recall scope (`where`, `kind`, `source_prefix`, `since`, `until`) is validated
and frozen before factories run. Recalled text is source-referenced, marked
untrusted, bounded and inserted only into a copy of the current request—not
shared history or transcript memory. `last_memory_receipt` reports prepared,
empty or failed lookup, not provider use. Lookup failure can continue without
recalled material; capture/output/provider failures cannot report success.
Raw audio and source blocks are not retained by this integration.

The session is single-use. `state` is `new`, `starting`, `running`, `ended`,
`interrupted` or `failed`. Observe the `run()` task alongside the `started`
event because startup can fail. `stored_count` counts acknowledged writes.
Call `close()` from a host task, not from a provider callback. Repeated Close
or caller cancellation joins the same owned cleanup. The deadline requests
shutdown; noncooperative resource cleanup can delay return. No hard process
termination is claimed.

Defaults: 8 queued input packets, 64,000 bytes per PCM packet, 32,000 bytes per
transcript, 64,000 reply bytes, 128,000 JSON-encoded history bytes, bounded recall,
30 seconds per response and 30 minutes per session. The input producer may hold
one additional packet while the queue is full. These bounds do not control
provider-internal queues, network buffers or physical playback.

Run the native regressions from the repository root in the ordinary project environment:

```sh
packages/memory/.venv/bin/python -m pytest packages/memory/tests/media/test_voice.py -q
```

Tests use scripted protocol adapters and real isolated Scone memory, not live
recognition/model calls. Concrete provider adapters, authenticated browser audio
transport, React voice controls and video remain release gates. HTTP capabilities
still advertise `voice: false` until that entire path works. The text runtime and voice runtime share Scone-owned public events and scoped
context preparation; neither depends on an external conversation framework.

## Browsing retained sources

`GET /v1/sources?limit=25&kind=file&before=123` enumerates retained episodes in
descending episode-ID order. Omit `kind` to browse all kinds; omit `before` to
start at the newest ID. This is inventory, not semantic search, source-date order,
or a frozen database snapshot. Newer inserts appear on refresh; deleting a page's
boundary record does not invalidate its `before` value. Keep the same kind filter
while following `next_before`.

The response has `items`, `has_more` and nullable `next_before`. Each item contains
`episode_id`, `kind`, `source`, `created_at`, `byte_count` (UTF-8 stored text),
`preview` (at most 500 Unicode scalar values), and `preview_truncated`. A preview
is not an original file; retrieve retained text with `GET /v1/episodes/{id}` and
original media through the separate attachment routes. The bearer key selects
the space; query/body fields cannot change it. Page limits are 1–100; an invalid
or unknown query field is rejected. Existing `GET /v1/episodes?ids=...` remains a
separate bounded batch read.

Native async and sync engines expose `source_page(space, before=None, limit=25,
kind=None)`, returning a `SourcePage` of full episode records. Built-in document
stores implement the optional `EpisodeInventory.page_episodes` port. Custom
stores without it advertise `episodes.list: false` and HTTP returns 501; there is
no fallback that scans the whole export or disguises ranked recall as inventory.
Adapters can run `scone_memory.testing.contract_inventory` with their usual
engine fixture. In-memory inventory scans resident entries with bounded selection;
database adapters apply scope, kind, ID boundary, ordering and limit in their
native queries. Read cost and remote-store certification are separate from the
bounded response contract. The Documents browser UI is subsequent work.

## Direct speech providers

Install `pip install 'scone-memory[speech]'` to use the native
`scone_memory.providers.speech.CartesiaSpeech` and `ElevenLabsSpeech` adapters.
They implement `realtime.audio.SpeechSynthesizer` directly without a provider
orchestration SDK. Select the provider, model and voice explicitly:

```python
import os
from contextlib import aclosing
from scone_memory.providers.speech import ElevenLabsSpeech

async def speak(text, play_pcm):
    # The caller supplies play_pcm(AudioChunk). No device is opened implicitly.
    async with aclosing(ElevenLabsSpeech(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        model=os.environ["SCONE_SPEECH_MODEL"],
        voice=os.environ["SCONE_SPEECH_VOICE"],
        sample_rate=24000,
    )) as speech:
        async with aclosing(speech.synthesize(text)) as audio:
            async for chunk in audio:
                await play_pcm(chunk)
```

For Cartesia use `CartesiaSpeech` with `CARTESIA_API_KEY`; the constructor options
are the same. Pass a fresh adapter factory to `VoiceSession(tts_factory=...)` or
register it under the exact `(provider, model, voice)` in `ProviderRegistry`.
Keep keys in operator configuration, never persona JSON or browser settings.
Instantiating an adapter makes no request; consuming `synthesize` sends text to
the selected provider and may incur charges. Provider retention and account/model
availability still apply; Scone does not claim zero retention at either provider.

Output is mono signed 16-bit little-endian PCM at the selected sample rate. No
resampling, compression decoding, voice fallback or automatic request retry is
performed. Supported rates are 8000, 16000, 22050, 24000, 44100 and 48000 Hz;
provider/account availability may be narrower. Output chunks default to at most
4096 bytes and the utterance cap defaults to 24 MB. `timeout` limits HTTP I/O
inactivity, not total generation time; `VoiceSession` supplies the turn deadline.
Close each iterator on interruption and the adapter at session end. Only one
utterance may be active per adapter; it can serve successive utterances.

The adapters follow [Cartesia's bytes API](https://docs.cartesia.ai/api-reference/tts/bytes)
(pinned version `2026-08-14`) and [ElevenLabs' streaming API](https://elevenlabs.io/docs/api-reference/text-to-speech/stream).
Transport-contract and native-runtime tests use scripted HTTP peers, not paid
provider calls. These adapters alone do not enable a browser microphone, audio
playback, speech recognition or a complete live voice conversation.
