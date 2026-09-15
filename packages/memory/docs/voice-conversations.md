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
| `AudioTransport` | `receive()` audio stream (which may carry `Keypress` events, see [Keys from the phone](#keys-from-the-phone-dtmf)), `send(chunk, turn_id)`, `clear(turn_id)`, `aclose()` |
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

### Semantic end of turn

A recognizer ends an utterance at a pause (`VoiceGate` waits 400 ms), and every
final `Transcript` used to be a turn. People pause longer than that mid-sentence,
so "I want to book a table for … four people." was answered as two questions.
`turn_detector_factory` adds a second judgement on the words: an
`EndOfTurnDetector` (`scone_memory.realtime.turn_end`) with `judge(text)` returning
a `Judgement(verdict, cue)` and `aclose()`, created fresh per session.

```python
from scone_memory.realtime.turn_end import LexicalEndOfTurn

session = VoiceSession(memory, "authorized-space", "voice-session-1", ...,
                       turn_detector_factory=LexicalEndOfTurn,
                       turn_hold=1.5, turn_max_duration=10, turn_judge_timeout=0.5)
```

| Verdict | When (lexical rules, in order) | What the session does |
| --- | --- | --- |
| `incomplete` | open `(` or quote; trailing filler (`um`, `uh`, `er`, `hmm`); trailing `,` `...` `;` `:` or dash; trailing conjunction (`and`, `but`, `or`, `because`, `if`, …), article (`the`, `a`, `my`, …) or preposition (`to`, `for`, `of`, `from`, …) | holds the turn up to `turn_hold` seconds past the recognizer's pause |
| `complete` | ends in `?`, `.` or `!` (closing quotes allowed) | ends the turn at the pause, as before |
| `unsure` | anything else: unpunctuated text with no cue, a last word that is a number or in another script (`set a timer for 20`), and a trailing preposition right after a pronoun in text that opens with a wh-phrase that can be its object (`who`, `what`, `which`, `where`, `how much`, …: `who is this for`, `where did you send it to`) | ends the turn at the pause, as before |

The rules are conservative on purpose: `on`, `in`, `up` and `that` end ordinary
sentences, so they never hold a turn, and unpunctuated text is only held on a
positive cue. A question word first is not such a cue (`what I really need is`),
and `when`, `why` and a bare `how` cannot take a trailing preposition as their
object, so `how do I get from` is held. A wh-phrase excuses a trailing
preposition only when a pronoun comes just before it: a noun, verb or adjective
there can still go on (`what's the best way to`, `who should I talk to about`,
`what would you recommend for`), so those are held, and so is a finished
`what are you waiting for`, which pays the hold. While a turn is held, speech
starting again (from the recognizer or the activity detector), or a partial
(`final=False`) transcript with words in it, keeps it open until
`turn_max_duration` after its first final transcript, not `turn_hold`; the next
final transcript joins it and the joined text is judged again. A recognizer that
reports speech ending without words as an empty final `Transcript` runs
`turn_hold` again from that transcript, capped by `turn_max_duration`. The HTTP
recognizers `scone serve` uses (`BufferedTranscription`) do not: an empty
transcription is their error, so a cough they send for transcription ends the
session, and a held turn is stored as `session_ended` without being answered. A
noise they transcribe as words ("Thank you.") joins the held turn instead and is
judged with it; that was not measured. A
noise onset that never produces a final transcript at all waits the full
`turn_max_duration` and is released as `max_duration`. A held turn is never
recorded or answered until it is released, and new speech still interrupts any
reply that is playing.

Every user turn records why it ended, as `metadata["turn_end"]` on its stored
episode and as `session.last_turn_receipt` (`reason`, `verdict`, `cue`,
`fragments`, `held_ms` after its last final transcript with words arrived, `turn_ms`
from its first; both start after the recognizer's pause, not at the first spoken word):

| `reason` | Meaning |
| --- | --- |
| `silence` | the recognizer's pause, with no evidence either way, no detector, or a detector that failed or took longer than `turn_judge_timeout` (the cue says which) |
| `semantic_complete` | the words finished a sentence or question |
| `semantic_incomplete_timeout` | the clause stayed open and `turn_hold` ran out |
| `max_duration` | the turn reached `turn_max_duration` while held |
| `max_bytes` | joining the next transcript would pass 32 000 bytes; the held turn was released first |
| `speaker_changed` | another speaker's transcript arrived; the held turn was released first |
| `input_ended` | audio input ended while a turn was held; it is answered before the session ends |
| `session_ended` | the session stopped (closed, timed out or failed) while a turn was held; its words are recorded and not answered. If they cannot be stored, that is the session's error, or a note on the error that stopped it |
| `keypad` | keys from the phone joined the turn and finished it (see [Keys from the phone](#keys-from-the-phone-dtmf)) |
| `strategy_timeout` | a turn strategy held the turn (the cue names it) and `turn_hold` ran out (see [When the bot may take its turn](#when-the-bot-may-take-its-turn)) |

A turn's timing (`conversation_turn` `latency_ms`) is measured from its last final
transcript's arrival, so a held turn's latency includes the hold and the
detector's own time.

A detector that returns something other than a `Judgement` fails the session, as a
malformed activity detector does. `ChatEndOfTurn(chat)` puts any `ChatModel`
(`complete(system, user)`) behind the same seam; an answer other than `complete`
or `incomplete` is `unsure`. A model detector is judged while the session's
controller is held, so its time comes out of every turn: keep `turn_judge_timeout`
short.

`scone serve` gives served voice sessions the lexical detector when
`SCONE_SEMANTIC_TURN=1` (default off) and reports `voice_turn_end`
(`semantic` or `silence`) in `/v1/conversations/capabilities`. The hold and turn
bounds are the defaults above there. The pure turn state (`TurnHold`) takes the
time as an argument, so it can be driven with a scripted clock.

Measured on a scripted fixture of 32 utterances, 16 cut mid-clause
([`semantic-turn-v1.json`](../benchmarks/semantic-turn-v1.json), results in
[`semantic-turn-v1.results.md`](../benchmarks/semantic-turn-v1.results.md),
replayed by `scone_memory.bench.semantic_turn`). The fixture models the energy
gate's timing; it is not a recording of real speech, and no real recognizer or
model ran.

Events are assumed to arrive in order: the recognizer's final transcript before
the `SpeechStarted` of the speech after it, as `BufferedTranscription` and the
local activity detector deliver them. With a duplex recognizer that reports new
speech before the transcript of the old, that speech start finds nothing held; the
hold then runs from the transcript although the speaker is already talking, and a
continuation longer than `turn_hold` is released as `semantic_incomplete_timeout`.

### Keys from the phone (DTMF)

A caller can answer with the keypad as well as the voice: "press 1 for sales", an
account number, a PIN ended with `#`. A transport may yield
`realtime.keypad.Keypress(key, source, offset_ms, tone_ms)` events among its audio
chunks; `key` is one of the sixteen DTMF keys `0-9 * # A-D`, `source` is `event`
(the transport's own message) or `inband` (tones heard in the audio), and
`offset_ms` places the key in the caller's audio from the start of the call. Keys
never reach the recognizer. A session without a keypad policy (the default)
counts them in `session.keypad_ignored` and does nothing else.

```python
from scone_memory.realtime.keypad import KeypadPolicy

session = VoiceSession(memory, "authorized-space", "call-1", ...,
                       keypad=KeypadPolicy("collect", terminator="#", timeout=3, max_digits=32,
                                           speech_wait=5))
```

| Mode | What a key does |
| --- | --- |
| `append` | each key is given to the user's turn at once, as `[keypad] 5`: it joins whatever the caller has said and not finished (a held clause) and the turn is answered |
| `collect` | keys are held until the terminator (included in the entry), `timeout` seconds with no key, or `max_digits` keys, then given to the turn as one entry, `[keypad] 1234#` |

`template` (default `[keypad] {keys}`) is the text the turn is given. The first key
of an entry stops a reply that is playing, as speech starting does. While keys are
being collected a held spoken clause waits for them (as it does while the caller
speaks), so "my card number is um" and `4111#` are one turn. Keys held when input
ends are answered (`input_ended`); keys held when the session stops are stored and
not answered (`session_ended`). Keys that would take a held turn past 32 000 bytes
release it first (`max_bytes`).

Keys and words are put in the order the caller made them, not the order they
reached the session. A key arrives at once; the words arrive only when the
recognizer has heard the caller stop and transcribed them. So the session marks
when speech began (the recognizer's `SpeechStarted`, a partial transcript, or the
session's own activity detector) and, when the words come:

* keys being collected that were pressed before that speech began are given to the
  turn first (ended by `speech`), and the words are their own turn;
* an entry finished after that speech began waits for its words, and is given to
  the turn after them: it joins them when a detector holds them as an open clause,
  and is the next turn otherwise. Keys still being collected carry on, and hold the
  clause open until they end.

An entry waits at most `speech_wait` seconds (default 5, at most 60) for the words.
When that bound cuts, the entry is given on its own and those words are no longer
waited for; speech that ends with no words, input ending and the session stopping
also let it go. A recognizer that gives no sign that speech began gives the session
nothing to order by, and keys go to the turn as they arrive. `Keypress.offset_ms`
cannot order keys against words: a transcript does not say where in the audio it was
spoken.

A user turn that keys finished has `metadata["turn_end"] = "keypad"` and says how
and when each key came; `session.last_keypad_receipt` keeps the whole entry,
audio offsets and tone lengths included:

| Metadata | Meaning |
| --- | --- |
| `keypad_keys` | the keys, in order, `12#` |
| `keypad_ended` | `key` (append), `terminator`, `timeout`, `max_digits` (the bound bit), `speech`, `input_ended` or `session_ended` |
| `keypad_sources` | one letter per key: `e` from the transport's event, `i` heard in the audio |
| `keypad_started_ms` | the first key, in milliseconds from the start of the conversation |
| `keypad_at_ms` | each key from the first, comma-separated: `0,812,1604` |
| `keypad_waited` | only for an entry that waited for words spoken before it: `words` (given after them), `no_words` (speech, input or the session ended without them) or `timeout` (`speech_wait` cut the wait) |

`timeout` is at most 60 seconds and `max_digits` at most 32, so the longest entry
still fits one metadata value (256 characters).

A voice pipeline (`pipeline.voice`) does not act on keys: `CallerStage` feeds each
`Keypress` down the line as a frame of its own, where a stage that takes keys can
read it, and the call goes on.

#### Carrier calls

`telephony.CarrierTransport(socket, dialect, rate=16000, keypad="off")` yields
keys to the session when `keypad` is on:

| `keypad` | Keys the session gets |
| --- | --- |
| `off` (default) | none; carrier digits are still kept in `transport.digits` (at most 64, then counted in `dropped_digits`) |
| `events` | the carrier's `dtmf` messages (Twilio, Telnyx, Plivo and Exotel put the key in `dtmf.digit`) |
| `inband` | keys heard as tones in the caller's audio; carrier messages are not taken |
| `both` | both, with a press heard both ways reported once |

A carrier digit that is not a keypad key (`"12"`, `"x"`, an empty value) is not
passed on and is counted in `transport.stream.unreadable_digits`. With `both`, a
key reported one way is paired with the same key heard the other way within
`DUPLICATE_WINDOW_MS` (1000 ms) of the call's audio, counted from where a key's
tones were last heard, so a key held down and reported by the carrier when it is
let go is still one key; a paired hearing is used up,
so a key pressed twice is still two keys, and `transport.stream.duplicates` counts
the second hearings. At most `MAX_UNPAIRED` (64) reported keys wait to be paired;
past that the oldest is forgotten and counted in `unpaired_forgotten`.

In-band detection (`audio.dtmf.ToneDetector`) runs Goertzel's recurrence at the
eight DTMF frequencies over 25.6 ms blocks (205 samples at 8 kHz) started a quarter
block apart, on the line's audio before any resampling. A block is a key when both
tones are at least -35 dBFS, they hold at least 60 % of the block's energy, the high
tone is at most 4 dB louder than the low and the low at most 8 dB louder than the
high (ITU-T Q.24 twist), each tone is 10 dB above the next strongest in its group,
and each is stronger at its nominal frequency than 5 % either side of it. A key is
reported once, when agreeing blocks span `min_tone_ms` (40), and not again until
two blocks in a row are not that key. The tones are not removed: the recognizer
still hears them, and what it makes of them is its own.

Measured on synthesized calls companded to mu-law
([`dtmf-inband-v1.results.md`](../benchmarks/dtmf-inband-v1.results.md), replayed by
`benchmarks/dtmf_inband.py`): every key heard down to 6 dB SNR, 483 of 500 at 3 dB,
4 of 500 at 0 dB, and no wrong or phantom key at any SNR; every 40 ms press and no
20 ms one; every tone 1.5 % off nominal and none 3.5 % off; no key in noise, a
440 Hz tone or steady synthesized vowels. With the detector on, reading a call cost
33.1 ms of CPU per second of audio against 3.3 ms off. The signals are synthesized;
real lines, codecs other than G.711 and recorded speech were not measured.

`scone serve` does not accept phone calls. Its browser audio socket takes keys
when `SCONE_VOICE_KEYPAD=append` or `collect` (default `off`): the client sends
`{"type": "keypad", "key": "5"}` and the served session uses that mode with the
defaults above; `GET /v1/conversations/capabilities` reports `voice_keypad`.

### When the user goes quiet

A caller can put the phone down, walk away from the browser, or wait for the
assistant to speak first. Without a rule the session listens until its own
deadline (30 minutes by default). `idle` (`scone_memory.realtime.idle.IdlePolicy`)
counts an idle after `timeout` seconds with no speech from the user while the bot is
not speaking, says `prompt` then, and ends the session at the `end_after`-th idle in
a row instead. It is off by default.

```python
from scone_memory.realtime.idle import IdlePolicy

session = VoiceSession(memory, "authorized-space", "call-1", ...,
                       idle=IdlePolicy(8, prompt="Are you still there?", end_after=3))
```

| Field | Default | Meaning |
| --- | --- | --- |
| `timeout` | (required) | seconds the conversation waits on the user before an idle |
| `prompt` | `Are you still there?` | what an idle says; `None` says nothing and only notes the idle |
| `end_after` | `3` | the idle in a row that ends the session, not prompting; `None` never ends. Both `None` is refused |

Only time spent waiting on the user counts:

* the wait starts with the conversation, so a user who never speaks is prompted;
* the user speaking stops it, however long they talk (the recognizer's or the
  activity detector's speech start, or a partial transcript), and the end of that
  speech (a final transcript with or without words, or the activity detector hearing
  it stop), or a key when the session has a keypad policy, starts it again from then
  and clears the count of idles in a row;
* a reply pauses it from the moment the turn is released until the reply task ends
  (context, model, speech, the stored reply and its timing note), and it starts again
  from then; an idle's prompt is spoken the same way, so the next idle is a whole
  `timeout` after the prompt. Pauses are counted, so anything else the session runs
  for the user holds it too; the voice session runs no tools of its own today;
* a clause held open by the end-of-turn detector, a turn held by a strategy, or keys
  being collected are a user in the middle of a turn: an idle that comes due then is
  not counted, and the wait starts again.

A prompt is the bot speaking: it gets its own turn ID, clears audio still queued from
the turn before, is interrupted by speech like any reply, and is bounded by
`turn_timeout`. Once spoken it is stored as an assistant episode with
`idle_count`, `idle_action` (`prompt`) and `idle_silent_ms`, and is added to the
history the model sees, so the next answer knows what was asked. A prompt the user
talks over is cleared and not stored, and their speech clears the count. An idle
with no prompt clears nothing. A prompt
that would take the history past `max_history_bytes` fails the session, as a reply
would.

Every idle is a `conversation_idle` event in the engine's event log
(`MemoryEngine.record_idle`): `session_id`, `count` (its place in the run of idles),
`action` (`prompt`, `noted` when there is no prompt, or `end`), `silent_ms` (how long
the user had been silent while the conversation waited on them) and, for a prompt,
its `turn_id`. The event is written before the prompt is spoken, so an idle the user
talks over is still in the log. A log that fails or takes longer than 2 seconds is
logged as `voice_idle.failed` and the conversation goes on. `session.last_idle_receipt`
keeps the latest idle and `session.idles` counts them.

`session.end_reason` says why any session ended:

| `end_reason` | `state` | Meaning |
| --- | --- | --- |
| `input_ended` | `ended` | the audio input ended and the last reply finished |
| `idle` | `ended` | the `end_after`-th idle in a row ended it |
| `closed` | `interrupted` | the host closed or cancelled it |
| `session_timeout` | `failed` | `session_timeout` ran out |
| `failed` | `failed` | anything else stopped it (a provider, a store, a malformed event) |

After a reply, the wait is timed from when the reply task ended, after its last audio
was accepted by the transport, not from when the listener heard it: a client still
playing a long reply can be prompted early by up to its playback buffer. A speech start that is never followed by a final
transcript, or by the activity detector hearing speech stop, keeps the wait stopped;
the session's own deadline still ends it.

`scone serve` gives served voice sessions an idle policy when
`SCONE_VOICE_IDLE_TIMEOUT` is more than 0 (default `0`, off), with
`SCONE_VOICE_IDLE_PROMPT` (blank for the default) and `SCONE_VOICE_IDLE_END_AFTER`
(default `3`, `0` for never). A served session an idle ends is journaled `ended`;
`GET /v1/conversations/capabilities` reports `voice_idle` (`timeout_s`, `prompt` as a
boolean, `end_after`) or `null`.

### When the bot may take its turn

A released user turn is answered at once. `turn_strategy`
(`scone_memory.realtime.turn_strategy`) is asked about every final transcript after the
end-of-turn detector, and may hold the turn instead. A held turn is joined by what the
user says next and released by the same bounds and receipts as a clause the detector
held (`max_duration`, `max_bytes`, `speaker_changed`, `input_ended`, `session_ended`).

| Strategy | The bot takes its turn |
| --- | --- |
| `EndOfTurn()` (default) | when the turn ends, as before; the stored records do not change |
| `MinSpeech(seconds=0.8)` | when the turn's speech has lasted `seconds`; a shorter turn ("mm-hm", a noise taken for words) waits up to `turn_hold` for more and is then answered as `strategy_timeout` |
| `KeypadSubmit(key="#")` | when the caller presses `key`; needs a keypad policy |

```python
from scone_memory.realtime.keypad import KeypadPolicy
from scone_memory.realtime.turn_strategy import KeypadSubmit, MinSpeech

session = VoiceSession(memory, "authorized-space", "call-1", ...,
                       turn_strategy=MinSpeech(0.8))
session = VoiceSession(memory, "authorized-space", "call-2", ...,
                       keypad=KeypadPolicy("append"), turn_strategy=KeypadSubmit())
```

`MinSpeech` times a turn's speech from its first sign (the recognizer's or activity
detector's speech start, or a partial transcript) to the arrival of its last final
transcript, across the fragments it joined. That time includes the recognizer's pause
and its transcription, so set `seconds` above what those take for the shortest turn that
should be answered. A recognizer that gives no sign of speech leaves the duration
unknown: the turn is taken, and its receipt's cue ends `min_speech: speech duration
unknown`. A turn held for being short has the verdict `incomplete` and a cue such as
`min_speech: 312 ms < 800 ms`. A clause the detector already holds is left to it, and
keys finish a turn as before. `seconds` is at most 30.

`KeypadSubmit` holds every spoken turn, finished sentence or not, until an entry of keys
that ends with the submit key: in `append` mode the key itself, in `collect` mode an
entry ended by its terminator. Other entries (a key in `append` mode, a `collect` entry
that timed out) join the turn and wait too. The turn is then answered as `keypad`, and
its record carries the receipt of the entry that submitted it. A held turn waits up to
`turn_max_duration` (default 10 seconds) from its first transcript or key, speech with
no words does not cut that short, and it is then answered as `max_duration`.

A strategy is any object with a `name`, a `submit` key or `None`, and
`decide(judgement, speech_ms)` returning the detector's `Judgement` unchanged to take
the turn or an `incomplete` one to hold it. One that returns anything else fails the
session.

`scone serve` takes `SCONE_VOICE_TURN_STRATEGY` = `end_of_turn` (default),
`min_speech` (with `SCONE_VOICE_MIN_SPEECH` seconds, 0.8 when blank) or
`keypad_submit` (needs `SCONE_VOICE_KEYPAD=append` or `collect`); capabilities
report `voice_turn_strategy`.

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
