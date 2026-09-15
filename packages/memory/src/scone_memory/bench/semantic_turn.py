"""End of turn by silence alone and with the lexical detector, on scripted utterances.

Each utterance in the fixture is one turn a person meant to take, written
as the fragments a recognizer transcribes between pauses. The replay
models the recognizer the voice path ships with (``audio.gate.VoiceGate``
defaults): a pause of ``STOP_MS`` or more ends a transcript, which arrives
``STOP_MS`` after the words stopped; speech starting again is noticed
``START_MS`` after it starts; a shorter pause is bridged into one
transcript. Words take ``WORD_MS`` each, which only places events in time.
Transcription itself is taken as instant, so the numbers are the turn
logic's, not a provider's.

Both modes run the same ``TurnHold`` the session runs, with the session's
defaults. Silence alone is every judgement ``unsure``, which releases each
transcript as it arrives, exactly as a session without a detector does.
A premature ending is a turn released before the utterance's last words.
Added latency is how long after its last transcript an utterance's final
turn was released.

``session_latency`` is the other half: the wall-clock time from a final
transcript to the model being asked, through a real ``VoiceSession`` with
in-memory storage, with and without the detector, interleaved.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..realtime.turn_end import HOLD_S, MAX_DURATION_S, UNSURE, Judgement, TurnEnd, TurnHold, judge_text

STOP_MS = 400
START_MS = 120
WORD_MS = 300
MODES = ("silence", "semantic")


def load(path: Path) -> list[dict[str, Any]]:
    utterances = json.loads(Path(path).read_text(encoding="utf-8"))["utterances"]
    for utterance in utterances:
        if len(utterance["pauses_ms"]) != len(utterance["fragments"]) - 1 or utterance["kind"] not in ("cut", "complete"):
            raise ValueError(f"utterance {utterance['id']} is malformed")
    return utterances


def events(utterance: dict[str, Any]) -> list[tuple[float, str, str]]:
    """What the session would see, in order: (ms, "transcript", text) and (ms, "speech", "")."""
    out: list[tuple[float, str, str]] = []
    at, words = 0.0, []
    fragments, pauses = utterance["fragments"], utterance["pauses_ms"]
    for index, fragment in enumerate(fragments):
        at += len(fragment.split()) * WORD_MS
        words.append(fragment)
        pause = pauses[index] if index < len(pauses) else None
        if pause is not None and pause < STOP_MS:
            at += pause
            continue
        out.append((at + STOP_MS, "transcript", " ".join(words)))
        words = []
        if pause is not None:
            at += pause
            out.append((at + START_MS, "speech", ""))
    return out


@dataclass
class Replay:
    id: str
    kind: str
    released: list[tuple[float, TurnEnd]]
    last_transcript_s: float

    @property
    def premature(self) -> int:
        return len(self.released) - 1

    @property
    def added_ms(self) -> float:
        return round((self.released[-1][0] - self.last_transcript_s) * 1000, 3)


def replay(utterance: dict[str, Any], *, semantic: bool, hold: float = HOLD_S,
           max_duration: float = MAX_DURATION_S) -> Replay:
    turns = TurnHold(hold=hold, max_duration=max_duration)
    released: list[tuple[float, TurnEnd]] = []
    last = 0.0

    def expire_before(at: float) -> None:
        # A deadline that comes at the same moment as an event is taken
        # first: the pessimistic order for the speaker.
        deadline = turns.deadline
        if deadline is not None and deadline <= at:
            released.extend((deadline, end) for end in turns.expire(deadline))

    for at_ms, kind, text in events(utterance):
        at = at_ms / 1000
        expire_before(at)
        if kind == "speech":
            turns.speech_started(at)
            continue
        judgement = judge_text(turns.text_with(text, "user")) if semantic else Judgement(UNSURE, "no detector")
        released.extend((at, end) for end in turns.heard(text, "user", judgement, at))
        last = at
    expire_before(float("inf"))
    return Replay(utterance["id"], utterance["kind"], released, last)


@dataclass
class Tally:
    utterances: int = 0
    cut: int = 0
    #: mode -> total turns released before an utterance's last words
    premature: dict[str, int] = field(default_factory=dict)
    #: mode -> ids of utterances answered before their last words
    answered_early: dict[str, list[str]] = field(default_factory=dict)
    #: mode -> added latency (ms) of each complete utterance's turn
    complete_added_ms: dict[str, list[float]] = field(default_factory=dict)
    #: mode -> added latency (ms) of each cut utterance's final turn
    cut_added_ms: dict[str, list[float]] = field(default_factory=dict)
    #: mode -> reason -> released turns
    reasons: dict[str, Counter[str]] = field(default_factory=dict)


def measure(path: Path, **options: float) -> Tally:
    utterances = load(path)
    tally = Tally(utterances=len(utterances), cut=sum(u["kind"] == "cut" for u in utterances))
    for mode in MODES:
        runs = [replay(u, semantic=mode == "semantic", **options) for u in utterances]
        tally.premature[mode] = sum(run.premature for run in runs)
        tally.answered_early[mode] = [run.id for run in runs if run.premature]
        tally.complete_added_ms[mode] = [run.added_ms for run in runs if run.kind == "complete"]
        tally.cut_added_ms[mode] = [run.added_ms for run in runs if run.kind == "cut"]
        tally.reasons[mode] = Counter(end.receipt.reason for run in runs for _, end in run.released)
    return tally


def judge_cost_ns(path: Path, *, repeats: int = 5, loops: int = 2000) -> list[float]:
    """Nanoseconds per ``judge_text`` call over every transcript the replay
    produces, one figure per repeat."""
    texts = [text for u in load(path) for _, kind, text in events(u) if kind == "transcript"]
    out = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(loops):
            for text in texts:
                judge_text(text)
        out.append((time.perf_counter_ns() - started) / (loops * len(texts)))
    return out


def _spread(values: list[float]) -> str:
    if not values:
        return "none"
    held = sum(value > 0 for value in values)
    return (f"median {statistics.median(values):.0f} ms, mean {statistics.mean(values):.0f} ms, "
            f"max {max(values):.0f} ms, {held}/{len(values)} held")


def report(tally: Tally) -> str:
    lines = [f"{tally.utterances} utterances, {tally.cut} cut mid-clause, {tally.utterances - tally.cut} complete"]
    for mode in MODES:
        early = tally.answered_early[mode]
        lines.append(f"{mode}: {tally.premature[mode]} premature endings; "
                     f"{len(early)} utterances answered early ({', '.join(early) or 'none'})")
        lines.append(f"  added latency, complete: {_spread(tally.complete_added_ms[mode])}")
        lines.append(f"  added latency, cut (final turn): {_spread(tally.cut_added_ms[mode])}")
        lines.append("  reasons: " + ", ".join(f"{reason} {count}" for reason, count in sorted(tally.reasons[mode].items())))
    return "\n".join(lines)


async def session_latency(texts: list[str], *, semantic: bool) -> list[float]:
    """Milliseconds from each final transcript to the model being asked,
    through a real session with in-memory storage and scripted providers."""
    from ..backends import InMemoryDocumentStore, InMemoryVectorIndex
    from ..embedders.hash import HashEmbedder
    from ..memory.engine import MemoryEngine
    from ..realtime.audio import AudioChunk, ReplyCompleted, TextDelta, Transcript
    from ..realtime.turn_end import LexicalEndOfTurn
    from ..realtime.voice import VoiceSession

    heard: list[float] = []
    asked: list[float] = []
    spoken = asyncio.Event()
    queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()

    class Part:
        async def aclose(self) -> None:
            return None

    class Transport(Part):
        async def receive(self):
            while (chunk := await queue.get()) is not None:
                yield chunk

        async def send(self, audio, turn_id):
            spoken.set()

        async def clear(self, turn_id):
            return None

    class Recognizer(Part):
        async def transcribe(self, audio):
            async for _ in audio:
                heard.append(time.perf_counter())
                yield Transcript(texts[len(heard) - 1])

    class Model(Part):
        async def respond(self, messages):
            asked.append(time.perf_counter())
            yield TextDelta("Noted.")
            yield ReplyCompleted()

    class Voice(Part):
        async def synthesize(self, text):
            yield AudioChunk(b"\x01\x00" * 160, 16000)

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        session = VoiceSession(engine, "turn-latency", "turn-latency", transport_factory=Transport,
                               stt_factory=Recognizer, model_factory=Model, tts_factory=Voice, capture=True,
                               turn_detector_factory=LexicalEndOfTurn if semantic else None,
                               session_timeout=600, max_history_bytes=1_000_000)
        running = asyncio.create_task(session.run())
        await session.started.wait()
        for _ in texts:
            spoken.clear()
            await queue.put(AudioChunk(b"\x02\x00" * 320, 16000))
            await asyncio.wait_for(spoken.wait(), 30)
        await queue.put(None)
        await running
    finally:
        await engine.close()
    return [round((a - h) * 1000, 3) for h, a in zip(heard, asked)]
