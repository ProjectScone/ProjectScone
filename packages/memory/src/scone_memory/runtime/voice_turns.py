"""The served voice session's idle policy and turn strategy, from settings.

``SCONE_VOICE_IDLE_TIMEOUT`` (seconds; 0, the default, is off) gives each
served voice session a ``realtime.idle.IdlePolicy``: after that long with no
user speech while the bot is not speaking it says ``SCONE_VOICE_IDLE_PROMPT``,
and the ``SCONE_VOICE_IDLE_END_AFTER``-th idle in a row (0 for never) ends
the session. ``SCONE_VOICE_TURN_STRATEGY`` chooses when the bot may take its
turn (``realtime.turn_strategy``): ``end_of_turn`` (the default, as before),
``min_speech`` (``SCONE_VOICE_MIN_SPEECH`` seconds, 0.8 when unset) or
``keypad_submit`` (``#``; needs ``SCONE_VOICE_KEYPAD``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from ..core.errors import InvalidInput
from ..realtime.idle import IdlePolicy
from ..realtime.turn_strategy import MAX_MIN_SPEECH_S, MIN_SPEECH_S, STRATEGIES, KeypadSubmit, MinSpeech

if TYPE_CHECKING:
    from .config import Settings


def environment_seconds(name: str, raw: Optional[str]) -> Optional[float]:
    """A number of seconds from the environment, None when unset or blank."""
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        raise InvalidInput(f"{name} must be a number of seconds, got {raw!r}") from None


def validate_voice_settings(settings: Settings) -> None:
    timeout = settings.voice_idle_timeout
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
        raise InvalidInput(f"SCONE_VOICE_IDLE_TIMEOUT must be a number of seconds, 0 for off, got {timeout!r}")
    if not isinstance(settings.voice_idle_prompt, str) or not settings.voice_idle_prompt.strip():
        raise InvalidInput("SCONE_VOICE_IDLE_PROMPT must be text to say")
    if type(settings.voice_idle_end_after) is not int or settings.voice_idle_end_after < 0:
        raise InvalidInput("SCONE_VOICE_IDLE_END_AFTER must be a whole number of idles, 0 for never")
    if settings.voice_turn_strategy not in STRATEGIES:
        raise InvalidInput(f"SCONE_VOICE_TURN_STRATEGY must be one of {', '.join(STRATEGIES)}, "
                           f"got {settings.voice_turn_strategy!r}")
    if settings.voice_turn_strategy == "keypad_submit" and settings.voice_keypad == "off":
        raise InvalidInput("SCONE_VOICE_TURN_STRATEGY=keypad_submit needs SCONE_VOICE_KEYPAD=append or collect")
    if settings.voice_min_speech is not None:
        if settings.voice_turn_strategy != "min_speech":
            raise InvalidInput("SCONE_VOICE_MIN_SPEECH needs SCONE_VOICE_TURN_STRATEGY=min_speech")
        try:
            MinSpeech(settings.voice_min_speech)
        except ValueError:
            raise InvalidInput(f"SCONE_VOICE_MIN_SPEECH must be a number of seconds in (0, {MAX_MIN_SPEECH_S:g}], "
                               f"got {settings.voice_min_speech!r}") from None


def build_voice_turns(settings: Settings) -> dict[str, object]:
    """The conversation service's voice_idle and voice_turn_strategy; empty when both are off."""
    options: dict[str, object] = {}
    if settings.voice_idle_timeout > 0:
        options["voice_idle"] = IdlePolicy(settings.voice_idle_timeout, settings.voice_idle_prompt,
                                           settings.voice_idle_end_after or None)
    if settings.voice_turn_strategy == "min_speech":
        seconds = settings.voice_min_speech
        options["voice_turn_strategy"] = MinSpeech(MIN_SPEECH_S if seconds is None else seconds)
    elif settings.voice_turn_strategy == "keypad_submit":
        options["voice_turn_strategy"] = KeypadSubmit()
    return options
