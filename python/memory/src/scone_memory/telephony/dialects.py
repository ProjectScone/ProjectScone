"""What differs between carriers is field names, not behaviour.

Each carrier streams the same call over a socket: a start message, media
messages carrying base64 audio, digits, a stop, and a way to say "drop
what you are still holding" for barge-in. Only the spelling changes, so
the spelling is data and the handling is written once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Dialect:
    """One carrier's spelling of the media-stream protocol."""

    name: str
    #: Where the stream's identifier is written, at the top level of a
    #: message or inside its start object.
    stream_key: str
    #: Names the call identifier may go by inside the start object.
    call_keys: Sequence[str] = ()
    codec: str = "ulaw"
    rate: int = 8000
    start_event: str = "start"
    media_event: str = "media"
    stop_event: str = "stop"
    dtmf_event: str = "dtmf"
    #: What this carrier calls "drop the audio you have queued", or None
    #: where it offers no such message and barge-in cannot be immediate.
    clear_event: Optional[str] = "clear"
    #: Where the base64 audio sits inside a media message.
    payload_keys: Sequence[str] = ("payload",)
    #: Where a digit sits inside a dtmf message.
    digit_keys: Sequence[str] = ("digit", "digits")


DIALECTS: dict[str, Dialect] = {
    d.name: d for d in (
        Dialect(name="twilio", stream_key="streamSid", call_keys=("callSid",)),
        Dialect(name="telnyx", stream_key="stream_id", call_keys=("call_control_id", "call_leg_id")),
        Dialect(name="plivo", stream_key="streamId", call_keys=("callId",), clear_event="clearAudio"),
        Dialect(name="exotel", stream_key="stream_sid", call_keys=("call_sid",), codec="pcm"),
        Dialect(name="genesys", stream_key="id", call_keys=("conversationId",)),
    )
}
