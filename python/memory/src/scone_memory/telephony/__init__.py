"""Phone calls: G.711 companding and the carriers' media-stream dialects."""

from . import g711
from .dialects import DIALECTS, Dialect
from .stream import CallEnded, CallStarted, Dtmf, MediaStream

__all__ = ["CallEnded", "CallStarted", "DIALECTS", "Dialect", "Dtmf", "MediaStream", "g711"]
