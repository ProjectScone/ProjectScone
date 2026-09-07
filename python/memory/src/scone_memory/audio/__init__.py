"""Audio a call needs before a provider sees it: exact frames, honest
rate conversion, loudness and a speech gate. Standard library only."""

from . import pcm
from .framer import Framer
from .gate import VoiceGate
from .rate import QUALITY, Resampler

__all__ = ["Framer", "QUALITY", "Resampler", "VoiceGate", "pcm"]
