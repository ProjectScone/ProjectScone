"""Direct speech providers for Scone's native audio protocol."""

from .cartesia import CartesiaSpeech
from .elevenlabs import ElevenLabsSpeech
from ._http import SpeechProviderError

__all__ = ['CartesiaSpeech', 'ElevenLabsSpeech', 'SpeechProviderError']
