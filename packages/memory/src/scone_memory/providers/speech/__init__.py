"""Direct speech providers for Scone's native audio protocol."""

from .cartesia import CartesiaSpeech
from .elevenlabs import ElevenLabsSpeech
from .self_hosted_openai import SelfHostedOpenAISpeech
from .local_openai import LocalOpenAISpeech
from ._http import SpeechProviderError

__all__ = ['CartesiaSpeech', 'ElevenLabsSpeech', 'SelfHostedOpenAISpeech', 'LocalOpenAISpeech', 'SpeechProviderError']
