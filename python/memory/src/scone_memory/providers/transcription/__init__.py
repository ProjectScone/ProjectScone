"""Direct transcription providers for Scone's native audio protocol."""

from ._http import BufferedTranscription, TranscriptionProviderError
from .openai import OpenAITranscription

__all__ = ['BufferedTranscription', 'OpenAITranscription', 'TranscriptionProviderError']
