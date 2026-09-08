"""Direct transcription providers for Scone's native audio protocol."""

from ._http import BufferedTranscription, TranscriptionProviderError
from .openai import OpenAITranscription
from .self_hosted_openai import SelfHostedOpenAITranscription
from .local_openai import LocalOpenAITranscription

__all__ = ['BufferedTranscription', 'OpenAITranscription', 'SelfHostedOpenAITranscription', 'LocalOpenAITranscription', 'TranscriptionProviderError']
