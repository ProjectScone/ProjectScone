"""Compatibility import for the self-hosted OpenAI-compatible speech adapter."""
from .self_hosted_openai import SelfHostedOpenAISpeech

LocalOpenAISpeech = SelfHostedOpenAISpeech

__all__ = ['LocalOpenAISpeech']
