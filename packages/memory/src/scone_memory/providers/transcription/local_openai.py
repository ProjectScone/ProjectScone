"""Compatibility import for the self-hosted OpenAI-compatible transcription adapter."""
from .self_hosted_openai import SelfHostedOpenAITranscription

LocalOpenAITranscription = SelfHostedOpenAITranscription

__all__ = ['LocalOpenAITranscription']
