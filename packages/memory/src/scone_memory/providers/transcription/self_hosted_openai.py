"""Utterance transcription through an explicitly configured self-hosted HTTP service."""

import httpx
from typing import Optional

from ...realtime.audio import SpeechActivityDetector
from ..self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier
from ._http import BufferedTranscription


class SelfHostedOpenAITranscription(BufferedTranscription):
    """OpenAI-compatible multipart audio/transcriptions, JSON {'text': ...}.

    A self-hosted Whisper service must support this specific contract. This is one
    bounded mono WAV request per utterance, not streaming partial recognition.
    No default endpoint, model, credential or cloud fallback is supplied.
    """

    provider = 'self-hosted-openai'
    requires_api_key = False

    def __init__(self, *, base_url: str, model: str, api_key: Optional[str] = None,
                 rate: int = 16000, timeout: float = 30, max_audio_bytes: int = 8_000_000,
                 max_transcript_bytes: int = 64000, detector: Optional[SpeechActivityDetector] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        self._url = validate_self_hosted_endpoint(base_url) + 'audio/transcriptions'
        validate_self_hosted_identifier(model)
        super().__init__(api_key=api_key, model=model, rate=rate, timeout=timeout,
                         max_audio_bytes=max_audio_bytes, max_transcript_bytes=max_transcript_bytes,
                         detector=detector, transport=transport)

    def _request(self) -> tuple[str, dict[str, str], dict[str, str]]:
        headers = {'Authorization': 'Bearer ' + self._api_key} if self._api_key else {}
        return self._url, headers, {'model': self._model, 'response_format': 'json'}

    def _text(self, payload: object) -> str:
        if not isinstance(payload, dict):
            return ''
        text = payload.get('text', '')
        return text if isinstance(text, str) else ''
