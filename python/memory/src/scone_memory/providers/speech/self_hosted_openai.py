"""Raw PCM from an explicitly selected self-hosted OpenAI-compatible TTS service."""

import httpx

from ..self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier
from ._http import PCMSpeech


class SelfHostedOpenAISpeech(PCMSpeech):
    """POST audio/speech with model, voice, input and response_format=pcm.

    The service must return mono signed 16-bit little-endian PCM at sample_rate.
    Standard OpenAI PCM is 24000 Hz; a self-hosted service may use another
    rate. This option describes the service output, it does not request a rate
    change or resample. WAV/MP3 responses are rejected. Bare Piper's HTTP API is
    not this contract: use an explicitly compatible self-hosted wrapper instead.
    """

    provider = 'self-hosted-openai'
    requires_api_key = False
    opaque_voice = False
    sample_rates = None

    def __init__(self, *, base_url: str, model: str, voice: str, api_key: str | None = None,
                 sample_rate: int = 24000, timeout: float = 30, chunk_bytes: int = 4096,
                 max_audio_bytes: int = 24_000_000,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._url = validate_self_hosted_endpoint(base_url) + 'audio/speech'
        self._model = validate_self_hosted_identifier(model)
        self._voice = validate_self_hosted_identifier(voice, max_length=120)
        super().__init__(api_key=api_key, model='self-hosted-model', voice='self-hosted-voice', sample_rate=sample_rate,
                         timeout=timeout, chunk_bytes=chunk_bytes, max_audio_bytes=max_audio_bytes,
                         transport=transport)

    def _request(self, text: str) -> tuple[str, dict[str, str], dict[str, object]]:
        headers = {'Authorization': 'Bearer ' + self._api_key} if self._api_key else {}
        return (self._url, headers,
                {'model': self._model, 'voice': self._voice,
                 'input': text, 'response_format': 'pcm'})
