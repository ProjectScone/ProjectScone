"""OpenAI-shaped audio transcription; no vendor SDK is imported."""

from ._http import BufferedTranscription


class OpenAITranscription(BufferedTranscription):
    """The multipart transcription contract several hosts now speak, so a
    compatible gateway works by changing the URL alone."""

    provider = 'openai'

    def __init__(self, *, url: str = 'https://api.openai.com/v1/audio/transcriptions', **options):
        super().__init__(**options)
        self._url = url

    def _request(self):
        return (self._url,
                {'Authorization': 'Bearer ' + self._api_key},
                {'model': self._model, 'response_format': 'json'})

    def _text(self, payload):
        return payload.get('text', '') if isinstance(payload, dict) else ''
