"""ElevenLabs streaming speech endpoint, translated to Scone PCM chunks."""

from ._http import PCMSpeech


class ElevenLabsSpeech(PCMSpeech):
    """Explicit voice/model; no account, model or voice fallback."""

    provider = 'elevenlabs'

    def _request(self, text):
        return (f'https://api.elevenlabs.io/v1/text-to-speech/{self._choice.voice}/stream?output_format=pcm_{self._sample_rate}',
                {'xi-api-key': self._api_key}, {'model_id': self._choice.model, 'text': text})
