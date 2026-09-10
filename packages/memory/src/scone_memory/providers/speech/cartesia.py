"""Cartesia streaming bytes endpoint; no Cartesia/Pipecat runtime dependency."""

from ._http import PCMSpeech


class CartesiaSpeech(PCMSpeech):
    """Explicit voice/model on Cartesia's pinned 2026-08-14 HTTP contract."""

    provider = 'cartesia'

    def _request(self, text):
        return ('https://api.cartesia.ai/tts/bytes',
                {'Authorization': 'Bearer ' + self._api_key, 'Cartesia-Version': '2026-08-14'},
                {'model_id': self._choice.model, 'transcript': text, 'voice': self._choice.voice,
                 'output_format': {'container': 'raw', 'encoding': 'pcm_s16le', 'sample_rate': self._sample_rate}})
