"""Explicit reads and interpretation of an immutable retained video catalogue."""
from __future__ import annotations

import json
import hashlib
import struct
from typing import Mapping, Protocol

from ._wire import ResourceClient, WorkflowTransport, integer, invalid
from .video_models import VideoCatalogue, VideoInterpretation, characters


class VideoTransport(WorkflowTransport, Protocol):
    def _video_frame(self, episode_id: int, ordinal: int, maximum: int) -> tuple[bytes, Mapping[str, str]]: ...


class VideoDocuments(ResourceClient):
    def __init__(self, client: VideoTransport, *, expected_space: str) -> None:
        super().__init__(client, expected_space=expected_space)
        self._transport = client

    def frame(self, source: VideoCatalogue, ordinal: int) -> bytes:
        self._validate(source, ordinal)
        self._current(source)
        selected = source.frame(ordinal)
        data, headers = self._transport._video_frame(source.episode_id, ordinal, selected.png_bytes)
        expected = {'x-scone-video-frame-sha256': selected.png_sha256,
                    'x-scone-video-frame-ordinal': str(ordinal),
                    'x-scone-video-pts': str(selected.presentation_timestamp),
                    'x-scone-video-time-base': source.time_base}
        if (headers.get('content-type', '').split(';')[0].strip() != 'image/png'
                or any(headers.get(key) != value for key, value in expected.items())
                or len(data) != selected.png_bytes or hashlib.sha256(data).hexdigest() != selected.png_sha256
                or len(data) < 33 or data[:8] != b'\x89PNG\r\n\x1a\n'
                or struct.unpack('>I4sII', data[8:24]) != (13, b'IHDR', selected.width, selected.height)):
            raise invalid('video PNG does not match retained frame evidence')
        self._current(source)
        return data

    def catalogue(self, episode_id: int) -> VideoCatalogue:
        integer(episode_id, 1)
        self._check('documents.provenance', mutation=True)
        return VideoCatalogue.from_json(self._client._request('GET', f'/v1/episodes/{episode_id}/document/video/catalogue'),
                                        expected_space=self.expected_space, episode_id=episode_id)

    def _validate(self, source: VideoCatalogue, ordinal: int) -> None:
        if not isinstance(source, VideoCatalogue) or source.space != self.expected_space:
            raise invalid('video catalogue space')
        source.frame(ordinal)
        try:
            original = VideoCatalogue.from_json(json.loads(source._encoded), expected_space=self.expected_space, episode_id=source.episode_id)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            raise invalid('video catalogue snapshot') from None
        if original != source:
            raise invalid('video catalogue snapshot was modified')

    def _current(self, source: VideoCatalogue) -> None:
        if self.catalogue(source.episode_id) != source:
            raise invalid('video evidence changed; read a new catalogue')

    def interpret(self, source: VideoCatalogue, ordinal: int, *, prompt: str) -> VideoInterpretation:
        characters(prompt, 16000)
        self._validate(source, ordinal)
        self._check('documents.video.understand', mutation=True)
        self._current(source)
        body = self._client._request('POST', f'/v1/episodes/{source.episode_id}/document/video/frames/{ordinal}/understand',
                                     json={'prompt': prompt})
        result = VideoInterpretation.from_json(body, source, ordinal)
        self._current(source)
        return result
