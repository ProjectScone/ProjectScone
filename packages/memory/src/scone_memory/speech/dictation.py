"""Bounded local speech transcription with no retained audio or cloud fallback."""
from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
import re
import signal
import wave

from ..agents.workflow import WorkflowError

MAX_AUDIO_BYTES = 5 * 1024 * 1024
MAX_SECONDS = 60
_FORMATS = {'audio/webm': 'matroska', 'audio/mp4': 'mov', 'audio/ogg': 'ogg',
            'audio/wav': 'wav', 'audio/x-wav': 'wav'}


def validate_dictation_format(media_type: str, language: str) -> None:
    if media_type not in _FORMATS or not re.fullmatch(r'[a-z]{2,3}', language):
        raise WorkflowError('invalid_dictation_format')


async def _run(arguments: list[str], content: bytes, *, timeout: float, limit: int) -> bytes:
    process = await asyncio.create_subprocess_exec(*arguments, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
        env={'PATH': os.defpath, 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
             'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false',
             'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1'})
    try:
        async with asyncio.timeout(timeout):
            output, _ = await process.communicate(content)
        if process.returncode or len(output) > limit:
            raise WorkflowError('dictation_processing_failed')
        return output
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()


class LocalDictation:
    """Uses explicitly configured local executables and already-downloaded weights."""
    def __init__(self, python: str, model: str, ffmpeg: str, *, backend: str = 'mlx') -> None:
        if backend not in ('mlx', 'faster-whisper'):
            raise ValueError('Local dictation backend must be mlx or faster-whisper')
        self.python, self.model, self.ffmpeg = python, model, ffmpeg
        self.backend = backend
        self._active = False
        if not self.available:
            raise ValueError('Local dictation requires executable Python/ffmpeg and downloaded model files')

    @property
    def available(self) -> bool:
        model = Path(self.model)
        if self.backend == 'faster-whisper':
            weights_available = all((model / name).is_file() for name in
                                    ('model.bin', 'tokenizer.json', 'vocabulary.txt'))
        else:
            weights_available = any((model / name).is_file() for name in ('weights.npz', 'model.safetensors'))
        return (all(Path(path).is_absolute() and Path(path).is_file() and os.access(path, os.X_OK)
                    for path in (self.python, self.ffmpeg))
                and model.is_absolute() and (model / 'config.json').is_file() and weights_available)

    async def transcribe(self, content: bytes, media_type: str, language: str) -> str:
        if not self.available:
            raise WorkflowError('dictation_unavailable')
        validate_dictation_format(media_type, language)
        if not content or len(content) > MAX_AUDIO_BYTES:
            raise WorkflowError('invalid_dictation_size')
        if self._active:
            raise WorkflowError('dictation_busy')
        self._active = True
        try:
            audio = await _run([self.ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                '-protocol_whitelist', 'pipe', '-f', _FORMATS[media_type], '-i', 'pipe:0',
                '-vn', '-sn', '-dn', '-t', str(MAX_SECONDS + 1), '-ac', '1', '-ar', '16000',
                '-acodec', 'pcm_s16le', '-f', 'wav', '-fs', '2100000', 'pipe:1'], content, timeout=20, limit=2100000)
            try:
                with wave.open(io.BytesIO(audio)) as reader:
                    if reader.getnchannels() != 1 or reader.getframerate() != 16000 or reader.getsampwidth() != 2:
                        raise ValueError('audio format')
                    # Pipe WAV headers can advertise unknown frame counts.
                    samples = reader.readframes((MAX_SECONDS + 1) * 16000)
                    if len(samples) > MAX_SECONDS * 32000 or not samples:
                        raise WorkflowError('invalid_dictation_duration')
            except (wave.Error, EOFError, ValueError):
                raise WorkflowError('invalid_dictation_audio') from None
            worker = str(Path(__file__).with_name('whisper_worker.py'))
            raw = await _run([self.python, '-I', worker, self.model, language, self.backend], audio, timeout=90, limit=128000)
            try:
                value = json.loads(raw)
                if not isinstance(value, dict) or not isinstance(value.get('text'), str) or len(value['text']) > 16000:
                    raise ValueError('text limit')
                return str(value['text']).strip()
            except (ValueError, UnicodeError):
                raise WorkflowError('dictation_processing_failed') from None
        except TimeoutError:
            raise WorkflowError('dictation_timeout') from None
        except OSError:
            raise WorkflowError('dictation_unavailable') from None
        finally:
            self._active = False
