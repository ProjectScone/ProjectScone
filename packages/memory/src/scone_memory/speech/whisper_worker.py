"""Isolated local model worker; stdin WAV, stdout bounded JSON, no saved audio."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import import_module
import io
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Protocol, cast
import wave

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class _Segment(Protocol):
    text: str


class _CpuModel(Protocol):
    def transcribe(self, samples: NDArray[np.float32], *, language: str, temperature: int,
                   condition_on_previous_text: bool, beam_size: int,
                   vad_filter: bool) -> tuple[Iterable[_Segment], object]: ...


class _CpuBackend(Protocol):
    def WhisperModel(self, path: str, *, device: str, compute_type: str,
                     cpu_threads: int, num_workers: int, local_files_only: bool) -> _CpuModel: ...


class _MlxBackend(Protocol):
    def transcribe(self, samples: NDArray[np.float32], *, path_or_hf_repo: str,
                   language: str, temperature: int, verbose: None,
                   condition_on_previous_text: bool) -> Mapping[str, object]: ...


def main() -> None:
    import numpy as np

    model = Path(sys.argv[1])
    backend = sys.argv[3] if len(sys.argv) > 3 else 'mlx'
    if backend not in ('mlx', 'faster-whisper'):
        raise ValueError('Unknown local dictation backend')
    required = ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt') if backend == 'faster-whisper' else ('config.json',)
    if not model.is_absolute() or not all((model / name).is_file() for name in required):
        raise ValueError('A downloaded local model is required')
    audio = sys.stdin.buffer.read(2100001)
    if len(audio) > 2100000:
        raise ValueError('Audio limit')
    with wave.open(io.BytesIO(audio)) as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2 or reader.getframerate() != 16000:
            raise ValueError('Audio format')
        data = reader.readframes(960001)
    samples = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0
    if not len(samples) or len(samples) > 960000:
        raise ValueError('Audio duration')
    text = ''
    if float(np.sqrt(np.mean(samples * samples))) > 0.0001:
        if backend == 'faster-whisper':
            backend_module = cast(_CpuBackend, import_module('faster_whisper'))

            transcriber = backend_module.WhisperModel(str(model), device='cpu', compute_type='int8',
                cpu_threads=2, num_workers=1, local_files_only=True)
            segments, _ = transcriber.transcribe(samples, language=sys.argv[2], temperature=0,
                condition_on_previous_text=False, beam_size=1, vad_filter=False)
            for segment in segments:
                text += segment.text
                if len(text) > 16000:
                    raise ValueError('Transcript limit')
            text = text.strip()
        else:
            mlx_whisper = cast(_MlxBackend, import_module('mlx_whisper'))

            result = mlx_whisper.transcribe(samples, path_or_hf_repo=str(model), language=sys.argv[2],
                temperature=0, verbose=None, condition_on_previous_text=False)
            text = str(result.get('text', '')).strip()
    if len(text) > 16000:
        raise ValueError('Transcript limit')
    print(json.dumps({'text': text}, ensure_ascii=False))


if __name__ == '__main__':
    main()
