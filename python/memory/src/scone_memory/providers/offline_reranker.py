"""Explicitly provisioned CPU cross-encoder scoring without network access.

Only built-in plain ONNX cross encoders are supported. Scores are raw logits,
not probabilities or accuracy judgments. Full query/document pairs are checked
with an untruncated clone of the loaded tokenizer before any batch is scored.
Loaded padding configurations that could exceed the same cap are rejected.

Inference runs off the event loop with at most one outstanding worker per
instance. Cancellation cannot interrupt an ONNX CPU kernel: callers cancel
promptly, late results are discarded, and the slot stays occupied until that
worker exits. ONNX Runtime telemetry is disabled before session construction.
No private executor, downloads, or environment changes are used.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import asdict
import hashlib
import importlib
from itertools import islice
import json
import math
from pathlib import Path
import threading
from typing import Protocol, cast

from ..retrieval.reranking import MAX_RERANK_BYTES, MAX_RERANK_LIMIT, RerankCandidate, RerankScore

_MODELS = frozenset({"Xenova/ms-marco-MiniLM-L-6-v2", "Xenova/ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-base"})
_REQUIRED = ("config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "onnx/model.onnx")
_ERROR = "offline reranking unavailable"


class _Encoding(Protocol):
    @property
    def ids(self) -> list[int]: ...


class _Tokenizer(Protocol):
    @property
    def truncation(self) -> object: ...
    @property
    def padding(self) -> object: ...
    def to_str(self) -> str: ...
    def no_truncation(self) -> None: ...
    def no_padding(self) -> None: ...
    def encode(self, sequence: str, pair: str, add_special_tokens: bool = True) -> _Encoding: ...


class _TokenizerFactory(Protocol):
    def from_str(self, value: str) -> _Tokenizer: ...


class _Inner(Protocol):
    @property
    def tokenizer(self) -> _Tokenizer: ...


class _Encoder(Protocol):
    @property
    def model(self) -> _Inner: ...
    def rerank(self, query: str, documents: list[str], batch_size: int) -> Iterable[object]: ...


class _EncoderFactory(Protocol):
    def __call__(self, *, model_name: str, specific_model_path: str, local_files_only: bool,
                 providers: list[str], cuda: bool, threads: int, lazy_load: bool) -> _Encoder: ...


def _integer(value: int, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(_ERROR)


def _object(pairs: list[tuple[str,object]]) -> dict[str,object]:
    result: dict[str,object] = {}
    for key,value in pairs:
        if key in result:
            raise ValueError(_ERROR)
        result[key] = value
    return result


def _json(path: Path) -> dict[str,object]:
    if path.stat().st_size > 2_000_000:
        raise ValueError(_ERROR)
    value: object = json.loads(path.read_text(),object_pairs_hook=_object)
    if not isinstance(value,dict):
        raise ValueError(_ERROR)
    return cast(dict[str,object],value)


def _file_state(path: Path) -> tuple[int,int,int,int]:
    state = path.stat()
    return state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns


def _provisioned(path: Path) -> tuple[dict[str,str],dict[str,tuple[int,int,int,int]]]:
    if not path.is_dir():
        raise ValueError(_ERROR)
    names = set(_REQUIRED)
    extras = list(islice((path/"onnx").iterdir(),33))
    if len(extras) > 32:
        raise ValueError(_ERROR)
    names.update(f"onnx/{entry.name}" for entry in extras)
    hashes: dict[str,str] = {}
    states: dict[str,tuple[int,int,int,int]] = {}
    total = 0
    for name in sorted(names):
        entry = path/name
        if not entry.is_file() or not entry.resolve().is_relative_to(path):
            raise ValueError(_ERROR)
        state = _file_state(entry)
        maximum = 2_000_000_000 if name.startswith("onnx/") else 32_000_000
        if not 0 < state[2] <= maximum:
            raise ValueError(_ERROR)
        total += state[2]
        if total > 4_000_000_000:
            raise ValueError(_ERROR)
        digest = hashlib.sha256()
        read_bytes = 0
        with entry.open("rb") as handle:
            for block in iter(lambda: handle.read(1024*1024),b""):
                read_bytes += len(block)
                if read_bytes > state[2]:
                    raise ValueError(_ERROR)
                digest.update(block)
        if _file_state(entry) != state:
            raise ValueError(_ERROR)
        hashes[name],states[name] = digest.hexdigest(),state
    return hashes,states


def _configured_limit(path: Path, operator: int) -> int:
    config,tokenizer = _json(path/"config.json"),_json(path/"tokenizer_config.json")
    limits = [operator]
    found = False
    for mapping,key in ((config,"max_position_embeddings"),(tokenizer,"model_max_length"),(tokenizer,"max_length")):
        if key not in mapping:
            continue
        value = mapping[key]
        if type(value) is not int or value < 2:
            raise ValueError(_ERROR)
        if value <= 8192:
            limits.append(value)
            found = True
        # Hugging Face uses a huge positive sentinel for unknown token limits.
        # It is not usable as proof; another recognized finite cap is required.
    if not found or not ("model_max_length" in tokenizer or "max_length" in tokenizer):
        raise ValueError(_ERROR)
    return min(limits)


def _padding(padding: object, maximum: int) -> None:
    if padding is None:
        return
    if not isinstance(padding,dict):
        raise ValueError(_ERROR)
    length: object = padding.get("length")
    multiple: object = padding.get("pad_to_multiple_of")
    if length is not None and (type(length) is not int or not 1 <= length <= maximum):
        raise ValueError(_ERROR)
    # Full pairs up to the cap must remain within it after batch padding.
    # Do not silently normalize a preprovisioned tokenizer's behavior.
    if multiple is not None and (type(multiple) is not int or not 1 <= multiple <= maximum or maximum % multiple):
        raise ValueError(_ERROR)


def _finite(value: object) -> bool:
    return type(value) in (int,float) and math.isfinite(cast(float,value))


def _bounded_text(value: object, maximum: int, *, nonblank: bool = False) -> bool:
    return (type(value) is str and len(value) <= maximum and (not nonblank or bool(value.strip()))
            and len(value.encode("utf-8")) <= maximum)


def _input(query: str, candidates: tuple[RerankCandidate,...]) -> tuple[RerankCandidate,...]:
    if not _bounded_text(query,8000,nonblank=True) or type(candidates) is not tuple or len(candidates) > MAX_RERANK_LIMIT:
        raise ValueError(_ERROR)
    copied: list[RerankCandidate] = []
    ids: set[int] = set()
    payload_bytes = len(json.dumps({"query":query,"candidates":[]},ensure_ascii=False,separators=(",",":")).encode())
    for candidate in candidates:
        if type(candidate) is not RerankCandidate or set(vars(candidate)) != set(RerankCandidate.__dataclass_fields__):
            raise ValueError(_ERROR)
        _integer(candidate.chunk_id,1,2**63-1)
        _integer(candidate.episode_id,1,2**63-1)
        if (candidate.chunk_id in ids or not _bounded_text(candidate.text,MAX_RERANK_BYTES,nonblank=True)
                or not _bounded_text(candidate.created_at,128,nonblank=True)
                or (candidate.source is not None and not _bounded_text(candidate.source,MAX_RERANK_BYTES))
                or not _finite(candidate.baseline_score)
                or (candidate.similarity is not None and not _finite(candidate.similarity))
                or type(candidate.lanes) is not tuple or len(candidate.lanes) > 16):
            raise ValueError(_ERROR)
        for lane in candidate.lanes:
            if type(lane) is not tuple or len(lane) != 2 or not _bounded_text(lane[0],128,nonblank=True):
                raise ValueError(_ERROR)
            _integer(lane[1],1,1_000_000)
        ids.add(candidate.chunk_id)
        snapshot = RerankCandidate(candidate.chunk_id,candidate.episode_id,candidate.text,candidate.source,
            candidate.created_at,candidate.baseline_score,candidate.similarity,candidate.lanes)
        payload_bytes += len(json.dumps(asdict(snapshot),ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()) + bool(copied)
        if payload_bytes > MAX_RERANK_BYTES:
            raise ValueError(_ERROR)
        copied.append(snapshot)
    return tuple(copied)


class OfflineCrossEncoderReranker:
    """Rank supplied candidate IDs with explicit local CPU assets only.

    Construction synchronously validates, fingerprints, and loads the model;
    request-time token checks and inference use the serialized worker. Install
    ``scone-memory[offline-rerank]`` to enable the optional model runtime.
    """
    def __init__(self, model_dir: Path | str, *, model_name: str, max_pair_tokens: int = 512,
                 threads: int = 2, batch_size: int = 8) -> None:
        _integer(max_pair_tokens,2,8192)
        _integer(threads,1,16)
        _integer(batch_size,1,128)
        if type(model_name) is not str or model_name not in _MODELS or not isinstance(model_dir,(str,Path)):
            raise ValueError(_ERROR)
        try:
            path = Path(model_dir).resolve(strict=True)
            hashes,states = _provisioned(path)
            maximum = _configured_limit(path,max_pair_tokens)
        except Exception:
            raise ValueError(_ERROR) from None
        try:
            encoder_factory = cast(_EncoderFactory,getattr(importlib.import_module("fastembed.rerank.cross_encoder"),"TextCrossEncoder"))
            tokenizer_factory = cast(_TokenizerFactory,getattr(importlib.import_module("tokenizers"),"Tokenizer"))
            disable_telemetry = cast(Callable[[],None],getattr(importlib.import_module("onnxruntime"),"disable_telemetry_events"))
        except ImportError:
            raise ValueError("offline reranking requires scone-memory[offline-rerank]") from None
        except Exception:
            raise ValueError(_ERROR) from None
        try:
            disable_telemetry()
            self._encoder = encoder_factory(model_name=model_name,specific_model_path=str(path),local_files_only=True,
                providers=["CPUExecutionProvider"],cuda=False,threads=threads,lazy_load=False)
            loaded = self._encoder.model.tokenizer
            truncation = loaded.truncation
            if not isinstance(truncation,dict):
                raise ValueError(_ERROR)
            loaded_limit: object = truncation.get("max_length")
            if type(loaded_limit) is not int or not 2 <= loaded_limit <= 8192:
                raise ValueError(_ERROR)
            self._max_pair_tokens = min(maximum,loaded_limit)
            _padding(loaded.padding,self._max_pair_tokens)
            self._tokenizer = tokenizer_factory.from_str(loaded.to_str())
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()
            if any(_file_state(path/name) != state for name,state in states.items()):
                raise ValueError(_ERROR)
        except Exception:
            raise ValueError(_ERROR) from None
        self._hashes,self._model_name = hashes,model_name
        self._batch_size = batch_size
        self._calls = 0
        self._gate = asyncio.Lock()

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def model_identity(self) -> dict[str,object]:
        return {"model_name":self._model_name,"sha256":self._hashes.copy(),"max_pair_tokens":self._max_pair_tokens}

    def _score(self, query: str, candidates: tuple[RerankCandidate,...], cancelled: threading.Event) -> tuple[RerankScore,...]:
        try:
            for candidate in candidates:
                if cancelled.is_set():
                    raise ValueError(_ERROR)
                tokens = self._tokenizer.encode(query,candidate.text,add_special_tokens=True)
                if len(tokens.ids) > self._max_pair_tokens:
                    raise ValueError(_ERROR)
            if cancelled.is_set():
                raise ValueError(_ERROR)
            self._calls += 1
            raw = self._encoder.rerank(query,[candidate.text for candidate in candidates],batch_size=self._batch_size)
            values = list(islice(raw,len(candidates)+1))
            if cancelled.is_set() or len(values) != len(candidates) or not all(_finite(value) for value in values):
                raise ValueError(_ERROR)
            return tuple(RerankScore(candidate.chunk_id,float(cast(float,value))) for candidate,value in zip(candidates,values))
        except Exception:
            raise ValueError(_ERROR) from None

    async def rerank(self, query: str, candidates: tuple[RerankCandidate,...]) -> tuple[RerankScore,...]:
        try:
            fixed = _input(query,candidates)
        except Exception:
            raise ValueError(_ERROR) from None
        if not fixed:
            return ()
        await self._gate.acquire()
        cancelled = threading.Event()
        worker = asyncio.create_task(asyncio.to_thread(self._score,query,fixed,cancelled))
        release_here = True
        try:
            result = await asyncio.shield(worker)
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise asyncio.CancelledError()
            return result
        except asyncio.CancelledError:
            cancelled.set()
            release_here = False
            def finished(completed: asyncio.Task[tuple[RerankScore,...]]) -> None:
                if not completed.cancelled():
                    completed.exception()
                self._gate.release()
            worker.add_done_callback(finished)
            raise
        finally:
            if release_here:
                self._gate.release()
