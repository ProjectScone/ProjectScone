"""Offline reranking uses fake runtimes: no model inference or downloads."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from scone_memory.providers.offline_reranker import OfflineCrossEncoderReranker
from scone_memory.retrieval.reranking import RerankCandidate

NAME = "Xenova/ms-marco-MiniLM-L-6-v2"


def candidate(number: int = 1, text: str = "matching document") -> RerankCandidate:
    return RerankCandidate(number,number,text,None,"2025-01-01T00:00:00Z",.5,None,(("text",1),))


@dataclass
class Encoding:
    ids: list[int]


class Tokenizer:
    clones: list[Tokenizer] = []

    def __init__(self) -> None:
        self.truncation: dict[str,object] | None = {"max_length":512}
        self.padding: dict[str,object] | None = {"length":None,"pad_to_multiple_of":None}
        self.pairs: list[tuple[str,str]] = []

    def to_str(self) -> str:
        return json.dumps(dict(truncation=self.truncation,padding=self.padding))

    @classmethod
    def from_str(cls,value: str) -> Tokenizer:
        parsed = json.loads(value)
        instance = cls()
        instance.truncation,instance.padding = parsed["truncation"],parsed["padding"]
        cls.clones.append(instance)
        return instance

    def no_truncation(self) -> None: self.truncation = None
    def no_padding(self) -> None: self.padding = None

    def encode(self,sequence: str,pair: str,add_special_tokens: bool = True) -> Encoding:
        self.pairs.append((sequence,pair))
        assert self.truncation is None and self.padding is None
        # Fake expansion makes punctuation expensive despite short raw text.
        count = len(sequence.split())+len(pair.split())+pair.count("!")*10+int(add_special_tokens)*3
        return Encoding(list(range(count)))


class Runtime:
    def __init__(self) -> None:
        self.model = SimpleNamespace(tokenizer=Tokenizer())
        self.kwargs: dict[str,object] = {}
        self.scores: list[object] = [-4.5,8.0]
        self.score_calls: list[tuple[str,list[str],int]] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.worker_ids: list[int] = []
        self.telemetry_disabled = False

    def rerank(self,query: str,documents: list[str],batch_size: int) -> list[object]:
        self.worker_ids.append(threading.get_ident())
        self.score_calls.append((query,documents,batch_size))
        self.entered.set()
        if self.block:
            assert self.release.wait(3)
        return self.scores


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Runtime:
    result = Runtime()
    Tokenizer.clones = []
    def factory(**kwargs: object) -> Runtime:
        assert result.telemetry_disabled
        result.kwargs = kwargs
        return result
    fastembed = ModuleType("fastembed.rerank.cross_encoder")
    setattr(fastembed,"TextCrossEncoder",factory)
    tokenizers = ModuleType("tokenizers")
    setattr(tokenizers,"Tokenizer",Tokenizer)
    monkeypatch.setitem(sys.modules,"fastembed.rerank.cross_encoder",fastembed)
    monkeypatch.setitem(sys.modules,"tokenizers",tokenizers)
    onnx = ModuleType("onnxruntime")
    def disable_telemetry() -> None:
        result.telemetry_disabled = True
    setattr(onnx,"disable_telemetry_events",disable_telemetry)
    monkeypatch.setitem(sys.modules,"onnxruntime",onnx)
    return result


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    (tmp_path/"onnx").mkdir()
    (tmp_path/"config.json").write_text(json.dumps(dict(max_position_embeddings=512,pad_token_id=0)))
    (tmp_path/"tokenizer_config.json").write_text(json.dumps(dict(model_max_length=512,pad_token="[PAD]")))
    (tmp_path/"tokenizer.json").write_text("{}")
    (tmp_path/"special_tokens_map.json").write_text("{}")
    (tmp_path/"onnx/model.onnx").write_bytes(b"fake preprovisioned model")
    return tmp_path


async def test_cpu_local_flags_full_tokenizer_clone_and_raw_score_identity(model_dir: Path,runtime: Runtime) -> None:
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    scores = await reranker.rerank("a query",(candidate(7),candidate(3,"different")))
    assert [(score.chunk_id,score.score) for score in scores] == [(7,-4.5),(3,8.0)]
    assert reranker.calls == 1
    assert runtime.kwargs["local_files_only"] is True
    assert runtime.kwargs["specific_model_path"] == str(model_dir)
    assert runtime.kwargs["providers"] == ["CPUExecutionProvider"] and runtime.kwargs["cuda"] is False
    assert runtime.kwargs["threads"] == 2 and runtime.telemetry_disabled
    assert runtime.score_calls[0][2] == 8
    assert runtime.worker_ids == [runtime.worker_ids[0]] and runtime.worker_ids[0] != threading.get_ident()
    assert runtime.model.tokenizer.truncation is not None and runtime.model.tokenizer.padding is not None
    assert Tokenizer.clones[0].pairs == [("a query","matching document"),("a query","different")]


def test_model_identity_hashes_local_files_without_paths(model_dir: Path,runtime: Runtime) -> None:
    reranker = OfflineCrossEncoderReranker(str(model_dir),model_name=NAME)
    identity = reranker.model_identity
    assert identity["model_name"] == NAME
    hashes = cast(dict[str,str],identity["sha256"])
    assert hashes["onnx/model.onnx"] == hashlib.sha256(b"fake preprovisioned model").hexdigest()
    assert set(hashes) >= {"config.json","tokenizer_config.json","tokenizer.json","special_tokens_map.json","onnx/model.onnx"}
    assert str(model_dir) not in json.dumps(identity)
    hashes.clear()
    assert reranker.model_identity["sha256"] != {}


@pytest.mark.parametrize("problem",["missing","unknown_model","outside_symlink","bad_config"])
def test_files_and_name_rejected_before_runtime_load(model_dir: Path,runtime: Runtime,tmp_path: Path,problem: str) -> None:
    name = NAME
    if problem == "missing": (model_dir/"tokenizer.json").unlink()
    if problem == "unknown_model": name = "https://unsafe.example/model.py"
    if problem == "outside_symlink":
        (model_dir/"tokenizer.json").unlink()
        (model_dir/"tokenizer.json").symlink_to("/etc/hosts")
    if problem == "bad_config": (model_dir/"config.json").write_text('{"max_position_embeddings":true}')
    with pytest.raises(ValueError) as caught: OfflineCrossEncoderReranker(model_dir,model_name=name)
    assert runtime.kwargs == {} and str(model_dir) not in str(caught.value)


@pytest.mark.parametrize("option,value",[("max_pair_tokens",True),("max_pair_tokens",1),("max_pair_tokens",8193),
    ("threads",0),("threads",17),("batch_size",0),("batch_size",129)])
def test_strict_constructor_bounds(model_dir: Path,runtime: Runtime,option: str,value: int) -> None:
    with pytest.raises(ValueError): OfflineCrossEncoderReranker(model_dir,model_name=NAME,**{option:value})
    assert runtime.kwargs == {}


async def test_complete_pair_rejected_before_any_scoring(model_dir: Path,runtime: Runtime) -> None:
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME,max_pair_tokens=8)
    with pytest.raises(ValueError,match="offline reranking unavailable"):
        await reranker.rerank("q",(candidate(1,"fits"),candidate(2,"!")))
    assert runtime.score_calls == [] and reranker.calls == 0


async def test_token_limit_is_minimum_of_operator_config_and_loaded_tokenizer(model_dir: Path,runtime: Runtime) -> None:
    (model_dir/"config.json").write_text('{"max_position_embeddings":8}')
    runtime.model.tokenizer.truncation = {"max_length":7}
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME,max_pair_tokens=100)
    assert reranker.model_identity["max_pair_tokens"] == 7
    runtime.scores = [1.0]
    assert len(await reranker.rerank("q",(candidate(1,"one two three"),))) == 1
    with pytest.raises(ValueError): await reranker.rerank("q",(candidate(1,"one two three four"),))
    assert reranker.calls == 1


@pytest.mark.parametrize("bad",[[],(candidate(),candidate()),(replace(candidate(),chunk_id=True),),
    (replace(candidate(),episode_id=0),),(replace(candidate(),baseline_score=float("nan")),),
    (replace(candidate(),lanes=(("text",True),)),),tuple(candidate(i) for i in range(1,130))])
async def test_strict_candidate_shape_ids_and_bounds(model_dir: Path,runtime: Runtime,bad: object) -> None:
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    with pytest.raises(ValueError): await reranker.rerank("q",cast(tuple[RerankCandidate,...],bad))
    assert reranker.calls == 0 and runtime.score_calls == []


@pytest.mark.parametrize("query",["", "q"*8001, "海"*3000])
async def test_query_byte_budget(model_dir: Path,runtime: Runtime,query: str) -> None:
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    with pytest.raises(ValueError): await reranker.rerank(query,(candidate(),))
    assert reranker.calls == 0


async def test_total_candidate_json_utf8_budget(model_dir: Path,runtime: Runtime) -> None:
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    with pytest.raises(ValueError): await reranker.rerank("q",(candidate(1,"海"*50000),candidate(2,"海"*50000)))
    assert Tokenizer.clones[0].pairs == [] and reranker.calls == 0


@pytest.mark.parametrize("scores",[[],[1.0],[1.0,2.0,3.0],[True,1.0],[float("inf"),1.0],[float("nan"),1.0],["2",1.0]])
async def test_score_shape_and_finiteness_fail_closed(model_dir: Path,runtime: Runtime,scores: list[object]) -> None:
    runtime.scores = scores
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    with pytest.raises(ValueError): await reranker.rerank("q",(candidate(1),candidate(2)))
    assert reranker.calls == 1


async def test_empty_input_does_not_score(model_dir: Path,runtime: Runtime) -> None:
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    assert await reranker.rerank("q",()) == () and reranker.calls == 0


async def test_cancelled_cpu_worker_keeps_slot_until_done(model_dir: Path,runtime: Runtime) -> None:
    runtime.block = True
    runtime.scores = [3.0]
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    first = asyncio.create_task(reranker.rerank("first",(candidate(),)))
    assert await asyncio.to_thread(runtime.entered.wait,2)
    first.cancel()
    with pytest.raises(asyncio.CancelledError): await first
    second = asyncio.create_task(reranker.rerank("second",(candidate(),)))
    await asyncio.sleep(.02)
    assert len(runtime.score_calls) == 1
    runtime.release.set()
    result = await asyncio.wait_for(second,2)
    assert result[0].score == 3.0 and len(runtime.score_calls) == 2


async def test_waiting_cancel_does_not_enqueue_extra_inference(model_dir: Path,runtime: Runtime) -> None:
    runtime.block = True
    runtime.scores = [3.0]
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    first = asyncio.create_task(reranker.rerank("first",(candidate(),)))
    assert await asyncio.to_thread(runtime.entered.wait,2)
    waiting = asyncio.create_task(reranker.rerank("waiting",(candidate(),)))
    await asyncio.sleep(.01)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError): await waiting
    runtime.release.set()
    await asyncio.wait_for(first,2)
    assert len(runtime.score_calls) == 1


async def test_exact_full_json_budget_boundary(model_dir: Path,runtime: Runtime) -> None:
    from dataclasses import asdict
    from scone_memory.retrieval.reranking import MAX_RERANK_BYTES
    base = candidate(text="word")
    overhead = len(json.dumps({"query":"q","candidates":[asdict(base)]},ensure_ascii=False,separators=(",",":")).encode()) - len(base.text)
    runtime.scores = [2.0]
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    exact = replace(base,text="x"*(MAX_RERANK_BYTES-overhead))
    assert len(await reranker.rerank("q",(exact,))) == 1
    with pytest.raises(ValueError): await reranker.rerank("q",(replace(exact,text=exact.text+"x"),))
    assert reranker.calls == 1


async def test_full_question_tokens_count_toward_pair_limit(model_dir: Path,runtime: Runtime) -> None:
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME,max_pair_tokens=6)
    with pytest.raises(ValueError): await reranker.rerank("one two three",(candidate(text="one two"),))
    assert runtime.score_calls == []


async def test_queued_candidate_snapshot_is_immutable(model_dir: Path,runtime: Runtime) -> None:
    runtime.block = True
    runtime.scores = [3.0]
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    first = asyncio.create_task(reranker.rerank("first",(candidate(),)))
    assert await asyncio.to_thread(runtime.entered.wait,2)
    queued = candidate(7,"original queued text")
    second = asyncio.create_task(reranker.rerank("second",(queued,)))
    await asyncio.sleep(.01)
    object.__setattr__(queued,"chunk_id",99)
    object.__setattr__(queued,"text","mutated text")
    runtime.release.set()
    await first
    scores = await second
    assert scores[0].chunk_id == 7
    assert runtime.score_calls[-1][1] == ["original queued text"]


def test_extra_model_weights_are_hashed(model_dir: Path,runtime: Runtime) -> None:
    (model_dir/"onnx/model.onnx_data").write_bytes(b"external local weights")
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    hashes = cast(dict[str,str],reranker.model_identity["sha256"])
    assert hashes["onnx/model.onnx_data"] == hashlib.sha256(b"external local weights").hexdigest()


def test_dependency_error_names_install_extra_without_local_path(model_dir: Path,runtime: Runtime,monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    def missing(name: str) -> ModuleType: raise ImportError("private installation path")
    monkeypatch.setattr(importlib,"import_module",missing)
    with pytest.raises(ValueError) as caught: OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    assert str(caught.value) == "offline reranking requires scone-memory[offline-rerank]"


def test_load_errors_are_content_free(model_dir: Path,runtime: Runtime,monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(**kwargs: object) -> Runtime: raise RuntimeError("private model directory")
    monkeypatch.setattr(sys.modules["fastembed.rerank.cross_encoder"],"TextCrossEncoder",failed)
    with pytest.raises(ValueError) as caught: OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    assert str(caught.value) == "offline reranking unavailable"


def test_changed_model_during_load_rejects_identity(model_dir: Path,runtime: Runtime,monkeypatch: pytest.MonkeyPatch) -> None:
    def changed(**kwargs: object) -> Runtime:
        (model_dir/"onnx/model.onnx").write_bytes(b"different weights during load")
        return runtime
    monkeypatch.setattr(sys.modules["fastembed.rerank.cross_encoder"],"TextCrossEncoder",changed)
    with pytest.raises(ValueError): OfflineCrossEncoderReranker(model_dir,model_name=NAME)


async def test_inference_error_releases_slot_and_is_sanitized(model_dir: Path,runtime: Runtime,monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    original = runtime.rerank
    def failed(query: str,documents: list[str],batch_size: int) -> list[object]: raise RuntimeError("private question")
    monkeypatch.setattr(runtime,"rerank",failed)
    with pytest.raises(ValueError) as caught: await reranker.rerank("q",(candidate(),))
    assert str(caught.value) == "offline reranking unavailable"
    monkeypatch.setattr(runtime,"rerank",original)
    runtime.scores = [1.0]
    assert len(await reranker.rerank("q",(candidate(),))) == 1


async def test_mutated_candidate_extra_fields_are_rejected(model_dir: Path,runtime: Runtime) -> None:
    item = candidate()
    object.__setattr__(item,"unexpected","private metadata")
    reranker = OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    with pytest.raises(ValueError): await reranker.rerank("q",(item,))
    assert reranker.calls == 0


@pytest.mark.parametrize("padding",[{"length":1024},{"length":True},{"length":-1},
    {"length":None,"pad_to_multiple_of":1000},{"length":None,"pad_to_multiple_of":3},
    {"length":None,"pad_to_multiple_of":True},{"length":None,"pad_to_multiple_of":0}])
def test_incompatible_loaded_padding_is_rejected(model_dir: Path,runtime: Runtime,padding: dict[str,object]) -> None:
    runtime.model.tokenizer.padding = padding
    with pytest.raises(ValueError): OfflineCrossEncoderReranker(model_dir,model_name=NAME)
    assert runtime.score_calls == []


@pytest.mark.parametrize("padding",[None,{"length":None,"pad_to_multiple_of":None},
    {"length":256,"pad_to_multiple_of":8}])
def test_bounded_dynamic_and_fixed_padding_remain_supported(model_dir: Path,runtime: Runtime,padding: dict[str,object] | None) -> None:
    runtime.model.tokenizer.padding = padding
    assert OfflineCrossEncoderReranker(model_dir,model_name=NAME).model_identity["max_pair_tokens"] == 512


def test_native_tokenizer_fixed_padding_counterexample_is_rejected(model_dir: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    # Real tokenizer only, without model loading/inference or network access.
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer as NativeTokenizer
    native = NativeTokenizer.from_str(json.dumps({"version":"1.0","truncation":None,"padding":None,
        "added_tokens":[],"normalizer":None,"pre_tokenizer":{"type":"Whitespace"},"post_processor":None,
        "decoder":None,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"q":1,"d":2},"unk_token":"[UNK]"}}))
    native.enable_truncation(max_length=512)
    native.enable_padding(length=1024)
    clone = NativeTokenizer.from_str(native.to_str())
    clone.no_truncation()
    clone.no_padding()
    assert len(clone.encode("q","d").ids) == 2
    assert len(native.encode_batch([("q","d")])[0].ids) == 1024
    def factory(**kwargs: object) -> SimpleNamespace: return SimpleNamespace(model=SimpleNamespace(tokenizer=native))
    module = ModuleType("fastembed.rerank.cross_encoder")
    setattr(module,"TextCrossEncoder",factory)
    monkeypatch.setitem(sys.modules,"fastembed.rerank.cross_encoder",module)
    onnx = ModuleType("onnxruntime")
    setattr(onnx,"disable_telemetry_events",lambda: None)
    monkeypatch.setitem(sys.modules,"onnxruntime",onnx)
    with pytest.raises(ValueError,match="offline reranking unavailable"):
        OfflineCrossEncoderReranker(model_dir,model_name=NAME)
