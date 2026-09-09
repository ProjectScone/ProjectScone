"""Runtime initialization must disable telemetry before native imports."""
from __future__ import annotations

import builtins
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest

from scone_memory.embedders.local import LocalEmbedder
from scone_memory.testing.edge_retrieval_benchmark import _CachedBGE


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('prior', [None, '0'])
def test_embedding_disables_telemetry_before_import_and_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached: bool, prior: str | None,
) -> None:
    if prior is None:
        monkeypatch.delenv('ORT_DISABLE_TELEMETRY', raising=False)
    else:
        monkeypatch.setenv('ORT_DISABLE_TELEMETRY', prior)
    events: list[str] = []
    onnx = ModuleType('onnxruntime')
    def disable() -> None:
        assert os.environ.get('ORT_DISABLE_TELEMETRY') == '1'
        events.append('disable')
    setattr(onnx, 'disable_telemetry_events', disable)
    fastembed = ModuleType('fastembed')
    def factory(**kwargs: object) -> object:
        assert events == ['disable']
        events.append('construct')
        return object()
    setattr(fastembed, 'TextEmbedding', factory)
    monkeypatch.setitem(sys.modules, 'onnxruntime', onnx)
    monkeypatch.setitem(sys.modules, 'fastembed', fastembed)
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in ('onnxruntime', 'fastembed'):
            assert os.environ.get('ORT_DISABLE_TELEMETRY') == '1', 'telemetry enabled at import'
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    if cached:
        _CachedBGE(tmp_path)
    else:
        LocalEmbedder(cache_dir=str(tmp_path))
    assert events == ['disable', 'construct']
