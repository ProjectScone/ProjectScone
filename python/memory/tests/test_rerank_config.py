"""Optional reranking settings reach production and in-process engines equally."""

import sys
from types import ModuleType

import pytest

from scone_memory import HashEmbedder, InvalidInput
from scone_memory.runtime import config


def test_optional_reranking_settings_are_disabled_by_default():
    settings = config.Settings.from_env({})
    assert settings.candidate_limit is None and settings.reranker_factory is None
    assert (settings.rerank_limit, settings.rerank_max_bytes, settings.rerank_timeout) == (32, 64000, 1.0)


def test_reranking_settings_read_explicit_bounded_values():
    settings = config.Settings.from_env({"SCONE_RECALL_CANDIDATES": "1000", "SCONE_RERANKER_FACTORY": "trusted:factory",
        "SCONE_RERANK_LIMIT": "128", "SCONE_RERANK_MAX_BYTES": "256000", "SCONE_RERANK_TIMEOUT": "10"})
    assert settings.candidate_limit == 1000 and settings.reranker_factory == "trusted:factory"
    assert (settings.rerank_limit, settings.rerank_max_bytes, settings.rerank_timeout) == (128, 256000, 10.0)


@pytest.mark.parametrize("name,value", [
    ("SCONE_RECALL_CANDIDATES", "0"), ("SCONE_RECALL_CANDIDATES", "1001"), ("SCONE_RECALL_CANDIDATES", "true"),
    ("SCONE_RERANK_LIMIT", "0"), ("SCONE_RERANK_LIMIT", "129"), ("SCONE_RERANK_LIMIT", "1.5"),
    ("SCONE_RERANK_MAX_BYTES", "0"), ("SCONE_RERANK_MAX_BYTES", "511"), ("SCONE_RERANK_MAX_BYTES", "256001"),
    ("SCONE_RERANK_TIMEOUT", "0"), ("SCONE_RERANK_TIMEOUT", "10.01"),
    ("SCONE_RERANK_TIMEOUT", "nan"), ("SCONE_RERANK_TIMEOUT", "inf"),
    ("SCONE_RERANKER_FACTORY", "missing-colon"), ("SCONE_RERANKER_FACTORY", "a:b:c"),
    ("SCONE_RECALL_CANDIDATES", False), ("SCONE_RERANK_TIMEOUT", True), ("SCONE_RERANK_LIMIT", 32),
])
def test_invalid_reranking_configuration_fails_before_startup(name, value):
    with pytest.raises(InvalidInput, match=name):
        config.Settings.from_env({name: value})


@pytest.mark.parametrize("options", [{"candidate_limit": True}, {"rerank_limit": False},
    {"rerank_max_bytes": 1.5}, {"rerank_timeout": True}, {"rerank_timeout": float("nan")}])
def test_direct_settings_construction_enforces_the_same_bounds(options):
    with pytest.raises(InvalidInput):
        config.Settings(**options)


@pytest.mark.parametrize("in_process", [False, True])
async def test_every_engine_builder_forwards_candidate_and_reranker_settings(monkeypatch, in_process):
    from scone_memory.retrieval.reranking import RerankScore

    calls = []

    class LocalReranker:
        closed = False

        async def rerank(self, query, candidates):
            calls.append((query, candidates))
            return [RerankScore(chunk_id=candidate.chunk_id, score=float(index))
                    for index, candidate in enumerate(candidates)]

        async def aclose(self):
            self.closed = True

    reranker = LocalReranker()
    module = ModuleType("scone_test_local_reranker")
    module.factory = lambda: reranker
    monkeypatch.setitem(sys.modules, module.__name__, module)
    settings = config.Settings.from_env({"SCONE_RECALL_CANDIDATES": "3",
        "SCONE_RERANKER_FACTORY": module.__name__ + ":factory", "SCONE_RERANK_LIMIT": "2",
        "SCONE_RERANK_MAX_BYTES": "4096", "SCONE_RERANK_TIMEOUT": ".5"})
    engine = (await config.build_in_process_engine(settings, HashEmbedder()) if in_process
              else await config.build_engine(settings))
    try:
        assert engine.candidate_limit == 3 and engine.reranker is reranker
        assert (engine.rerank_limit, engine.rerank_max_bytes, engine.rerank_timeout) == (2, 4096, .5)
        await engine.remember("alpha", "Project telescope calibration checklist")
        await engine.remember("alpha", "Project telescope calibration runbook")
        result = await engine.recall("alpha", "telescope calibration", limit=2)
        assert calls and len(calls[0][1]) == 2
        assert all(item.rerank_score is not None for item in result.items)
    finally:
        await engine.close()
    assert not reranker.closed, "the injecting operator owns adapter shutdown"
    await reranker.aclose()


@pytest.mark.parametrize("invalid", ["sync_method", "async_factory", "required_argument", "factory_exception"])
async def test_invalid_factory_is_refused_before_opening_backends(monkeypatch, invalid):
    class SyncReranker:
        def rerank(self, query, candidates):
            return []

    async def async_factory():
        return SyncReranker()

    def raises():
        raise RuntimeError("private-provider-secret")

    module = ModuleType("scone_test_invalid_reranker")
    module.factory = {"sync_method": lambda: SyncReranker(), "async_factory": async_factory,
                      "required_argument": lambda required: required, "factory_exception": raises}[invalid]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def forbidden(*args):
        pytest.fail("stores were initialized before reranker validation")

    monkeypatch.setattr(config, "build_documents", forbidden)
    with pytest.raises(InvalidInput, match="SCONE_RERANKER_FACTORY") as error:
        await config.build_engine(config.Settings(reranker_factory=module.__name__ + ":factory"))
    assert "private-provider-secret" not in str(error.value)
