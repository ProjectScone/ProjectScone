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


def test_offline_cross_encoder_env_is_explicit_and_keeps_caps():
    settings = config.Settings.from_env({"SCONE_RERANKER_CROSS_ENCODER_DIR": "/models/provisioned",
        "SCONE_RERANKER_CROSS_ENCODER_MODEL": "Xenova/ms-marco-MiniLM-L-6-v2",
        "SCONE_RERANKER_CROSS_ENCODER_MAX_PAIR_TOKENS": "256", "SCONE_RERANKER_CROSS_ENCODER_THREADS": "3",
        "SCONE_RERANKER_CROSS_ENCODER_BATCH_SIZE": "4", "SCONE_RERANK_LIMIT": "12", "SCONE_RERANK_TIMEOUT": ".5"})
    assert settings.reranker_cross_encoder_dir == "/models/provisioned"
    assert settings.reranker_cross_encoder_model == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert (settings.reranker_cross_encoder_max_pair_tokens, settings.reranker_cross_encoder_threads,
            settings.reranker_cross_encoder_batch_size) == (256, 3, 4)
    assert settings.reranker_factory is None and (settings.rerank_limit, settings.rerank_timeout) == (12, .5)


@pytest.mark.parametrize("options", [
    {"SCONE_RERANKER_CROSS_ENCODER_DIR": "/private-model"},
    {"SCONE_RERANKER_CROSS_ENCODER_MODEL": "private-model"},
    {"SCONE_RERANKER_CROSS_ENCODER_DIR": "  ", "SCONE_RERANKER_CROSS_ENCODER_MODEL": "model"},
    {"SCONE_RERANKER_CROSS_ENCODER_DIR": "/models", "SCONE_RERANKER_CROSS_ENCODER_MODEL": "  "},
    {"SCONE_RERANKER_CROSS_ENCODER_DIR": "/models", "SCONE_RERANKER_CROSS_ENCODER_MODEL": "model",
     "SCONE_RERANKER_FACTORY": "trusted:factory"},
    {"SCONE_RERANKER_CROSS_ENCODER_MAX_PAIR_TOKENS": "512"},
    {"SCONE_RERANKER_CROSS_ENCODER_THREADS": "2"},
    {"SCONE_RERANKER_CROSS_ENCODER_BATCH_SIZE": "8"},
    {"SCONE_RERANKER_CROSS_ENCODER_DIR": True},
    {"SCONE_RERANKER_CROSS_ENCODER_MODEL": 123},
    {"SCONE_RERANKER_CROSS_ENCODER_THREADS": True},
])
def test_offline_pair_conflicts_or_orphans_are_refused_without_private_values(options):
    with pytest.raises(InvalidInput, match="SCONE_RERANKER") as error:
        config.Settings.from_env(options)
    assert "private-model" not in str(error.value)


@pytest.mark.parametrize("name,value", [
    ("MAX_PAIR_TOKENS", "1"), ("MAX_PAIR_TOKENS", "8193"), ("MAX_PAIR_TOKENS", "private-secret"),
    ("THREADS", "0"), ("THREADS", "17"), ("THREADS", "1.5"),
    ("BATCH_SIZE", "0"), ("BATCH_SIZE", "129"), ("BATCH_SIZE", ""),
])
def test_offline_tuning_bounds_match_provider(name, value):
    with pytest.raises(InvalidInput, match=f"SCONE_RERANKER_CROSS_ENCODER_{name}") as error:
        config.Settings.from_env({"SCONE_RERANKER_CROSS_ENCODER_DIR": "/models",
            "SCONE_RERANKER_CROSS_ENCODER_MODEL": "model", f"SCONE_RERANKER_CROSS_ENCODER_{name}": value})
    assert "private-secret" not in str(error.value)


@pytest.mark.parametrize("options", [
    {"reranker_cross_encoder_dir": True, "reranker_cross_encoder_model": "model"},
    {"reranker_cross_encoder_dir": "/models", "reranker_cross_encoder_model": False},
    {"reranker_cross_encoder_threads": 2},
    {"reranker_cross_encoder_dir": "/models", "reranker_cross_encoder_model": "model", "reranker_cross_encoder_threads": True},
    {"reranker_cross_encoder_dir": "/models", "reranker_cross_encoder_model": "model", "reranker_cross_encoder_max_pair_tokens": 2.0},
])
def test_offline_direct_settings_enforce_strict_types(options):
    with pytest.raises(InvalidInput, match="SCONE_RERANKER"):
        config.Settings(**options)


def test_unset_cross_encoder_does_not_import_optional_provider(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not any(part in name for part in ("offline_reranker", "onnxruntime", "tokenizers"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    settings = config.Settings.from_env({"SCONE_RERANKER_CROSS_ENCODER_DIR": "", "SCONE_RERANKER_CROSS_ENCODER_MODEL": ""})
    assert settings.reranker_cross_encoder_dir is None and settings.reranker_cross_encoder_model is None
    assert config.build_reranker(settings) is None


@pytest.mark.parametrize("in_process", [False, True])
@pytest.mark.parametrize("tuned", [False, True])
async def test_every_engine_builder_constructs_explicit_offline_reranker(monkeypatch, in_process, tuned):
    from scone_memory.retrieval.reranking import RerankScore

    initialized, calls = [], []

    class OfflineReranker:
        def __init__(self, model_dir, **options):
            initialized.append((model_dir, options))

        async def rerank(self, query, candidates):
            calls.append((query, candidates))
            return [RerankScore(chunk_id=candidate.chunk_id, score=float(index)) for index, candidate in enumerate(candidates)]

    module = ModuleType("scone_memory.providers.offline_reranker")
    module.OfflineCrossEncoderReranker = OfflineReranker
    monkeypatch.setitem(sys.modules, module.__name__, module)
    env = {"SCONE_RERANKER_CROSS_ENCODER_DIR": "/models/provisioned",
        "SCONE_RERANKER_CROSS_ENCODER_MODEL": "model", "SCONE_RECALL_CANDIDATES": "3", "SCONE_RERANK_LIMIT": "2"}
    if tuned:
        env.update(SCONE_RERANKER_CROSS_ENCODER_MAX_PAIR_TOKENS="256", SCONE_RERANKER_CROSS_ENCODER_THREADS="3",
                   SCONE_RERANKER_CROSS_ENCODER_BATCH_SIZE="4")
    settings = config.Settings.from_env(env)
    engine = (await config.build_in_process_engine(settings, HashEmbedder()) if in_process else await config.build_engine(settings))
    try:
        expected = {"model_name": "model", "max_pair_tokens": 256 if tuned else 512,
                    "threads": 3 if tuned else 2, "batch_size": 4 if tuned else 8}
        assert initialized == [("/models/provisioned", expected)]
        assert engine.candidate_limit == 3 and engine.rerank_limit == 2
        await engine.remember("alpha", "Project telescope calibration checklist")
        await engine.remember("alpha", "Project telescope calibration runbook")
        result = await engine.recall("alpha", "telescope calibration", limit=2)
        assert calls and len(calls[0][1]) == 2 and all(item.rerank_score is not None for item in result.items)
    finally:
        await engine.close()


async def test_offline_load_failure_is_sanitized_before_backend_open(monkeypatch):
    module = ModuleType("scone_memory.providers.offline_reranker")

    def broken(*args, **kwargs):
        raise RuntimeError("/private/model/path private-credential")

    def forbidden(*args):
        pytest.fail("stores initialized before offline reranker validation")

    module.OfflineCrossEncoderReranker = broken
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(config, "build_documents", forbidden)
    with pytest.raises(InvalidInput, match="SCONE_RERANKER_CROSS_ENCODER") as error:
        await config.build_engine(config.Settings(reranker_cross_encoder_dir="/private/model/path", reranker_cross_encoder_model="model"))
    assert "private" not in str(error.value) and error.value.__suppress_context__


LISTWISE_CHAT = {"SCONE_CHAT_URL": "http://127.0.0.1:11434/v1", "SCONE_CHAT_MODEL": "llama3.2-ctx8k"}


def test_listwise_reranker_is_off_unless_asked_and_reads_its_bounds():
    assert config.Settings.from_env(LISTWISE_CHAT).reranker_listwise is False
    assert config.build_reranker(config.Settings.from_env(LISTWISE_CHAT)) is None
    settings = config.Settings.from_env({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE": "1",
        "SCONE_RERANKER_LISTWISE_WINDOW": "12", "SCONE_RERANKER_LISTWISE_STEP": "4",
        "SCONE_RERANKER_LISTWISE_PASSAGE_BYTES": "700", "SCONE_RERANKER_LISTWISE_TIMEOUT": "45"})
    assert settings.reranker_listwise is True
    assert (settings.reranker_listwise_window, settings.reranker_listwise_step,
            settings.reranker_listwise_passage_bytes, settings.reranker_listwise_timeout) == (12, 4, 700, 45.0)


@pytest.mark.parametrize("options,named", [
    ({"SCONE_RERANKER_LISTWISE": "1"}, "SCONE_CHAT"),
    ({"SCONE_RERANKER_LISTWISE": "1", "SCONE_CHAT_URL": "http://127.0.0.1:11434/v1"}, "SCONE_CHAT"),
    ({"SCONE_RERANKER_LISTWISE": "1", "SCONE_CHAT_MODEL": "llama3.2-ctx8k"}, "SCONE_CHAT"),
    ({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE": "maybe"}, "SCONE_RERANKER_LISTWISE"),
    ({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE": "1", "SCONE_RERANKER_FACTORY": "trusted:factory"}, "SCONE_RERANKER_LISTWISE"),
    ({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE": "1", "SCONE_RERANKER_CROSS_ENCODER_DIR": "/models",
      "SCONE_RERANKER_CROSS_ENCODER_MODEL": "model"}, "SCONE_RERANKER_LISTWISE"),
    ({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE_WINDOW": "10"}, "SCONE_RERANKER_LISTWISE_WINDOW"),
    ({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE_TIMEOUT": "10"}, "SCONE_RERANKER_LISTWISE_TIMEOUT"),
    ({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE": "1", "SCONE_RERANKER_LISTWISE_WINDOW": "21"}, "SCONE_RERANKER_LISTWISE_WINDOW must"),
    ({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE": "1", "SCONE_RERANKER_LISTWISE_STEP": "20"}, "SCONE_RERANKER_LISTWISE_STEP must"),
    ({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE": "1", "SCONE_RERANKER_LISTWISE_PASSAGE_BYTES": "63"}, "SCONE_RERANKER_LISTWISE_PASSAGE_BYTES must"),
    ({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE": "1", "SCONE_RERANKER_LISTWISE_TIMEOUT": "601"}, "SCONE_RERANKER_LISTWISE_TIMEOUT must"),
    ({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE": "1", "SCONE_RERANKER_LISTWISE_WINDOW": "ten"}, "SCONE_RERANKER_LISTWISE_WINDOW"),
])
def test_listwise_settings_that_cannot_be_honoured_are_refused(options, named):
    with pytest.raises(InvalidInput, match=named):
        config.Settings.from_env(options)


@pytest.mark.parametrize("in_process", [False, True])
async def test_every_engine_builder_constructs_the_listwise_reranker_on_the_chat_model(in_process):
    from scone_memory.providers.llm import OpenAICompatibleChat
    from scone_memory.retrieval.listwise import ListwiseReranker

    settings = config.Settings.from_env({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE": "1",
        "SCONE_RERANKER_LISTWISE_WINDOW": "12", "SCONE_RERANKER_LISTWISE_STEP": "4",
        "SCONE_RERANKER_LISTWISE_PASSAGE_BYTES": "700", "SCONE_RERANKER_LISTWISE_TIMEOUT": "45"})
    engine = (await config.build_in_process_engine(settings, HashEmbedder()) if in_process
              else await config.build_engine(settings))
    try:
        ranker = engine.reranker
        assert isinstance(ranker, ListwiseReranker) and isinstance(ranker.chat, OpenAICompatibleChat)
        assert ranker.chat.model == "llama3.2-ctx8k"
        assert (ranker.window, ranker.step, ranker.passage_bytes, ranker.timeout) == (12, 4, 700, 45.0)
    finally:
        await engine.close()


def test_listwise_switch_must_be_a_boolean_when_set_directly():
    with pytest.raises(InvalidInput, match="SCONE_RERANKER_LISTWISE must be 1 or 0"):
        config.Settings(chat_url="http://127.0.0.1:11434/v1", chat_model="llama3.2-ctx8k", reranker_listwise=1)


def test_listwise_step_defaults_to_half_a_configured_window():
    ranker = config.build_reranker(config.Settings.from_env({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE": "1",
                                                             "SCONE_RERANKER_LISTWISE_WINDOW": "12"}))
    assert (ranker.window, ranker.step) == (12, 6)


def test_listwise_defaults_apply_when_only_switched_on():
    from scone_memory.retrieval.listwise import DEFAULT_PASSAGE_BYTES, DEFAULT_STEP, DEFAULT_TIMEOUT, DEFAULT_WINDOW

    ranker = config.build_reranker(config.Settings.from_env({**LISTWISE_CHAT, "SCONE_RERANKER_LISTWISE": "1"}))
    assert (ranker.window, ranker.step, ranker.passage_bytes, ranker.timeout) == (
        DEFAULT_WINDOW, DEFAULT_STEP, DEFAULT_PASSAGE_BYTES, DEFAULT_TIMEOUT)
