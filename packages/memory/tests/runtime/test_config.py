import os

import pytest

from scone_memory import InvalidInput
from scone_memory.runtime.config import Settings, build_engine, parse_keys
from scone_memory.embedders.hash import HashEmbedder


def test_keys_parse_and_refuse_duplicates():
    assert parse_keys("k1:alpha, k2:beta", None) == {"k1": "alpha", "k2": "beta"}
    assert parse_keys(None, "solo") == {"solo": "default"}
    assert parse_keys("", "") == {}
    with pytest.raises(InvalidInput):
        parse_keys("k1:alpha,k1:beta", None)
    with pytest.raises(InvalidInput):
        parse_keys("nocolon", None)


def test_settings_read_the_environment():
    env = {"SCONE_DOCUMENTS": "sqlite", "SCONE_API_KEY": "k", "SCONE_PORT": "9000"}
    s = Settings.from_env(env)
    assert (s.documents, s.vectors, s.embedder, s.port, s.keys) == ("sqlite", "memory", "hash", 9000, {"k": "default"})
    assert s.events_queries == "hash", "query text is never logged unless asked"


async def test_build_engine_wires_the_named_parts(tmp_path):
    env = {"SCONE_DOCUMENTS": "sqlite", "SCONE_SQLITE_PATH": str(tmp_path / "m.db"), "SCONE_VECTORS": "memory"}
    engine = await build_engine(Settings.from_env(env))
    added = await engine.remember("default", "wired through the environment")
    assert (await engine.recall("default", "environment")).items[0].episode_id == added.episode_id
    status = await engine.status("default")
    assert (status.document_store, status.vector_index, status.embedder) == ("sqlite", "memory", HashEmbedder(256).id)


def test_missing_url_is_a_configuration_error():
    with pytest.raises(InvalidInput):
        build_engine_sync({"SCONE_DOCUMENTS": "mongo"})
    with pytest.raises(InvalidInput):
        build_engine_sync({"SCONE_VECTORS": "qdrant"})
    with pytest.raises(InvalidInput):
        build_engine_sync({"SCONE_EMBEDDER": "remote"})


def build_engine_sync(env):
    import asyncio

    return asyncio.run(build_engine(Settings.from_env(env)))


@pytest.mark.mongo
@pytest.mark.skipif("SCONE_TEST_MONGO_URL" not in os.environ, reason="needs a live MongoDB")
async def test_production_shape_mongo_plus_qdrant_round_trips():
    env = {
        "SCONE_DOCUMENTS": "mongo",
        "SCONE_MONGO_URL": os.environ["SCONE_TEST_MONGO_URL"],
        "SCONE_MONGO_DB": "scone_test_shape",
        "SCONE_VECTORS": "qdrant",
        "SCONE_QDRANT_URL": os.environ.get("SCONE_TEST_QDRANT_URL", ":memory:"),
    }
    engine = await build_engine(Settings.from_env(env))
    try:
        await engine.remember("default", "the production shape stores documents in mongo and vectors in qdrant")
        result = await engine.recall("default", "where are vectors stored")
        assert result.items and result.items[0].similarity is not None
        assert result.degraded == []
        status = await engine.status("default")
        assert (status.document_store, status.vector_index) == ("mongo", "qdrant")
    finally:
        await engine.documents.drop()
        await engine.vectors.drop()


def test_event_sink_follows_the_document_store_by_default(tmp_path):
    import importlib.util

    from scone_memory.runtime.config import build_events
    from scone_memory.observability.events import InMemoryEventLog, SqliteEventLog

    assert isinstance(build_events(Settings()), InMemoryEventLog)
    assert isinstance(build_events(Settings(documents="sqlite", sqlite_path=str(tmp_path / "e.db"))), SqliteEventLog)
    assert isinstance(build_events(Settings(documents="mongo", mongo_url="mongodb://localhost:1", events="memory")), InMemoryEventLog)
    assert build_events(Settings(events="none")) is None
    with pytest.raises(InvalidInput):
        build_events(Settings(events="mongo"))
    if importlib.util.find_spec("pymongo") is None:
        pytest.skip("pymongo not installed; the mongo default needs the driver to construct")
    from scone_memory.observability.events import MongoEventLog

    assert isinstance(build_events(Settings(documents="mongo", mongo_url="mongodb://localhost:1")), MongoEventLog)


def test_a_slow_model_host_can_be_given_longer_before_it_is_given_up_on():
    """Three of 455 episodes timed out at the fixed 180 seconds during a
    benchmark, on a machine that was busy, and there was no way to say
    "wait longer" short of editing the source. A local model on a loaded
    machine is the ordinary case, not the exotic one."""
    from scone_memory.runtime.config import Settings, build_chat

    settings = Settings.from_env({"SCONE_CHAT_URL": "http://127.0.0.1:11434/v1",
                             "SCONE_CHAT_MODEL": "llama3.1", "SCONE_CHAT_TIMEOUT": "900"})
    assert settings.chat_timeout == 900.0
    assert build_chat(settings).timeout == 900.0


def test_the_wait_defaults_to_what_it_was_before():
    from scone_memory.runtime.config import Settings, build_chat

    settings = Settings.from_env({"SCONE_CHAT_URL": "http://127.0.0.1:11434/v1",
                             "SCONE_CHAT_MODEL": "llama3.1"})
    assert build_chat(settings).timeout == 180.0


@pytest.mark.parametrize("value", ["0", "-5", "soon", "", "1e999"])
def test_a_wait_that_is_not_a_length_of_time_is_refused(value):
    """Silently falling back to the default would leave a benchmark
    timing out for the reason it was configured not to."""
    from scone_memory.runtime.config import Settings

    with pytest.raises(InvalidInput, match="SCONE_CHAT_TIMEOUT"):
        Settings.from_env({"SCONE_CHAT_URL": "http://x/v1", "SCONE_CHAT_MODEL": "m",
                      "SCONE_CHAT_TIMEOUT": value})


async def test_many_valued_predicates_are_configured_by_name_and_reach_every_engine():
    from scone_memory.runtime.config import ENGINE_SETTINGS, build_in_process_engine
    from scone_memory import HashEmbedder

    from scone_memory.core.extracted import MANY_VALUED

    settings = Settings.from_env({"SCONE_MANY_VALUED": " knows, Owns ,"})
    assert settings.many_valued == ("knows", "Owns")
    engine = await build_engine(settings)
    # What a person configures sits beside what the framework extracts,
    # which is many-valued by nature and needs no setting.
    assert engine.many_valued - MANY_VALUED == frozenset({"knows", "owns"}) and MANY_VALUED <= engine.many_valued
    unconfigured = await build_engine(Settings.from_env({}))
    assert Settings.from_env({}).many_valued == () and unconfigured.many_valued == MANY_VALUED
    assert "many_valued" in ENGINE_SETTINGS
    in_process = await build_in_process_engine(settings, HashEmbedder())
    assert in_process.many_valued - MANY_VALUED == frozenset({"knows", "owns"})


async def test_table_context_embedding_policy_reaches_standard_and_in_process_engines():
    from scone_memory.runtime.config import build_in_process_engine
    settings = Settings.from_env({'SCONE_TABLE_CONTEXT_EMBEDDINGS': '1'})
    assert settings.table_context_embeddings is True
    assert Settings.from_env({}).table_context_embeddings is False
    engine = await build_engine(settings)
    try:
        assert engine.table_context_embeddings is True
    finally:
        await engine.close()
    local = await build_in_process_engine(settings, HashEmbedder())
    try:
        assert local.table_context_embeddings is True
    finally:
        await local.close()


def test_recency_settings_come_from_the_environment_reach_the_engine_and_refuse_bad_values(tmp_path):
    import asyncio

    import pytest

    from scone_memory.core.errors import InvalidInput
    from scone_memory.retrieval.fusion import RECENCY_HALF_LIFE_DAYS, W_RECENCY

    base = {"SCONE_SQLITE_PATH": str(tmp_path / "m.db"), "SCONE_EMBEDDER": "hash"}
    plain = Settings.from_env(base)
    assert (plain.recency_weight, plain.recency_half_life_days) == (W_RECENCY, RECENCY_HALF_LIFE_DAYS), "the defaults are the constants"
    tuned = Settings.from_env(base | {"SCONE_RECENCY_WEIGHT": "0.02", "SCONE_RECENCY_HALF_LIFE_DAYS": "7"})
    assert (tuned.recency_weight, tuned.recency_half_life_days) == (0.02, 7.0)
    engine = asyncio.run(build_engine(tuned))
    try:
        assert (engine.recency_weight, engine.recency_half_life_days) == (0.02, 7.0)
    finally:
        asyncio.run(engine.close())
    off = asyncio.run(build_engine(Settings.from_env(base | {"SCONE_RECENCY_WEIGHT": "0"})))
    try:
        assert off.recency_weight == 0.0
    finally:
        asyncio.run(off.close())
    for env, named in (({"SCONE_RECENCY_WEIGHT": "-1"}, "SCONE_RECENCY_WEIGHT"), ({"SCONE_RECENCY_WEIGHT": "two"}, "SCONE_RECENCY_WEIGHT"),
                       ({"SCONE_RECENCY_HALF_LIFE_DAYS": "0"}, "SCONE_RECENCY_HALF_LIFE_DAYS"), ({"SCONE_RECENCY_HALF_LIFE_DAYS": "inf"}, "SCONE_RECENCY_HALF_LIFE_DAYS"),
                       ({"SCONE_RECENCY_HALF_LIFE_DAYS": "36501"}, "SCONE_RECENCY_HALF_LIFE_DAYS")):
        with pytest.raises(InvalidInput, match=named):
            Settings.from_env(base | env)
    blank = Settings.from_env(base | {"SCONE_RECENCY_WEIGHT": "", "SCONE_RECENCY_HALF_LIFE_DAYS": ""})
    assert (blank.recency_weight, blank.recency_half_life_days) == (W_RECENCY, RECENCY_HALF_LIFE_DAYS), "an empty value is unset, as the neighbours treat it"


async def test_a_synonym_file_is_read_at_build_time_and_reaches_every_engine(tmp_path):
    from scone_memory import HashEmbedder
    from scone_memory.runtime.config import FILE_SETTINGS, build_in_process_engine, build_synonyms

    path = tmp_path / "synonyms.txt"
    path.write_text("car, automobile\n")
    settings = Settings.from_env({"SCONE_SYNONYMS": str(path)})
    assert settings.synonyms == str(path) and "synonyms" in FILE_SETTINGS
    assert Settings.from_env({}).synonyms is None and build_synonyms(Settings.from_env({})) is None
    engine = await build_engine(settings)
    try:
        assert engine.synonyms is not None and engine.synonyms.record() == {"groups": 1, "terms": 2}
    finally:
        await engine.close()
    in_process = await build_in_process_engine(settings, HashEmbedder())
    assert in_process.synonyms is not None and in_process.synonyms.expand("a car").added == ("automobile",)
    with pytest.raises(InvalidInput, match="not found"):
        await build_engine(Settings.from_env({"SCONE_SYNONYMS": str(tmp_path / "missing.txt")}))


async def test_the_context_lane_is_a_flag_that_reaches_every_engine():
    from scone_memory import HashEmbedder
    from scone_memory.runtime.config import ENGINE_SETTINGS, build_in_process_engine

    settings = Settings.from_env({"SCONE_CONTEXT_LANE": "1"})
    assert settings.context_lane is True and Settings.from_env({}).context_lane is False
    assert "context_lane" in ENGINE_SETTINGS
    engine = await build_engine(settings)
    try:
        assert engine.context_lane is True
    finally:
        await engine.close()
    assert (await build_in_process_engine(settings, HashEmbedder())).context_lane is True
    with pytest.raises(InvalidInput, match="SCONE_CONTEXT_LANE"):
        Settings.from_env({"SCONE_CONTEXT_LANE": "maybe"})


async def test_the_vector_weight_is_a_number_that_reaches_every_engine():
    from scone_memory import HashEmbedder
    from scone_memory.runtime.config import ENGINE_SETTINGS, build_in_process_engine

    settings = Settings.from_env({"SCONE_VECTOR_WEIGHT": "0.5"})
    assert settings.vector_weight == 0.5 and Settings.from_env({}).vector_weight is None and "vector_weight" in ENGINE_SETTINGS
    assert (await build_in_process_engine(Settings.from_env({}), HashEmbedder())).vector_weight == 0.25, "unset follows the embedder"
    engine = await build_engine(settings)
    try:
        assert engine.vector_weight == 0.5
    finally:
        await engine.close()
    assert (await build_in_process_engine(settings, HashEmbedder())).vector_weight == 0.5
    for bad in ("0", "5", "many", "nan"):
        with pytest.raises(InvalidInput, match="SCONE_VECTOR_WEIGHT"):
            Settings.from_env({"SCONE_VECTOR_WEIGHT": bad})


async def test_a_token_chunk_target_is_a_number_that_reaches_every_engine():
    from scone_memory import HashEmbedder
    from scone_memory.runtime.config import ENGINE_SETTINGS, build_in_process_engine

    settings = Settings.from_env({"SCONE_CHUNK_TOKENS": " 512 ", "SCONE_CHUNK_OVERLAP_TOKENS": "32"})
    assert (settings.chunk_tokens, settings.chunk_overlap_tokens) == (512, 32)
    assert (Settings.from_env({}).chunk_tokens, Settings.from_env({}).chunk_overlap_tokens) == (None, 0)
    unset = Settings.from_env({"SCONE_CHUNK_TOKENS": "", "SCONE_CHUNK_OVERLAP_TOKENS": " "})
    assert (unset.chunk_tokens, unset.chunk_overlap_tokens) == (None, 0), "an empty value is unset, as elsewhere"
    assert {"chunk_tokens", "chunk_overlap_tokens"} <= set(ENGINE_SETTINGS)
    engine = await build_engine(settings)
    try:
        assert (engine.chunk_tokens, engine.chunk_overlap_tokens) == (512, 32)
    finally:
        await engine.close()
    in_process = await build_in_process_engine(settings, HashEmbedder())
    assert (in_process.chunk_tokens, in_process.chunk_overlap_tokens) == (512, 32)


@pytest.mark.parametrize("env, match", [
    ({"SCONE_CHUNK_TOKENS": "0"}, "SCONE_CHUNK_TOKENS/SCONE_CHUNK_OVERLAP_TOKENS: chunk_tokens must be"),
    ({"SCONE_CHUNK_TOKENS": "8"}, "SCONE_CHUNK_TOKENS/SCONE_CHUNK_OVERLAP_TOKENS: chunk_tokens must be"),
    ({"SCONE_CHUNK_TOKENS": "many"}, "SCONE_CHUNK_TOKENS must be a whole number of tokens, got 'many'"),
    ({"SCONE_CHUNK_TOKENS": "512.5"}, "SCONE_CHUNK_TOKENS must be a whole number"),
    ({"SCONE_CHUNK_TOKENS": "64", "SCONE_CHUNK_OVERLAP_TOKENS": "64"}, "chunk_overlap_tokens must be"),
    ({"SCONE_CHUNK_TOKENS": "64", "SCONE_CHUNK_OVERLAP_TOKENS": "-1"}, "chunk_overlap_tokens must be"),
    ({"SCONE_CHUNK_TOKENS": "64", "SCONE_CHUNK_OVERLAP_TOKENS": "some"}, "SCONE_CHUNK_OVERLAP_TOKENS must be a whole number"),
    ({"SCONE_CHUNK_OVERLAP_TOKENS": "16"}, "SCONE_CHUNK_OVERLAP_TOKENS needs SCONE_CHUNK_TOKENS"),
])
def test_a_token_chunk_target_that_cannot_work_is_refused_by_name(env, match):
    with pytest.raises(InvalidInput, match=match):
        Settings.from_env(env)
