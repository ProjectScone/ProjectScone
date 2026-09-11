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
