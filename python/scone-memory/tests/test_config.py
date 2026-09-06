import os

import pytest

from scone_memory import InvalidInput
from scone_memory.config import Settings, build_engine, parse_keys


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


async def test_build_engine_wires_the_named_parts(tmp_path):
    env = {"SCONE_DOCUMENTS": "sqlite", "SCONE_SQLITE_PATH": str(tmp_path / "m.db"), "SCONE_VECTORS": "memory"}
    engine = await build_engine(Settings.from_env(env))
    added = await engine.remember("default", "wired through the environment")
    assert (await engine.recall("default", "environment")).items[0].episode_id == added.episode_id
    status = await engine.status("default")
    assert (status.document_store, status.vector_index, status.embedder) == ("sqlite", "memory", "hash-256")


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
