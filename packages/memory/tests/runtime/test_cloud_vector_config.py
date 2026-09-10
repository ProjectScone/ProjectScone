"""Configuration contracts; no cloud services are contacted."""
import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.config import Settings, build_vectors


def test_elasticache_environment_is_explicit_and_hides_password():
    settings = Settings.from_env({
        "SCONE_VECTORS": "elasticache", "SCONE_ELASTICACHE_URL": "rediss://cache.example.test:6379",
        "SCONE_ELASTICACHE_PREFIX": "memory_vectors", "SCONE_ELASTICACHE_USERNAME": "app",
        "SCONE_ELASTICACHE_PASSWORD": "private-password", "SCONE_ELASTICACHE_CLUSTER_MODE": "0",
        "SCONE_ELASTICACHE_ALGORITHM": "HNSW", "SCONE_ELASTICACHE_BATCH_SIZE": "32",
        "SCONE_ELASTICACHE_TIMEOUT": "12.5", "SCONE_ELASTICACHE_CA_CERTS": "/local/ca.pem",
    })
    assert settings.elasticache_url == "rediss://cache.example.test:6379"
    assert settings.elasticache_prefix == "memory_vectors"
    assert settings.elasticache_username == "app"
    assert settings.elasticache_password == "private-password"
    assert settings.elasticache_cluster_mode is False
    assert settings.elasticache_algorithm == "HNSW"
    assert settings.elasticache_batch_size == 32
    assert settings.elasticache_timeout == 12.5
    assert settings.elasticache_ca_certs == "/local/ca.pem"
    assert "private-password" not in repr(settings)


def test_elasticache_requires_explicit_endpoint():
    with pytest.raises(InvalidInput, match="SCONE_ELASTICACHE_URL"):
        build_vectors(Settings(vectors="elasticache"))


def test_qdrant_metadata_index_environment_parses_trimmed_names():
    settings = Settings.from_env({"SCONE_QDRANT_METADATA_INDEXES": "source_type, document_id ,language"})
    assert settings.qdrant_metadata_indexes == ("source_type", "document_id", "language")
    assert Settings.from_env({}).qdrant_metadata_indexes == ()


async def test_qdrant_metadata_index_configuration_reaches_adapter():
    pytest.importorskip("qdrant_client")
    index = build_vectors(Settings.from_env({"SCONE_VECTORS": "qdrant", "SCONE_QDRANT_URL": ":memory:",
        "SCONE_QDRANT_METADATA_INDEXES": "document_format,entity_id"}))
    try:
        assert index.metadata_indexes == ("document_format", "entity_id")
    finally:
        await index.close()


async def test_elasticache_configuration_constructs_native_adapter_without_connecting():
    pytest.importorskip("redis")
    from scone_memory.backends import ElastiCacheVectorIndex

    index = build_vectors(Settings.from_env({
        "SCONE_VECTORS": "elasticache", "SCONE_ELASTICACHE_URL": "rediss://cache.example.test:6379",
        "SCONE_ELASTICACHE_PREFIX": "memory_vectors", "SCONE_ELASTICACHE_USERNAME": "app",
        "SCONE_ELASTICACHE_PASSWORD": "private-password", "SCONE_ELASTICACHE_CLUSTER_MODE": "0",
        "SCONE_ELASTICACHE_ALGORITHM": "HNSW", "SCONE_ELASTICACHE_BATCH_SIZE": "32",
        "SCONE_ELASTICACHE_TIMEOUT": "12.5",
    }))
    try:
        assert isinstance(index, ElastiCacheVectorIndex)
        assert index.prefix == "memory_vectors"
        assert index.algorithm == "HNSW"
        assert index.batch_size == 32
        assert index.timeout == 12.5
    finally:
        await index.close()


async def test_qdrant_hnsw_ef_setting_reaches_adapter():
    pytest.importorskip("qdrant_client")
    index = build_vectors(Settings.from_env({"SCONE_VECTORS": "qdrant", "SCONE_QDRANT_URL": ":memory:",
        "SCONE_QDRANT_HNSW_EF": "128"}))
    try:
        assert index.hnsw_ef == 128
    finally:
        await index.close()
    assert Settings.from_env({}).qdrant_hnsw_ef is None
    assert Settings.from_env({"SCONE_QDRANT_HNSW_EF": ""}).qdrant_hnsw_ef is None


@pytest.mark.parametrize("value", ["many", "1.5", "0", "-1"])
def test_invalid_qdrant_hnsw_ef_setting(value: str):
    with pytest.raises(InvalidInput, match="SCONE_QDRANT_HNSW_EF"):
        Settings.from_env({"SCONE_QDRANT_HNSW_EF": value})
