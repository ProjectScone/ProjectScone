import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.config import Settings, build_vectors


def test_opensearch_requires_explicit_endpoint():
    with pytest.raises(InvalidInput, match="SCONE_OPENSEARCH_URL"):
        build_vectors(Settings(vectors="opensearch"))


async def test_opensearch_settings_wire_native_adapter_and_keep_password_private():
    from scone_memory.backends import OpenSearchVectorIndex

    settings = Settings.from_env({
        "SCONE_VECTORS": "opensearch", "SCONE_OPENSEARCH_URL": "https://localhost:9200",
        "SCONE_OPENSEARCH_INDEX": "image_vectors", "SCONE_OPENSEARCH_USERNAME": "reader",
        "SCONE_OPENSEARCH_PASSWORD": "private-password", "SCONE_OPENSEARCH_BATCH_SIZE": "32",
        "SCONE_OPENSEARCH_MAX_BATCH_BYTES": "65536", "SCONE_OPENSEARCH_TIMEOUT": "12.5",
    })
    index = build_vectors(settings)
    try:
        assert isinstance(index, OpenSearchVectorIndex)
        assert index.index == "image_vectors"
        assert index.batch_size == 32
        assert index.max_batch_bytes == 65536
        assert index.client.timeout.read == 12.5
        assert "private-password" not in repr(settings)
    finally:
        await index.close()


@pytest.mark.parametrize("name,value", [
    ("SCONE_OPENSEARCH_BATCH_SIZE", "0"), ("SCONE_OPENSEARCH_BATCH_SIZE", "many"),
    ("SCONE_OPENSEARCH_MAX_BATCH_BYTES", "0"), ("SCONE_OPENSEARCH_TIMEOUT", "nan"),
    ("SCONE_OPENSEARCH_INDEX", "*"), ("SCONE_OPENSEARCH_USERNAME", "without-password"),
])
def test_invalid_opensearch_settings_fail_before_transport(name: str, value: str):
    with pytest.raises(InvalidInput, match="OPENSEARCH"):
        build_vectors(Settings.from_env({"SCONE_VECTORS": "opensearch",
            "SCONE_OPENSEARCH_URL": "http://localhost:9200", name: value}))


async def test_explicit_aws_credentials_construct_domain_signer_without_discovery():
    from scone_memory.providers.aws_auth import AwsSigV4Auth

    settings = Settings.from_env({
        "SCONE_VECTORS": "opensearch", "SCONE_OPENSEARCH_URL": "https://domain.example.test",
        "SCONE_OPENSEARCH_AWS_REGION": "us-east-1", "SCONE_OPENSEARCH_AWS_ACCESS_KEY_ID": "AKIDEXAMPLE",
        "SCONE_OPENSEARCH_AWS_SECRET_ACCESS_KEY": "private-secret", "SCONE_OPENSEARCH_AWS_SESSION_TOKEN": "private-token",
    })
    index = build_vectors(settings)
    try:
        assert isinstance(index.client.auth, AwsSigV4Auth)
        assert index.client.auth.region == "us-east-1"
        assert index.client.auth.service == "es"
        for secret in ("AKIDEXAMPLE", "private-secret", "private-token"):
            assert secret not in repr(settings)
    finally:
        await index.close()


@pytest.mark.parametrize("environment", [
    {"SCONE_OPENSEARCH_AWS_REGION": "us-east-1"},
    {"SCONE_OPENSEARCH_AWS_SESSION_TOKEN": "only-token"},
    {"SCONE_OPENSEARCH_AWS_REGION": "us-east-1", "SCONE_OPENSEARCH_AWS_ACCESS_KEY_ID": "AKIDEXAMPLE",
     "SCONE_OPENSEARCH_AWS_SECRET_ACCESS_KEY": "private-secret", "SCONE_OPENSEARCH_USERNAME": "u",
     "SCONE_OPENSEARCH_PASSWORD": "p"},
])
def test_partial_or_conflicting_aws_credentials_are_rejected(environment: dict[str, str]):
    with pytest.raises(InvalidInput, match="OPENSEARCH"):
        build_vectors(Settings.from_env({"SCONE_VECTORS": "opensearch",
            "SCONE_OPENSEARCH_URL": "https://domain.example.test"} | environment))
