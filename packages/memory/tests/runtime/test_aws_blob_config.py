"""AWS attachment configuration is explicit and makes no network calls."""
from unittest.mock import patch

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.backends.blobs import FileBlobStore, InMemoryBlobStore
from scone_memory.runtime.config import Settings, build_blobs

AWS = {
    "SCONE_BLOBS": "s3", "SCONE_S3_BUCKET": "synthetic-attachments",
    "SCONE_DYNAMODB_BLOB_TABLE": "synthetic-blob-metadata", "SCONE_AWS_REGION": "us-east-1",
}


def test_default_and_existing_file_configuration_remain_unchanged(tmp_path):
    assert isinstance(build_blobs(Settings()), InMemoryBlobStore)
    assert isinstance(build_blobs(Settings(blob_dir=str(tmp_path / "bytes"))), FileBlobStore)
    assert isinstance(build_blobs(Settings(documents="sqlite", sqlite_path=str(tmp_path / "memory.db"))), FileBlobStore)
    assert isinstance(build_blobs(Settings(blobs="memory", documents="sqlite")), InMemoryBlobStore)


def test_s3_is_wired_without_creating_sdk_clients():
    with patch("scone_memory.backends.aws_blobs.S3BlobStore") as factory:
        settings = Settings.from_env({**AWS, "SCONE_S3_PREFIX": "tenant-uploads/"})
        assert build_blobs(settings) is factory.return_value
        factory.assert_called_once_with("synthetic-attachments", "synthetic-blob-metadata",
                                        region_name="us-east-1", prefix="tenant-uploads/")


@pytest.mark.parametrize("missing", ["SCONE_S3_BUCKET", "SCONE_DYNAMODB_BLOB_TABLE", "SCONE_AWS_REGION"])
def test_aws_backend_requires_complete_configuration(missing):
    with pytest.raises(InvalidInput, match=missing):
        Settings.from_env({key: value for key, value in AWS.items() if key != missing})


@pytest.mark.parametrize("env", [
    {"SCONE_BLOBS": "typo"}, {"SCONE_BLOBS": "file"},
    {**AWS, "SCONE_BLOB_DIR": "/synthetic"},
    {"SCONE_BLOBS": "memory", "SCONE_BLOB_DIR": "/synthetic"},
    {"SCONE_S3_BUCKET": "synthetic-attachments"},
    {**AWS, "SCONE_S3_PREFIX": "../outside/"},
])
def test_invalid_or_ambiguous_storage_configuration_is_rejected(env):
    with pytest.raises(InvalidInput):
        build_blobs(Settings.from_env(env))


def test_ambient_aws_region_does_not_enable_cloud_storage():
    settings = Settings.from_env({"AWS_REGION": "us-east-1"})
    assert isinstance(build_blobs(settings), InMemoryBlobStore)


def test_s3_construction_does_not_resolve_credentials_or_contact_services(monkeypatch):
    boto3 = pytest.importorskip("boto3")

    def forbidden(*args, **kwargs):
        raise AssertionError("configuration must not create SDK clients")

    monkeypatch.setattr(boto3, "client", forbidden)
    monkeypatch.setattr(boto3.session.Session, "client", forbidden)
    assert build_blobs(Settings.from_env(AWS)).name == "s3"


async def test_invalid_blob_configuration_fails_before_opening_other_backends(monkeypatch):
    from scone_memory.runtime import config

    def forbidden(settings):
        raise AssertionError("invalid storage configuration must not open other backends")

    monkeypatch.setattr(config, "build_documents", forbidden)
    with pytest.raises(InvalidInput, match="S3 prefix"):
        await config.build_engine(Settings.from_env({**AWS, "SCONE_S3_PREFIX": "../outside/"}))
