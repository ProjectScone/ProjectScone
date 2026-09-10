"""S3 links verify immutable object checksums without downloading image bodies."""
from __future__ import annotations

from collections import Counter
import io
import json
from typing import cast

import pytest

from scone_memory.backends.aws_blobs import AwsClient
from scone_memory.core.errors import NotFound, SconeError
from test_aws_blobs import aws, store  # noqa: F401


def count_downloads(client: AwsClient, monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    counts: Counter[str] = Counter()
    original = client.get_object

    def counted(**kwargs: object) -> dict[str, object]:
        result = original(**kwargs)
        counts["requests"] += 1
        counts["bytes"] += cast(int, result["ContentLength"])
        return result

    monkeypatch.setattr(client, "get_object", counted)
    return counts


async def test_new_and_repeated_links_verify_checksum_without_body_download(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"image" * 200_000, "image/png")
    downloads = count_downloads(aws[0], monkeypatch)
    await blobs.link("alpha", attachment.attachment_id, 1)
    await blobs.link("alpha", attachment.attachment_id, 1)
    assert await blobs.for_episode("alpha", 1) == [attachment]
    assert downloads["bytes"] == 0
    assert downloads["requests"] == 0


@pytest.mark.parametrize("checksum", [None, "composite"])
async def test_legacy_and_composite_checksums_fall_back_to_verified_download(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch, checksum: str | None,
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"legacy body", "image/png")
    original = aws[0].head_object

    def legacy(**kwargs: object) -> dict[str, object]:
        result = original(**kwargs)
        result.pop("ChecksumSHA256", None)
        if checksum == "composite":
            result.update(ChecksumSHA256="multipart-hash-2", ChecksumType="COMPOSITE")
        return result

    monkeypatch.setattr(aws[0], "head_object", legacy)
    downloads = count_downloads(aws[0], monkeypatch)
    await blobs.link("alpha", attachment.attachment_id, 1)
    assert await blobs.for_episode("alpha", 1) == [attachment]
    assert downloads == {"requests": 1, "bytes": 11}


@pytest.mark.parametrize("field,value", [
    ("ChecksumSHA256", "corrupt"), ("ContentLength", 999), ("VersionId", "other-version"),
])
async def test_link_rejects_checksum_length_and_version_mismatch(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch, field: str, value: object,
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"correct", "image/png")
    original = aws[0].head_object

    def corrupted(**kwargs: object) -> dict[str, object]:
        return {**original(**kwargs), field: value}

    monkeypatch.setattr(aws[0], "head_object", corrupted)
    with pytest.raises(SconeError, match="integrity"):
        await blobs.link("alpha", attachment.attachment_id, 1)
    assert await blobs.for_episode("alpha", 1) == []


@pytest.mark.parametrize("changed", ["hold", "root"])
async def test_link_rechecks_ownership_after_checksum_read(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"correct", "image/png")
    original = aws[0].head_object

    def removed(**kwargs: object) -> dict[str, object]:
        result = original(**kwargs)
        if changed == "hold":
            aws[1].delete_item(TableName="scone-test-blobs", Key={
                "pk": {"S": "S#alpha"}, "sk": {"S": "H#" + attachment.attachment_id}})
        else:
            key = {"pk": {"S": "B#" + attachment.attachment_id}, "sk": {"S": "R"}}
            item = cast(dict[str, dict[str, str]], aws[1].get_item(TableName="scone-test-blobs", Key=key)["Item"])
            payload = json.loads(item["data"]["S"])
            payload["size"] += 1
            item["data"]["S"] = json.dumps(payload)
            aws[1].put_item(TableName="scone-test-blobs", Item=item)
        return result

    monkeypatch.setattr(aws[0], "head_object", removed)
    with pytest.raises(SconeError, match="changed during read"):
        await blobs.link("alpha", attachment.attachment_id, 1)


async def test_link_rejects_missing_object_and_other_space_and_released_hold(
    aws: tuple[AwsClient, AwsClient],
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"correct", "image/png")
    with pytest.raises(NotFound):
        await blobs.link("beta", attachment.attachment_id, 1)
    versions = aws[0].list_object_versions(Bucket="scone-test-blobs")["Versions"]
    version = cast(list[dict[str, object]], versions)[0]
    aws[0].delete_object(Bucket="scone-test-blobs", Key=version["Key"], VersionId=version["VersionId"])
    with pytest.raises(NotFound):
        await blobs.link("alpha", attachment.attachment_id, 1)
    await blobs.release_space("alpha")
    with pytest.raises(NotFound):
        await blobs.link("alpha", attachment.attachment_id, 1)


@pytest.mark.parametrize("code", ["NotImplemented", "AccessDenied"])
async def test_checksum_head_unavailable_falls_back_and_still_rejects_corrupt_bytes(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch, code: str,
) -> None:
    from botocore.exceptions import ClientError

    blobs = store(aws)
    attachment = await blobs.put("alpha", b"correct", "image/png")

    def unavailable(**kwargs: object) -> dict[str, object]:
        raise ClientError({"Error": {"Code": code}}, "HeadObject")

    monkeypatch.setattr(aws[0], "head_object", unavailable)
    await blobs.link("alpha", attachment.attachment_id, 1)
    monkeypatch.setattr(aws[0], "get_object", lambda **kwargs: {"Body": io.BytesIO(b"changed")})
    with pytest.raises(SconeError, match="integrity"):
        await blobs.link("alpha", attachment.attachment_id, 1)


async def test_link_checks_the_held_version_when_a_newer_version_exists(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"original", "image/png")
    version = cast(list[dict[str, object]], aws[0].list_object_versions(Bucket="scone-test-blobs")["Versions"])[0]
    aws[0].put_object(Bucket="scone-test-blobs", Key=version["Key"], Body=b"changed", ChecksumAlgorithm="SHA256")
    downloads = count_downloads(aws[0], monkeypatch)
    await blobs.link("alpha", attachment.attachment_id, 1)
    assert downloads["bytes"] == 0
    assert await blobs.for_episode("alpha", 1) == [attachment]
    assert (await blobs.get("alpha", attachment.attachment_id))[1] == b"original"


async def test_provider_without_checksum_upload_support_keeps_legacy_verification(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.exceptions import ClientError

    original = aws[0].put_object

    def unsupported(**kwargs: object) -> dict[str, object]:
        if "ChecksumAlgorithm" in kwargs:
            raise ClientError({"Error": {"Code": "NotImplemented"}}, "PutObject")
        return original(**kwargs)

    monkeypatch.setattr(aws[0], "put_object", unsupported)
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"original", "image/png")
    downloads = count_downloads(aws[0], monkeypatch)
    await blobs.link("alpha", attachment.attachment_id, 1)
    assert downloads == {"requests": 1, "bytes": 8}
    assert await blobs.for_episode("alpha", 1) == [attachment]


@pytest.mark.parametrize("record", ["root", "hold"])
async def test_forged_identity_cannot_read_or_link_another_same_size_blob(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch, record: str,
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"correct", "image/png")
    other = await blobs.put("beta", b"changed", "image/png")
    key = ({"pk": {"S": "B#" + attachment.attachment_id}, "sk": {"S": "R"}}
           if record == "root" else {"pk": {"S": "S#alpha"}, "sk": {"S": "H#" + attachment.attachment_id}})
    item = cast(dict[str, dict[str, str]], aws[1].get_item(TableName="scone-test-blobs", Key=key)["Item"])
    payload = json.loads(item["data"]["S"])
    if record == "root":
        other_key = {"pk": {"S": "B#" + other.attachment_id}, "sk": {"S": "R"}}
        other_item = cast(dict[str, dict[str, str]], aws[1].get_item(TableName="scone-test-blobs", Key=other_key)["Item"])
        payload = json.loads(other_item["data"]["S"])
    else:
        payload["attachment"]["attachment_id"] = other.attachment_id
    item["data"]["S"] = json.dumps(payload)
    aws[1].put_item(TableName="scone-test-blobs", Item=item)

    def forbidden(**kwargs: object) -> dict[str, object]:
        pytest.fail("forged ownership must be rejected before requesting any object")

    monkeypatch.setattr(aws[0], "head_object", forbidden)
    monkeypatch.setattr(aws[0], "get_object", forbidden)
    with pytest.raises(SconeError, match="identity|integrity"):
        await blobs.get("alpha", attachment.attachment_id)
    with pytest.raises(SconeError, match="identity|integrity"):
        await blobs.link("alpha", attachment.attachment_id, 1)
