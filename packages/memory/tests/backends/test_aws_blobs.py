"""Attachment contract against in-process AWS emulators, never AWS services."""
from __future__ import annotations

from collections.abc import Callable, Iterator
import asyncio
import hashlib
import importlib
import json
import os
import re
from threading import Event, RLock
from pathlib import Path
from time import monotonic
from typing import cast
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from scone_memory.backends.aws_blobs import AwsClient, S3BlobStore
from scone_memory.core.errors import NotFound, SconeError


class AtomicDynamoEmulator:
    """Make each Moto request atomic while preserving workflow interleavings.

    Moto 5.2.3 backs up the entire table in transact_write_items and restores
    that snapshot on conflict. Concurrent requests can erase a successful
    transaction when another request rolls back. Moto documents that it is
    not concurrency-safe: https://docs.getmoto.org/en/latest/docs/faq.html.
    This lock covers one emulator request, never a Scone operation or S3 call.
    DynamoDB Local concurrency tests use its real client without this wrapper.
    """

    def __init__(self, client: AwsClient) -> None:
        self.client = client
        self.lock = RLock()

    def __getattr__(self, name: str) -> Callable[..., dict[str, object]]:
        method = cast(Callable[..., dict[str, object]], getattr(self.client, name))

        def request(*args: object, **kwargs: object) -> dict[str, object]:
            with self.lock:
                try:
                    return method(*args, **kwargs)
                finally:
                    # Moto's dashboard trackers retain every transaction's
                    # deep-copied table and item until mock teardown. Backend
                    # tables own live state; drop only dashboard references.
                    from moto.dynamodb.models.dynamo_type import Item
                    from moto.dynamodb.models.table import Table

                    for model in (Table, Item):
                        cast(list[object], getattr(model, "instances_tracked")).clear()

        return request


@pytest.fixture
def aws() -> Iterator[tuple[AwsClient, AwsClient]]:
    moto = pytest.importorskip("moto")
    boto3 = pytest.importorskip("boto3")
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test")
        ddb = boto3.client("dynamodb", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test")
        s3.create_bucket(Bucket="scone-test-blobs")
        s3.put_bucket_versioning(Bucket="scone-test-blobs", VersioningConfiguration={"Status": "Enabled"})
        ddb.create_table(TableName="scone-test-blobs", KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"}], AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"}], BillingMode="PAY_PER_REQUEST")
        yield cast(AwsClient, s3), AtomicDynamoEmulator(cast(AwsClient, ddb))


def store(aws: tuple[AwsClient, AwsClient]) -> S3BlobStore:
    return S3BlobStore("scone-test-blobs", "scone-test-blobs", region_name="us-east-1", s3_client=aws[0], dynamodb_client=aws[1])


def test_emulator_releases_dashboard_snapshots_and_preserves_transaction_rollback(
    aws: tuple[AwsClient, AwsClient],
) -> None:
    from botocore.exceptions import ClientError
    from moto.dynamodb.models.dynamo_type import Item
    from moto.dynamodb.models.table import Table

    ddb = aws[1]
    table = "scone-test-blobs"

    def key(number: int) -> dict[str, dict[str, str]]:
        return {"pk": {"S": str(number)}, "sk": {"S": "R"}}

    def put(number: int, value: str) -> dict[str, object]:
        return {"Put": {"TableName": table, "Item": {**key(number), "value": {"S": value}}}}

    def value(number: int) -> dict[str, str]:
        response = ddb.get_item(TableName=table, Key=key(number))
        return cast(dict[str, dict[str, str]], response["Item"])["value"]

    for number in range(3):
        ddb.put_item(TableName=table, Item={**key(number), "value": {"S": "old"}})
    ddb.transact_write_items(TransactItems=[put(0, "committed"), put(1, "committed")])
    for model in (Table, Item):
        assert not getattr(model, "instances_tracked")
    assert value(0) == {"S": "committed"}

    with pytest.raises(ClientError) as failed:
        ddb.transact_write_items(TransactItems=[put(0, "rolled back"), {"ConditionCheck": {
            "TableName": table, "Key": key(2), "ConditionExpression": "attribute_not_exists(pk)"}}])
    assert failed.value.response["Error"]["Code"] == "TransactionCanceledException"
    for model in (Table, Item):
        assert not getattr(model, "instances_tracked")
    assert value(0) == {"S": "committed"}
    assert ddb.scan(TableName=table)["Count"] == 3


async def test_shared_holds_links_restart_and_final_release(aws: tuple[AwsClient, AwsClient]) -> None:
    first, restarted = store(aws), store(aws)
    attachment = await first.put("alpha", b"same exact bytes", "text/plain", "first.txt")
    assert await first.put("alpha", b"same exact bytes", "image/png", "later.png") == attachment
    await first.put("beta", b"same exact bytes", "text/plain")
    with pytest.raises(NotFound):
        await restarted.get("gamma", attachment.attachment_id)
    await first.link("alpha", attachment.attachment_id, 1)
    await first.link("alpha", attachment.attachment_id, 2)
    await first.link("alpha", attachment.attachment_id, 1)
    assert await restarted.for_episode("alpha", 1) == [attachment]
    assert await restarted.linked("alpha") == {attachment.attachment_id}
    assert await restarted.released_by("alpha", 1) == []
    assert await restarted.unlink("alpha", 1) == []
    assert await restarted.unlink("alpha", 2) == [attachment.attachment_id]
    assert await restarted.held("alpha") == []
    assert (await restarted.get("beta", attachment.attachment_id))[1] == b"same exact bytes"
    assert await restarted.release_space("beta", preview=True) == ([attachment.attachment_id], [])
    assert await restarted.release_space("beta") == ([attachment.attachment_id], [])
    with pytest.raises(NotFound):
        await restarted.get("beta", attachment.attachment_id)


async def test_release_preview_reports_shared_bytes_without_mutation(aws: tuple[AwsClient, AwsClient]) -> None:
    blobs = store(aws)
    shared = await blobs.put("alpha", b"shared", "text/plain")
    alone = await blobs.put("alpha", b"alone", "text/plain")
    await blobs.put("beta", b"shared", "text/plain")
    assert await blobs.release_space("alpha", preview=True) == ([alone.attachment_id], [shared.attachment_id])
    assert len(await blobs.held("alpha")) == 2
    assert await blobs.release_space("alpha") == ([alone.attachment_id], [shared.attachment_id])
    assert (await blobs.get("beta", shared.attachment_id))[1] == b"shared"


@pytest.mark.parametrize("space", ["", "../escape", "Upper", "a/b"])
async def test_invalid_space_fails_before_sdk(aws: tuple[AwsClient, AwsClient], space: str) -> None:
    with pytest.raises((ValueError, SconeError)):
        await store(aws).put(space, b"data", "text/plain")


def metadata(aws: tuple[AwsClient, AwsClient], pk: str) -> list[dict[str, object]]:
    response = aws[1].query(TableName="scone-test-blobs", ConsistentRead=True,
        KeyConditionExpression="pk = :pk", ExpressionAttributeValues={":pk": {"S": pk}})
    return cast(list[dict[str, object]], response.get("Items", []))


def payload(row: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], json.loads(cast(dict[str, str], row["data"])["S"]))


async def test_independent_workers_preserve_all_holds_and_links(aws: tuple[AwsClient, AwsClient]) -> None:
    stores = [store(aws) for _ in range(4)]
    attachments = await asyncio.gather(*(blobs.put(f"space{number}", b"shared concurrent bytes", "text/plain")
        for number, blobs in enumerate(stores)))
    identifier = attachments[0].attachment_id
    assert {attachment.attachment_id for attachment in attachments} == {identifier}
    root = payload(metadata(aws, "B#" + identifier)[0])
    assert root["holds"] == 4
    await asyncio.gather(*(blobs.link("space0", identifier, number + 1) for number, blobs in enumerate(stores)))
    for number in range(4):
        assert await stores[0].for_episode("space0", number + 1) == [attachments[0]]
    for number in range(3):
        assert await stores[0].unlink("space0", number + 1) == []
    assert await stores[0].unlink("space0", 4) == [identifier]
    for number in range(1, 4):
        assert (await stores[number].get(f"space{number}", identifier))[1] == b"shared concurrent bytes"


async def test_for_episode_preserves_link_order_across_restart(aws: tuple[AwsClient, AwsClient]) -> None:
    blobs = store(aws)
    first = await blobs.put("alpha", b"first", "text/plain")
    second = await blobs.put("alpha", b"second", "text/plain")
    await blobs.link("alpha", second.attachment_id, 1)
    await blobs.link("alpha", first.attachment_id, 1)
    await blobs.link("alpha", second.attachment_id, 1)
    assert await store(aws).for_episode("alpha", 1) == [second, first]
    assert await blobs.released_by("alpha", 1) == [second.attachment_id, first.attachment_id]


async def test_failed_delete_is_durable_and_does_not_delete_new_generation(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"same bytes again", "text/plain")
    await blobs.link("alpha", attachment.attachment_id, 1)
    original_delete = aws[0].delete_object

    def fail_delete(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("private provider detail")

    monkeypatch.setattr(aws[0], "delete_object", fail_delete)
    with pytest.raises(SconeError, match="AWS blob operation failed"):
        await blobs.unlink("alpha", 1)
    intent = payload(metadata(aws, "I#alpha")[0])
    assert intent["kind"] == "delete"
    assert await blobs.held("alpha") == []
    replacement = await store(aws).put("beta", b"same bytes again", "text/plain")
    current = payload(metadata(aws, "B#" + attachment.attachment_id)[0])
    assert current["generation"] != intent["generation"]
    monkeypatch.setattr(aws[0], "delete_object", original_delete)
    assert await store(aws).unlink("alpha", 1) == [attachment.attachment_id]
    assert metadata(aws, "I#alpha") == []
    assert (await store(aws).get("beta", replacement.attachment_id))[1] == b"same bytes again"


async def test_partial_release_resumes_and_blocks_new_space_writes(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = store(aws)
    attachments = [await blobs.put("alpha", content, "text/plain") for content in (b"one", b"two")]
    original_delete = aws[0].delete_object
    deleted = 0

    def fail_second(**kwargs: object) -> dict[str, object]:
        nonlocal deleted
        deleted += 1
        if deleted == 2:
            raise RuntimeError("synthetic deletion outage")
        return original_delete(**kwargs)

    monkeypatch.setattr(aws[0], "delete_object", fail_second)
    with pytest.raises(SconeError):
        await blobs.release_space("alpha")
    with pytest.raises(SconeError, match="unfinished release"):
        await store(aws).put("alpha", b"blocked", "text/plain")
    monkeypatch.setattr(aws[0], "delete_object", original_delete)
    assert await store(aws).release_space("alpha") == (sorted(a.attachment_id for a in attachments), [])
    assert metadata(aws, "I#alpha") == []


async def test_versioned_bytes_are_physically_removed_without_delete_marker(
    aws: tuple[AwsClient, AwsClient],
) -> None:
    blobs = store(aws)
    await blobs.put("alpha", b"versioned body", "text/plain")
    await blobs.release_space("alpha")
    response = aws[0].list_object_versions(Bucket="scone-test-blobs", Prefix="attachments/")
    assert response.get("Versions", []) == []
    assert response.get("DeleteMarkers", []) == []


async def test_tampered_bytes_are_rejected_even_if_metadata_is_retained(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"correct", "text/plain")

    def tampered(**kwargs: object) -> dict[str, object]:
        return {"Body": io.BytesIO(b"changed")}

    monkeypatch.setattr(aws[0], "get_object", tampered)
    with pytest.raises(SconeError, match="integrity"):
        await blobs.get("alpha", attachment.attachment_id)


async def test_source_hold_removed_during_s3_read_is_not_returned(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"current bytes", "text/plain")
    original_get = aws[0].get_object

    def remove_hold(**kwargs: object) -> dict[str, object]:
        result = original_get(**kwargs)
        aws[1].delete_item(TableName="scone-test-blobs", Key={"pk": {"S": "S#alpha"}, "sk": {"S": "H#" + attachment.attachment_id}})
        return result

    monkeypatch.setattr(aws[0], "get_object", remove_hold)
    with pytest.raises(SconeError, match="changed during read"):
        await blobs.get("alpha", attachment.attachment_id)


async def test_failed_upload_has_durable_intent_and_recovery_can_repeat(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = store(aws)
    original_put = aws[0].put_object

    def uploaded_then_failed(**kwargs: object) -> dict[str, object]:
        original_put(**kwargs)
        raise RuntimeError("response lost after upload")

    monkeypatch.setattr(aws[0], "put_object", uploaded_then_failed)
    with pytest.raises(SconeError):
        await blobs.put("alpha", b"orphan after lost response", "text/plain")
    assert await blobs.held("alpha") == []
    assert payload(metadata(aws, "I#alpha")[0])["kind"] == "upload"
    monkeypatch.setattr(aws[0], "put_object", original_put)
    assert (await store(aws).recover_uploads("alpha")).visited == 1
    assert payload(metadata(aws, "I#alpha")[0])["kind"] == "abandoned"
    assert (await store(aws).recover_uploads("alpha")).visited == 1
    assert aws[0].list_object_versions(Bucket="scone-test-blobs").get("Versions", []) == []


async def test_cancelled_worker_keeps_slot_until_network_returns(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = store(aws)
    entered, finish = Event(), Event()
    original_put = aws[0].put_object

    def blocked_put(**kwargs: object) -> dict[str, object]:
        entered.set()
        assert finish.wait(5)
        return original_put(**kwargs)

    monkeypatch.setattr(aws[0], "put_object", blocked_put)
    task = asyncio.create_task(blobs.put("alpha", b"cancelled upload", "text/plain"))
    assert await asyncio.to_thread(entered.wait, 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    second = asyncio.create_task(blobs.held("alpha"))
    await asyncio.sleep(.02)
    assert not second.done()
    finish.set()
    assert await second == []
    assert payload(metadata(aws, "I#alpha")[0])["kind"] == "upload"
    assert (await blobs.recover_uploads("alpha")).visited == 1


async def test_unknown_digest_and_boolean_episode_are_rejected(aws: tuple[AwsClient, AwsClient]) -> None:
    blobs = store(aws)
    with pytest.raises(SconeError):
        await blobs.get("alpha", "../invalid")
    with pytest.raises(SconeError):
        await blobs.link("alpha", hashlib.sha256(b"x").hexdigest(), True)


async def test_sqlite_memory_engine_attachment_lifecycle_survives_restart(
    aws: tuple[AwsClient, AwsClient], tmp_path: Path,
) -> None:
    from scone_memory import HashEmbedder, MemoryEngine
    from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex

    path = tmp_path / "memory.db"
    engine = await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder(), blobs=store(aws)).open()
    attachment = await engine.attach("alpha", b"source attachment bytes", "text/plain", "source.txt")
    source = await engine.remember("alpha", "Juniper maintenance uses cobalt.", attachment_ids=[attachment.attachment_id])
    await engine.attach("beta", b"source attachment bytes", "text/plain")
    await engine.close()
    restarted = await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder(), blobs=store(aws)).open()
    try:
        recalled = await restarted.recall("alpha", "Juniper cobalt")
        assert recalled.items[0].episode_id == source.episode_id
        assert (await restarted.episode("alpha", source.episode_id)).attachments == (attachment,)
        assert (await restarted.attachment("alpha", attachment.attachment_id))[1] == b"source attachment bytes"
        forgotten = await restarted.forget("alpha", source.episode_id)
        assert forgotten.attachments_released == [attachment.attachment_id]
        with pytest.raises(NotFound):
            await restarted.attachment("alpha", attachment.attachment_id)
        assert (await restarted.attachment("beta", attachment.attachment_id))[1] == b"source attachment bytes"
        await restarted.delete_space("beta")
        assert aws[0].list_object_versions(Bucket="scone-test-blobs").get("Versions", []) == []
    finally:
        await restarted.close()


async def test_timeout_does_not_release_worker_or_publish_late_upload(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = S3BlobStore("scone-test-blobs", "scone-test-blobs", region_name="us-east-1", timeout_s=1,
                       s3_client=aws[0], dynamodb_client=aws[1])
    entered, finish = Event(), Event()
    original_put = aws[0].put_object

    def slow_put(**kwargs: object) -> dict[str, object]:
        entered.set()
        assert finish.wait(4)
        return original_put(**kwargs)

    monkeypatch.setattr(aws[0], "put_object", slow_put)
    started = monotonic()
    task = asyncio.create_task(blobs.put("alpha", b"late timeout bytes", "text/plain"))
    assert await asyncio.to_thread(entered.wait, 2)
    with pytest.raises(SconeError, match="timed out"):
        await task
    assert monotonic() - started < 1.8
    waiting = asyncio.create_task(blobs.held("alpha"))
    await asyncio.sleep(.02)
    assert not waiting.done()
    finish.set()
    assert await waiting == []
    assert len(metadata(aws, "I#alpha")) == 1


async def test_recovery_fences_uploader_and_cleans_a_late_s3_completion(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploader, recovery = store(aws), store(aws)
    entered, finish = Event(), Event()
    original_put = aws[0].put_object

    def paused_upload(**kwargs: object) -> dict[str, object]:
        entered.set()
        assert finish.wait(5)
        return original_put(**kwargs)

    monkeypatch.setattr(aws[0], "put_object", paused_upload)
    task = asyncio.create_task(uploader.put("alpha", b"paused before upload", "text/plain"))
    assert await asyncio.to_thread(entered.wait, 3)
    assert (await recovery.recover_uploads("alpha")).visited == 1
    assert payload(metadata(aws, "I#alpha")[0])["kind"] == "abandoned"
    finish.set()
    with pytest.raises(SconeError, match="fenced"):
        await task
    assert await recovery.held("alpha") == []
    assert len(cast(list[object], aws[0].list_object_versions(Bucket="scone-test-blobs").get("Versions", []))) == 1
    assert (await recovery.recover_uploads("alpha")).visited == 1
    assert aws[0].list_object_versions(Bucket="scone-test-blobs").get("Versions", []) == []


async def test_deleted_and_recreated_root_cannot_pass_stale_generation_cas(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"ABA generation bytes", "text/plain")
    original_get = aws[1].get_item
    injected = False

    def replace_after_root_read(**kwargs: object) -> dict[str, object]:
        nonlocal injected
        result = original_get(**kwargs)
        key = cast(dict[str, dict[str, str]], kwargs["Key"])
        if key["pk"]["S"] == "B#" + attachment.attachment_id and not injected:
            injected = True

            async def replace() -> None:
                other = store(aws)
                await other.release_space("alpha")
                await other.put("gamma", b"ABA generation bytes", "text/plain")

            asyncio.run(replace())
        return result

    monkeypatch.setattr(aws[1], "get_item", replace_after_root_read)
    await blobs.put("beta", b"ABA generation bytes", "text/plain")
    assert injected
    assert (await blobs.get("gamma", attachment.attachment_id))[1] == b"ABA generation bytes"
    assert (await blobs.get("beta", attachment.attachment_id))[1] == b"ABA generation bytes"
    assert payload(metadata(aws, "B#" + attachment.attachment_id)[0])["holds"] == 2


def test_constructor_does_not_import_sdk_or_resolve_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_import(name: str) -> object:
        raise AssertionError("constructor attempted SDK import")

    monkeypatch.setattr(importlib, "import_module", forbidden_import)
    assert S3BlobStore("scone-test-blobs", "scone-test-blobs", region_name="us-east-1").name == "s3"


async def test_engine_space_erase_failure_leaves_retryable_cleanup_and_documents(
    aws: tuple[AwsClient, AwsClient], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scone_memory import HashEmbedder, InMemoryVectorIndex, MemoryEngine
    from scone_memory.backends.sqlite import SqliteDocumentStore

    engine = await MemoryEngine(SqliteDocumentStore(tmp_path / "erase.db"), InMemoryVectorIndex(), HashEmbedder(), blobs=store(aws)).open()
    attachment = await engine.attach("alpha", b"erase after retry", "text/plain")
    source = await engine.remember("alpha", "Retain documents until cleanup succeeds.", attachment_ids=[attachment.attachment_id])
    original_delete = aws[0].delete_object

    def unavailable(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("private service outage")

    try:
        monkeypatch.setattr(aws[0], "delete_object", unavailable)
        with pytest.raises(SconeError, match="AWS blob operation failed"):
            await engine.delete_space("alpha")
        assert await engine.documents.get_episode("alpha", source.episode_id) is not None
        assert payload(metadata(aws, "S#alpha")[0])["mode"] == "release"
        assert len(metadata(aws, "I#alpha")) == 1
        monkeypatch.setattr(aws[0], "delete_object", original_delete)
        receipt = await engine.delete_space("alpha")
        assert receipt.attachments_released == [attachment.attachment_id]
        assert await engine.documents.get_episode("alpha", source.episode_id) is None
        assert metadata(aws, "I#alpha") == []
    finally:
        await engine.close()


async def test_upload_intent_failure_never_writes_s3(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("private DynamoDB outage")

    monkeypatch.setattr(aws[1], "transact_write_items", unavailable)
    with pytest.raises(SconeError, match="AWS blob operation failed"):
        await store(aws).put("alpha", b"never uploaded", "text/plain")
    assert aws[0].list_object_versions(Bucket="scone-test-blobs").get("Versions", []) == []


@pytest.mark.parametrize("setting,value", [("timeout_s", True), ("timeout_s", float("nan")),
    ("timeout_s", .5), ("prefix", "../"), ("prefix", "/root/"), ("region_name", "not-region")])
def test_invalid_configuration_is_rejected(setting: str, value: object) -> None:
    options: dict[str, object] = {"bucket": "scone-test-blobs", "table": "scone-test-blobs", "region_name": "us-east-1"}
    options[setting] = value
    constructor = cast(Callable[..., S3BlobStore], S3BlobStore)
    with pytest.raises(ValueError):
        constructor(**options)


async def test_legacy_count_at_old_capacity_does_not_block_upload(aws: tuple[AwsClient, AwsClient]) -> None:
    # The persisted space counter is transactionally maintained with hold rows.
    # Seed its boundary directly, without 1024 unrelated object uploads.
    data = {"mode": "active", "count": 1024, "operation": "", "episode": 0,
            "pending": [], "released": [], "kept": [], "gc": []}
    aws[1].put_item(TableName="scone-test-blobs", Item={"pk": {"S": "S#alpha"}, "sk": {"S": "C"},
        "rev": {"N": "1"}, "data": {"S": json.dumps(data)}})
    attachment = await store(aws).put("alpha", b"past the old capacity", "text/plain")
    assert attachment.bytes == len(b"past the old capacity")
    assert payload(metadata(aws, "S#alpha")[0])["count"] == 1025
    assert metadata(aws, "I#alpha") == []


def seed_intents(aws: tuple[AwsClient, AwsClient], count: int, *, kind: str = "abandoned") -> str:
    digest = hashlib.sha256(b"late orphan").hexdigest()
    for offset in range(0, count, 25):
        requests: list[dict[str, object]] = []
        for number in range(offset, min(offset + 25, count)):
            generation = f"{number + 1:032x}"
            data = {"kind": kind, "attachment_id": digest, "generation": generation,
                    "key": f"attachments/{digest}/{generation}", "version_id": None}
            requests.append({"PutRequest": {"Item": {"pk": {"S": "I#alpha"}, "sk": {"S": generation},
                "rev": {"N": "1"}, "data": {"S": json.dumps(data, separators=(",", ":"))}}}})
        result = aws[1].batch_write_item(RequestItems={"scone-test-blobs": requests})
        assert not result.get("UnprocessedItems")
    return f"attachments/{digest}/{count:032x}"


async def test_recovery_pages_reach_late_orphan_beyond_1024_retained_intents(
    aws: tuple[AwsClient, AwsClient],
) -> None:
    key = seed_intents(aws, 1030)
    aws[0].put_object(Bucket="scone-test-blobs", Key=key, Body=b"late orphan", IfNoneMatch="*")
    blobs = store(aws)
    cursor: str | None = None
    visited = 0
    pages = 0
    while True:
        page = await blobs.recover_uploads("alpha", after=cursor, limit=100)
        assert 0 <= page.visited <= 100
        visited += page.visited
        pages += 1
        if page.next_cursor is None:
            break
        assert cursor is None or page.next_cursor > cursor
        cursor = page.next_cursor
        assert pages <= 11
    assert visited == 1030
    assert pages == 11
    assert aws[0].list_object_versions(Bucket="scone-test-blobs").get("Versions", []) == []
    assert aws[1].describe_table(TableName="scone-test-blobs")["Table"]
    assert len(metadata(aws, "I#alpha")) == 1030


async def test_recovery_cursor_remains_valid_after_page_intents_are_deleted(
    aws: tuple[AwsClient, AwsClient],
) -> None:
    seed_intents(aws, 3, kind="delete")
    first = await store(aws).recover_uploads("alpha", limit=2)
    assert first.visited == 2
    assert first.next_cursor == f"{2:032x}"
    second = await store(aws).recover_uploads("alpha", after=first.next_cursor, limit=2)
    assert second.visited == 1
    assert second.next_cursor is None
    assert metadata(aws, "I#alpha") == []


@pytest.mark.parametrize("after,limit", [("../", 100), ("A" * 32, 100), ("0" * 33, 100),
    (None, 0), (None, 101), (None, True)])
async def test_recovery_rejects_invalid_cursor_and_page_limit(
    aws: tuple[AwsClient, AwsClient], after: str | None, limit: int,
) -> None:
    with pytest.raises(SconeError):
        await store(aws).recover_uploads("alpha", after=after, limit=limit)


@pytest.mark.parametrize("cancel", [False, True])
async def test_late_get_response_closes_stream_after_timeout_or_cancellation(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch, cancel: bool,
) -> None:
    import io
    blobs = S3BlobStore("scone-test-blobs", "scone-test-blobs", region_name="us-east-1", timeout_s=1,
                       s3_client=aws[0], dynamodb_client=aws[1])
    attachment = await blobs.put("alpha", b"body bytes", "text/plain")
    entered, finish = Event(), Event()
    body = io.BytesIO(b"body bytes")

    def late_body(**kwargs: object) -> dict[str, object]:
        entered.set()
        assert finish.wait(4)
        return {"Body": body}

    monkeypatch.setattr(aws[0], "get_object", late_body)
    task = asyncio.create_task(blobs.get("alpha", attachment.attachment_id))
    assert await asyncio.to_thread(entered.wait, 2)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(SconeError, match="timed out"):
            await task
    finish.set()
    await blobs.held("alpha")
    assert body.closed


async def test_corrupted_root_cannot_delete_another_attachment_generation(
    aws: tuple[AwsClient, AwsClient],
) -> None:
    blobs = store(aws)
    first = await blobs.put("alpha", b"alpha bytes", "text/plain")
    second = await blobs.put("beta", b"beta bytes", "text/plain")
    first_row = metadata(aws, "B#" + first.attachment_id)[0]
    second_root = payload(metadata(aws, "B#" + second.attachment_id)[0])
    corrupted = payload(first_row)
    for key in ("key", "generation", "version_id"):
        corrupted[key] = second_root[key]
    first_row["data"] = {"S": json.dumps(corrupted, separators=(",", ":"))}
    aws[1].put_item(TableName="scone-test-blobs", Item=first_row)
    for row in metadata(aws, "S#alpha"):
        if cast(dict[str, str], row["sk"])["S"].startswith("H#"):
            data = payload(row)
            data["generation"] = second_root["generation"]
            row["data"] = {"S": json.dumps(data, separators=(",", ":"))}
            aws[1].put_item(TableName="scone-test-blobs", Item=row)
    with pytest.raises(SconeError):
        await blobs.release_space("alpha")
    assert (await blobs.get("beta", second.attachment_id))[1] == b"beta bytes"


async def test_recovery_access_denied_is_not_treated_as_absent(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_error = cast(type[Exception], getattr(importlib.import_module("botocore.exceptions"), "ClientError"))
    seed_intents(aws, 1)

    def denied(**kwargs: object) -> dict[str, object]:
        raise client_error({"Error": {"Code": "403", "Message": "private access detail"}}, "HeadObject")

    monkeypatch.setattr(aws[0], "head_object", denied)
    with pytest.raises(SconeError, match="AWS blob operation failed"):
        await store(aws).recover_uploads("alpha")
    assert len(metadata(aws, "I#alpha")) == 1


async def test_close_drains_active_worker_and_closes_only_owned_clients(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto3 = pytest.importorskip("boto3")
    closed: list[str] = []

    def factory(service: str, **kwargs: object) -> AwsClient:
        assert kwargs["region_name"] == "us-east-1"
        return aws[0] if service == "s3" else aws[1]

    def close_s3() -> None:
        closed.append("s3")

    def close_ddb() -> None:
        closed.append("dynamodb")

    monkeypatch.setattr(boto3, "client", factory)
    monkeypatch.setattr(aws[0], "close", close_s3)
    monkeypatch.setattr(aws[1], "close", close_ddb)
    owned = S3BlobStore("scone-test-blobs", "scone-test-blobs", region_name="us-east-1")
    entered, finish = Event(), Event()
    original_put = aws[0].put_object

    def paused_upload(**kwargs: object) -> dict[str, object]:
        entered.set()
        assert finish.wait(5)
        return original_put(**kwargs)

    monkeypatch.setattr(aws[0], "put_object", paused_upload)
    task = asyncio.create_task(owned.put("alpha", b"drain on shutdown", "text/plain"))
    assert await asyncio.to_thread(entered.wait, 3)
    closing = asyncio.create_task(owned.close())
    await asyncio.sleep(.02)
    assert not closing.done()
    assert closed == []
    with pytest.raises(SconeError, match="closed"):
        await owned.held("alpha")
    finish.set()
    with pytest.raises(SconeError):
        await task
    await closing
    await owned.close()
    assert closed == ["s3", "dynamodb"]
    await store(aws).close()
    assert closed == ["s3", "dynamodb"]


async def test_close_unused_store_does_not_load_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(name: str) -> object:
        raise AssertionError("close loaded unused SDK")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    blobs = S3BlobStore("scone-test-blobs", "scone-test-blobs", region_name="us-east-1")
    await blobs.close()
    await blobs.close()
    with pytest.raises(SconeError, match="closed"):
        await blobs.held("alpha")


async def test_forged_cleanup_intent_cannot_delete_published_generation(
    aws: tuple[AwsClient, AwsClient],
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("beta", b"still published", "text/plain")
    root = payload(metadata(aws, "B#" + attachment.attachment_id)[0])
    data = {key: root[key] for key in ("attachment_id", "key", "generation", "version_id")}
    data["kind"] = "delete"
    aws[1].put_item(TableName="scone-test-blobs", Item={"pk": {"S": "I#alpha"},
        "sk": {"S": root["generation"]}, "rev": {"N": "1"}, "data": {"S": json.dumps(data)}})
    with pytest.raises(SconeError, match="published generation"):
        await blobs.recover_uploads("alpha")
    assert (await blobs.get("beta", attachment.attachment_id))[1] == b"still published"


async def test_partial_client_initialization_reuses_owned_s3_client(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto3 = pytest.importorskip("boto3")
    calls: list[str] = []

    def factory(service: str, **kwargs: object) -> AwsClient:
        calls.append(service)
        if service == "dynamodb" and calls.count("dynamodb") == 1:
            raise RuntimeError("synthetic client initialization failure")
        return aws[0] if service == "s3" else aws[1]

    monkeypatch.setattr(boto3, "client", factory)
    blobs = S3BlobStore("scone-test-blobs", "scone-test-blobs", region_name="us-east-1")
    try:
        with pytest.raises(SconeError):
            await blobs.held("alpha")
        assert await blobs.held("alpha") == []
        assert calls == ["s3", "dynamodb", "dynamodb"]
    finally:
        await blobs.close()


def local_dynamodb_url() -> str:
    url = os.environ.get("SCONE_TEST_DYNAMODB_URL")
    if not url:
        pytest.skip("SCONE_TEST_DYNAMODB_URL is not configured")
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        pytest.fail("DynamoDB Local tests require an explicit loopback HTTP endpoint")
    return url


def local_table(ddb: AwsClient) -> str:
    table = "scone-test-" + uuid4().hex
    ddb.create_table(TableName=table, KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"}], AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"}], BillingMode="PAY_PER_REQUEST")
    return table


async def test_dynamodb_local_concurrent_cas_keeps_the_committed_winner() -> None:
    from threading import Barrier
    url = local_dynamodb_url()
    boto3 = pytest.importorskip("boto3")
    ddb = cast(AwsClient, boto3.client("dynamodb", endpoint_url=url, region_name="us-east-1",
               aws_access_key_id="test", aws_secret_access_key="test"))
    table = local_table(ddb)
    try:
        for _ in range(20):
            barrier = Barrier(2)
            key = {"pk": {"S": "counter"}, "sk": {"S": "counter"}}
            ddb.put_item(TableName=table, Item={**key, "rev": {"N": "1"}})

            def compete(value: int) -> int | None:
                barrier.wait(timeout=5)
                try:
                    ddb.transact_write_items(TransactItems=[{"Put": {"TableName": table,
                        "Item": {**key, "rev": {"N": str(value)}}, "ConditionExpression": "rev = :old",
                        "ExpressionAttributeValues": {":old": {"N": "1"}}}}])
                    return value
                except Exception as error:
                    response = cast(dict[str, dict[str, str]], getattr(error, "response"))
                    assert response["Error"]["Code"] == "TransactionCanceledException"
                    return None

            outcomes = await asyncio.gather(asyncio.to_thread(compete, 2), asyncio.to_thread(compete, 3))
            winners = [value for value in outcomes if value is not None]
            assert len(winners) == 1
            item = cast(dict[str, dict[str, str]], ddb.get_item(TableName=table, Key=key, ConsistentRead=True)["Item"])
            assert item["rev"]["N"] == str(winners[0])
    finally:
        ddb.delete_table(TableName=table)
        ddb.close()


async def test_dynamodb_local_concurrent_attachment_ownership() -> None:
    url = local_dynamodb_url()
    moto = pytest.importorskip("moto")
    boto3 = pytest.importorskip("boto3")
    with moto.mock_aws(config={"core": {"passthrough": {"urls": [re.escape(url) + r"/.*"]}}}):
        s3 = cast(AwsClient, boto3.client("s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test"))
        ddb = cast(AwsClient, boto3.client("dynamodb", endpoint_url=url, region_name="us-east-1",
                   aws_access_key_id="test", aws_secret_access_key="test"))
        s3.create_bucket(Bucket="scone-test-blobs")
        s3.put_bucket_versioning(Bucket="scone-test-blobs", VersioningConfiguration={"Status": "Enabled"})
        table = local_table(ddb)
        stores = [S3BlobStore("scone-test-blobs", table, region_name="us-east-1",
                  s3_client=s3, dynamodb_client=ddb) for _ in range(4)]
        try:
            for round_number in range(5):
                data = f"concurrent attachment round {round_number}".encode()
                attachments = await asyncio.gather(*(blobs.put(f"space{number}", data, "text/plain")
                    for number, blobs in enumerate(stores)))
                identifier = attachments[0].attachment_id
                result = ddb.get_item(TableName=table, ConsistentRead=True,
                    Key={"pk": {"S": "B#" + identifier}, "sk": {"S": "R"}})
                assert payload(cast(dict[str, object], result["Item"]))["holds"] == 4
                await asyncio.gather(*(blobs.link("space0", identifier, number + 1) for number, blobs in enumerate(stores)))
                for number in range(4):
                    assert await stores[number].for_episode("space0", number + 1) == [attachments[0]]
                results = await asyncio.gather(*(blobs.release_space(f"space{number}") for number, blobs in enumerate(stores)))
                assert sum(len(released) for released, _ in results) == 1
                assert sum(len(kept) for _, kept in results) == 3
                assert "Item" not in ddb.get_item(TableName=table, ConsistentRead=True,
                    Key={"pk": {"S": "B#" + identifier}, "sk": {"S": "R"}})
                assert s3.list_object_versions(Bucket="scone-test-blobs").get("Versions", []) == []
        finally:
            for blobs in stores:
                await blobs.close()
            ddb.delete_table(TableName=table)
            ddb.close()
            s3.close()
