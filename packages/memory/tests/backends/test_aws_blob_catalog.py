"""Indexed catalog and journal contracts against local Moto, never AWS."""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import cast

import pytest

from scone_memory.backends.aws_blobs import AwsClient, S3BlobStore
from scone_memory.core.errors import SconeError
from ..backends.test_aws_blobs import aws, metadata, payload, store  # noqa: F401

TABLE = "scone-test-blobs"


def seed_legacy(aws: tuple[AwsClient, AwsClient], count: int, *, episode: int = 7) -> list[str]:
    rows: list[dict[str, object]] = []
    identifiers: list[str] = []

    def row(pk: str, sk: str, value: dict[str, object]) -> dict[str, object]:
        return {"pk": {"S": pk}, "sk": {"S": sk}, "rev": {"N": "1"}, "data": {"S": json.dumps(value)}}

    for number in range(count):
        identifier = hashlib.sha256(f"legacy-{number}".encode()).hexdigest()
        identifiers.append(identifier)
        generation = f"{number + 1:032x}"
        attachment = {"attachment_id": identifier, "media_type": "text/plain", "bytes": 1, "filename": None}
        rows.extend([
            row("S#alpha", "H#" + identifier, {"attachment": attachment, "generation": generation,
                "links": [[episode, number + 1]] if episode else []}),
            row("B#" + identifier, "R", {"attachment_id": identifier, "generation": generation,
                "key": f"attachments/{identifier}/{generation}", "version_id": None, "size": 1, "holds": 1}),
        ])
    control = row("S#alpha", "C", {"mode": "active", "count": count, "operation": "", "episode": 0,
        "pending": [], "released": [], "kept": [], "gc": []})
    control["rev"] = {"N": str(count + 1)}
    rows.append(control)
    for offset in range(0, len(rows), 25):
        result = aws[1].batch_write_item(RequestItems={TABLE: [{"PutRequest": {"Item": item}} for item in rows[offset:offset + 25]]})
        assert not result.get("UnprocessedItems")
    return identifiers


async def test_episode_lookup_migrates_legacy_then_queries_only_target_partition(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifiers = seed_legacy(aws, 120)
    blobs = store(aws)
    assert [a.attachment_id for a in await blobs.for_episode("alpha", 7)] == identifiers
    calls: list[dict[str, object]] = []
    query = aws[1].query

    def record(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return query(**kwargs)

    monkeypatch.setattr(aws[1], "query", record)
    assert await blobs.for_episode("alpha", 99) == []
    assert calls and all(cast(dict[str, dict[str, str]], call["ExpressionAttributeValues"])[":pk"]["S"] == "E#alpha#99" for call in calls)
    assert all(call["ConsistentRead"] is True for call in calls)
    assert await store(aws).for_episode("beta", 7) == []


async def test_legacy_migration_and_reads_support_more_than_1024_holds(aws: tuple[AwsClient, AwsClient]) -> None:
    identifiers = seed_legacy(aws, 1030)
    blobs = S3BlobStore(TABLE, TABLE, region_name="us-east-1", timeout_s=180, s3_client=aws[0], dynamodb_client=aws[1])
    assert [a.attachment_id for a in await blobs.for_episode("alpha", 7)] == identifiers
    added = await blobs.put("alpha", b"after former capacity", "text/plain")
    await blobs.link("alpha", added.attachment_id, 7)
    assert (await blobs.for_episode("alpha", 7))[-1] == added
    assert len(await blobs.held("alpha")) == 1031


async def test_catalog_link_unlink_is_transactional_and_idempotent(aws: tuple[AwsClient, AwsClient]) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"indexed bytes", "text/plain")
    await blobs.link("alpha", attachment.attachment_id, 1)
    await blobs.link("alpha", attachment.attachment_id, 1)
    await blobs.link("alpha", attachment.attachment_id, 2)
    assert len(metadata(aws, "E#alpha#1")) == 1
    assert len(metadata(aws, "E#alpha#2")) == 1
    assert await blobs.unlink("alpha", 1) == []
    assert metadata(aws, "E#alpha#1") == []
    assert await blobs.for_episode("alpha", 2) == [attachment]
    assert await blobs.unlink("alpha", 2) == [attachment.attachment_id]
    assert metadata(aws, "E#alpha#2") == []
    assert await blobs.unlink("alpha", 2) == []


async def test_space_release_beyond_1024_uses_bounded_journal_rows(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scone_memory.backends._aws_blob_catalog import Catalog

    identifiers = seed_legacy(aws, 1030)
    blobs = S3BlobStore(TABLE, TABLE, region_name="us-east-1", timeout_s=60, s3_client=aws[0], dynamodb_client=aws[1])
    original = Catalog.release_hold
    steps: dict[object, int] = {}
    checkpoints = 0

    def bounded_attempt(self: Catalog, *args, **kwargs):
        nonlocal checkpoints
        result = original(self, *args, **kwargs)
        steps[self.s] = steps.get(self.s, 0) + int(result)
        if steps[self.s] == 10:
            # Stop only after a committed checkpoint, before completion. This
            # deterministically exercises recovery without sleeping near a timer.
            checkpoints += 1
            self.s.stop.set()
        return result

    monkeypatch.setattr(Catalog, "release_hold", bounded_attempt)
    previous_revision = 1031
    previous_count = 1030
    for attempt in range(20):
        try:
            released, kept = await blobs.release_space("alpha")
            break
        except SconeError as error:
            if "timed out" not in str(error):
                raise
            row = metadata(aws, "S#alpha")[0]
            state = payload(row)
            revision = int(cast(dict[str, str], row["rev"])["N"])
            assert revision > previous_revision, "each bounded attempt must persist progress"
            assert int(state["count"]) <= previous_count
            previous_revision, previous_count = revision, int(state["count"])
    else:
        pytest.fail("paged release did not make sufficient resumable progress")
    assert checkpoints >= 1
    assert released == sorted(identifiers)
    assert kept == []
    assert await blobs.held("alpha") == []
    assert await blobs.for_episode("alpha", 7) == []
    control = payload(metadata(aws, "S#alpha")[0])
    assert control["mode"] == "active"
    assert len(json.dumps(control)) < 2048
    assert control["pending"] == control["released"] == control["kept"] == control["gc"] == []


async def test_migration_checkpoint_survives_lost_transaction_response(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifiers = seed_legacy(aws, 120)
    original = aws[1].transact_write_items
    failed = False

    def lose_response(**kwargs: object) -> dict[str, object]:
        nonlocal failed
        result = original(**kwargs)
        changes = cast(list[dict[str, dict[str, object]]], kwargs["TransactItems"])
        if not failed and any(str(change.get("Put", {}).get("Item", "")).find("E#alpha#7") >= 0 for change in changes):
            failed = True
            raise RuntimeError("synthetic lost migration response")
        return result

    monkeypatch.setattr(aws[1], "transact_write_items", lose_response)
    with pytest.raises(SconeError):
        await store(aws).for_episode("alpha", 7)
    control = payload(metadata(aws, "S#alpha")[0])
    assert control["mode"] == "migrate"
    assert control["catalog_after"]
    monkeypatch.setattr(aws[1], "transact_write_items", original)
    assert [a.attachment_id for a in await store(aws).for_episode("alpha", 7)] == identifiers
    assert len(metadata(aws, "E#alpha#7")) == 120


async def test_concurrent_link_helps_paused_migration_without_lost_links(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import Event

    first, helper = store(aws), store(aws)
    attachment = await first.put("alpha", b"migrate alongside link", "text/plain")
    await first.link("alpha", attachment.attachment_id, 1)
    control_row = metadata(aws, "S#alpha")[0]
    current = payload(control_row)
    legacy = {name: current[name] for name in ("mode", "count", "operation", "episode", "pending", "released", "kept", "gc")}
    control_row["data"] = {"S": json.dumps(legacy)}
    aws[1].put_item(TableName=TABLE, Item=control_row)
    aws[1].delete_item(TableName=TABLE, Key={"pk": {"S": "E#alpha#1"}, "sk": {"S": attachment.attachment_id}})
    entered, finish = Event(), Event()
    original = aws[1].transact_write_items
    paused = False

    def pause_first_backfill(**kwargs: object) -> dict[str, object]:
        nonlocal paused
        changes = cast(list[dict[str, dict[str, object]]], kwargs["TransactItems"])
        if not paused and any("E#alpha#1" in str(change.get("Put", {}).get("Item", "")) for change in changes):
            paused = True
            entered.set()
            assert finish.wait(5)
        return original(**kwargs)

    monkeypatch.setattr(aws[1], "transact_write_items", pause_first_backfill)
    task = asyncio.create_task(first.for_episode("alpha", 1))
    assert await asyncio.to_thread(entered.wait, 3)
    try:
        await helper.link("alpha", attachment.attachment_id, 2)
    finally:
        finish.set()
    assert await task == [attachment]
    assert await helper.for_episode("alpha", 1) == [attachment]
    assert await helper.for_episode("alpha", 2) == [attachment]


async def test_stale_catalog_generation_is_rejected_without_cross_space_exposure(
    aws: tuple[AwsClient, AwsClient],
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"generation scoped", "text/plain")
    await blobs.link("alpha", attachment.attachment_id, 1)
    row = metadata(aws, "E#alpha#1")[0]
    value = payload(row)
    value["generation"] = "f" * 32
    row["data"] = {"S": json.dumps(value)}
    aws[1].put_item(TableName=TABLE, Item=row)
    with pytest.raises(SconeError, match="catalog ownership"):
        await blobs.for_episode("alpha", 1)
    assert await blobs.for_episode("beta", 1) == []
    assert (await blobs.get("alpha", attachment.attachment_id))[1] == b"generation scoped"


@pytest.mark.parametrize("cursor", ["../outside", "H#not-a-digest", "E#alpha#7"])
async def test_invalid_migration_cursor_is_rejected(
    aws: tuple[AwsClient, AwsClient], cursor: str,
) -> None:
    identifiers = seed_legacy(aws, 2)
    row = metadata(aws, "S#alpha")[0]
    value = payload(row)
    value.update(mode="migrate", catalog_after=cursor)
    row["data"] = {"S": json.dumps(value)}
    aws[1].put_item(TableName=TABLE, Item=row)
    with pytest.raises(SconeError, match="cursor"):
        await store(aws).for_episode("alpha", 7)
    assert sorted(identifiers) == await store(aws).held("alpha")
    assert metadata(aws, "E#alpha#7") == []


async def test_old_unfinished_release_journal_resumes_before_catalog_migration(
    aws: tuple[AwsClient, AwsClient],
) -> None:
    identifiers = seed_legacy(aws, 3)
    row = metadata(aws, "S#alpha")[0]
    value = payload(row)
    value.update(mode="unlink", episode=7, operation="a" * 32, pending=identifiers)
    row["data"] = {"S": json.dumps(value)}
    aws[1].put_item(TableName=TABLE, Item=row)
    blobs = store(aws)
    assert [a.attachment_id for a in await blobs.for_episode("alpha", 7)] == identifiers
    assert await blobs.unlink("alpha", 7) == identifiers
    assert await blobs.for_episode("alpha", 7) == []
    assert payload(metadata(aws, "S#alpha")[0])["catalog_version"] == 1


async def test_unlink_and_gc_resume_from_small_rows_after_delete_failure(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = store(aws)
    first = await blobs.put("alpha", b"one resumable", "text/plain")
    second = await blobs.put("alpha", b"two resumable", "text/plain")
    await blobs.link("alpha", second.attachment_id, 7)
    await blobs.link("alpha", first.attachment_id, 7)
    original = aws[0].delete_object

    def fail(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("offline deletion fault")

    monkeypatch.setattr(aws[0], "delete_object", fail)
    with pytest.raises(SconeError):
        await blobs.unlink("alpha", 7)
    state = payload(metadata(aws, "S#alpha")[0])
    assert state["mode"] == "unlink" and state["release_phase"] == "gc"
    assert state["pending"] == state["released"] == state["gc"] == []
    results_pk = f"J#alpha#{state['operation']}"
    assert len(metadata(aws, results_pk)) == 2
    assert metadata(aws, "E#alpha#7") == []
    monkeypatch.setattr(aws[0], "delete_object", original)
    assert await store(aws).unlink("alpha", 7) == [second.attachment_id, first.attachment_id]
    assert metadata(aws, "I#alpha") == []
    assert await store(aws).unlink("alpha", 7) == []
    assert metadata(aws, results_pk) == []


@pytest.mark.parametrize("total_holds", [1, 101, 1030])
async def test_fixed_episode_lookup_request_count_is_independent_of_space_size(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch, total_holds: int,
) -> None:
    identifiers = seed_legacy(aws, total_holds, episode=0)
    hold = next(row for row in metadata(aws, "S#alpha") if cast(dict[str, str], row["sk"])["S"] == "H#" + identifiers[0])
    value = payload(hold)
    value["links"] = [[7, 1]]
    hold["data"] = {"S": json.dumps(value)}
    aws[1].put_item(TableName=TABLE, Item=hold)
    counts = {"query": 0, "get_item": 0, "transact_write_items": 0}

    def recorder(name: str):
        original = getattr(aws[1], name)

        def request(**kwargs: object) -> dict[str, object]:
            counts[name] += 1
            return original(**kwargs)

        return request

    for name in counts:
        monkeypatch.setattr(aws[1], name, recorder(name))
    blobs = store(aws)
    await blobs._run(lambda session: session.catalog().ensure("alpha"))
    migration = dict(counts)
    counts.update({name: 0 for name in counts})
    assert [a.attachment_id for a in await blobs.for_episode("alpha", 7)] == [identifiers[0]]
    assert counts == {"query": 1, "get_item": 4, "transact_write_items": 0}
    print({"total_holds": total_holds, "migration": migration, "steady_state": dict(counts)})


async def test_large_reference_hold_keeps_separate_cap_and_resumes_paged_unlink(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = store(aws)
    attachment = await blobs.put("alpha", b"many episode references", "text/plain")
    rows = metadata(aws, "S#alpha")
    for row in rows:
        value = payload(row)
        if cast(dict[str, str], row["sk"])["S"] == "C":
            value = {name: value[name] for name in ("mode", "count", "operation", "episode", "pending", "released", "kept", "gc")}
            row["rev"] = {"N": "1025"}
        else:
            value["links"] = [[number, number] for number in range(1, 1025)]
        row["data"] = {"S": json.dumps(value)}
        aws[1].put_item(TableName=TABLE, Item=row)
    assert await blobs.for_episode("alpha", 1024) == [attachment]
    with pytest.raises(SconeError, match="reference capacity"):
        await blobs.link("alpha", attachment.attachment_id, 1025)
    sizes: list[int] = []
    original = aws[1].transact_write_items
    failed = False

    def lose_first_removal(**kwargs: object) -> dict[str, object]:
        nonlocal failed
        changes = cast(list[dict[str, dict[str, object]]], kwargs["TransactItems"])
        sizes.append(len(changes))
        result = original(**kwargs)
        if not failed and any("E#alpha#" in str(change.get("Delete", {}).get("Key", "")) for change in changes):
            failed = True
            raise RuntimeError("lost partial hold removal response")
        return result

    monkeypatch.setattr(aws[1], "transact_write_items", lose_first_removal)
    with pytest.raises(SconeError):
        await blobs.release_space("alpha")
    assert await blobs.for_episode("alpha", 1) == []
    assert await blobs.for_episode("alpha", 1024) == [attachment]
    monkeypatch.setattr(aws[1], "transact_write_items", original)
    assert await store(aws).release_space("alpha") == ([attachment.attachment_id], [])
    assert max(sizes) <= 100
    assert await blobs.for_episode("alpha", 1024) == []


async def test_cancelled_release_keeps_durable_receipt_and_worker_until_sdk_returns(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import Event

    blobs = store(aws)
    attachment = await blobs.put("alpha", b"cancelled catalog cleanup", "text/plain")
    entered, finish = Event(), Event()
    original = aws[0].delete_object

    def pause(**kwargs: object) -> dict[str, object]:
        entered.set()
        assert finish.wait(5)
        return original(**kwargs)

    monkeypatch.setattr(aws[0], "delete_object", pause)
    task = asyncio.create_task(blobs.release_space("alpha"))
    assert await asyncio.to_thread(entered.wait, 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    queued = asyncio.create_task(blobs.held("alpha"))
    await asyncio.sleep(.02)
    assert not queued.done()
    finish.set()
    assert await queued == []
    monkeypatch.setattr(aws[0], "delete_object", original)
    assert await store(aws).release_space("alpha") == ([attachment.attachment_id], [])
    assert metadata(aws, "I#alpha") == []


async def test_helpers_of_same_release_return_same_retained_receipt(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import Event
    from scone_memory.backends._aws_blob_catalog import Catalog

    first, helper = store(aws), store(aws)
    attachment = await first.put("alpha", b"cooperative cleanup", "text/plain")
    await first.link("alpha", attachment.attachment_id, 7)
    entered, finish = Event(), Event()
    original = Catalog.collect_page
    paused = False

    def pause_first(self: Catalog, *args, **kwargs):
        nonlocal paused
        if self.s.store is first and not paused:
            paused = True
            entered.set()
            assert finish.wait(5)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Catalog, "collect_page", pause_first)
    task = asyncio.create_task(first.unlink("alpha", 7))
    assert await asyncio.to_thread(entered.wait, 3)
    try:
        assert await helper.unlink("alpha", 7) == [attachment.attachment_id]
    finally:
        finish.set()
    assert await task == [attachment.attachment_id]


async def test_receipt_pruning_is_fenced_before_rows_disappear(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import Event
    from scone_memory.backends._aws_blob_catalog import Catalog

    first, helper, next_release = store(aws), store(aws), store(aws)
    attachment = await first.put("alpha", b"receipt pruning race", "text/plain")
    await first.link("alpha", attachment.attachment_id, 7)
    collecting, resume_collect = Event(), Event()
    reading, resume_read = Event(), Event()
    pruned, resume_prune = Event(), Event()
    collect = Catalog.collect_page
    receipt = Catalog.receipt
    prune = Catalog.prune_completed
    paused = False

    def pause_collect(self: Catalog, *args, **kwargs):
        nonlocal paused
        if self.s.store is first and not paused:
            paused = True
            collecting.set()
            assert resume_collect.wait(5)
        return collect(self, *args, **kwargs)

    def pause_read(self: Catalog, *args, **kwargs):
        if self.s.store is first:
            reading.set()
            assert resume_read.wait(5)
        return receipt(self, *args, **kwargs)

    def pause_prune(self: Catalog, *args, **kwargs):
        result = prune(self, *args, **kwargs)
        if self.s.store is next_release:
            pruned.set()
            assert resume_prune.wait(5)
        return result

    monkeypatch.setattr(Catalog, "collect_page", pause_collect)
    monkeypatch.setattr(Catalog, "receipt", pause_read)
    monkeypatch.setattr(Catalog, "prune_completed", pause_prune)
    task = asyncio.create_task(first.unlink("alpha", 7))
    assert await asyncio.to_thread(collecting.wait, 3)
    assert await helper.unlink("alpha", 7) == [attachment.attachment_id]
    resume_collect.set()
    assert await asyncio.to_thread(reading.wait, 3)
    later = asyncio.create_task(next_release.release_space("alpha"))
    try:
        assert await asyncio.to_thread(pruned.wait, 3)
        assert payload(metadata(aws, "S#alpha")[0])["mode"] == "release"
        resume_read.set()
        with pytest.raises(SconeError, match="retired"):
            await task
    finally:
        resume_read.set()
        resume_prune.set()
    assert await later == ([], [])


async def test_lost_prune_response_resumes_from_prior_operation(
    aws: tuple[AwsClient, AwsClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = store(aws)
    first = await blobs.put("alpha", b"previous receipt", "text/plain")
    assert await blobs.release_space("alpha") == ([first.attachment_id], [])
    previous = payload(metadata(aws, "S#alpha")[0])["operation"]
    second = await blobs.put("alpha", b"next receipt", "text/plain")
    original = aws[1].transact_write_items
    failed = False

    def lose_response(**kwargs: object) -> dict[str, object]:
        nonlocal failed
        result = original(**kwargs)
        changes = cast(list[dict[str, dict[str, object]]], kwargs["TransactItems"])
        if not failed and any(f"J#alpha#{previous}" in str(change.get("Delete", {}).get("Key", "")) for change in changes):
            failed = True
            raise RuntimeError("lost journal prune response")
        return result

    monkeypatch.setattr(aws[1], "transact_write_items", lose_response)
    with pytest.raises(SconeError):
        await blobs.release_space("alpha")
    state = payload(metadata(aws, "S#alpha")[0])
    assert state["mode"] == "release" and state["prior_operation"] == previous
    assert metadata(aws, f"J#alpha#{previous}") == []
    monkeypatch.setattr(aws[1], "transact_write_items", original)
    assert await store(aws).release_space("alpha") == ([second.attachment_id], [])
    assert payload(metadata(aws, "S#alpha")[0])["prior_operation"] == ""


@pytest.mark.parametrize("phase,cursor", [("holds", "H#../escape"), ("gc", "R#bad")])
async def test_malformed_release_cursor_does_not_mutate_holds(
    aws: tuple[AwsClient, AwsClient], phase: str, cursor: str,
) -> None:
    identifiers = seed_legacy(aws, 1)
    await store(aws).for_episode("alpha", 7)
    row = metadata(aws, "S#alpha")[0]
    state = payload(row)
    state.update(mode="release", operation="e" * 32, journal_version=1,
                 release_phase=phase, release_cursor=cursor)
    row["data"] = {"S": json.dumps(state)}
    aws[1].put_item(TableName=TABLE, Item=row)
    with pytest.raises(SconeError, match="cursor"):
        await store(aws).release_space("alpha")
    assert await store(aws).held("alpha") == identifiers
