"""S3 attachment bytes with transactional DynamoDB ownership metadata.

This is a BlobStore, not a document/vector store. AWS resources must already
exist. Construction imports no SDK and resolves no credentials. Each instance
runs one bounded worker; cancellation stops new requests but cannot interrupt an
in-flight SDK request, and its worker slot stays occupied until that request ends.

Objects have immutable, random generation keys. Last-holder removal commits a
DynamoDB deletion intent before deleting that exact S3 version. Failed cleanup
raises and leaves a resumable space journal. No cross-service atomicity is claimed.
Upload intents precede S3 writes. Explicit recover_uploads fences unpublished
uploads; abandoned intents remain durable because a paused uploader can finish
late. Repeating recovery removes such late orphan bytes without touching any
published generation. Recovery is an attempt, not proof no paused writer exists.

Bounds: 25 MiB per attachment, 1024 holds per space, 1024 episode references per
hold, 300 KiB per metadata item, 12 optimistic retries. Listing uses strongly
consistent partition queries, never table scans. Capacity failures are explicit.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import importlib
import math
import re
from threading import Event
from time import monotonic
from typing import Generic, Literal, Protocol, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import InvalidInput, NotFound, SconeError
from ..core.models import Attachment

MAX_BYTES = 25 * 1024 * 1024
MAX_HOLDS = 1024
MAX_ITEM_BYTES = 300 * 1024
RETRIES = 12


class AwsClient(Protocol):
    """The small dynamic call surface exposed by boto3 clients."""
    def __getattr__(self, name: str) -> Callable[..., dict[str, object]]: ...


class _Model(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class _Control(_Model):
    mode: Literal["active", "unlink", "release"] = "active"
    count: int = Field(default=0, ge=0, le=MAX_HOLDS)
    operation: str = ""
    episode: int = 0
    pending: tuple[str, ...] = ()
    released: tuple[str, ...] = ()
    kept: tuple[str, ...] = ()
    gc: tuple[str, ...] = ()


class _Root(_Model):
    attachment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: str = Field(pattern=r"^[0-9a-f]{32}$")
    key: str
    version_id: str | None
    size: int = Field(ge=1, le=MAX_BYTES)
    holds: int = Field(ge=1)


class _Hold(_Model):
    attachment: Attachment
    generation: str
    links: tuple[tuple[int, int], ...] = Field(default=(), max_length=1024)


class _Intent(_Model):
    kind: Literal["upload", "delete", "abandoned"]
    attachment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: str = Field(pattern=r"^[0-9a-f]{32}$")
    key: str
    version_id: str | None = None


class RecoveryPage(_Model):
    """A bounded cleanup sweep page; visited is not proof of reclaimed bytes."""
    visited: int = Field(ge=0, le=100)
    next_cursor: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True)
class _Row(Generic[M]):
    pk: str
    sk: str
    revision: int
    data: M


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise SconeError("AWS blob metadata is invalid")
    return cast(dict[str, object], value)


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise SconeError("AWS blob metadata is invalid")
    return value


def _code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        detail = response.get("Error")
        if isinstance(detail, dict) and isinstance(detail.get("Code"), str):
            return cast(str, detail["Code"])
    return None


def _space(space: str) -> None:
    if type(space) is not str or re.fullmatch(r"[a-z0-9_-]{1,64}", space) is None:
        raise InvalidInput("invalid attachment space")


def _digest(identifier: str) -> None:
    if type(identifier) is not str or re.fullmatch(r"[0-9a-f]{64}", identifier) is None:
        raise InvalidInput("invalid attachment digest")


def _episode(identifier: int) -> None:
    if type(identifier) is not int or not 1 <= identifier <= 2**63 - 1:
        raise InvalidInput("invalid episode id")


class S3BlobStore:
    name = "s3"

    def __init__(self, bucket: str, table: str, *, region_name: str,
                 prefix: str = "attachments/", timeout_s: float = 30,
                 s3_client: AwsClient | None = None, dynamodb_client: AwsClient | None = None) -> None:
        if type(bucket) is not str or re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) is None:
            raise ValueError("invalid S3 bucket name")
        if type(table) is not str or re.fullmatch(r"[A-Za-z0-9_.-]{3,255}", table) is None:
            raise ValueError("invalid DynamoDB table name")
        if type(region_name) is not str or re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-\d", region_name) is None:
            raise ValueError("invalid AWS region")
        if (type(prefix) is not str or not 1 <= len(prefix) <= 128 or not prefix.endswith("/")
                or re.fullmatch(r"[A-Za-z0-9_/-]+", prefix) is None or prefix.startswith("/")):
            raise ValueError("invalid S3 prefix")
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 1 <= timeout_s <= 180:
            raise ValueError("timeout_s must be finite from 1 to 180")
        if (s3_client is None) != (dynamodb_client is None):
            raise ValueError("provide both injected AWS clients")
        self.bucket, self.table, self.region_name, self.prefix = bucket, table, region_name, prefix
        self.timeout_s = float(timeout_s)
        self._s3, self._ddb = s3_client, dynamodb_client
        self._gate = asyncio.Lock()
        self._owns_clients = s3_client is None
        self._closed = False
        self._active_stop: Event | None = None
        self._close_task: asyncio.Task[None] | None = None

    def _clients(self) -> tuple[AwsClient, AwsClient]:
        if self._s3 is None or self._ddb is None:
            try:
                sdk = importlib.import_module("boto3")
                config_module = importlib.import_module("botocore.config")
            except ImportError as error:
                raise ImportError("S3BlobStore requires scone-memory[aws]") from error
            factory = cast(Callable[..., AwsClient], getattr(sdk, "client"))
            config_factory = cast(Callable[..., object], getattr(config_module, "Config"))
            config = config_factory(connect_timeout=5, read_timeout=10,
                                    retries={"mode": "standard", "total_max_attempts": 2}, max_pool_connections=1)
            if self._s3 is None:
                self._s3 = factory("s3", region_name=self.region_name, config=config)
            if self._ddb is None:
                self._ddb = factory("dynamodb", region_name=self.region_name, config=config)
        return self._s3, self._ddb

    async def _run(self, operation: Callable[[_Session], T]) -> T:
        if self._closed:
            raise SconeError("AWS blob store is closed")
        deadline = monotonic() + self.timeout_s
        stop = Event()
        try:
            await asyncio.wait_for(self._gate.acquire(), self.timeout_s)
        except TimeoutError:
            raise SconeError("AWS blob operation timed out") from None
        if self._closed:
            self._gate.release()
            raise SconeError("AWS blob store is closed")
        self._active_stop = stop

        def work() -> T:
            try:
                if stop.is_set() or monotonic() >= deadline:
                    raise SconeError("AWS blob operation timed out")
                s3, ddb = self._clients()
                return operation(_Session(self, s3, ddb, stop, deadline))
            except (SconeError, ImportError):
                raise
            except Exception:
                raise SconeError("AWS blob operation failed") from None

        task = asyncio.create_task(asyncio.to_thread(work))

        def done(finished: asyncio.Task[T]) -> None:
            self._active_stop = None
            self._gate.release()
            if not finished.cancelled():
                finished.exception()

        task.add_done_callback(done)
        try:
            return await asyncio.wait_for(asyncio.shield(task), max(0, deadline - monotonic()))
        except asyncio.CancelledError:
            stop.set()
            raise
        except TimeoutError:
            stop.set()
            raise SconeError("AWS blob operation timed out") from None

    async def close(self) -> None:
        """Drain the current worker, then close owned clients exactly once.

        Injected clients remain caller-owned. A cancelled close caller does not
        cancel background draining. A never-used store closes without SDK import
        or credential resolution. New and queued operations are rejected.
        """
        if self._close_task is None:
            self._closed = True
            if self._active_stop is not None:
                self._active_stop.set()
            self._close_task = asyncio.create_task(self._finish_close())
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        async with self._gate:
            if not self._owns_clients:
                return
            clients = tuple(client for client in (self._s3, self._ddb) if client is not None)

            def finish() -> None:
                failed = False
                for client in clients:
                    try:
                        client.close()
                    except Exception:
                        failed = True
                if failed:
                    raise SconeError("AWS blob client shutdown failed")

            if clients:
                await asyncio.to_thread(finish)

    async def put(self, space: str, data: bytes, media_type: str, filename: str | None = None) -> Attachment:
        _space(space)
        if type(data) is not bytes or not 1 <= len(data) <= MAX_BYTES:
            raise InvalidInput("attachment bytes exceed allowed bounds")
        if type(media_type) is not str or not 1 <= len(media_type.encode("utf-8")) <= 255:
            raise InvalidInput("invalid attachment media type")
        if filename is not None and (type(filename) is not str or len(filename.encode("utf-8")) > 1024):
            raise InvalidInput("invalid attachment filename")
        attachment = Attachment(attachment_id=hashlib.sha256(data).hexdigest(), media_type=media_type,
                                bytes=len(data), filename=filename)
        return await self._run(lambda session: session.put(space, data, attachment))

    async def get(self, space: str, attachment_id: str) -> tuple[Attachment, bytes]:
        _space(space)
        _digest(attachment_id)
        return await self._run(lambda session: session.get(space, attachment_id))

    async def link(self, space: str, attachment_id: str, episode_id: int) -> None:
        _space(space)
        _digest(attachment_id)
        _episode(episode_id)
        await self._run(lambda session: session.link(space, attachment_id, episode_id))

    async def for_episode(self, space: str, episode_id: int) -> list[Attachment]:
        _space(space)
        _episode(episode_id)
        return await self._run(lambda session: session.for_episode(space, episode_id))

    async def held(self, space: str) -> list[str]:
        _space(space)
        return await self._run(lambda session: sorted(row.data.attachment.attachment_id for row in session.snapshot(space)[1]))

    async def linked(self, space: str) -> set[str]:
        _space(space)
        return await self._run(lambda session: {row.data.attachment.attachment_id for row in session.snapshot(space)[1] if row.data.links})

    async def released_by(self, space: str, episode_id: int) -> list[str]:
        _space(space)
        _episode(episode_id)
        return await self._run(lambda session: session.released_by(space, episode_id))

    async def unlink(self, space: str, episode_id: int) -> list[str]:
        _space(space)
        _episode(episode_id)
        return (await self._run(lambda session: session.release(space, episode_id)))[0]

    async def release_space(self, space: str, *, preview: bool = False) -> tuple[list[str], list[str]]:
        _space(space)
        if type(preview) is not bool:
            raise InvalidInput("preview must be a boolean")
        return await self._run(lambda session: session.preview(space) if preview else session.release(space, None))

    async def recover_uploads(self, space: str, *, after: str | None = None, limit: int = 100) -> RecoveryPage:
        """Fence and attempt cleanup of unpublished generations; repeat after crashes.

        May abort a concurrent put. Follow next_cursor until None; repeat a full
        sweep after crashes or late writes. Each page visits at most limit intents.
        This is not a promise an already in-flight uploader cannot finish later.
        Published data/references are unaffected. Abandoned intents are kept.
        """
        _space(space)
        if after is not None and (type(after) is not str or re.fullmatch(r"[0-9a-f]{32}", after) is None):
            raise InvalidInput("invalid AWS blob recovery cursor")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise InvalidInput("recovery page limit must be from 1 to 100")
        return await self._run(lambda session: session.recover(space, after, limit))


class _Session:
    def __init__(self, store: S3BlobStore, s3: AwsClient, ddb: AwsClient, stop: Event, deadline: float) -> None:
        self.store, self.s3, self.ddb, self.stop, self.deadline = store, s3, ddb, stop, deadline

    def check(self) -> None:
        if self.stop.is_set() or monotonic() >= self.deadline:
            raise SconeError("AWS blob operation timed out")

    def call(self, client: AwsClient, method: str, **kwargs: object) -> dict[str, object]:
        self.check()
        result = _mapping(getattr(client, method)(**kwargs))
        try:
            self.check()
        except SconeError:
            closer = getattr(result.get("Body"), "close", None)
            if callable(closer):
                closer()
            raise
        return result

    @staticmethod
    def key(pk: str, sk: str) -> dict[str, object]:
        return {"pk": {"S": pk}, "sk": {"S": sk}}

    def read(self, pk: str, sk: str, model: type[M]) -> _Row[M] | None:
        result = self.call(self.ddb, "get_item", TableName=self.store.table, Key=self.key(pk, sk), ConsistentRead=True)
        item = result.get("Item")
        if item is None:
            return None
        row = self.row(item, model)
        if (row.pk, row.sk) != (pk, sk):
            raise SconeError("AWS blob metadata identity is invalid")
        return row

    def row(self, value: object, model: type[M]) -> _Row[M]:
        item = _mapping(value)
        pk = _text(_mapping(item["pk"])["S"])
        sk = _text(_mapping(item["sk"])["S"])
        revision = int(_text(_mapping(item["rev"])["N"]))
        payload = _text(_mapping(item["data"])["S"])
        if revision < 1 or len(payload.encode("utf-8")) > MAX_ITEM_BYTES:
            raise SconeError("AWS blob metadata is invalid")
        data = model.model_validate_json(payload, strict=True)
        if isinstance(data, _Root):
            if (pk, sk) != ("B#" + data.attachment_id, "R"):
                raise SconeError("AWS blob root identity is invalid")
            self.object_args(data)
        if isinstance(data, _Intent):
            if sk != data.generation:
                raise SconeError("AWS blob intent identity is invalid")
            self.object_args(data)
        return _Row(pk, sk, revision, data)

    def query(self, pk: str, prefix: str, model: type[M], *, limit: int = MAX_HOLDS) -> list[_Row[M]]:
        found: list[_Row[M]] = []
        cursor: object = None
        for _ in range(limit + 1):
            options: dict[str, object] = {"TableName": self.store.table, "ConsistentRead": True,
                "KeyConditionExpression": "pk = :pk",
                "ExpressionAttributeValues": {":pk": {"S": pk}},
                "Limit": min(100, limit + 1 - len(found))}
            if prefix:
                options["KeyConditionExpression"] = "pk = :pk AND begins_with(sk, :prefix)"
                options["ExpressionAttributeValues"] = {":pk": {"S": pk}, ":prefix": {"S": prefix}}
            if cursor is not None:
                options["ExclusiveStartKey"] = cursor
            result = self.call(self.ddb, "query", **options)
            items = result.get("Items", [])
            if not isinstance(items, list):
                raise SconeError("AWS blob metadata is invalid")
            found.extend(self.row(item, model) for item in items)
            if len(found) > limit:
                raise SconeError("AWS blob query exceeds capacity")
            cursor = result.get("LastEvaluatedKey")
            if not cursor:
                return found
        raise SconeError("AWS blob query exceeds capacity")

    def change(self, pk: str, sk: str, old: _Row[M] | None, value: BaseModel | None) -> dict[str, object]:
        body: dict[str, object] = {"TableName": self.store.table}
        if old is None:
            body["ConditionExpression"] = "attribute_not_exists(pk)"
        else:
            # A deleted root can be recreated at revision 1. Comparing its
            # complete payload also fences that generation-level ABA race.
            body["ConditionExpression"] = "rev = :revision AND #payload = :payload"
            body["ExpressionAttributeNames"] = {"#payload": "data"}
            body["ExpressionAttributeValues"] = {":revision": {"N": str(old.revision)},
                                                  ":payload": {"S": old.data.model_dump_json()}}
        if value is None:
            body["Key"] = self.key(pk, sk)
            return {"Delete": body}
        payload = value.model_dump_json()
        if len(payload.encode("utf-8")) > MAX_ITEM_BYTES:
            raise SconeError("AWS blob metadata exceeds capacity")
        body["Item"] = {**self.key(pk, sk), "rev": {"N": str(1 if old is None else old.revision + 1)}, "data": {"S": payload}}
        return {"Put": body}

    def transaction(self, changes: list[dict[str, object]]) -> bool:
        try:
            self.call(self.ddb, "transact_write_items", TransactItems=changes, ClientRequestToken=uuid4().hex)
            return True
        except Exception as error:
            if _code(error) in {"TransactionCanceledException", "TransactionConflictException", "ConditionalCheckFailedException"}:
                return False
            raise

    def control(self, space: str) -> _Row[_Control] | None:
        return self.read("S#" + space, "C", _Control)

    @staticmethod
    def active(row: _Row[_Control] | None) -> _Control:
        value = row.data if row else _Control()
        if value.mode != "active":
            raise SconeError("AWS blob space has unfinished release; retry that release")
        return value

    def snapshot(self, space: str) -> tuple[_Row[_Control] | None, list[_Row[_Hold]]]:
        for _ in range(RETRIES):
            before = self.control(space)
            rows = self.query("S#" + space, "H#", _Hold)
            if self.control(space) == before:
                for row in rows:
                    _digest(row.data.attachment.attachment_id)
                    if row.pk != "S#" + space or row.sk != "H#" + row.data.attachment.attachment_id:
                        raise SconeError("AWS blob metadata is invalid")
                return before, rows
        raise SconeError("AWS blob metadata remained busy")

    def intent(self, space: str, generation: str) -> _Row[_Intent] | None:
        return self.read("I#" + space, generation, _Intent)

    def put(self, space: str, data: bytes, attachment: Attachment) -> Attachment:
        identifier = attachment.attachment_id
        upload: _Row[_Intent] | None = None
        candidate: _Root | None = None
        for _ in range(RETRIES):
            control = self.control(space)
            current = self.active(control)
            hold = self.read("S#" + space, "H#" + identifier, _Hold)
            root = self.read("B#" + identifier, "R", _Root)
            if hold is not None:
                if root is None or hold.data.generation != root.data.generation:
                    raise SconeError("AWS blob ownership is inconsistent")
                if upload is not None:
                    self.dispose_upload(space, upload)
                return hold.data.attachment
            if current.count >= MAX_HOLDS:
                raise SconeError("AWS blob space exceeds capacity")
            if root is None and candidate is None:
                generation = uuid4().hex
                key = self.store.prefix + identifier + "/" + generation
                value = _Intent(kind="upload", attachment_id=identifier, generation=generation, key=key)
                if not self.transaction([self.change("I#" + space, generation, None, value)]):
                    continue
                upload = self.intent(space, generation)
                if upload is None:
                    raise SconeError("AWS blob upload intent is missing")
                result = self.call(self.s3, "put_object", Bucket=self.store.bucket, Key=key, Body=data,
                                   ContentType=attachment.media_type, IfNoneMatch="*")
                version = result.get("VersionId")
                if version is not None:
                    version = _text(version)
                candidate = _Root(attachment_id=identifier, generation=generation, key=key,
                                  version_id=version, size=len(data), holds=1)
                # Persist the version before publication, so a failed transaction
                # can be retried or recovered after a process restart.
                updated = value.model_copy(update={"version_id": version})
                if not self.transaction([self.change(upload.pk, upload.sk, upload, updated)]):
                    raise SconeError("AWS blob upload was fenced by recovery")
                upload = self.intent(space, generation)
                continue
            selected = root.data if root is not None else candidate
            if selected is None:
                raise SconeError("AWS blob upload is unavailable")
            if selected.size != attachment.bytes:
                raise SconeError("AWS blob digest metadata is inconsistent")
            changes = [self.change("S#" + space, "C", control, current.model_copy(update={"count": current.count + 1})),
                       self.change("S#" + space, "H#" + identifier, None, _Hold(attachment=attachment, generation=selected.generation)),
                       self.change("B#" + identifier, "R", root, selected.model_copy(update={"holds": selected.holds + (1 if root else 0)}))]
            if root is None:
                if upload is None or upload.data.kind != "upload":
                    raise SconeError("AWS blob upload was fenced by recovery")
                changes.append(self.change(upload.pk, upload.sk, upload, None))
            if self.transaction(changes):
                if root is not None and upload is not None:
                    self.dispose_upload(space, upload)
                return attachment
        raise SconeError("AWS blob metadata remained busy")

    def object_args(self, root: _Root | _Intent) -> dict[str, object]:
        if root.key != self.store.prefix + root.attachment_id + "/" + root.generation:
            raise SconeError("AWS blob object identity is invalid")
        args: dict[str, object] = {"Bucket": self.store.bucket, "Key": root.key}
        if root.version_id is not None:
            args["VersionId"] = root.version_id
        return args

    def get(self, space: str, identifier: str) -> tuple[Attachment, bytes]:
        hold = self.read("S#" + space, "H#" + identifier, _Hold)
        if hold is None:
            raise NotFound("attachment not found in space")
        root = self.read("B#" + identifier, "R", _Root)
        if root is None or hold.data.generation != root.data.generation:
            raise SconeError("AWS blob ownership is inconsistent")
        try:
            result = self.call(self.s3, "get_object", **self.object_args(root.data))
        except Exception as error:
            if _code(error) in {"NoSuchKey", "NoSuchVersion", "404"}:
                raise NotFound("attachment bytes are unavailable") from None
            raise
        body = result.get("Body")
        reader = cast(Callable[[int], object], getattr(body, "read"))
        closer = cast(Callable[[], object], getattr(body, "close"))
        try:
            data = reader(MAX_BYTES + 1)
        finally:
            closer()
        self.check()
        if (type(data) is not bytes or len(data) != root.data.size or len(data) != hold.data.attachment.bytes
                or hashlib.sha256(data).hexdigest() != identifier or hold.data.attachment.attachment_id != identifier):
            raise SconeError("AWS blob bytes failed integrity verification")
        current_hold = self.read(hold.pk, hold.sk, _Hold)
        current_root = self.read(root.pk, root.sk, _Root)
        if (current_hold is None or current_root is None
                or current_hold.data.attachment != hold.data.attachment
                or current_hold.data.generation != hold.data.generation
                or current_root.data.model_copy(update={"holds": root.data.holds}) != root.data):
            raise SconeError("AWS blob ownership changed during read")
        return hold.data.attachment, data

    def link(self, space: str, identifier: str, episode: int) -> None:
        self.get(space, identifier)
        for _ in range(RETRIES):
            control = self.control(space)
            current = self.active(control)
            hold = self.read("S#" + space, "H#" + identifier, _Hold)
            if hold is None:
                raise NotFound("attachment not found in space")
            if any(number == episode for number, _ in hold.data.links):
                return
            if len(hold.data.links) >= 1024:
                raise SconeError("AWS blob reference capacity exceeded")
            sequence = 1 if control is None else control.revision + 1
            value = hold.data.model_copy(update={"links": (*hold.data.links, (episode, sequence))})
            if self.transaction([self.change("S#" + space, "C", control, current),
                                 self.change(hold.pk, hold.sk, hold, value)]):
                return
        raise SconeError("AWS blob metadata remained busy")

    def for_episode(self, space: str, episode: int) -> list[Attachment]:
        _, rows = self.snapshot(space)
        ordered = [(sequence, row.data.attachment) for row in rows for number, sequence in row.data.links if number == episode]
        return [attachment for _, attachment in sorted(ordered, key=lambda item: item[0])]

    def released_by(self, space: str, episode: int) -> list[str]:
        _, rows = self.snapshot(space)
        ordered = [(sequence, row.data.attachment.attachment_id) for row in rows
                   for number, sequence in row.data.links if number == episode and len(row.data.links) == 1]
        return [identifier for _, identifier in sorted(ordered)]

    def preview(self, space: str) -> tuple[list[str], list[str]]:
        for _ in range(RETRIES):
            control, rows = self.snapshot(space)
            released: list[str] = []
            kept: list[str] = []
            for row in rows:
                root = self.read("B#" + row.data.attachment.attachment_id, "R", _Root)
                if root is None or root.data.generation != row.data.generation:
                    raise SconeError("AWS blob ownership is inconsistent")
                (released if root.data.holds == 1 else kept).append(row.data.attachment.attachment_id)
            if self.control(space) == control:
                return sorted(released), sorted(kept)
        raise SconeError("AWS blob metadata remained busy")

    def release(self, space: str, episode: int | None) -> tuple[list[str], list[str]]:
        mode: Literal["release", "unlink"] = "release" if episode is None else "unlink"
        for _ in range(RETRIES):
            control, rows = self.snapshot(space)
            current = control.data if control else _Control()
            if current.mode != "active":
                if current.mode != mode or current.episode != (episode or 0):
                    raise SconeError("AWS blob space has another unfinished release")
                break
            if episode is not None:
                rows = sorted((row for row in rows if any(number == episode for number, _ in row.data.links)),
                              key=lambda row: next(sequence for number, sequence in row.data.links if number == episode))
            pending = tuple(row.data.attachment.attachment_id for row in rows)
            if not pending:
                return [], []
            journal = _Control(mode=mode, count=current.count, operation=uuid4().hex,
                               episode=episode or 0, pending=pending)
            if self.transaction([self.change("S#" + space, "C", control, journal)]):
                break
        else:
            raise SconeError("AWS blob metadata remained busy")
        for _ in range(MAX_HOLDS * RETRIES + RETRIES):
            control = self.control(space)
            if control is None or control.data.mode != mode or control.data.episode != (episode or 0):
                raise SconeError("AWS blob release journal changed")
            journal = control.data
            if not journal.pending:
                for generation in journal.gc:
                    intent = self.intent(space, generation)
                    if intent is not None:
                        self.delete_intent(intent)
                finished = _Control(count=journal.count)
                if self.transaction([self.change(control.pk, control.sk, control, finished)]):
                    return list(journal.released), list(journal.kept)
                continue
            identifier = journal.pending[0]
            hold = self.read("S#" + space, "H#" + identifier, _Hold)
            if hold is None:
                raise SconeError("AWS blob release ownership is inconsistent")
            remaining = tuple(link for link in hold.data.links if link[0] != episode) if episode else ()
            updated = journal.model_copy(update={"pending": journal.pending[1:]})
            if remaining:
                changes = [self.change(hold.pk, hold.sk, hold, hold.data.model_copy(update={"links": remaining}))]
            else:
                root = self.read("B#" + identifier, "R", _Root)
                if root is None or root.data.generation != hold.data.generation:
                    raise SconeError("AWS blob release ownership is inconsistent")
                last = root.data.holds == 1
                updates: dict[str, object] = {"count": journal.count - 1}
                if episode is not None or last:
                    updates["released"] = (*journal.released, identifier)
                else:
                    updates["kept"] = (*journal.kept, identifier)
                changes = [self.change(hold.pk, hold.sk, hold, None),
                           self.change(root.pk, root.sk, root, None if last else root.data.model_copy(update={"holds": root.data.holds - 1}))]
                if last:
                    deletion = _Intent(kind="delete", attachment_id=identifier, generation=root.data.generation,
                                       key=root.data.key, version_id=root.data.version_id)
                    changes.append(self.change("I#" + space, deletion.generation, None, deletion))
                    updates["gc"] = (*journal.gc, deletion.generation)
                updated = updated.model_copy(update=updates)
            changes.append(self.change(control.pk, control.sk, control, updated))
            self.transaction(changes)
        raise SconeError("AWS blob release exceeded retry capacity")

    def delete_object(self, value: _Intent) -> None:
        current = self.read("B#" + value.attachment_id, "R", _Root)
        if current is not None and current.data.generation == value.generation:
            raise SconeError("AWS blob cleanup cannot delete a published generation")
        args = self.object_args(value)
        # An interrupted upload may not have persisted its returned VersionId.
        # HEAD discovers only that immutable generation, never another upload.
        if value.version_id is None:
            try:
                head = self.call(self.s3, "head_object", **args)
            except Exception as error:
                if _code(error) in {"NoSuchKey", "404"}:
                    return
                raise
            if head.get("VersionId") is not None:
                args["VersionId"] = _text(head["VersionId"])
        self.call(self.s3, "delete_object", **args)

    def delete_intent(self, intent: _Row[_Intent]) -> None:
        if intent.data.kind != "delete":
            raise SconeError("AWS blob cleanup intent is invalid")
        self.delete_object(intent.data)
        if not self.transaction([self.change(intent.pk, intent.sk, intent, None)]):
            raise SconeError("AWS blob cleanup acknowledgement failed")

    def dispose_upload(self, space: str, intent: _Row[_Intent]) -> None:
        current = self.intent(space, intent.sk)
        if current is None:
            return
        if current.data.kind == "upload":
            value = current.data.model_copy(update={"kind": "delete"})
            if not self.transaction([self.change(current.pk, current.sk, current, value)]):
                raise SconeError("AWS blob upload cleanup remained busy")
            current = self.intent(space, intent.sk)
        if current is not None and current.data.kind == "delete":
            self.delete_intent(current)

    def recover(self, space: str, after: str | None, limit: int) -> RecoveryPage:
        options: dict[str, object] = {"TableName": self.store.table, "ConsistentRead": True,
            "KeyConditionExpression": "pk = :pk", "ExpressionAttributeValues": {":pk": {"S": "I#" + space}},
            "Limit": limit}
        if after is not None:
            options["ExclusiveStartKey"] = self.key("I#" + space, after)
        result = self.call(self.ddb, "query", **options)
        items = result.get("Items", [])
        if not isinstance(items, list) or len(items) > limit:
            raise SconeError("AWS blob recovery page is invalid")
        intents = [self.row(item, _Intent) for item in items]
        cursor: str | None = None
        if result.get("LastEvaluatedKey"):
            key = _mapping(result["LastEvaluatedKey"])
            if _mapping(key["pk"]).get("S") != "I#" + space:
                raise SconeError("AWS blob recovery cursor is invalid")
            cursor = _text(_mapping(key["sk"])["S"])
            if re.fullmatch(r"[0-9a-f]{32}", cursor) is None or (after is not None and cursor <= after):
                raise SconeError("AWS blob recovery cursor is invalid")
        for intent in intents:
            if intent.pk != "I#" + space or intent.sk != intent.data.generation:
                raise SconeError("AWS blob recovery intent is invalid")
            if intent.data.kind == "delete":
                self.delete_intent(intent)
                continue
            if intent.data.kind == "upload":
                value = intent.data.model_copy(update={"kind": "abandoned"})
                if not self.transaction([self.change(intent.pk, intent.sk, intent, value)]):
                    continue
                refreshed = self.intent(space, intent.sk)
                if refreshed is None:
                    raise SconeError("AWS blob upload recovery lost its intent")
                intent = refreshed
            self.delete_object(intent.data)
        return RecoveryPage(visited=len(intents), next_cursor=cursor)
