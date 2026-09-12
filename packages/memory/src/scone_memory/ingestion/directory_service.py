"""Host-configured local sync execution, with explicit recovery and ownership."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat

from ..agents.workflow import WorkflowError, _integer, _name, _private_file
from ..core.validation import check_space
from ..memory.engine import MemoryEngine
from .directory_run_types import TERMINAL_RESULTS, SyncRunRecord, SyncRunSpec
from .directory_runs import DirectoryRunStore, SyncOutcomePage
from .directory_sync import DirectorySync


@dataclass(frozen=True)
class DirectoryCollection:
    collection_id: str
    label: str
    sync: DirectorySync
    allow_delete_missing: bool = False

    def __post_init__(self) -> None:
        _name(self.collection_id)
        if not isinstance(self.label, str) or not 1 <= len(self.label) <= 160 or not self.label.strip():
            raise ValueError('bounded collection label required')
        if not isinstance(self.sync, DirectorySync) or type(self.allow_delete_missing) is not bool:
            raise ValueError('configured directory sync and boolean policy required')

    def configuration(self) -> str:
        sync = self.sync
        info = sync.scanner.root.stat()
        values = [str(sync.scanner.root), info.st_dev, info.st_ino, str(sync.journal.path),
                  sync.journal._binding.hex(), sync.parser_revision, self.allow_delete_missing,
                  sync.limits.model_dump(mode='json'), sync.scanner.limits.model_dump(mode='json'),
                  sorted(sync.scanner.extensions)]
        return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class _CollectionBinding:
    space: str
    configuration: str
    references: tuple[object, ...]


@dataclass(frozen=True)
class SyncCollectionInfo:
    collection_id: str
    label: str
    allow_delete_missing: bool
    configuration: str


@dataclass(frozen=True)
class SyncServiceStatus:
    record: SyncRunRecord
    status: str
    active_local: bool
    active_elsewhere: bool
    outcome_unknown: bool


@dataclass(frozen=True)
class SyncStatusPage:
    items: tuple[SyncServiceStatus, ...]
    next_after: str | None


class DirectorySyncService:
    """Own local workers, never the engine. Reads and reopen do not execute.

    Run and collection locks protect admission. SourceJournal still owns source
    transitions, including direct CLI callers. All cooperating service instances
    must share this private state directory. Resume rescans the current inventory.
    """
    def __init__(self, directory: str | Path, *, key: bytes, memory: MemoryEngine,
                 collections: tuple[DirectoryCollection, ...], max_active: int = 2,
                 max_runs: int = 4096, deadline_s: float = 300.0, max_attempts: int = 3) -> None:
        _integer(max_active, 1, 16)
        SyncRunSpec(collection_id='validation', configuration='0' * 64,
                    deadline_s=deadline_s, max_attempts=max_attempts)
        if type(key) is not bytes or len(key) != 32:
            raise WorkflowError('key_must_be_32_bytes')
        if not isinstance(collections, tuple) or not 1 <= len(collections) <= 256:
            raise ValueError('one to 256 configured collections required')
        configured: dict[tuple[str, str], DirectoryCollection] = {}
        roots: set[tuple[str, int, int]] = set()
        for collection in collections:
            if not isinstance(collection, DirectoryCollection) or collection.sync.memory is not memory:
                raise ValueError('collections must use the host memory engine')
            identity = (collection.sync.space, collection.collection_id)
            check_space(identity[0])
            info = collection.sync.scanner.root.stat()
            root = (identity[0], info.st_dev, info.st_ino)
            if identity in configured or root in roots:
                raise ValueError('duplicate collection or source root')
            roots.add(root)
            configured[identity] = collection
        target = Path(directory).absolute()
        target.mkdir(mode=0o700, exist_ok=True)
        info = target.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise WorkflowError('private_directory_required')
        self._directory, self._key, self._memory = target, key, memory
        self._collections, self._maximum = configured, max_active
        self._bindings = {id(value): _CollectionBinding(scope, value.configuration(),
            (value.sync.memory, value.sync.parser, value.sync.scanner, value.sync.journal))
            for (scope, _), value in configured.items()}
        self._deadline, self._attempts = deadline_s, max_attempts
        self._runs = DirectoryRunStore(target / 'runs.sqlite', key=key, max_runs=max_runs)
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._owners: dict[tuple[str, str], tuple[int, int]] = {}
        self._closing = self._closed = False
        self._shutdown: asyncio.Task[None] | None = None

    def require_host(self, memory: MemoryEngine) -> None:
        if memory is not self._memory:
            raise ValueError('directory sync must use the host memory engine')

    async def _space(self, space: str) -> None:
        if self._closing or self._closed:
            raise WorkflowError('sync_service_closed')
        check_space(space)
        if await self._memory.space_deleted(space) is not None:
            raise WorkflowError('space_deleted')
        if self._closing or self._closed:
            raise WorkflowError('sync_service_closed')

    def _collection(self, space: str, collection_id: str) -> DirectoryCollection:
        _name(collection_id)
        value = self._collections.get((space, collection_id))
        if value is None:
            raise WorkflowError('sync_collection_not_found')
        return value

    def _configuration(self, collection: DirectoryCollection) -> str:
        binding = self._bindings[id(collection)]
        sync = collection.sync
        references = (sync.memory, sync.parser, sync.scanner, sync.journal)
        if (sync.space != binding.space
                or any(current is not saved for current, saved in zip(references, binding.references))
                or collection.configuration() != binding.configuration):
            raise WorkflowError('sync_configuration_changed')
        return binding.configuration

    def _execution(self, collection: DirectoryCollection) -> DirectorySync:
        # Pin mutable coordinator fields while sharing the host engine/parser.
        # Fresh journal/scanner objects retain the same durable identity and locks.
        sync = collection.sync
        if sync.journal._directory_fd is not None:
            raise WorkflowError('sync_busy')
        execution = copy(sync)
        execution.scanner = copy(sync.scanner)
        execution.journal = copy(sync.journal)
        execution.journal._collection = None
        return execution

    def _claim(self, kind: str, space: str, identity: str) -> int:
        token = hmac.new(self._key, json.dumps([kind, space, identity]).encode(), hashlib.sha256).hexdigest()
        descriptor = _private_file(self._directory / (token + '.owner'))
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return descriptor
        except BlockingIOError:
            os.close(descriptor)
            raise WorkflowError('sync_busy') from None
        except BaseException:
            os.close(descriptor)
            raise

    def _status(self, record: SyncRunRecord, *, owned_idle: bool = False) -> SyncServiceStatus:
        local = (record.space, record.run_id) in self._tasks
        elsewhere = False
        if not local and not owned_idle and record.status not in TERMINAL_RESULTS:
            try:
                descriptor = self._claim('run', record.space, record.run_id)
            except WorkflowError as error:
                if error.code != 'sync_busy':
                    raise
                elsewhere = True
            else:
                os.close(descriptor)
        status: str = record.status
        if not local and not elsewhere:
            if status == 'running' or (status == 'failed' and record.error_code == 'sync_interrupted'):
                status = 'interrupted'
            if record.cancel_requested_at is not None:
                status = 'cancelled'
        unknown = bool(record.attempt) and not local and not elsewhere and record.status not in TERMINAL_RESULTS
        return SyncServiceStatus(record, status, local, elsewhere, unknown)

    async def catalog(self, space: str) -> tuple[SyncCollectionInfo, ...]:
        await self._space(space)
        return tuple(SyncCollectionInfo(value.collection_id, value.label, value.allow_delete_missing,
                                         self._configuration(value))
                     for (scope, _), value in sorted(self._collections.items()) if scope == space)

    async def request(self, space: str, run_id: str) -> SyncRunRecord | None:
        await self._space(space)
        return self._runs.get(space, run_id)

    async def status(self, space: str, run_id: str) -> SyncServiceStatus | None:
        record = await self.request(space, run_id)
        return self._status(record) if record else None

    async def list(self, space: str, *, limit: int = 20, after: str | None = None) -> SyncStatusPage:
        await self._space(space)
        page = self._runs.list(space, limit=limit, after=after)
        return SyncStatusPage(tuple(self._status(record) for record in page.items), page.next_after)

    async def result(self, space: str, run_id: str, *, limit: int = 20,
                     after: int | None = None) -> SyncOutcomePage:
        await self._space(space)
        return self._runs.outcomes(space, run_id, limit=limit, after=after)

    async def start(self, space: str, run_id: str, *, collection_id: str, delete_missing: bool = False,
                    expected_configuration: str | None = None,
                    admission_guard: Callable[[], None] | None = None) -> SyncServiceStatus:
        await self._space(space)
        collection = self._collection(space, collection_id)
        spec = SyncRunSpec(collection_id=collection_id, configuration=self._configuration(collection),
            delete_missing=delete_missing, deadline_s=self._deadline, max_attempts=self._attempts)
        if expected_configuration is not None and expected_configuration != spec.configuration:
            raise WorkflowError('sync_configuration_changed')
        if delete_missing and not collection.allow_delete_missing:
            raise WorkflowError('sync_delete_forbidden')
        prior = self._runs.get(space, run_id)
        if prior is not None:
            if prior.spec != spec:
                raise WorkflowError('sync_request_conflict')
            return self._status(prior)
        return self._admit(space, run_id, collection, spec, None, admission_guard)

    def _admit(self, space: str, run_id: str, collection: DirectoryCollection, spec: SyncRunSpec,
               expected_revision: int | None, guard: Callable[[], None] | None) -> SyncServiceStatus:
        identity = (space, run_id)
        if identity in self._tasks or len(self._tasks) >= self._maximum:
            raise WorkflowError('sync_busy')
        run_lock = self._claim('run', space, run_id)
        collection_lock: int | None = None
        try:
            collection_lock = self._claim('collection', space, str(collection.sync.scanner.root))
            if guard is not None:
                guard()
            if self._configuration(collection) != spec.configuration:
                raise WorkflowError('sync_configuration_changed')
            if expected_revision is None:
                record = self._runs.register(space, run_id, spec)
                if record.attempt or record.cancel_requested_at is not None:
                    return self._status(record, owned_idle=True)
                expected_revision = record.revision
                resume = False
            else:
                resume = True
            execution = self._execution(collection)
            current = self._runs.start_attempt(space, run_id, expected_revision=expected_revision, resume=resume)
            task = asyncio.create_task(self._execute(current, collection, execution))
            self._tasks[identity] = task
            self._owners[identity] = (run_lock, collection_lock)
            task.add_done_callback(lambda finished: self._finished(identity, finished))
            return self._status(current)
        finally:
            if identity not in self._owners:
                os.close(run_lock)
                if collection_lock is not None:
                    os.close(collection_lock)

    async def resume(self, space: str, run_id: str, *, expected_revision: int,
                     admission_guard: Callable[[], None] | None = None) -> SyncServiceStatus:
        _integer(expected_revision, 0, 2**31 - 1)
        record = await self.request(space, run_id)
        if record is None:
            raise WorkflowError('sync_not_found')
        if record.revision != expected_revision:
            raise WorkflowError('sync_request_conflict')
        collection = self._collection(space, record.spec.collection_id)
        if self._configuration(collection) != record.spec.configuration:
            raise WorkflowError('sync_configuration_changed')
        return self._admit(space, run_id, collection, record.spec, expected_revision, admission_guard)

    async def cancel(self, space: str, run_id: str, *, expected_revision: int,
                     admission_guard: Callable[[], None] | None = None) -> SyncServiceStatus:
        _integer(expected_revision, 0, 2**31 - 1)
        record = await self.request(space, run_id)
        if record is None:
            raise WorkflowError('sync_not_found')
        identity = (space, run_id)
        descriptor: int | None = None
        if identity not in self._tasks:
            try:
                descriptor = self._claim('run', space, run_id)
            except WorkflowError as error:
                if error.code == 'sync_busy':
                    raise WorkflowError('sync_owned_elsewhere') from None
                raise
        try:
            if admission_guard is not None:
                admission_guard()
            task = self._tasks.get(identity)
            try:
                record = self._runs.request_cancel(space, run_id, expected_revision=expected_revision)
            except Exception as error:
                refused = isinstance(error, WorkflowError) and error.code in {
                    'sync_request_conflict', 'sync_result_terminal', 'sync_not_found'}
                if task is not None and not refused:
                    task.cancel()
                raise
            if task is not None:
                task.cancel()
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return self._status(record)

    async def _execute(self, record: SyncRunRecord, collection: DirectoryCollection,
                       execution: DirectorySync) -> None:
        try:
            async with asyncio.timeout(record.spec.deadline_s):
                await self._space(record.space)
                if self._configuration(collection) != record.spec.configuration:
                    raise WorkflowError('sync_configuration_changed')
                result = await execution.synchronize(delete_missing=record.spec.delete_missing)
                await self._space(record.space)
                if self._configuration(collection) != record.spec.configuration:
                    raise WorkflowError('sync_configuration_changed')
                self._runs.finish(record.space, record.run_id, expected_revision=record.revision, result=result)
        except asyncio.CancelledError:
            self._fail(record.space, record.run_id, 'sync_interrupted')
        except TimeoutError:
            self._fail(record.space, record.run_id, 'sync_deadline')
        except WorkflowError as error:
            code = 'sync_interrupted' if error.code == 'sync_service_closed' else error.code
            self._fail(record.space, record.run_id, code)
        except Exception:
            self._fail(record.space, record.run_id, 'sync_execution_failed')

    def _fail(self, space: str, run_id: str, code: str) -> None:
        record = self._runs.get(space, run_id)
        if record is not None and record.status == 'running':
            self._runs.fail(space, run_id, expected_revision=record.revision, error_code=code)

    def _finished(self, identity: tuple[str, str], task: asyncio.Task[None]) -> None:
        try:
            if not task.cancelled():
                task.exception()
            # Cancellation before the coroutine starts never enters its finally.
            self._fail(*identity, 'sync_interrupted')
        except Exception:
            # An unpersistable outcome remains running on disk and is reported as
            # interrupted after ownership ends. Never manufacture successful rows.
            pass
        finally:
            for descriptor in self._owners.pop(identity):
                os.close(descriptor)
            self._tasks.pop(identity)

    def close_idle(self) -> None:
        if self._tasks or self._owners or (self._shutdown is not None and not self._shutdown.done()):
            raise WorkflowError('sync_busy')
        if not self._closed:
            self._closing = True
            self._runs.close()
            self._closed = True

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._shutdown is None:
            self._closing = True
            self._shutdown = asyncio.create_task(self._drain())
        interrupted = False
        while not self._shutdown.done():
            try:
                await asyncio.shield(self._shutdown)
            except asyncio.CancelledError:
                interrupted = True
        self._shutdown.result()
        if interrupted:
            raise asyncio.CancelledError

    async def _drain(self) -> None:
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._runs.close()
        self._closed = True
