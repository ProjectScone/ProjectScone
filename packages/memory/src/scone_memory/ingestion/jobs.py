"""Durable ingestion receipts and progress, independent of retrieval orchestration.

Searchable and consolidated remain separate milestones. Cancelling a receipt
stops expectations for pending work; it does not remove stored source records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Optional, Protocol, Sequence
from uuid import uuid4

from ..core.errors import InvalidInput, NotFound
from ..core.models import Added, IngestJob, JobItem
from ..core.ports import DocumentStore, NewJob
from ..core.validation import check_space
from .records import Record

RECORDS_JOBS = ("create_job", "update_job")
READS_JOBS = ("get_job", "list_jobs")
MAX_JOBS_PAGE = 100


class AvailableMethods(Protocol):
    def __call__(self, *methods: str) -> bool: ...


class RecordJob(Protocol):
    def __call__(self, space: str, added: Sequence[Added], *,
                 request_id: Optional[str] = None) -> Awaitable[IngestJob]: ...


@dataclass(frozen=True)
class JobRuntime:
    documents: DocumentStore
    clock: Callable[[], str]
    emit: Callable[[str, str, dict[str, object]], Awaitable[object]]
    living: Callable[[str], Awaitable[None]]
    able: AvailableMethods
    keeps_jobs: Callable[[], None]
    reads_jobs: Callable[[], None]
    remember_many: Callable[[str, Iterable[Record]], Awaitable[list[Added]]]
    record_job: RecordJob
    job: Callable[[str, str], Awaitable[IngestJob]]
    max_page: int


def able(documents: DocumentStore, *methods: str) -> bool:
    return all(callable(getattr(documents, name, None)) for name in methods)


def keeps_jobs(available: AvailableMethods, methods: tuple[str, ...]) -> None:
    if not available(*methods):
        raise InvalidInput("this document store does not record ingest jobs")


def reads_jobs(available: AvailableMethods, methods: tuple[str, ...]) -> None:
    if not available(*methods):
        raise InvalidInput("this document store does not read ingest jobs")


async def ingest_batch(runtime: JobRuntime, space: str, records: Iterable[Record], *,
                       request_id: Optional[str] = None) -> "IngestJob":
    """Ingest a batch and keep the receipt of what it became.

    Two receipts per record, never one: ``searchable_at`` is set here,
    because chunks and vectors land with the batch, and
    ``consolidated_at`` only when a model has read claims out of the
    record, which happens later and may never happen. A request id
    makes a retry the same job rather than a second one."""
    check_space(space)
    await runtime.living(space)
    runtime.keeps_jobs()
    if request_id:
        already = await runtime.documents.job_by_request(space, request_id)
        if already is not None:
            return already
    added = await runtime.remember_many(space, list(records))
    return await runtime.record_job(space, added, request_id=request_id)


async def record_job(runtime: JobRuntime, space: str, added: Sequence[Added], *,
                     request_id: Optional[str] = None) -> "IngestJob":
    """Keep the receipt for records that have just landed. Separate
    from ingesting them, because a caller may have written the batch
    its own way and still owes the person a receipt."""
    check_space(space)
    runtime.keeps_jobs()
    when = runtime.clock()
    items = tuple(
        JobItem(index=index, episode_id=one.episode_id, outcome=one.outcome,
                state="searchable", searchable_at=when)
        for index, one in enumerate(added)
    )
    job = await runtime.documents.create_job(NewJob(
        job_id=uuid4().hex, space=space, created_at=when, request_id=request_id, items=items))
    await runtime.emit(space, "ingest_job", {
        "job_id": job.job_id, "records": len(items), "request_id": request_id,
    })
    return job


async def job_for_request(runtime: JobRuntime, space: str, request_id: str) -> Optional["IngestJob"]:
    """The job this request already made, if it made one."""
    check_space(space)
    if not runtime.able("job_by_request"):
        return None
    return await runtime.documents.job_by_request(space, request_id)


async def job(runtime: JobRuntime, space: str, job_id: str) -> "IngestJob":
    """One batch's receipt."""
    check_space(space)
    runtime.reads_jobs()
    found = await runtime.documents.get_job(space, job_id)
    if found is None:
        raise NotFound(f"job {job_id!r} in {space!r}")
    return found


async def jobs(runtime: JobRuntime, space: str, limit: int = 20, before: Optional[str] = None) -> list["IngestJob"]:
    """Recent batches, newest first. ``before`` continues from the last
    job of a previous page; a cursor naming no job is refused rather
    than quietly returning the newest page again."""
    check_space(space)
    runtime.reads_jobs()
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= runtime.max_page:
        raise InvalidInput(f"limit must be an integer from 1 through {runtime.max_page}")
    if before is not None and await runtime.documents.get_job(space, before) is None:
        raise NotFound(f"job {before!r} in {space!r}")
    return await runtime.documents.list_jobs(space, limit, before)


async def cancel_job(runtime: JobRuntime, space: str, job_id: str) -> "IngestJob":
    """Stop expecting more of this batch. What has already been read
    stays read and what was searchable stays searchable: cancelling a
    job abandons the work still to come, it does not delete memory."""
    job = await runtime.job(space, job_id)
    if job.cancelled_at:
        raise InvalidInput(f"job {job_id!r} was already cancelled at {job.cancelled_at}")
    when = runtime.clock()
    stopped = job.model_copy(update={
        "cancelled_at": when,
        "items": [item if item.consolidated_at else item.model_copy(update={"state": "cancelled"})
                  for item in job.items],
    })
    await runtime.documents.update_job(stopped)
    await runtime.emit(space, "ingest_job", {"job_id": job_id, "cancelled": len(stopped.items)})
    return stopped


async def note_failed(runtime: JobRuntime, space: str, episode_id: int, error: str) -> int:
    """Record that reading this record failed, against the record it
    failed on rather than against the batch. The attempt is counted,
    so a retry that works still shows it took two goes."""
    check_space(space)
    if not callable(getattr(runtime.documents, "mark_failed", None)):
        return 0
    return await runtime.documents.mark_failed(space, episode_id, error[:500], runtime.clock())


async def note_consolidated(runtime: JobRuntime, space: str, episode_ids: Sequence[int]) -> int:
    """Record that these episodes have been read into claims. Marking
    the same episode twice moves nothing, so a re-run of the extractor
    does not rewrite a receipt that already stands."""
    check_space(space)
    if not callable(getattr(runtime.documents, "mark_consolidated", None)):
        return 0
    return await runtime.documents.mark_consolidated(space, list(episode_ids), runtime.clock())
