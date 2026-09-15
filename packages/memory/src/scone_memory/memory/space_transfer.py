"""Move retained space evidence through explicit copy, verification and closure.

Independent storage observations do not provide a distributed transaction. Source
and destination writers must be quiescent throughout a whole-space movement.
"""
from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from ..core.errors import InvalidInput, NotFound
from ..core.models import Attachment
from ..ingestion.batch import validated_record
from ..ingestion.records import Record, RetainedVideoRecord, content_hash
from . import archive, attachment_archive

if TYPE_CHECKING:
    from .engine import MemoryEngine


@dataclass(frozen=True)
class Snapshot:
    rows: list[dict]
    transfer: attachment_archive.Transfer
    unlinked: dict[str, tuple[Attachment, bytes]]
    source_rows: list[dict]
    forgotten_source_references: int

    @property
    def blobs(self) -> dict[str, tuple[Attachment, bytes]]:
        return {**self.transfer.blobs, **self.unlinked}

    def identities(self) -> dict[tuple[str, str, str], dict]:
        """Storage enumeration order and export timestamps are not source changes."""
        keys = {'episode': 'episode_id', 'fact': 'fact_id', 'fact_link': 'link_id',
                'affirmation': 'affirmation_id', 'attachment': 'attachment_id'}
        return {(group, row['type'], str(row.get(keys.get(row['type'], ''), ''))):
                {key: value for key, value in row.items() if row['type'] != 'archive' or key != 'wrote_at'}
                for group, records in (('source', self.source_rows), ('transfer', self.rows))
                for row in records}



async def capture(memory: MemoryEngine, space: str) -> Snapshot:
    from .engine import ATTACHMENT_TYPES

    raw: list[dict] = []
    async for row in archive.export_records(memory.documents, space, wrote_at=memory.clock()):
        if len(raw) >= attachment_archive.MAX_ROWS:
            raise InvalidInput('space transfer exceeds its archive record limit')
        raw.append(row)
    source_rows = attachment_archive.bounded_rows(raw)
    normalized, forgotten = await retained_claims(memory, space, source_rows)
    async def records() -> AsyncIterator[dict]:
        for row in normalized:
            yield row
    rows = await attachment_archive.export_records(records(), memory.blobs, memory.documents, space,
                                                    memory.max_attachment_bytes, ATTACHMENT_TYPES)
    transfer = attachment_archive.prepare(rows, memory.max_attachment_bytes, ATTACHMENT_TYPES)
    held = set(await memory.blobs.held(space))
    if not set(transfer.blobs) <= held:
        raise InvalidInput('source attachment holds changed during space transfer')
    if len(held) > attachment_archive.MAX_ATTACHMENTS:
        raise InvalidInput('space transfer exceeds its attachment count limit')
    total = sum(attachment.bytes for attachment, _ in transfer.blobs.values())
    unlinked: dict[str, tuple[Attachment, bytes]] = {}
    for identity in sorted(held - set(transfer.blobs)):
        attachment, data = await memory.blobs.get(space, identity)
        if not 1 <= len(data) <= memory.max_attachment_bytes or attachment.bytes != len(data):
            raise InvalidInput('space transfer exceeds its attachment byte limit or has invalid metadata')
        total += len(data)
        if total > attachment_archive.MAX_TOTAL_BYTES:
            raise InvalidInput('space transfer exceeds its total attachment byte limit')
        row = {'type': 'attachment', **attachment.model_dump(),
               'data_base64': base64.b64encode(data).decode('ascii')}
        checked = attachment_archive._blob(row, memory.max_attachment_bytes, ATTACHMENT_TYPES)
        if checked[0].attachment_id != identity:
            raise InvalidInput('source attachment identity does not match its hold')
        unlinked[identity] = checked
    if len(rows) + len(unlinked) > attachment_archive.MAX_ROWS:
        raise InvalidInput('space transfer exceeds its archive record limit')
    if held != set(await memory.blobs.held(space)):
        raise InvalidInput('source attachment holds changed during space transfer')
    return Snapshot(rows, transfer, unlinked, source_rows, forgotten)


async def retained_claims(memory: MemoryEngine, space: str,
                          records: list[dict]) -> tuple[list[dict], int]:
    """Preserve legacy claim retention without inventing a replacement source."""
    episodes = {row['episode_id'] for row in records if row['type'] == 'episode'}
    normalized: list[dict] = []
    forgotten = 0
    for row in records:
        source = row.get('source_episode_id')
        if row['type'] in ('fact', 'fact_link', 'affirmation') and source is not None and source not in episodes:
            if type(source) is not int or await memory.documents.tombstone(space, source) is None:
                raise InvalidInput('source ledger has an unknown episode reference')
            row = {**row, 'source_episode_id': None}
            forgotten += 1
        normalized.append(row)
    return normalized, forgotten


def episode_record(row: dict, space: str) -> Record:
    record = archive._rederived(Record.from_dict(row), row.get('space'), space)
    if record.kind == 'file' and record.content == '':
        return RetainedVideoRecord(**asdict(record))
    return record


async def selected_episodes(memory: MemoryEngine, snapshot: Snapshot, into: str) -> tuple[dict[int, Record], int]:
    """The episodes the import will store, and how many it will leave because
    their own schedule has come -- judged in the order the import judges."""
    selected: dict[int, Record] = {}
    overdue = 0
    for row in snapshot.transfer.records:
        if row['type'] != 'episode':
            continue
        record = episode_record(row, into)
        digest = record.content_hash or content_hash(into, record.content, record.dedup_key)
        if await memory.documents.tombstone_by_hash(into, digest) is not None:
            continue
        if archive.overdue(record, memory.clock()):
            overdue += 1
            continue
        selected[row['episode_id']] = record
    return selected, overdue


def receipt(space: str, into: str, snapshot: Snapshot, selected: dict[int, Record], overdue: int = 0) -> archive.MergeReceipt:
    requested = {identity for episode_id in selected for identity in snapshot.transfer.links[episode_id]}
    skipped = set(snapshot.transfer.blobs) - requested
    moved = requested | set(snapshot.unlinked)
    blobs = snapshot.blobs
    return archive.MergeReceipt(
        space=space, into=into, episodes=len(selected),
        facts=sum(row['type'] == 'fact' for row in snapshot.rows),
        attachments=len(moved), attachment_bytes=sum(blobs[identity][0].bytes for identity in moved),
        unlinked_attachments=len(snapshot.unlinked), tombstoned=len(snapshot.transfer.links) - len(selected) - overdue,
        past_forget_after=overdue,
        attachments_skipped=len(skipped), forgotten_source_references=snapshot.forgotten_source_references)


async def verify_destination(memory: MemoryEngine, snapshot: Snapshot, into: str,
                             selected: dict[int, Record]) -> None:
    wanted = dict(snapshot.unlinked)
    for old_id, record in selected.items():
        # Re-deriving a stored record's identity, not writing it: a schedule
        # that came due since the import is not a reason to refuse the check.
        expected = validated_record(into, record, memory.clock(),
                                    verified_visual=isinstance(record, RetainedVideoRecord), past_schedule=True)
        current = await memory.documents.episode_by_hash(into, expected.content_hash)
        if current is None or attachment_archive._episode_signature(current) != attachment_archive._episode_signature(expected):
            raise InvalidInput('destination source evidence changed during space transfer')
        attached = {value.attachment_id: value for value in await memory.blobs.for_episode(into, current.episode_id)}
        for identity in snapshot.transfer.links[old_id]:
            evidence = snapshot.transfer.blobs[identity]
            if attached.get(identity) != evidence[0]:
                raise InvalidInput('destination attachment evidence is missing or changed')
            wanted[identity] = evidence
    for identity, expected_blob in wanted.items():
        try:
            current_blob = await memory.blobs.get(into, identity)
        except NotFound as error:
            raise InvalidInput('destination attachment evidence is missing') from error
        if current_blob != expected_blob:
            raise InvalidInput('destination attachment evidence changed during space transfer')


async def merge(memory: MemoryEngine, space: str, into: str, *, preview: bool,
                authorize: Callable[[], None] | None = None) -> archive.MergeReceipt:
    def check() -> None:
        if authorize is not None:
            authorize()

    check()
    snapshot = await capture(memory, space)
    check()
    selected, overdue = await selected_episodes(memory, snapshot, into)
    result = receipt(space, into, snapshot, selected, overdue)
    check()
    if preview:
        return result
    await attachment_archive.preflight_blobs(memory.blobs, into, snapshot.unlinked)
    check()
    imported = await memory.import_records(into, snapshot.rows)
    check()
    if imported.tombstoned != result.tombstoned:
        raise InvalidInput('destination tombstone policy changed during space transfer')
    if imported.past_forget_after != result.past_forget_after:
        raise InvalidInput('a source came due to be forgotten during space transfer; source remains open, merge again')
    await attachment_archive.stage_blobs(memory.blobs, into, snapshot.unlinked)
    check()
    await verify_destination(memory, snapshot, into, selected)
    check()
    current = await capture(memory, space)
    check()
    if snapshot.identities() != current.identities() or snapshot.blobs != current.blobs:
        raise InvalidInput('source changed during space transfer; source remains open')
    await verify_destination(memory, snapshot, into, selected)
    await memory._living(into)
    check()
    await memory.delete_space(space)
    check()
    result.moved = True
    return result
