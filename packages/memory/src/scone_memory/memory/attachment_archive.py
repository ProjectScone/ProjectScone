"""Bounded, verified attachment transfer for the opt-in archive profile."""
from __future__ import annotations

import base64
import binascii
import copy
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass
import re

from ..backends.blobs import BlobStore, digest_of
from ..core.errors import InvalidInput, NotFound
from ..core.models import Attachment, Episode, DEPENDENCY_KINDS, LINK_KINDS
from ..core.ports import DocumentStore, NewEpisode
from ..ingestion.batch import validated_record
from ..ingestion.records import Record, RetainedVideoRecord

PROFILE = 'scone.archive/2'
MAX_ROWS = 100_000
MAX_ATTACHMENTS = 10_000
MAX_TOTAL_BYTES = 256 * 1024 * 1024
_BLOB_FIELDS = frozenset({'type', 'attachment_id', 'media_type', 'bytes', 'filename', 'data_base64'})
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')


def bounded_rows(records: Iterable[Mapping]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        if len(rows) >= MAX_ROWS:
            raise InvalidInput('attachment archive exceeds its record limit')
        if not isinstance(record, Mapping):
            raise InvalidInput('attachment archive records must be objects')
        try:
            rows.append(copy.deepcopy(dict(record)))
        except RecursionError as error:
            raise InvalidInput("attachment archive nesting exceeds its limit") from error
    return rows


def _identity(value: object) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise InvalidInput('attachment identity must be a lowercase SHA-256 digest')
    return value


def _episode_id(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 2**63 - 1:
        raise InvalidInput('attachment archive episode_id must be a positive int64')
    return value


def _blob(row: Mapping, max_bytes: int, media_types: Sequence[str]) -> tuple[Attachment, bytes]:
    if set(row) != _BLOB_FIELDS:
        raise InvalidInput('attachment archive has missing or unknown attachment fields')
    identity = _identity(row['attachment_id'])
    size = row['bytes']
    if type(size) is not int or not 1 <= size <= max_bytes:
        raise InvalidInput('attachment archive exceeds its per-attachment byte limit')
    media_type, filename, encoded = row['media_type'], row['filename'], row['data_base64']
    if not isinstance(media_type, str) or media_type not in media_types:
        raise InvalidInput('attachment archive media_type is unsupported')
    if filename is not None and (not isinstance(filename, str) or len(filename) > 4096):
        raise InvalidInput('attachment archive filename is invalid')
    try:
        if filename is not None:
            filename.encode('utf-8')
        if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
            raise InvalidInput('attachment archive base64 length does not match its bytes')
        data = base64.b64decode(encoded, validate=True)
    except (UnicodeError, ValueError, binascii.Error) as error:
        raise InvalidInput('attachment archive base64 or filename is invalid') from error
    if (len(data) != size or digest_of(data) != identity
            or base64.b64encode(data).decode('ascii') != encoded):
        raise InvalidInput('attachment archive bytes do not match their identity')
    return Attachment(attachment_id=identity, media_type=media_type, bytes=size, filename=filename), data


def _episode_signature(episode: Episode | NewEpisode) -> tuple[object, ...]:
    return (episode.content, episode.kind, episode.source, episode.created_at,
            tuple(episode.tags), dict(episode.metadata))


@dataclass(frozen=True)
class Transfer:
    records: list[dict]
    blobs: dict[str, tuple[Attachment, bytes]]
    links: dict[int, tuple[str, ...]]

    async def preflight_episodes(self, documents: DocumentStore, space: str,
                                 records: Sequence[Record], when: str) -> None:
        seen: dict[str, tuple[object, ...]] = {}
        for record in records:
            # Only validate the record shape here. Actual visual evidence is
            # checked by normal ingestion after its bytes have been staged.
            new = validated_record(space, record, when, verified_visual=isinstance(record, RetainedVideoRecord))
            signature = _episode_signature(new)
            existing = await documents.episode_by_hash(space, new.content_hash)
            if ((existing is not None and _episode_signature(existing) != signature)
                    or (new.content_hash in seen and seen[new.content_hash] != signature)):
                raise InvalidInput('attachment archive episode identity conflicts with retained source')
            seen[new.content_hash] = signature

    async def stage(self, store: BlobStore, space: str, active_ids: Sequence[int | None]) -> int:
        """Preflight target collisions, then store only evidence for accepted episodes."""
        wanted = {identity for episode_id in active_ids if episode_id is not None
                  for identity in self.links[episode_id]}
        absent: list[str] = []
        for identity in sorted(wanted):
            expected = self.blobs[identity]
            try:
                current = await store.get(space, identity)
            except NotFound:
                absent.append(identity)
                continue
            if current != expected:
                raise InvalidInput('target attachment conflicts with archive metadata or bytes')
        for identity in absent:
            attachment, data = self.blobs[identity]
            stored = await store.put(space, data, attachment.media_type, attachment.filename)
            if stored != attachment or await store.get(space, identity) != (attachment, data):
                raise InvalidInput('stored attachment does not match the archive')
        return len(wanted)

    async def link(self, store: BlobStore, space: str, old_id: int, new_id: int) -> int:
        for identity in self.links[old_id]:
            if await store.get(space, identity) != self.blobs[identity]:
                raise InvalidInput('attachment changed before archive episode linking')
            await store.link(space, identity, new_id)
        return len(self.links[old_id])


def prepare(rows: list[dict], max_bytes: int, media_types: Sequence[str]) -> Transfer:
    """Validate the complete evidence graph before any target write."""
    from .archive import ARCHIVE_CARRIES, ARCHIVE_PROFILE, KNOWN_FIELDS, _check_fields

    header = rows[0]
    if (header.get('type') != 'archive' or header.get('profile') != PROFILE
            or not isinstance(header.get('space'), str)
            or header.get('carries') != [*ARCHIVE_CARRIES, 'attachments']):
        raise InvalidInput('attachment archive requires a complete profile header')
    _check_fields('archive', header)
    records = [{**header, 'profile': ARCHIVE_PROFILE, 'carries': list(ARCHIVE_CARRIES)}]
    blobs: dict[str, tuple[Attachment, bytes]] = {}
    links: dict[int, tuple[str, ...]] = {}
    total = 0
    for row in rows[1:]:
        kind = row.get('type')
        if kind == 'attachment':
            size = row.get('bytes')
            if type(size) is not int or size < 1 or total + size > MAX_TOTAL_BYTES:
                raise InvalidInput('attachment archive exceeds its total byte limit')
            if len(blobs) >= MAX_ATTACHMENTS:
                raise InvalidInput('attachment archive exceeds its attachment count limit')
            attachment, data = _blob(row, max_bytes, media_types)
            if attachment.attachment_id in blobs:
                raise InvalidInput('attachment archive repeats an attachment identity')
            blobs[attachment.attachment_id] = (attachment, data)
            total += size
            continue
        if not isinstance(kind, str) or kind not in KNOWN_FIELDS or kind == 'archive':
            raise InvalidInput('attachment archive has an unknown record or repeated header')
        clean = dict(row)
        if kind == 'episode':
            episode_id = _episode_id(row.get('episode_id'))
            if episode_id in links or row.get('space') != header['space']:
                raise InvalidInput('attachment archive repeats an episode or mixes source spaces')
            values = clean.pop('attachment_ids', None)
            if not isinstance(values, list):
                raise InvalidInput('attachment archive episode requires attachment_ids')
            identities = tuple(_identity(value) for value in values)
            if len(identities) != len(set(identities)):
                raise InvalidInput('attachment archive repeats an episode attachment link')
            links[episode_id] = identities
            source, metadata = row.get('source'), row.get('metadata', {})
            if not isinstance(metadata, Mapping):
                raise InvalidInput('attachment archive episode metadata must be an object')
            referenced = [metadata[key] for key in ('document_original', 'document_manifest') if key in metadata]
            if isinstance(source, str) and source.startswith('attachment:'):
                referenced.append(source.removeprefix('attachment:'))
            if any(_identity(value) not in identities for value in referenced):
                raise InvalidInput('attachment archive omits linked document evidence')
        _check_fields(kind, clean)
        records.append(clean)
    wanted = {identity for identities in links.values() for identity in identities}
    if wanted != set(blobs):
        raise InvalidInput('attachment archive has dangling links or unreferenced attachments')
    _check_ledger_references(records, set(links), header["space"])
    return Transfer(records, blobs, links)


def _check_ledger_references(records: Sequence[Mapping], episodes: set[int], space: str) -> None:
    identities: dict[str, set[int]] = {'fact': set(), 'fact_link': set(), 'affirmation': set()}
    keys = {'fact': 'fact_id', 'fact_link': 'link_id', 'affirmation': 'affirmation_id'}
    for record in records:
        kind = record['type']
        if kind not in identities:
            continue
        if record.get('space', space) != space:
            raise InvalidInput('attachment archive ledger mixes source spaces')
        identity = _episode_id(record.get(keys[kind]))
        if identity in identities[kind]:
            raise InvalidInput('attachment archive repeats a ledger identity')
        identities[kind].add(identity)

    def reference(value: object, known: set[int]) -> None:
        if _episode_id(value) not in known:
            raise InvalidInput('attachment archive has a dangling ledger reference')

    for record in records:
        kind = record['type']
        if kind not in identities:
            continue
        if record.get('source_episode_id') is not None:
            reference(record['source_episode_id'], episodes)
        if kind == 'fact' and record.get('superseded_by') is not None:
            reference(record['superseded_by'], identities['fact'])
        if kind == 'fact_link':
            for key in ('from_fact', 'to_fact'):
                reference(record.get(key), identities['fact'])
            if record.get('kind') not in LINK_KINDS or record['from_fact'] == record['to_fact']:
                raise InvalidInput('attachment archive fact link is invalid')
        if kind == 'affirmation':
            reference(record.get('fact_id'), identities['fact'])
            premises = record.get('links', [])
            if not isinstance(premises, (list, tuple)):
                raise InvalidInput('attachment archive affirmation links must be a list')
            for premise in premises:
                if (not isinstance(premise, (list, tuple)) or len(premise) != 2
                        or premise[0] not in DEPENDENCY_KINDS):
                    raise InvalidInput('attachment archive affirmation link is invalid')
                reference(premise[1], identities['fact'])


async def export_records(records: AsyncIterator[dict], store: BlobStore, documents: DocumentStore, space: str,
                         max_bytes: int, media_types: Sequence[str]) -> list[dict]:
    """Build and verify before yielding a header; this is not a database snapshot."""
    rows: list[dict] = []
    blobs: dict[str, dict[str, object]] = {}
    total = 0
    for_row_links: dict[int, list[Attachment]] = {}
    async for row in records:
        if len(rows) + len(blobs) >= MAX_ROWS:
            raise InvalidInput('attachment archive exceeds its record limit')
        if row['type'] == 'archive':
            row = {**row, 'profile': PROFILE, 'carries': [*row['carries'], 'attachments'],
                   'not_carried': {}}
        if row['type'] == 'episode':
            episode_id = _episode_id(row['episode_id'])
            attached = await store.for_episode(space, episode_id)
            for_row_links[episode_id] = attached
            row = {**row, 'attachment_ids': [attachment.attachment_id for attachment in attached]}
            for attachment in attached:
                if attachment.attachment_id in blobs:
                    continue
                if (len(blobs) >= MAX_ATTACHMENTS or attachment.bytes > max_bytes
                        or total + attachment.bytes > MAX_TOTAL_BYTES):
                    raise InvalidInput('attachment archive exceeds its attachment byte or count limit')
                current, data = await store.get(space, attachment.attachment_id)
                if current != attachment or len(data) != attachment.bytes or digest_of(data) != attachment.attachment_id:
                    raise InvalidInput('source attachment does not match its retained identity')
                blobs[attachment.attachment_id] = {'type': 'attachment', **attachment.model_dump(),
                                                  'data_base64': base64.b64encode(data).decode('ascii')}
                total += len(data)
        rows.append(row)
    for row in rows:
        if row['type'] != 'episode':
            continue
        current_episode = await documents.episode_by_hash(space, row['content_hash'])
        expected = (row['content'], row['kind'], row['source'], row['created_at'],
                    tuple(row['tags']), row['metadata'])
        if (current_episode is None or current_episode.episode_id != row['episode_id']
                or _episode_signature(current_episode) != expected):
            raise InvalidInput('source episode changed during attachment export')
    for episode_id, attached in for_row_links.items():
        if await store.for_episode(space, episode_id) != attached:
            raise InvalidInput('source attachment links changed during export')
    if await store.linked(space) != set(blobs):
        raise InvalidInput("source attachment links are missing evidence or changed during export")
    left = set(await store.held(space)) - set(blobs)
    rows[0]['not_carried'] = {'unlinked_attachments': len(left)} if left else {}
    rows.extend(blobs.values())
    prepare(bounded_rows(rows), max_bytes, media_types)
    return rows
