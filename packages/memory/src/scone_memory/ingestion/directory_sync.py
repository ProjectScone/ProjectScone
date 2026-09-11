"""Recoverable source replacement over existing memory and attachment stores."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from ..core.errors import Gone, InvalidInput, NotFound
from ..core.models import Episode
from ..memory.engine import MemoryEngine
from .document_source import DocumentSource, source_revision_key
from .files import (FILE_MEDIA_TYPES, DocumentManifest, digest, document_provenance,
                    encode_manifest, prepare_document, store_document)
from .formats.registry import BuiltinDocumentParser, DocumentParser, extension
from .formats.types import DocumentLimits
from .source_journal import SourceEntry, SourceJournal, SourceRevision, SourceState
from .source_scan import DirectoryScanner, ScanIssue, ScanLimits, ScannedFile


@dataclass(frozen=True)
class SourceReceipt:
    path: str
    status: Literal['added', 'updated', 'unchanged', 'deleted', 'absent', 'suppressed', 'failed']
    episode_id: int | None = None
    previous_episode_id: int | None = None
    code: str | None = None


@dataclass(frozen=True)
class DirectorySyncResult:
    collection_id: str
    complete: bool
    receipts: tuple[SourceReceipt, ...]
    issues: tuple[ScanIssue, ...]
    skipped: int


class DirectorySync:
    """One journal per collection. Serialize other writes to its managed sources.

    Replacements retain and verify the new source before retiring the previous
    episode. Stages are recoverable; the stores do not provide a global atomic
    swap. Extracted claims stand when source episodes are forgotten. An external
    forget suppresses the path; managed missing-file deletion permits its later
    return. Missing evidence or ownership mismatches fail closed.
    """

    def __init__(self, memory: MemoryEngine, root: str | Path, *, space: str,
                 journal: str | Path, key: bytes, store_id: str, parser_revision: str,
                 parser: DocumentParser | None = None, limits: DocumentLimits = DocumentLimits(),
                 scan_limits: ScanLimits = ScanLimits(), extensions: frozenset[str] | None = None):
        DocumentSource('0' * 32, 'validation.txt', parser_revision)
        if not isinstance(store_id, str) or not 1 <= len(store_id) <= 512:
            raise InvalidInput('directory synchronization requires a bounded stable store identity')
        self.scanner = DirectoryScanner(root, limits=scan_limits, extensions=extensions)
        # Changes to the catalog, root inode or supported suffixes require an
        # explicit new collection, never an apparent mass deletion in this one.
        info = self.scanner.root.stat()
        binding = json.dumps([store_id, info.st_dev, info.st_ino, sorted(self.scanner.extensions)], separators=(',', ':'))
        self.journal = SourceJournal(journal, root=self.scanner.root, space=space, store_id=digest(binding.encode()), key=key)
        if self.journal.path.is_relative_to(self.scanner.root):
            raise InvalidInput('directory journal must be outside the source root')
        try:
            self.limits = DocumentLimits.model_validate_json(limits.model_dump_json())
        except (ValidationError, AttributeError):
            raise InvalidInput('directory document limits are invalid') from None
        self.memory = memory
        self.space = space
        self.parser = parser or BuiltinDocumentParser()
        self.parser_revision = parser_revision

    def _source(self, state: SourceState, path: str, revision: SourceRevision) -> DocumentSource:
        return DocumentSource(state.collection_id, path, revision.parser_revision, revision.generation)

    async def _episode(self, state: SourceState, path: str, revision: SourceRevision,
                       *, provenance: bool = True) -> Episode:
        source = self._source(state, path, revision)
        key = source_revision_key(source, revision.original_sha256, revision.manifest_sha256)
        episode = await self.memory.episode_by_key(self.space, key)
        expected = source.metadata() | {'document_original': revision.original_sha256,
                                       'document_manifest': revision.manifest_sha256}
        if (episode.kind != 'file' or episode.source != 'attachment:' + revision.original_sha256
                or (revision.episode_id is not None and revision.episode_id != episode.episode_id)
                or any(episode.metadata.get(name) != value for name, value in expected.items())
                or episode.metadata.get('source_generation', '0') != str(revision.generation)):
            raise InvalidInput('managed source ownership does not match its journal')
        if provenance:
            evidence = await document_provenance(self.memory, self.space, episode.episode_id)
            if evidence.filename != path:
                raise InvalidInput('managed source provenance path does not match its journal')
            await self._index(episode)
        return episode

    async def _index(self, episode: Episode) -> None:
        if (self.space, episode.content_hash) in await self.memory.documents.inflight():
            raise InvalidInput('managed source indexing is unfinished; recover the engine before retrying')
        chunks = await self.memory.documents.chunks_of(self.space, episode.episode_id)
        encoded = episode.content.encode('utf-8')
        end = 0
        if not chunks:
            raise InvalidInput('managed source has no indexed chunks')
        for ordinal, chunk in enumerate(chunks):
            if (chunk.ordinal != ordinal or chunk.space != self.space or chunk.episode_id != episode.episode_id
                    or not 0 <= chunk.start < chunk.end <= len(encoded)
                    or encoded[chunk.start:chunk.end] != chunk.text.encode('utf-8')):
                raise InvalidInput('managed source chunks do not match its retained text')
        for chunk in sorted(chunks, key=lambda value: value.start):
            if chunk.start > end and encoded[end:chunk.start].decode('utf-8').strip():
                raise InvalidInput('managed source index is missing retained text')
            end = max(end, chunk.end)
        if encoded[end:].decode('utf-8').strip():
            raise InvalidInput('managed source index is missing retained text')

    def _save(self, state: SourceState, path: str, entry: SourceEntry) -> None:
        state.entries[path] = entry
        self.journal.save(state)

    async def _preflight(self, state: SourceState) -> None:
        for path, entry in state.entries.items():
            if entry.current is None or entry.state in ('absent', 'suppressed'):
                continue
            try:
                await self._episode(state, path, entry.current, provenance=False)
            except Gone:
                continue
            except NotFound:
                raise InvalidInput('directory journal references missing sources; verify the store identity and retained data') from None

    async def _suppress(self, state: SourceState, path: str, entry: SourceEntry) -> SourceReceipt:
        self._save(state, path, SourceEntry(state='suppress', current=entry.current, pending=entry.pending))
        for revision in (entry.current, entry.pending):
            if revision is None:
                continue
            try:
                episode = await self._episode(state, path, revision)
            except Gone:
                continue
            except NotFound:
                if revision.episode_id is not None:
                    raise
            else:
                await self.memory.forget(self.space, episode.episode_id)
        revision = entry.pending or entry.current
        if revision is None:
            raise InvalidInput('source suppression has no revision')
        self._save(state, path, SourceEntry(state='suppressed', current=revision))
        return SourceReceipt(path, 'suppressed', revision.episode_id)

    async def _finish(self, state: SourceState, path: str, entry: SourceEntry) -> SourceReceipt:
        if entry.state == 'suppress':
            return await self._suppress(state, path, entry)
        pending = entry.pending
        if pending is None:
            raise InvalidInput('managed source replacement has no pending revision')
        current = entry.current
        if entry.state == 'replace':
            if current is not None:
                try:
                    await self._episode(state, path, current)
                except Gone:
                    return await self._suppress(state, path, entry)
            try:
                await self._episode(state, path, pending, provenance=False)
            except Gone:
                return await self._suppress(state, path, entry)
            except NotFound:
                if pending.episode_id is not None:
                    raise InvalidInput('indexed pending source is missing') from None
            original, raw = await self.memory.attachment(self.space, pending.original_sha256)
            retained, encoded = await self.memory.attachment(self.space, pending.manifest_sha256)
            if (digest(raw) != pending.original_sha256 or digest(encoded) != pending.manifest_sha256
                    or retained.media_type != 'application/json'):
                raise InvalidInput('pending source evidence changed')
            manifest = DocumentManifest.model_validate_json(encoded)
            if manifest.filename != path or manifest.original_sha256 != pending.original_sha256:
                raise InvalidInput('pending manifest does not match its source')
            added = await store_document(self.memory, self.space, original, manifest,
                                         source=self._source(state, path, pending))
            pending = pending.model_copy(update={'episode_id': added.added.episode_id})
            await self._episode(state, path, pending)
            entry = SourceEntry(state='retire', current=current, pending=pending)
            self._save(state, path, entry)
        # A recorded retire intent permits an already-forgotten old episode on
        # retry. A missing record without its tombstone remains an error.
        try:
            await self._episode(state, path, pending)
        except Gone:
            return await self._suppress(state, path, entry)
        if current is not None:
            try:
                old = await self._episode(state, path, current)
            except Gone:
                pass
            else:
                await self.memory.forget(self.space, old.episode_id)
        self._save(state, path, SourceEntry(state='active', current=pending))
        return SourceReceipt(path, 'updated' if current is not None else 'added', pending.episode_id,
                             current.episode_id if current else None)

    async def _prepare(self, state: SourceState, item: ScannedFile, current: SourceRevision | None,
                       generation: int) -> SourceReceipt:
        data = await asyncio.to_thread(self.scanner.read, item)
        if len(data) > self.memory.max_attachment_bytes:
            raise InvalidInput('directory document exceeds its attachment limit')
        manifest = await prepare_document(data, item.path, parser=self.parser, limits=self.limits)
        encoded = encode_manifest(manifest)
        if len(encoded) > self.memory.max_attachment_bytes:
            raise InvalidInput('directory manifest exceeds its attachment limit')
        await asyncio.to_thread(self.scanner.read, item)
        pending = SourceRevision(original_sha256=digest(data), manifest_sha256=digest(encoded),
                                 parser_revision=self.parser_revision, generation=generation)
        entry = SourceEntry(state='replace', current=current, pending=pending)
        # Validate the bounded journal entry before uploads; persist the intent
        # after both exact inputs are retained and before indexing starts.
        checked = SourceState(collection_id=state.collection_id, entries=state.entries | {item.path: entry})
        original = await self.memory.attach(self.space, data, FILE_MEDIA_TYPES.get(extension(item.path), 'application/octet-stream'),
                                           filename=item.path)
        retained = await self.memory.attach(self.space, encoded, 'application/json', filename='document-provenance.json')
        if original.attachment_id != pending.original_sha256 or retained.attachment_id != pending.manifest_sha256:
            raise InvalidInput('retained directory source identity does not match its bytes')
        self._save(state, item.path, checked.entries[item.path])
        return await self._finish(state, item.path, entry)

    async def _file(self, state: SourceState, item: ScannedFile) -> SourceReceipt:
        entry = state.entries.get(item.path)
        recovered: SourceReceipt | None = None
        if entry is not None and entry.state in ('replace', 'retire', 'suppress'):
            recovered = await self._finish(state, item.path, entry)
            entry = state.entries[item.path]
        if entry is not None and entry.state == 'suppressed':
            return SourceReceipt(item.path, 'suppressed', entry.current.episode_id if entry.current else None)
        current = entry.current if entry else None
        generation = current.generation + 1 if current else 0
        if entry is not None and entry.state == 'absent':
            current = None
        elif current is not None:
            try:
                await self._episode(state, item.path, current)
            except Gone:
                if entry is not None and entry.state == 'delete':
                    self._save(state, item.path, SourceEntry(state='absent', current=current))
                    current = None
                else:
                    return await self._suppress(state, item.path, SourceEntry(state='active', current=current))
            else:
                if entry is not None and entry.state == 'delete':
                    self._save(state, item.path, SourceEntry(state='active', current=current))
                if current.original_sha256 == item.sha256 and current.parser_revision == self.parser_revision:
                    return recovered or SourceReceipt(item.path, 'unchanged', current.episode_id)
        return await self._prepare(state, item, current, generation)

    async def _delete(self, state: SourceState, path: str, entry: SourceEntry) -> SourceReceipt:
        if entry.state in ('absent', 'suppressed'):
            return SourceReceipt(path, entry.state, entry.current.episode_id if entry.current else None)
        if entry.state in ('replace', 'retire', 'suppress'):
            await self._finish(state, path, entry)
            entry = state.entries[path]
            if entry.state == 'suppressed':
                return SourceReceipt(path, 'suppressed', entry.current.episode_id if entry.current else None)
        current = entry.current
        if current is None or current.episode_id is None:
            raise InvalidInput('missing managed source has no current episode')
        try:
            await self._episode(state, path, current)
        except Gone:
            if entry.state != 'delete':
                return await self._suppress(state, path, entry)
        else:
            self._save(state, path, SourceEntry(state='delete', current=current))
            if not await asyncio.to_thread(self.scanner.missing, path):
                raise InvalidInput('managed source reappeared before deletion')
            await self.memory.forget(self.space, current.episode_id)
        self._save(state, path, SourceEntry(state='absent', current=current))
        return SourceReceipt(path, 'deleted', previous_episode_id=current.episode_id)

    async def synchronize(self, *, delete_missing: bool = False) -> DirectorySyncResult:
        if type(delete_missing) is not bool:
            raise InvalidInput('delete_missing must be a boolean')
        with self.journal.locked():
            state = self.journal.load()
            await self._preflight(state)
            self.journal.save(state)
            first = await asyncio.to_thread(self.scanner.scan)
            receipts: list[SourceReceipt] = []
            present = {item.path for item in first.files}
            for path, entry in sorted(state.entries.items()):
                if path not in present and entry.state in ('replace', 'retire', 'suppress'):
                    try:
                        receipts.append(await self._finish(state, path, entry))
                    except Exception:
                        receipts.append(SourceReceipt(path, 'failed', code='source_transition_failed'))
            for item in first.files:
                try:
                    receipts.append(await self._file(state, item))
                except Exception:
                    receipts.append(SourceReceipt(item.path, 'failed', code='source_transition_failed'))
            final = await asyncio.to_thread(self.scanner.scan)
            stable = first.same_inventory(final)
            issues = list(first.issues)
            issues.extend(issue for issue in final.issues if issue not in issues)
            if not stable and not issues:
                issues.append(ScanIssue('', 'inventory_changed'))
            if delete_missing and stable:
                failed_paths = {receipt.path for receipt in receipts if receipt.status == 'failed'}
                for path, entry in sorted(state.entries.items()):
                    if path in present or path in failed_paths:
                        continue
                    try:
                        deleted = await self._delete(state, path, entry)
                        receipts = [receipt for receipt in receipts if receipt.path != path]
                        receipts.append(deleted)
                    except Exception:
                        receipts = [receipt for receipt in receipts if receipt.path != path]
                        receipts.append(SourceReceipt(path, 'failed', code='source_deletion_failed'))
            reported = {receipt.path for receipt in receipts}
            for path, entry in sorted(state.entries.items()):
                if entry.state == 'delete' and path not in reported:
                    receipts.append(SourceReceipt(path, 'failed', code='deletion_pending'))
            return DirectorySyncResult(state.collection_id, stable and all(item.status != 'failed' for item in receipts),
                                       tuple(sorted(receipts, key=lambda item: item.path)), tuple(issues), final.skipped)
