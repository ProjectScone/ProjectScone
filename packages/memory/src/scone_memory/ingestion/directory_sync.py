"""Recoverable source replacement over existing memory and attachment stores."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Mapping, cast

from pydantic import ValidationError

from ..core import forget_after as schedule
from ..core.errors import Gone, InvalidInput, NotFound
from ..core.retirement import RetirementStore
from ..core.timeutil import parse_rfc3339
from .records import key_hash
from ..core.models import Episode, Tombstone
from ..memory import source_keys
from ..memory.engine import MemoryEngine
from .document_source import DocumentSource, source_revision_key
from .files import (FILE_MEDIA_TYPES, DocumentManifest, digest, document_provenance,
                    encode_manifest, prepare_document, store_document)
from .formats.registry import BuiltinDocumentParser, DocumentParser, extension
from .formats.types import DocumentLimits
from .sensitive import screen
from .source_journal import MAX_CLAIMANTS, SourceEntry, SourceJournal, SourceRevision, SourceState
from .source_scan import DirectoryScanner, ScanIssue, ScanLimits, ScannedFile


@dataclass(frozen=True)
class SourceReceipt:
    path: str
    status: Literal['added', 'updated', 'unchanged', 'deleted', 'absent', 'suppressed', 'failed', 'withheld']
    episode_id: int | None = None
    previous_episode_id: int | None = None
    code: str | None = None
    #: Claims the stored revision makes (a source file's definitions,
    #: imports and calls; a manifest's dependencies); zero for a document
    #: that makes none.
    claims: int = 0
    #: Claims of the previous revision closed because the new one no
    #: longer makes them, or because the file was deleted. None when the
    #: store could not read the claims by episode: nothing was closed,
    #: and the receipt says so rather than saying zero.
    claims_closed: int | None = 0
    #: Earlier episodes this file's claims still cite that the journal
    #: stopped tracking, because its lineage is bounded; what they cite
    #: is not closed by later replacements or deletion. Zero unless the
    #: bound bit.
    claims_untracked: int = 0
    #: When the stored revision is to be forgotten: the run's schedule for a
    #: revision this run wrote, the one its episode holds for one it left
    #: unchanged. None when it holds none, and for a source not stored.
    forget_after: str | None = None


@dataclass(frozen=True)
class DirectorySyncResult:
    collection_id: str
    complete: bool
    receipts: tuple[SourceReceipt, ...]
    issues: tuple[ScanIssue, ...]
    skipped: int
    #: Totals over the receipts: claims the run recorded, claims it
    #: closed, and whether some could not be read and so were not closed.
    claims: int = 0
    claims_closed: int = 0
    claims_unread: bool = False
    #: The instant every revision this run wrote is to be forgotten, resolved
    #: once from what the caller asked; None when it asked for none.
    forget_after: str | None = None
    #: Unchanged sources -- not written, so they keep the schedule their
    #: episode holds -- holding another schedule than ``forget_after``.
    schedule_kept: int = 0


class DirectorySync:
    """One journal per collection. Serialize other writes to its managed sources.

    Replacements retain and verify the new source before retiring the previous
    episode. Stages are recoverable; the stores do not provide a global atomic
    swap. A managed replacement closes the claims the new revision no longer
    makes, and a managed deletion closes them all, each naming the file and
    why; the receipts count both. An external forget suppresses the path and
    leaves its claims standing, by the engine's contract for forget. Managed
    missing-file deletion permits the file's later return. Missing evidence or
    ownership mismatches fail closed.

    A run's ``forget_after`` is stored on every revision it prepares, and a
    pending revision keeps it whichever run finishes it; an unchanged source
    keeps the schedule it holds. A source whose schedule has come -- swept or
    not -- is read afresh from the file by the next run, as ``sync`` does,
    with that run's schedule; only a forget before its time suppresses it.
    """

    def __init__(self, memory: MemoryEngine, root: str | Path, *, space: str, include_sensitive: bool = False,
                 journal: str | Path, key: bytes, store_id: str, parser_revision: str,
                 parser: DocumentParser | None = None, limits: DocumentLimits = DocumentLimits(),
                 scan_limits: ScanLimits = ScanLimits(), extensions: frozenset[str] | None = None):
        DocumentSource('0' * 32, 'validation.txt', parser_revision)
        if not isinstance(store_id, str) or not 1 <= len(store_id) <= 512:
            raise InvalidInput('directory synchronization requires a bounded stable store identity')
        #: Whether a source that screens as a credential is taken anyway.
        #: Off unless asked for: a store that quietly held a key was the
        #: fault this exists to remove, and taking one must be a decision.
        self.include_sensitive = include_sensitive
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
        if revision.episode_id is not None and isinstance(self.memory.documents, RetirementStore):
            pending = await self.memory.documents.retirement(self.space, revision.episode_id)
            if pending is not None:
                if pending.content_hash != key_hash(self.space, key):
                    raise InvalidInput('source retirement does not match its directory journal')
                # The catalog already accepted this deletion. Finish its exact
                # identity before the normal Gone/ownership checks below.
                await self.memory.forget(self.space, revision.episode_id)
        # Read as stored, due or not, so a revision whose time has come is
        # still checked against the journal; callers decide what due means.
        episode = await source_keys.episode_by_key(self.memory.documents, self.memory.blobs, self.space, key)
        expected = source.metadata() | {'document_original': revision.original_sha256,
                                       'document_manifest': revision.manifest_sha256}
        if (episode.kind != 'file' or episode.source != 'attachment:' + revision.original_sha256
                or (revision.episode_id is not None and revision.episode_id != episode.episode_id)
                or any(episode.metadata.get(name) != value for name, value in expected.items())
                or episode.metadata.get('source_generation', '0') != str(revision.generation)):
            raise InvalidInput('managed source ownership does not match its journal')
        if provenance and not self._due(episode.metadata):
            evidence = await document_provenance(self.memory, self.space, episode.episode_id)
            if evidence.filename != path:
                raise InvalidInput('managed source provenance path does not match its journal')
            await self._index(episode)
        return episode

    def _due(self, metadata: Mapping[str, str]) -> bool:
        """Whether an episode's schedule has come."""
        return schedule.is_due(metadata, parse_rfc3339(self.memory.clock()))

    async def _swept(self, state: SourceState, path: str, revision: SourceRevision) -> Tombstone | None:
        """The tombstone of a revision read as Gone when it went because its
        time came: forgotten no earlier than its schedule. None for a forget
        before its time, a forget by hand, which suppresses the path. Read by
        the revision's identity, so a revision stored and swept before the
        journal recorded its episode is found too."""
        if revision.forget_after is None:
            return None
        source = self._source(state, path, revision)
        identity = key_hash(self.space, source_revision_key(source, revision.original_sha256, revision.manifest_sha256))
        tombstone = await self.memory.documents.tombstone_by_hash(self.space, identity)
        if tombstone is None or parse_rfc3339(tombstone.forgotten_at) < parse_rfc3339(revision.forget_after):
            return None
        return tombstone

    async def _drop(self, state: SourceState, path: str, entry: SourceEntry, swept: Tombstone) -> SourceReceipt:
        """Abandon a replacement whose pending revision was taken at its time
        before the replacement finished. Neither revision stands: a previous
        one still held goes too, by the ordinary forget, after the claims both
        were cited for are closed, as for a deletion. The swept revision is
        recorded as absent, so a file still there is read afresh under a new
        generation and states its claims again."""
        pending = cast(SourceRevision, entry.pending)
        episodes = [*entry.claimants]
        previous: Episode | None = None
        if entry.current is not None:
            try:
                previous = await self._episode(state, path, entry.current)
            except Gone:
                pass
            else:
                episodes.append(previous.episode_id)
        closed, claimants, untracked = await self._close_claims(
            (*episodes, swept.episode_id), path, kept=[],
            reason=f'{path} was taken at its forget_after before it replaced its last revision', kind='source_removed')
        if previous is not None:
            await self.memory.forget(self.space, previous.episode_id)
        self._save(state, path, SourceEntry(state='absent', current=pending.model_copy(update={'episode_id': swept.episode_id}),
                                            claimants=claimants))
        return SourceReceipt(path, 'absent', swept.episode_id, previous.episode_id if previous else None,
                             claims_closed=closed, claims_untracked=untracked, forget_after=pending.forget_after)

    async def _index(self, episode: Episode) -> None:
        if (self.space, episode.content_hash) in await self.memory.documents.inflight():
            raise InvalidInput('managed source indexing is unfinished; recover the engine before retrying')
        chunks = await self.memory.documents.chunks_of(self.space, episode.episode_id)
        encoded = episode.content.encode('utf-8')
        end = 0
        if not chunks:
            if episode.content == '':
                from .formats.types import visual_only, ParsedDocument
                evidence = await document_provenance(self.memory, self.space, episode.episode_id)
                if visual_only(ParsedDocument(format=evidence.format, parser=evidence.parser,
                        segments=evidence.segments, metadata=evidence.metadata, video=evidence.video)):
                    return
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
        self._save(state, path, SourceEntry(state='suppress', current=entry.current, pending=entry.pending, claimants=entry.claimants))
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
        self._save(state, path, SourceEntry(state='suppressed', current=revision, claimants=entry.claimants))
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
                    # Swept at its time, the old revision is simply gone.
                    if not await self._swept(state, path, current):
                        return await self._suppress(state, path, entry)
            try:
                await self._episode(state, path, pending, provenance=False)
            except Gone:
                if taken := await self._swept(state, path, pending):
                    return await self._drop(state, path, entry, taken)
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
            # The schedule it was prepared under, checked then, stored now even
            # if it has passed since; a replay of it holds the same one.
            added = await store_document(self.memory, self.space, original, manifest,
                                         source=self._source(state, path, pending),
                                         forget_after=schedule.Resolved(pending.forget_after)
                                         if pending.forget_after is not None else None)
            claims = added.claims
            pending = pending.model_copy(update={'episode_id': added.added.episode_id})
            await self._episode(state, path, pending)
            entry = SourceEntry(state='retire', current=current, pending=pending, claimants=entry.claimants)
            self._save(state, path, entry)
        else:
            # Resumed after the retire save: the claims were recorded with
            # the store; read how many stand, so the receipt says so.
            claims = (await self._claims_of(pending.episode_id) or 0) if pending.episode_id is not None else 0
        # A recorded retire intent permits an already-forgotten old episode on
        # retry. A missing record without its tombstone remains an error.
        try:
            await self._episode(state, path, pending)
        except Gone:
            if swept := await self._swept(state, path, pending):
                return await self._drop(state, path, entry, swept)
            return await self._suppress(state, path, entry)
        closed: int | None = 0
        untracked = 0
        claimants = entry.claimants
        if current is not None:
            try:
                old = await self._episode(state, path, current)
            except Gone:
                if await self._swept(state, path, current):
                    # Taken at its time, not by this replacement, so nothing
                    # closed its claims before it went: they are closed now.
                    closed, claimants, untracked = await self._close_claims(
                        (*claimants, cast(int, current.episode_id)), path, kept=await self._kept(pending),
                        reason=f'no longer stated by {path}', kind='source_changed')
            else:
                # What the new revision still says stays; the rest is closed
                # before the old episode goes, so a retry after a crash here
                # finds the episode and closes again (nothing more), instead
                # of finding it gone and closing nothing. A claim restated
                # across revisions is one fact cited to the episode that
                # first made it, so every earlier episode is closed against
                # the new revision too.
                closed, claimants, untracked = await self._close_claims(
                    (*claimants, old.episode_id), path, kept=await self._kept(pending),
                    reason=f'no longer stated by {path}', kind='source_changed')
                await self.memory.forget(self.space, old.episode_id)
        self._save(state, path, SourceEntry(state='active', current=pending, claimants=claimants))
        return SourceReceipt(path, 'updated' if current is not None else 'added', pending.episode_id,
                             current.episode_id if current else None, claims=claims, claims_closed=closed,
                             claims_untracked=untracked, forget_after=pending.forget_after)

    async def _kept(self, revision: SourceRevision) -> list[tuple[str, str, str]]:
        """The claims a revision makes, read from its retained manifest:
        what a replacement keeps of the previous revision's claims."""
        from .files import document_claims

        _retained, encoded = await self.memory.attachment(self.space, revision.manifest_sha256)
        manifest = DocumentManifest.model_validate_json(encoded)
        return [(claim.subject, claim.predicate, claim.object)
                for claim in document_claims(manifest.parsed, manifest.filename)]

    async def _claims_of(self, episode_id: int) -> int | None:
        """Active extracted claims cited to an episode; None when the store
        cannot read claims by episode."""
        from ..core import graph_read

        documents = self.memory.documents
        if not isinstance(documents, graph_read.GraphFactReader):
            return None
        rows = await documents.facts_for_graph(self.space, episode_id, graph_read.MAX_GRAPH_FACTS)
        return sum(1 for fact in rows if fact.status == 'active' and fact.origin == 'extracted')

    async def _close_claims(self, episodes: tuple[int, ...], path: str, *, kept: list[tuple[str, str, str]],
                            reason: str, kind: str) -> tuple[int | None, tuple[int, ...], int]:
        """Close what the given episodes' claims say that `kept` does not.
        Says how many were closed (None when a store could not read them),
        which episodes still have claims citing them, and how many such
        episodes were left untracked because the lineage is bounded by
        MAX_CLAIMANTS: the oldest are kept, the newest dropped, and what
        the dropped ones still cite is no longer closed by this sync."""
        from .files import says_claims

        if not says_claims(path):
            return 0, (), 0
        ids = tuple(dict.fromkeys(episodes))
        closed: int | None = 0
        still: list[int] = []
        for episode_id in ids:
            retired = await self.memory.close_unstated(self.space, episode_id, kept=kept, reason=reason, kind=kind)
            if retired.closed is None or retired.unread:
                closed = None
            elif closed is not None:
                closed += retired.closed
            standing = await self._claims_of(episode_id)
            if standing is None or standing:
                still.append(episode_id)
        untracked = max(0, len(still) - MAX_CLAIMANTS)
        return closed, tuple(still[:MAX_CLAIMANTS]), untracked

    async def _prepare(self, state: SourceState, item: ScannedFile, current: SourceRevision | None,
                       generation: int, *, forget_after: str | None = None) -> SourceReceipt:
        data = await asyncio.to_thread(self.scanner.read, item)
        # The bytes are in hand and nothing has been retained yet. A file
        # that screens as a credential goes no further: no attachment, no
        # episode, no journal entry to recover into one later. The last
        # clean revision, if any, is left standing and named.
        #
        # One place, on purpose. A cheaper name-only refusal in `_file`
        # was provably indistinguishable from this one -- `screen` runs
        # the name stage first anyway -- and it sat ahead of the journal
        # recovery above, so a named credential with a pending entry would
        # have been refused without that entry ever being settled.
        if not self.include_sensitive:
            found = screen(item.path, data)
            if found.reason is not None:
                return SourceReceipt(item.path, 'withheld', None,
                                     current.episode_id if current else None, found.reason)
        if len(data) > self.memory.max_attachment_bytes:
            raise InvalidInput('directory document exceeds its attachment limit')
        manifest = await prepare_document(data, item.path, parser=self.parser, limits=self.limits)
        encoded = encode_manifest(manifest)
        if len(encoded) > self.memory.max_attachment_bytes:
            raise InvalidInput('directory manifest exceeds its attachment limit')
        await asyncio.to_thread(self.scanner.read, item)
        pending = SourceRevision(original_sha256=digest(data), manifest_sha256=digest(encoded),
                                 parser_revision=self.parser_revision,
                                 forget_after=forget_after, generation=generation)
        entry = SourceEntry(state='replace', current=current, pending=pending,
                            claimants=state.entries[item.path].claimants if item.path in state.entries else ())
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

    async def _file(self, state: SourceState, item: ScannedFile, *, forget_after: str | None = None) -> SourceReceipt:
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
                held = await self._episode(state, item.path, current)
            except Gone:
                if entry is not None and entry.state == 'delete':
                    self._save(state, item.path, SourceEntry(state='absent', current=current, claimants=entry.claimants))
                    current = None
                elif not await self._swept(state, item.path, current):
                    # Forgotten by hand: the path is suppressed. One swept at its
                    # time is replaced below, which closes what it said.
                    return await self._suppress(state, item.path, SourceEntry(state='active', current=current,
                                                                              claimants=entry.claimants if entry else ()))
            else:
                if entry is not None and entry.state == 'delete':
                    self._save(state, item.path, SourceEntry(state='active', current=current, claimants=entry.claimants))
                # One whose time has come is replaced even when the file is
                # unchanged, as `sync` and `remember` store content again once
                # it is overdue; the replacement forgets it as any other.
                if (current.original_sha256 == item.sha256 and current.parser_revision == self.parser_revision
                        and not self._due(held.metadata)):
                    return recovered or SourceReceipt(item.path, 'unchanged', current.episode_id,
                                                      forget_after=held.metadata.get(schedule.KEY))
        return await self._prepare(state, item, current, generation, forget_after=forget_after)

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
        closed: int | None = 0
        untracked = 0
        claimants = entry.claimants
        try:
            await self._episode(state, path, current)
        except Gone:
            if entry.state != 'delete' and await self._swept(state, path, current):
                # Taken at its time and gone from disk since: what it said is
                # closed, as for a deletion.
                closed, claimants, untracked = await self._close_claims(
                    (*entry.claimants, current.episode_id), path, kept=[], reason=f'{path} was removed', kind='source_removed')
                self._save(state, path, SourceEntry(state='absent', current=current, claimants=claimants))
                return SourceReceipt(path, 'absent', current.episode_id, claims_closed=closed, claims_untracked=untracked)
            if entry.state != 'delete':
                return await self._suppress(state, path, entry)
        else:
            self._save(state, path, SourceEntry(state='delete', current=current, claimants=entry.claimants))
            if not await asyncio.to_thread(self.scanner.missing, path):
                raise InvalidInput('managed source reappeared before deletion')
            # Closed before the episode goes, for the same reason as a
            # replacement: a retry after a crash still finds it. Every
            # earlier episode a restated claim still cites is closed too.
            closed, claimants, untracked = await self._close_claims((*entry.claimants, current.episode_id), path, kept=[],
                                                                    reason=f'{path} was removed', kind='source_removed')
            await self.memory.forget(self.space, current.episode_id)
        self._save(state, path, SourceEntry(state='absent', current=current, claimants=claimants))
        return SourceReceipt(path, 'deleted', previous_episode_id=current.episode_id, claims_closed=closed,
                             claims_untracked=untracked)

    async def synchronize(self, *, delete_missing: bool = False,
                          forget_after: str | schedule.Resolved | None = None) -> DirectorySyncResult:
        """``forget_after`` schedules every revision this run prepares, as for
        ``remember``: resolved once, before the journal is read, so the run
        writes one instant, a walk that outlasts it still writes it (those
        revisions are due at once), and a refused schedule records nothing."""
        if type(delete_missing) is not bool:
            raise InvalidInput('delete_missing must be a boolean')
        when = schedule.instant(schedule.asked(forget_after, self.memory.clock()))
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
                    receipts.append(await self._file(state, item, forget_after=when))
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
                                       tuple(sorted(receipts, key=lambda item: item.path)), tuple(issues), final.skipped,
                                       claims=sum(item.claims for item in receipts),
                                       claims_closed=sum(item.claims_closed or 0 for item in receipts),
                                       claims_unread=any(item.claims_closed is None for item in receipts),
                                       forget_after=when,
                                       schedule_kept=sum(1 for item in receipts
                                                         if item.status == 'unchanged' and item.forget_after != when))
