"""Archive identity and evidence remapping over explicit storage ports.

The public engine checks space access. Episodes pass through its supplied normal
ingestion callback, rebuilding chunks/vectors and preserving event behavior.
"""
from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Optional

from ..backends.blobs import BlobStore
from . import attachment_archive, archive_supersession, archive_inventory
from ..core import forget_after
from ..core.timeutil import parse_rfc3339
from ..core.affirmations import Affirmation, NewAffirmation, affirmation_store
from ..core.errors import InvalidInput
from ..core.models import DEPENDENCY_KINDS, LINK_KINDS, Added, Fact, FactLink
from ..core.ports import DocumentStore, NewFact, NewFactLink
from ..core.validation import ORIGINS, STATUSES, normalise_term, normalise_time
from ..ingestion.records import Record, RetainedVideoRecord, content_hash


#: What this version of the archive format is called. An archive says it
#: in its first record, and an importer refuses a profile it does not
#: know rather than reading it hopefully.
ARCHIVE_PROFILE = "scone.archive/1"
#: What this profile carries. Everything a space holds that is not in
#: this list is named in the header as not carried, so that nobody reads
#: a dump of an illustrated space as the whole of it.
ARCHIVE_CARRIES = ("episodes", "facts", "fact_links", "affirmations")

#: The fields each kind of record may carry, taken from the models the
#: exporter writes from rather than written out here: a list kept by hand
#: goes stale the first time a model gains a field, and then refuses an
#: archive this engine could have read perfectly well. A record carrying
#: anything outside these is refused, because importing the part we
#: recognise would look like a success and quietly drop the rest.
KNOWN_FIELDS: dict[str, frozenset[str]] = {
    "archive": frozenset({"type", "profile", "space", "wrote_at", "engine", "carries", "not_carried"}),
    "episode": frozenset({"type", "space", "episode_id", "kind", "content", "content_hash",
                          "source", "tags", "metadata", "created_at", "dedup_key"}),
    "fact": frozenset({"type", "space"}) | frozenset(Fact.model_fields),
    "fact_link": frozenset({"type", "space"}) | frozenset(FactLink.model_fields),
    "affirmation": frozenset({"type", "space"}) | frozenset(Affirmation.model_fields),
}


@dataclass
class MergeReceipt:
    """What a merge would move, or did. ``moved`` says which it was, so a
    preview and a deed are never mistaken for each other."""

    space: str
    into: str
    episodes: int = 0
    facts: int = 0
    moved: bool = False
    attachments: int = 0
    attachment_bytes: int = 0
    unlinked_attachments: int = 0
    tombstoned: int = 0
    attachments_skipped: int = 0
    forgotten_source_references: int = 0
    #: Episodes left behind because their own ``forget_after`` had come; they
    #: go when the source space is deleted, as a sweep would have taken them.
    past_forget_after: int = 0

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class ImportSummary:
    #: The profile the archive was read as: what its header said, or the
    #: first profile when it had no header.
    profile: str = ARCHIVE_PROFILE
    episodes: int = 0
    #: Episodes already present in the target.
    deduplicated: int = 0
    facts: int = 0
    #: Relations stored, and those dropped or already present.
    links: int = 0
    links_skipped: int = 0
    #: Episodes skipped because this space forgot that content on purpose.
    tombstoned: int = 0
    #: Episodes skipped because their own ``forget_after`` had already come:
    #: restoring them would bring back what was scheduled to be gone.
    past_forget_after: int = 0
    #: Facts already present in the target (same subject, predicate,
    #: object, interval and status).
    facts_skipped: int = 0
    #: Restatements kept beside their facts, and those whose fact is not in
    #: the archive or whose target keeps none.
    affirmations: int = 0
    affirmations_skipped: int = 0
    #: Verified distinct attachments and links restored, including existing ones.
    attachments: int = 0
    attachment_links: int = 0
    #: Replacement edges restored after fact identity remapping.
    supersessions: int = 0

    def record(self) -> dict[str, object]:
        """Keep ordinary legacy receipts stable; disclose nonzero edge repairs."""
        result: dict[str, object] = dataclasses.asdict(self)
        if not self.supersessions:
            result.pop('supersessions')
        if not self.past_forget_after:
            result.pop('past_forget_after')
        if self.profile == ARCHIVE_PROFILE:
            result.pop('attachments')
            result.pop('attachment_links')
        return result


@dataclass(frozen=True)
class ArchiveRuntime:
    """Storage captured at dispatch; ingestion and clock supplied by the host."""

    documents: DocumentStore
    clock: Callable[[], str]
    remember_many: Callable[[str, Sequence[Record]], Awaitable[list[Added]]]
    blobs: BlobStore | None = None
    max_attachment_bytes: int = 25 * 1024 * 1024
    attachment_types: Sequence[str] = ()


async def export_records(documents: DocumentStore, space: str, *, wrote_at: str = "",
                         left_behind: Optional[Mapping[str, int]] = None) -> AsyncIterator[dict]:
    """Yield what the space is made of: a header saying what this archive
    is and what it does not carry, then original episodes, facts, unique
    links and restatements. No derived vectors: they are rebuilt by the
    ingestion that reads this.

    ``left_behind`` counts what the space holds that this profile does not
    carry. The host counts it, because the document store is not where
    those things live."""
    await archive_inventory.prepare(documents, space)
    counts = await documents.counts(space)
    yield {"type": "archive", "profile": ARCHIVE_PROFILE, "space": space, "wrote_at": wrote_at,
           "carries": list(ARCHIVE_CARRIES),
           "not_carried": {name: count for name, count in (left_behind or {}).items() if count}}
    async for episode in archive_inventory.episodes(documents, space, counts.episodes):
        yield {
            "type": "episode",
            "space": space,
            "episode_id": episode.episode_id,
            "kind": episode.kind,
            "content": episode.content,
            "content_hash": episode.content_hash,
            "source": episode.source,
            "tags": list(episode.tags),
            "metadata": dict(episode.metadata),
            "created_at": episode.created_at,
        }
    facts = await archive_inventory.facts(documents, space)
    for fact in facts:
        yield {"type": "fact", **fact.model_dump(exclude={"space"})}
    # Read each stored relationship once, even when an endpoint is missing.
    for link in await archive_inventory.links(documents, space):
        yield {"type": "fact_link", **link.model_dump(exclude={"space"})}
    for affirmation in await archive_inventory.affirmations(documents, space):
        yield {"type": "affirmation", **affirmation.model_dump(exclude={"space"})}


async def import_records(runtime: ArchiveRuntime, space: str, records: Iterable[Mapping], *,
                         resurrect: bool = False) -> ImportSummary:
    """Restore an archive through normal ingestion and remap its store-local IDs."""
    iterator = iter(records)
    first = next(iterator, None)
    rows: Iterable[Mapping] = chain((), iterator) if first is None else chain((first,), iterator)
    transfer = None
    if first is not None and first.get('profile') == attachment_archive.PROFILE:
        if runtime.blobs is None:
            raise InvalidInput('attachment archive requires a blob store')
        transfer = attachment_archive.prepare(attachment_archive.bounded_rows(rows),
                                               runtime.max_attachment_bytes, runtime.attachment_types)
        rows = transfer.records
    summary = ImportSummary()
    episodes: list[Record] = []
    source_ids: list[Optional[int]] = []
    facts: list[Mapping] = []
    links: list[Mapping] = []
    affirmations: list[Mapping] = []
    for record in rows:
        kind = record.get("type", "episode")
        _check_fields(kind, record)
        if kind == "archive":
            said = str(record.get("profile") or "")
            if said and said != ARCHIVE_PROFILE:
                raise InvalidInput(
                    f"this archive says it is {said}; this engine reads {ARCHIVE_PROFILE}, and reading "
                    f"a format it does not know would drop whatever it did not recognise")
            summary.profile = said or ARCHIVE_PROFILE
            continue
        if kind == "episode":
            episode = _rederived(Record.from_dict(record), record.get("space"), space)
            if episode.content == '' and episode.kind == 'file':
                if transfer is None:
                    raise InvalidInput('visual-only documents require attachment transfer; this archive profile omits attachments')
                episode = RetainedVideoRecord(**dataclasses.asdict(episode))
            # Forgetting was a decision; an archive that carries the
            # forgotten content does not undo it unless told to.
            digest = episode.content_hash or content_hash(space, episode.content, episode.dedup_key)
            if not resurrect and await runtime.documents.tombstone_by_hash(space, digest) is not None:
                summary.tombstoned += 1
                continue
            if overdue(episode, runtime.clock()):
                summary.past_forget_after += 1
                continue
            episodes.append(episode)
            source_ids.append(record.get("episode_id"))
        elif kind == "fact":
            facts.append(dict(record))
        elif kind == "fact_link":
            links.append(record)
        elif kind == "affirmation":
            affirmations.append(record)
        else:
            raise InvalidInput(f"unknown record type {kind!r}")
    supersessions = archive_supersession.parse(facts)
    # Ids are store-local (spec 3.1). Provenance in the dump names the
    # source store's episodes, so it is remapped through the ids this
    # import produced; a reference to an episode not in the dump is
    # dropped rather than pointed at an unrelated record.
    if transfer is not None:
        assert runtime.blobs is not None
        summary.profile = attachment_archive.PROFILE
        await transfer.preflight_episodes(runtime.documents, space, episodes, runtime.clock())
        summary.attachments = await transfer.stage(runtime.blobs, space, source_ids)
    id_map: dict[int, int] = {}
    imported: list[Added] = []
    if transfer is not None and any(isinstance(episode, RetainedVideoRecord) for episode in episodes):
        # Textless documents require the retained-evidence singleton path.
        for episode in episodes:
            imported.extend(await runtime.remember_many(space, [episode]))
    else:
        imported = await runtime.remember_many(space, episodes)
    for old_id, added in zip(source_ids, imported):
        summary.episodes += 0 if added.deduplicated else 1
        summary.deduplicated += 1 if added.deduplicated else 0
        if old_id is not None:
            id_map[int(old_id)] = added.episode_id
            if transfer is not None:
                assert runtime.blobs is not None
                summary.attachment_links += await transfer.link(runtime.blobs, space, int(old_id), added.episode_id)
    destination_facts = (await archive_supersession.inventory(runtime.documents, space) if supersessions
                         else await runtime.documents.list_facts(space, include_closed=True))
    existing = {_fact_identity(f): f.fact_id for f in destination_facts}
    # Fact ids are store-local too: a link's ends are remapped through
    # the ids this import produced or found already present.
    fact_map: dict[int, int] = {}
    prepared: list[tuple[Mapping, NewFact]] = []
    for f in facts:
        source = f.get("source_episode_id")
        new = NewFact(
            space=space,
            subject=normalise_term(str(f["subject"]), "subject"),
            predicate=normalise_term(str(f["predicate"]), "predicate"),
            object=str(f["object"]),
            valid_from=normalise_time(str(f["valid_from"])),
            confidence=float(f.get("confidence", 1.0)),
            valid_until=normalise_time(str(f["valid_until"])) if f.get("valid_until") else None,
            status=str(f.get("status", "active")),
            closed_reason=f.get("closed_reason"),
            source_episode_id=id_map.get(int(source)) if source is not None else None,
            origin=str(f.get("origin", "stated")),
            excluded_reason=f.get("excluded_reason"),
            superseded_by=None,  # Restored after all destination fact IDs are known.
            quote=f.get("quote"),
        )
        if new.status not in STATUSES or new.origin not in ORIGINS:
            raise InvalidInput(f"fact record has status {new.status!r} and origin {new.origin!r}")
        prepared.append((f, new))
    archive_supersession.preflight(supersessions, prepared, destination_facts, _fact_identity)
    for f, new in prepared:
        identity = _fact_identity(new)
        if identity in existing:
            summary.facts_skipped += 1
            if f.get("fact_id") is not None:
                fact_map[int(f["fact_id"])] = existing[identity]
            continue
        stored = await runtime.documents.insert_fact(new)
        existing[identity] = stored.fact_id
        if f.get("fact_id") is not None:
            fact_map[int(f["fact_id"])] = stored.fact_id
        summary.facts += 1
    summary.supersessions = await archive_supersession.restore(runtime.documents, space, supersessions, fact_map)
    for record in links:
        kind = str(record.get("kind", ""))
        from_fact = fact_map.get(int(record["from_fact"]))
        to_fact = fact_map.get(int(record["to_fact"]))
        if kind not in LINK_KINDS or from_fact is None or to_fact is None or from_fact == to_fact:
            # A link to a fact not in the dump is dropped, not pointed at
            # an unrelated record; a bad kind is a bad record.
            summary.links_skipped += 1
            continue
        source = record.get("source_episode_id")
        new_link = NewFactLink(
            space=space, from_fact=from_fact, to_fact=to_fact, kind=kind,
            created_at=normalise_time(str(record["created_at"])) if record.get("created_at") else runtime.clock(),
            source_episode_id=id_map.get(int(source)) if source is not None else None,
            quote=record.get("quote"),
        )
        already = any((l.to_fact, l.kind) == (to_fact, kind) for l in await runtime.documents.fact_links(space, from_fact) if l.from_fact == from_fact)
        if already:
            summary.links_skipped += 1
            continue
        await runtime.documents.insert_fact_link(new_link)
        summary.links += 1
    store = affirmation_store(runtime.documents)
    for record in affirmations:
        fact_id = fact_map.get(int(record["fact_id"]))
        valid_from = normalise_time(str(record["valid_from"]))
        if store is None or fact_id is None or any(
                kept.valid_from == valid_from for kept in await store.affirmations(space, fact_id)):
            summary.affirmations_skipped += 1
            continue
        # A premise is named by its new id; one not in the archive is
        # dropped, like a link to it, and counted with them.
        premises = []
        for kind, to_fact in record.get("links") or ():
            target = fact_map.get(int(to_fact))
            if kind in DEPENDENCY_KINDS and target is not None:
                premises.append((str(kind), target))
            else:
                summary.links_skipped += 1
        source = record.get("source_episode_id")
        await store.add_affirmation(NewAffirmation(
            space=space, fact_id=fact_id, valid_from=valid_from,
            recorded_at=normalise_time(str(record["recorded_at"])) if record.get("recorded_at") else runtime.clock(),
            confidence=float(record.get("confidence", 1.0)),
            source_episode_id=id_map.get(int(source)) if source is not None else None,
            origin=str(record.get("origin", "stated")), quote=record.get("quote"), links=tuple(premises)))
        summary.affirmations += 1
    if summary.facts or summary.links or summary.affirmations:
        await runtime.documents.bump_revision(space)
    return summary


def _fact_identity(fact: Fact | NewFact) -> tuple:
    """Import identity includes retained evidence after source-ID remapping.

    Store-local fact IDs are remapped; supersession edges are restored separately. Exclusion is mutable
    target policy: reimporting the same evidence must not create a fresh,
    unexcluded copy of a fact the target has suppressed.
    """
    return (
        fact.subject, fact.predicate, fact.object, fact.valid_from, fact.valid_until,
        fact.status, fact.closed_reason, fact.confidence,
        fact.source_episode_id, fact.origin, fact.quote,
    )


def _rederived(record: Record, source_space: Optional[str], space: str) -> Record:
    """The record with its content_hash dropped when it is the default
    derivation under the source space, so ingest derives it for the
    target space instead."""
    if record.content_hash is None or not source_space or source_space == space:
        return record
    if record.content_hash == content_hash(str(source_space), record.content):
        return dataclasses.replace(record, content_hash=None)
    return record


def overdue(record: Record, now: str) -> bool:
    """Whether an archived episode's own schedule has already come. A value
    that cannot be read is not judged here; ingestion refuses it by name."""
    return forget_after.is_due(record.metadata or {}, parse_rfc3339(now))


def _check_fields(kind: str, record: Mapping) -> None:
    """Refuse a record carrying anything this version cannot keep, naming
    every field it did not know."""
    known = KNOWN_FIELDS.get(kind)
    if known is None:
        return  # an unknown type is refused where types are decided
    strange = sorted(set(record) - known)
    if strange:
        raise InvalidInput(
            f"a {kind} record carries {', '.join(strange)}, which this engine does not keep; "
            f"importing the rest would drop it silently")
