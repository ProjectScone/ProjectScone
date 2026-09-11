"""Archive identity and evidence remapping over explicit storage ports.

The public engine checks space access. Episodes pass through its supplied normal
ingestion callback, rebuilding chunks/vectors and preserving event behavior.
"""
from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from ..core.affirmations import NewAffirmation, affirmation_store
from ..core.errors import InvalidInput
from ..core.models import DEPENDENCY_KINDS, LINK_KINDS, Added, Fact
from ..core.ports import DocumentStore, NewFact, NewFactLink
from ..core.validation import ORIGINS, STATUSES, normalise_term, normalise_time
from ..ingestion.records import Record, content_hash


@dataclass
class ImportSummary:
    episodes: int = 0
    #: Episodes already present in the target.
    deduplicated: int = 0
    facts: int = 0
    #: Relations stored, and those dropped or already present.
    links: int = 0
    links_skipped: int = 0
    #: Episodes skipped because this space forgot that content on purpose.
    tombstoned: int = 0
    #: Facts already present in the target (same subject, predicate,
    #: object, interval and status).
    facts_skipped: int = 0
    #: Restatements kept beside their facts, and those whose fact is not in
    #: the archive or whose target keeps none.
    affirmations: int = 0
    affirmations_skipped: int = 0


@dataclass(frozen=True)
class ArchiveRuntime:
    """Storage captured at dispatch; ingestion and clock supplied by the host."""

    documents: DocumentStore
    clock: Callable[[], str]
    remember_many: Callable[[str, Sequence[Record]], Awaitable[list[Added]]]


async def export_records(documents: DocumentStore, space: str) -> AsyncIterator[dict]:
    """Yield original episodes, facts and unique links; no derived vectors."""
    counts = await documents.counts(space)
    for episode in await documents.recent_episodes(space, max(counts.episodes, 1)):
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
    facts = await documents.list_facts(space, include_closed=True)
    for fact in facts:
        yield {"type": "fact", **fact.model_dump(exclude={"space"})}
    # A relation is part of the ledger's history, so it moves with it;
    # each link is read from both ends and written once.
    seen: set[int] = set()
    for fact in facts:
        for link in await documents.fact_links(space, fact.fact_id):
            if link.link_id not in seen:
                seen.add(link.link_id)
                yield {"type": "fact_link", **link.model_dump(exclude={"space"})}
    # A restatement's later start is history too: without it a backfill
    # in the new store would erase a value that returned.
    store = affirmation_store(documents)
    if store is not None:
        for affirmation in await store.space_affirmations(space):
            yield {"type": "affirmation", **affirmation.model_dump(exclude={"space"})}


async def import_records(runtime: ArchiveRuntime, space: str, records: Iterable[Mapping], *,
                         resurrect: bool = False) -> ImportSummary:
    """Restore an archive through normal ingestion and remap its store-local IDs."""
    summary = ImportSummary()
    episodes: list[Record] = []
    source_ids: list[Optional[int]] = []
    facts: list[Mapping] = []
    links: list[Mapping] = []
    affirmations: list[Mapping] = []
    for record in records:
        kind = record.get("type", "episode")
        if kind == "episode":
            episode = _rederived(Record.from_dict(record), record.get("space"), space)
            # Forgetting was a decision; an archive that carries the
            # forgotten content does not undo it unless told to.
            digest = episode.content_hash or content_hash(space, episode.content, episode.dedup_key)
            if not resurrect and await runtime.documents.tombstone_by_hash(space, digest) is not None:
                summary.tombstoned += 1
                continue
            episodes.append(episode)
            source_ids.append(record.get("episode_id"))
        elif kind == "fact":
            facts.append(record)
        elif kind == "fact_link":
            links.append(record)
        elif kind == "affirmation":
            affirmations.append(record)
        else:
            raise InvalidInput(f"unknown record type {kind!r}")
    # Ids are store-local (spec 3.1). Provenance in the dump names the
    # source store's episodes, so it is remapped through the ids this
    # import produced; a reference to an episode not in the dump is
    # dropped rather than pointed at an unrelated record.
    id_map: dict[int, int] = {}
    for old_id, added in zip(source_ids, await runtime.remember_many(space, episodes)):
        summary.episodes += 0 if added.deduplicated else 1
        summary.deduplicated += 1 if added.deduplicated else 0
        if old_id is not None:
            id_map[int(old_id)] = added.episode_id
    existing = {_fact_identity(f): f.fact_id for f in await runtime.documents.list_facts(space, include_closed=True)}
    # Fact ids are store-local too: a link's ends are remapped through
    # the ids this import produced or found already present.
    fact_map: dict[int, int] = {}
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
            superseded_by=None,  # ids are store-local; the reason text keeps the history
            quote=f.get("quote"),
        )
        if new.status not in STATUSES or new.origin not in ORIGINS:
            raise InvalidInput(f"fact record has status {new.status!r} and origin {new.origin!r}")
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

    Store-local fact/supersession IDs are not preserved. Exclusion is mutable
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
