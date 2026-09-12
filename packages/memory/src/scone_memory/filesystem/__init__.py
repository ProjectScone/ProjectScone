"""Memory as a tree of paths: a view of the ledger, never a second store.

An agent that can list and read paths can explore a space without being
taught an API for every kind of thing in it, which is what makes a
filesystem a good shape for memory. It is a bad shape if it becomes a
second place where things are true, so nothing here holds anything: a
path resolves to episodes, claims and entities the engine already has,
and reading one changes nothing.

Three rules keep it honest, and each is enforced rather than intended.

* **The space is not in the path.** A tree is opened for one space and
  every path is inside it, so there is no path that could name another
  space and nothing to escape from. `..` is refused rather than resolved,
  because resolving it is how a reader ends up somewhere it was not
  allowed.
* **Names are encoded, not cleaned.** A subject called `a/b c` is a real
  subject; its file is `a%2Fb%20c.md` and reading that file gives that
  subject back. Nothing is silently renamed, and two different names
  never become one file.
* **Every read is bounded and says so.** A listing carries how many
  entries there are and where to continue; a file says when it was cut.

What an agent may do with the tree, and what is written down when it
does, belong to the policy that wraps it; this module lists and reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional
from urllib.parse import quote, unquote

from ..core.errors import SconeError
from ..core.models import Episode
from ..core.validation import check_space

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

#: The longest path this will look at.
MAX_PATH = 512
#: Entries in one listing, by default and at most.
DEFAULT_ENTRIES = 100
MAX_ENTRIES = 1_000
#: Bytes of a file returned, by default and at most.
MAX_FILE_BYTES = 64_000
MAX_FILE_BYTES_LIMIT = 1_000_000
#: The directories a space has. Three are ways of looking at the ledger;
#: notes is the one an agent may write to, and a note is an episode.
DIRECTORIES = ("entities", "episodes", "facts", "notes")
#: Where a note's episodes say they came from.
NOTE_SOURCE = "fs:"
#: Bytes of one note, by default and at most.
MAX_NOTE_BYTES = 64_000
MAX_NOTE_BYTES_LIMIT = 1_000_000
#: What may appear in a path before it is encoded.
SAFE = "-._~"


class PathRefused(SconeError):
    """A path that cannot mean anything in this tree, or an action the
    policy does not allow, and why."""


class PathConflict(SconeError):
    """A write onto something that moved since it was read. Carries the
    version it now stands at, so a caller can re-read rather than guess."""

    def __init__(self, message: str, version: int = 0) -> None:
        super().__init__(message)
        self.version = version


@dataclass(frozen=True)
class Entry:
    """One thing in the tree."""

    path: str
    kind: Literal["directory", "file"]
    of: str
    bytes: Optional[int] = None
    modified: Optional[str] = None

    def record(self) -> dict[str, object]:
        return {"path": self.path, "kind": self.kind, "of": self.of,
                "bytes": self.bytes, "modified": self.modified}


@dataclass(frozen=True)
class Listing:
    """What is under a directory, and what was left out."""

    path: str
    entries: tuple[Entry, ...]
    #: How many there are, which is not how many were read: a listing
    #: reads a bounded number and says so rather than reporting its own
    #: bound as the answer.
    total: int
    truncated: bool = False
    next_offset: Optional[int] = None
    revision: int = 0
    #: How many the tree read, when that is fewer than there are. None
    #: when everything was read.
    capped: Optional[int] = None

    def record(self) -> dict[str, object]:
        return {"path": self.path, "entries": [entry.record() for entry in self.entries],
                "total": self.total, "truncated": self.truncated,
                "next_offset": self.next_offset, "revision": self.revision, "capped": self.capped}


@dataclass(frozen=True)
class FilePage:
    """A file's text, and what it is a view of."""

    path: str
    text: str
    of: str
    bytes: int
    #: Where the space stood when this was read. Everything in the tree is
    #: read at one of these, which is what makes a read repeatable.
    revision: int
    truncated: bool = False
    source: Optional[str] = None
    #: What this file itself stood at, for the files that can be written:
    #: a note's own version, which is the episode behind it. None for a
    #: view of the ledger, which is not a thing anyone writes to. A space
    #: revision will not do here: it moves when anything at all is
    #: written, so comparing it would refuse writes nobody conflicted with
    #: and, in a space where the two numbers drifted apart, let a real
    #: conflict through.
    version: Optional[int] = None

    def record(self) -> dict[str, object]:
        return {"path": self.path, "text": self.text, "of": self.of, "bytes": self.bytes,
                "revision": self.revision, "truncated": self.truncated, "source": self.source,
                "version": self.version}


@dataclass(frozen=True)
class Route:
    """What a path names: a directory, or a file and the name behind it."""

    kind: Literal["root", "directory", "file"]
    directory: str = ""
    name: str = ""


def encode(name: str) -> str:
    """A name as a filename, reversibly. Anything a path could read as
    structure is encoded, so no two names become one file."""
    return quote(name, safe=SAFE)


def decode(name: str) -> str:
    return unquote(name)


def route(path: str) -> Route:
    """What a path names here, or a refusal saying why it names nothing.

    Refusals come before any read, and `..` is never resolved: a tree that
    resolves it is one segment away from reading somewhere it should not."""
    if not isinstance(path, str) or not path:
        raise PathRefused("a path must be text, and must start with /")
    if len(path) > MAX_PATH:
        raise PathRefused(f"a path may be at most {MAX_PATH} characters long")
    if not path.startswith("/"):
        raise PathRefused("a path must start with / and is always inside its own space")
    if "\\" in path:
        raise PathRefused("a path separates its parts with / alone")
    if any(ch < " " or ch == "\x7f" for ch in path):
        raise PathRefused("a path may not hold control characters")
    if "//" in path:
        raise PathRefused("a path separates its parts with one slash")
    trimmed = path[:-1] if path.endswith("/") and path != "/" else path
    parts = [part for part in trimmed.split("/")[1:] if part]
    if any(part in ("..", ".") for part in parts):
        raise PathRefused("a path may not leave the space it was opened for")
    if not parts:
        return Route("root")
    if parts[0] not in DIRECTORIES:
        raise PathRefused(f"no such directory {parts[0]!r}; this space has {', '.join(DIRECTORIES)}")
    if len(parts) == 1:
        return Route("directory", parts[0])
    if not parts[-1].endswith(".md"):
        raise PathRefused("a file here is a .md view of what the ledger holds")
    if parts[0] == "notes":
        # A note is named by its whole path under notes, so an agent may
        # keep them in folders without the tree keeping folders.
        return Route("file", "notes", "/".join(decode(part) for part in parts[1:])[: -len(".md")])
    if len(parts) > 2:
        raise PathRefused("a directory here holds files, not more directories")
    return Route("file", parts[0], decode(parts[1][: -len(".md")]))


async def list_path(engine: "MemoryEngine", space: str, path: str, *, limit: int = DEFAULT_ENTRIES,
                    offset: int = 0) -> Listing:
    """What is under a path. Nothing is written, and what is left out is
    said rather than quietly dropped."""
    check_space(space)
    if not 1 <= limit <= MAX_ENTRIES:
        raise PathRefused(f"limit must be from 1 to {MAX_ENTRIES}")
    if offset < 0:
        raise PathRefused("offset must not be negative")
    where = route(path)
    revision = await engine.documents.revision(space)
    if where.kind == "file":
        raise PathRefused("that is a file; read it rather than listing it")
    held: Optional[int] = None
    if where.kind == "root":
        entries = [Entry(f"/{name}", "directory", name) for name in DIRECTORIES]
    elif where.directory == "episodes":
        held = (await engine.documents.counts(space)).episodes
        entries = [Entry(f"/episodes/{episode.episode_id}.md", "file", "episode",
                         len(episode.content.encode()), episode.ingested_at)
                   for episode in await _episodes(engine, space)]
    elif where.directory == "facts":
        entries = [Entry(f"/facts/{encode(subject)}.md", "file", "fact", None, None)
                   for subject in await _subjects(engine, space)]
    elif where.directory == "notes":
        entries = [Entry(path, "file", "note", len(episode.content.encode()), episode.ingested_at)
                   for path, episode in sorted((await _notes(engine, space)).items())]
    else:
        entries = [Entry(f"/entities/{encode(key)}.md", "file", "entity", None, None)
                   for key in await _entities(engine, space)]
    shown = entries[offset : offset + limit]
    total = max(held or 0, len(entries))
    more = offset + limit < len(entries)
    return Listing(path, tuple(shown), total,
                   truncated=more or total > len(entries) or (offset > 0 and bool(entries)),
                   next_offset=offset + limit if more else None, revision=revision,
                   capped=len(entries) if total > len(entries) else None)


async def read_file(engine: "MemoryEngine", space: str, path: str, *,
                    max_bytes: int = MAX_FILE_BYTES) -> FilePage:
    """One file. Reading is a read: nothing in the ledger moves."""
    check_space(space)
    if not 1 <= max_bytes <= MAX_FILE_BYTES_LIMIT:
        raise PathRefused(f"max_bytes must be from 1 to {MAX_FILE_BYTES_LIMIT}")
    where = route(path)
    if where.kind != "file":
        raise PathRefused("that is a directory; list it rather than reading it")
    revision = await engine.documents.revision(space)
    version: Optional[int] = None
    if where.directory == "episodes":
        text, source = await _episode_text(engine, space, where.name)
    elif where.directory == "notes":
        text, source, version = await _note_text(engine, space, where.name)
    elif where.directory == "facts":
        text, source = await _fact_text(engine, space, where.name)
    else:
        text, source = await _entity_text(engine, space, where.name)
    raw = text.encode()
    cut = len(raw) > max_bytes
    shown = raw[:max_bytes].decode(errors="ignore") if cut else text
    of = {"episodes": "episode", "facts": "fact", "entities": "entity", "notes": "note"}[where.directory]
    return FilePage(path, shown, of, len(shown.encode()), revision, truncated=cut, source=source,
                    version=version)


#: Episodes listed at most, so a tree over a large space stays a read.
MAX_LISTED = 5_000


async def _episodes(engine: "MemoryEngine", space: str) -> list[Episode]:
    """The space's episodes, oldest first, which is the order they read in."""
    counts = await engine.documents.counts(space)
    found = await engine.documents.recent_episodes(space, min(counts.episodes, MAX_LISTED))
    return sorted(found, key=lambda episode: episode.episode_id)


async def _subjects(engine: "MemoryEngine", space: str) -> list[str]:
    facts = await engine.documents.list_facts(space, include_closed=True)
    return sorted({fact.subject for fact in facts})


async def _entities(engine: "MemoryEngine", space: str) -> list[str]:
    from ..entities.read import load_projection

    projection, _ = await load_projection(engine, space, mode="current")
    return sorted(entity.key for entity in projection.entities)


async def _episode_text(engine: "MemoryEngine", space: str, name: str) -> tuple[str, str]:
    if not name.isdigit():
        raise PathRefused("an episode file is named by its episode number")
    episode = await engine.documents.get_episode(space, int(name))
    if episode is None:
        raise PathRefused(f"no such episode {name} in this space")
    return episode.content, f"episode {episode.episode_id}"


async def _fact_text(engine: "MemoryEngine", space: str, subject: str) -> tuple[str, str]:
    """Every claim about one subject, cited, newest first. A page of claims
    is not a summary of them: each line is one claim as the ledger holds
    it, with the fact behind it named."""
    facts = [fact for fact in await engine.documents.list_facts(space, include_closed=True)
             if fact.subject == subject]
    if not facts:
        raise PathRefused(f"no claims about {subject!r} in this space")
    lines = [f"# {subject}", ""]
    for fact in sorted(facts, key=lambda item: (item.valid_from, item.fact_id), reverse=True):
        held = f"from {fact.valid_from[:10]}" + (f" until {fact.valid_until[:10]}" if fact.valid_until else "")
        lines.append(f"- {fact.predicate} {fact.object} [fact {fact.fact_id}, {fact.status}, {held}]")
    return "\n".join(lines) + "\n", f"{len(facts)} claim(s)"


async def _entity_text(engine: "MemoryEngine", space: str, key: str) -> tuple[str, str]:
    """One entity as a page: what it is, what it relates to, what it has."""
    from ..entities.query import neighbourhood
    from ..entities.read import load_projection

    projection, _ = await load_projection(engine, space, mode="current")
    entity = next((item for item in projection.entities if item.key == key), None)
    if entity is None:
        raise PathRefused(f"no such entity {key!r} in this space")
    found = neighbourhood(projection, entity.entity_id)
    label = {item.entity_id: item.label for item in projection.entities}
    lines = [f"# {entity.label}", "", f"kind: {entity.kind or 'unknown'} ({entity.kind_status})", ""]
    for relation in (found.outgoing if found else ()):
        lines.append(f"- {relation.predicate} {label.get(relation.object_id, relation.object_id)} "
                     f"[{_cited(relation.fact_ids)}]")
    for relation in (found.incoming if found else ()):
        lines.append(f"- {label.get(relation.subject_id, relation.subject_id)} {relation.predicate} it "
                     f"[{_cited(relation.fact_ids)}]")
    for item in (found.follows if found else ()):
        lines.append(f"- follows ({item.follows}): {label.get(item.subject_id, item.subject_id)} "
                     f"{item.predicate} {label.get(item.object_id, item.object_id)} [{_cited(item.fact_ids)}]")
    for attribute in (found.attributes if found else ()):
        lines.append(f"- {attribute.predicate} {attribute.value} [{_cited(attribute.fact_ids)}]")
    return "\n".join(lines) + "\n", entity.entity_id


def _cited(fact_ids) -> str:
    ids = list(fact_ids)
    return f"fact {ids[0]}" if len(ids) == 1 else "facts " + ", ".join(str(i) for i in ids)


async def _notes(engine: "MemoryEngine", space: str) -> dict[str, Episode]:
    """The newest episode written at each note path. Writing a note again
    keeps the old episode and supersedes it, so the tree shows the last
    one and the ledger still holds every one."""
    newest: dict[str, Episode] = {}
    for episode in await _episodes(engine, space):
        if episode.source and episode.source.startswith(NOTE_SOURCE):
            newest[episode.source[len(NOTE_SOURCE):]] = episode
    return newest


async def _note_text(engine: "MemoryEngine", space: str, name: str) -> tuple[str, str, int]:
    found = (await _notes(engine, space)).get(f"/notes/{name}.md")
    if found is None:
        raise PathRefused(f"no note at /notes/{name}.md in this space")
    return found.content, f"episode {found.episode_id}", found.episode_id


#: Hits one search may answer with, by default and at most.
DEFAULT_HITS = 10
MAX_HITS = 200
#: Characters of a matching passage shown beside a path.
EXCERPT = 240


@dataclass(frozen=True)
class Hit:
    """One path a search found, and why."""

    path: str
    excerpt: str
    of: str
    score: float

    def record(self) -> dict[str, object]:
        return {"path": self.path, "excerpt": self.excerpt, "of": self.of, "score": self.score}


@dataclass(frozen=True)
class Found:
    """What a search found, and what it was."""

    query: str
    under: str
    hits: tuple[Hit, ...] = ()
    truncated: bool = False

    def record(self) -> dict[str, object]:
        return {"query": self.query, "under": self.under, "hits": [hit.record() for hit in self.hits],
                "truncated": self.truncated}


@dataclass(frozen=True)
class FilesystemPolicy:
    """What a tree allows. Nothing is writable unless it says so."""

    writable: bool = False
    max_note_bytes: int = MAX_NOTE_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.writable, bool):
            raise PathRefused("writable is a yes or a no")
        if (isinstance(self.max_note_bytes, bool) or not isinstance(self.max_note_bytes, int)
                or not 1 <= self.max_note_bytes <= MAX_NOTE_BYTES_LIMIT):
            raise PathRefused(f"max_note_bytes must be from 1 to {MAX_NOTE_BYTES_LIMIT}")

    def record(self) -> dict[str, object]:
        return {"writable": self.writable, "max_note_bytes": self.max_note_bytes}


@dataclass(frozen=True)
class Written:
    """What a write did: the note, the episode it became, and where the
    space stood afterwards."""

    path: str
    episode_id: int
    bytes: int
    revision: int

    @property
    def version(self) -> int:
        """What the note now stands at, to write on top of it next time."""
        return self.episode_id

    def record(self) -> dict[str, object]:
        return {"path": self.path, "episode_id": self.episode_id, "bytes": self.bytes,
                "revision": self.revision, "version": self.episode_id}


class MemoryFilesystem:
    """A tree opened for one space under one policy, which writes down what
    it did.

    The space is fixed when the tree is opened, so no path can name
    another; the policy is fixed with it, so what is allowed cannot change
    under a caller mid-read. Every action is recorded where the space
    keeps its events — reads and listings as well as writes, and refusals
    most of all, since a refusal is the interesting half of an audit."""

    def __init__(self, engine: "MemoryEngine", space: str,
                 policy: Optional[FilesystemPolicy] = None) -> None:
        check_space(space)
        self.engine = engine
        self.space = space
        self.policy = policy or FilesystemPolicy()

    async def list(self, path: str, *, limit: int = DEFAULT_ENTRIES, offset: int = 0) -> Listing:
        try:
            listing = await list_path(self.engine, self.space, path, limit=limit, offset=offset)
        except SconeError as refused:
            await self._said("refused", path, why=str(refused), doing="list")
            raise
        await self._said("list", path, entries=len(listing.entries), total=listing.total)
        return listing

    async def read(self, path: str, *, max_bytes: int = MAX_FILE_BYTES) -> FilePage:
        try:
            page = await read_file(self.engine, self.space, path, max_bytes=max_bytes)
        except SconeError as refused:
            await self._said("refused", path, why=str(refused), doing="read")
            raise
        await self._said("read", path, bytes=page.bytes, truncated=page.truncated, of=page.of)
        return page

    async def search(self, query: str, *, under: str = "/", limit: int = DEFAULT_HITS) -> Found:
        """Paths whose content answers a query, best first.

        The searching is the engine's ordinary recall, so there is no
        second index and no second idea of a good answer; what this adds
        is the tree's own terms. A passage that was written as a note is
        answered at its note path rather than at the episode behind it,
        because that is where its writer would look for it."""
        try:
            found = await self._search(query, under, limit)
        except SconeError as refused:
            await self._said("refused", under, why=str(refused), doing="search")
            raise
        # What was searched for is not recorded: a query is the reader's,
        # and an audit is for what was done, not for what was wondered.
        await self._said("search", under, hits=len(found.hits), truncated=found.truncated)
        return found

    async def _search(self, query: str, under: str, limit: int) -> Found:
        if not isinstance(query, str) or not query.strip():
            raise PathRefused("a search needs something to look for")
        if not 1 <= limit <= MAX_HITS:
            raise PathRefused(f"limit must be from 1 to {MAX_HITS}")
        where = route(under)
        if where.kind == "file":
            raise PathRefused("search under a directory, not under a file")
        prefix = "/" if where.kind == "root" else f"/{where.directory}/"
        result = await self.engine.recall(self.space, query, limit=min(limit * 3, MAX_HITS))
        notes = {episode.episode_id: path for path, episode in (await _notes(self.engine, self.space)).items()}
        hits: list[Hit] = []
        for item in result.items:
            path = notes.get(item.episode_id, f"/episodes/{item.episode_id}.md")
            hits.append(Hit(path, _excerpt(item.text), "note" if item.episode_id in notes else "episode",
                            item.score))
        for fact in result.facts:
            hits.append(Hit(f"/facts/{encode(fact.subject)}.md",
                            _excerpt(f"{fact.subject} {fact.predicate} {fact.object}"), "fact", 1.0))
        kept: list[Hit] = []
        seen: set[str] = set()
        for hit in hits:
            if not hit.path.startswith(prefix) or hit.path in seen:
                continue
            seen.add(hit.path)
            kept.append(hit)
        return Found(query, under, tuple(kept[:limit]), truncated=len(kept) > limit)

    async def write(self, path: str, text: str, *, if_version: Optional[int] = None) -> Written:
        """Write a note. Refused unless the policy allows writing, unless
        the path is under /notes, and unless the note is as the writer last
        read it — a write that would land on top of a newer one is refused
        rather than resolved, because there is no way to resolve it that is
        not somebody's work thrown away. ``if_version`` is the note's own
        version, as a read of it reports."""
        try:
            written = await self._write(path, text, if_version)
        except SconeError as refused:
            await self._said("refused", path, why=str(refused), doing="write")
            raise
        await self._said("write", path, bytes=written.bytes, episode_id=written.episode_id)
        return written

    async def _write(self, path: str, text: str, if_version: Optional[int]) -> Written:
        if not self.policy.writable:
            raise PathRefused("this tree is read only; nothing in it may be written")
        where = route(path)
        if where.kind != "file" or where.directory != "notes":
            raise PathRefused("only /notes/... may be written; everything else is a view of the ledger")
        if not isinstance(text, str) or not text.strip():
            raise PathRefused("a note must have something in it")
        if len(text.encode()) > self.policy.max_note_bytes:
            raise PathRefused(f"a note may be at most {self.policy.max_note_bytes} bytes")
        at = f"/notes/{where.name}.md"
        standing = (await _notes(self.engine, self.space)).get(at)
        if if_version is not None and standing is not None and standing.episode_id != if_version:
            raise PathConflict(
                f"{at} moved since it was read: it stands at version {standing.episode_id}, "
                f"not {if_version}", standing.episode_id)
        added = await self.engine.remember(self.space, text, kind="note", source=f"{NOTE_SOURCE}{at}")
        return Written(at, added.episode_id, len(text.encode()),
                       await self.engine.documents.revision(self.space))

    async def _said(self, what: str, path: str, **fields: object) -> None:
        await self.engine._emit(self.space, f"filesystem.{what}",
                                {"path": path, "space": self.space, **fields})


def _excerpt(text: str) -> str:
    """One line of what matched, short enough to read in a listing."""
    line = " ".join(text.split())
    return line if len(line) <= EXCERPT else line[: EXCERPT - 1] + "\u2026"
