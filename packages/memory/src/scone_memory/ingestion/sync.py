"""Keep a space in step with a directory, including what left it.

``map`` remembers the files under a directory and notices when it has
seen one before. What it cannot notice is that a file has **changed** or
that a file is **gone**, and those two are the difference between an
import you can run once and a sync you can run on a schedule. This is
the second: a directory read again, and the space brought into step with
it.

Removal is the dangerous half. Forgetting memory because a file is
missing is destructive, and a directory can be missing for reasons that
have nothing to do with intent — an unmounted volume, a half-finished
checkout, a typo in a path. So:

- **Nothing is written unless ``apply``.** The default is a plan: what
  would be added, updated and removed, with nothing touched.
- **Nothing is forgotten unless ``remove``**, separately from ``apply``.
  An ordinary sync reports what is missing and leaves it alone, because
  a caller who has not thought about deletion should not get it.
- **An empty directory is refused outright.** If the walk found no files
  at all and the marker has episodes, that is overwhelmingly a wrong
  path rather than a repository whose every file was deleted, and the
  refusal happens before anything is forgotten.

Identity comes from the ``marker``: every episode this writes carries it
in its metadata, so "what did the last sync of this directory leave
here" is an exact question with an exact answer. Two directories synced
into one space cannot delete each other's memories, and a directory can
be renamed without losing what it stored, because the marker is a name
the caller chooses rather than a path we hope stays put.

No model is called, nothing is fetched, and the writes are the engine's
own ``replace`` and ``forget`` — the same keyed-update and receipt
machinery any other caller gets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import TYPE_CHECKING, Optional, Sequence

from ..core.errors import InvalidInput, SconeError
from .code import BRACE_SUFFIXES, PYTHON_SUFFIXES
from .manifests import is_manifest
from .code_resolution import file_resolver
from .ignore import Ignore
from .records import Record

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

#: Files one sync will look at. A directory with more is read as far as
#: this and says so: ``files_found`` is what is there, ``files_read`` is
#: what was looked at, and the two are never conflated.
MAX_FILES = 100_000
#: Bytes read from one file. The rest is left out, and ``cut`` counts it.
MAX_BYTES = 1_000_000
#: What a sync reads when the caller does not say. Prose and code, not
#: archives or images: a file whose bytes are not text has nothing for a
#: lexical lane and would only bloat the space.
SUFFIXES: tuple[str, ...] = (*PYTHON_SUFFIXES, *BRACE_SUFFIXES, ".md", ".markdown", ".rst", ".txt")
#: Changes listed in a receipt before it stops listing them. The counts
#: stay exact; only the per-file list is bounded.
MAX_LISTED = 1_000


class NoEventLog(SconeError):
    """Raised when asked when a directory last synced and nothing keeps a
    record. Distinct from "never synced", which is an answer."""


@dataclass(frozen=True)
class Change:
    """One file and what the sync made of it."""

    path: str
    #: added, updated, unchanged, missing, empty, or cut.
    what: str


@dataclass(frozen=True)
class SyncReceipt:
    """What a sync did, or would do. Counts are exact; ``changes`` is a
    bounded list of the same events for reading."""

    root: str
    marker: str
    space: str
    applied: bool
    removing: bool
    #: Files under the root matching the suffixes, or package manifests
    #: when any suffix is code, and not excluded by the ignore rules. What
    #: is there to read.
    files_found: int = 0
    #: Files this sync actually read, which is fewer when capped. What we
    #: looked at, which is never reported as what is there.
    files_read: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    #: Files the suffixes selected that a `.gitignore` or `.sconeignore`
    #: excluded, and directories those rules pruned (each hiding an
    #: unknown number of files). Zero under ``ignore=False``.
    ignored: int = 0
    ignored_directories: int = 0
    #: The ignore files read, relative to the root, in reading order.
    ignore_files: tuple[str, ...] = ()
    #: An ignore file or pattern past the bound was left unread, so the
    #: tree's exclusions are not all known, and this run may have read
    #: what the tree said not to.
    ignore_truncated: bool = False
    #: Patterns that could not be read as one, passed over and named.
    ignore_unusable: tuple[str, ...] = ()
    #: Memories the marker holds for files that are on disk but that the
    #: ignore rules now exclude. Not missing: left alone, and counted.
    ignored_memories: int = 0
    #: Chunks of the added and updated files whose vector came from the
    #: embedding cache: text stored before under the same embedder. Zero
    #: without a cache, and on a plan.
    embeddings_reused: int = 0
    #: Episodes the marker holds whose file is no longer on disk. Only
    #: meaningful when ``checked_for_missing``: a walk that stopped at the
    #: file cap did not see the whole directory, so it cannot tell a file
    #: that is gone from one it never reached.
    removed: int = 0
    #: Of those, the ones actually forgotten. Zero unless ``remove``.
    forgotten: int = 0
    #: Symbolic links under the root, which are counted and not followed:
    #: what one points at is outside the root the caller named.
    links: int = 0
    #: Things matching the suffixes (or named as a manifest) that are not
    #: ordinary files -- named pipes, sockets, devices. Counted and never
    #: opened: reading one can block forever.
    special: int = 0
    #: Directories under the root this sync could not read. Each one hides
    #: an unknown number of files, so a run with any of these cannot say
    #: what is gone.
    unreadable: int = 0
    #: Episodes this marker holds whose file this run would not have looked
    #: for -- the suffixes no longer select it. Counted apart from missing,
    #: because a narrowed flag is not a deleted file.
    out_of_scope: int = 0
    #: Files with nothing but whitespace in them, which are not stored.
    #: Counted apart from a file that could not be read at all: a count
    #: that adds them together tells a reader neither.
    empty: int = 0
    #: Files longer than the byte limit, stored up to it.
    cut: int = 0
    changes: tuple[Change, ...] = ()
    listed_all: bool = True
    #: Whether this sync was in a position to say what is gone at all. A
    #: capped walk is not, and reporting ``removed: 0`` from one without
    #: saying so would read as "nothing is gone" rather than "we did not
    #: look".
    checked_for_missing: bool = True
    #: Extracted claims closed because a changed file no longer makes them
    #: or a removed file made them. Only counted with the code graph on.
    claims_closed: int = 0
    #: Some claims of a replaced or removed file could not be examined --
    #: more than one read returns, or a store that cannot read claims by
    #: episode -- and may still stand.
    claims_unread: bool = False

    @property
    def capped(self) -> bool:
        return self.files_read < self.files_found

    def record(self) -> dict[str, object]:
        return {"root": self.root, "marker": self.marker, "space": self.space,
                "applied": self.applied, "removing": self.removing,
                "files_found": self.files_found, "files_read": self.files_read,
                "capped": self.capped, "added": self.added, "updated": self.updated,
                "unchanged": self.unchanged, "embeddings_reused": self.embeddings_reused, "removed": self.removed,
                "ignored": self.ignored, "ignored_directories": self.ignored_directories,
                "ignore_files": list(self.ignore_files), "ignore_truncated": self.ignore_truncated,
                "ignore_unusable": list(self.ignore_unusable), "ignored_memories": self.ignored_memories,
                "out_of_scope": self.out_of_scope, "unreadable": self.unreadable,
                "links": self.links, "special": self.special,
                "forgotten": self.forgotten, "empty": self.empty, "cut": self.cut,
                "listed_all": self.listed_all, "checked_for_missing": self.checked_for_missing,
                "changes": [{"path": c.path, "what": c.what} for c in self.changes],
                # Only when there is something to say, so a sync without the
                # code graph reads exactly as it did.
                **({"claims_closed": self.claims_closed} if self.claims_closed else {}),
                **({"claims_unread": True} if self.claims_unread else {})}

    def text(self) -> str:
        did = "synced" if self.applied else "would sync"
        lines = [f"{did} {self.marker}: {self.files_read} of {self.files_found} file(s) read"
                 + (" (capped)" if self.capped else "")
                 + f"; {self.added} added, {self.updated} updated, {self.unchanged} unchanged"
                 + (f"; {self.embeddings_reused} chunk embedding(s) reused" if self.embeddings_reused else "")]
        if self.ignored or self.ignored_directories:
            lines.append(f"{self.ignored} file(s) and {self.ignored_directories} directory(ies) left unread by "
                         f"{', '.join(self.ignore_files) or 'the ignore rules'}; pass --no-ignore to read them")
        if self.ignored_memories:
            lines.append(f"{self.ignored_memories} memory(ies) are for files the ignore rules now exclude; they were "
                         f"left alone, neither read nor removed")
        if self.ignore_truncated:
            lines.append("the ignore rules were read only as far as the bound allows; some of the tree's exclusions "
                         "were not read, so this run may have read what the tree said not to")
        if self.ignore_unusable:
            lines.append(f"{len(self.ignore_unusable)} ignore pattern(s) could not be read and were passed over: "
                         + "; ".join(self.ignore_unusable[:5]))
        if self.unreadable:
            lines.append(f"{self.unreadable} directory(ies) under the root could not be read, so "
                         f"this run cannot say what is gone and forgot nothing")
        if not self.checked_for_missing and not self.unreadable:
            lines.append("this walk stopped at the file cap, so it did not look for missing files")
        if self.removed and self.removing:
            lines.append(f"{self.forgotten} of {self.removed} missing file(s) forgotten")
        elif self.removed:
            lines.append(f"{self.removed} file(s) are gone from disk but not removed from memory; "
                         f"pass remove to forget them")
        if self.claims_closed:
            lines.append(f"{self.claims_closed} claim(s) closed that changed or removed files no longer make")
        if self.claims_unread:
            lines.append("some claims of a changed or removed file could not be examined and may still stand")
        if self.out_of_scope:
            lines.append(f"{self.out_of_scope} memory(ies) are out of scope for this run's "
                         f"suffixes and were left alone, not treated as gone")
        if self.special:
            lines.append(f"{self.special} path(s) matching the suffixes or named as a manifest are not ordinary files "
                         f"and were not opened")
        if self.links:
            lines.append(f"{self.links} symbolic link(s) were left alone: what a link points at "
                         f"is outside the root")
        if self.empty:
            lines.append(f"{self.empty} file(s) had nothing in them")
        if self.cut:
            lines.append(f"{self.cut} file(s) were longer than the byte limit and stored up to it")
        if not self.listed_all:
            lines.append(f"only the first {len(self.changes)} change(s) are listed; the counts are all of them")
        return "; ".join(lines)


@dataclass
class _Tally:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    empty: int = 0
    cut: int = 0
    #: Chunks of added and updated files whose vector the embedding cache
    #: answered: what this run did not pay the embedder for.
    embeddings_reused: int = 0
    changes: list[Change] = field(default_factory=list)
    listed_all: bool = True
    claims_closed: int = 0
    claims_unread: bool = False

    def claims(self, closed: Optional[int], unread: bool) -> None:
        """What a replace or removal closed. None means the store could
        not read the claims by episode: nothing closed, and said so."""
        if closed is None:
            self.claims_unread = True
        else:
            self.claims_closed += closed
        self.claims_unread = self.claims_unread or unread

    def saw(self, path: str, what: str) -> None:
        if len(self.changes) < MAX_LISTED:
            self.changes.append(Change(path, what))
        else:
            self.listed_all = False


def _wanted(suffixes: Sequence[str]) -> set[str]:
    return {suffix.lower() for suffix in suffixes}


#: A sync that reads code reads the project's manifests too, whatever
#: their suffix: `map` does, and the two commands walk one tree the
#: same way. A sync of notes alone (`--suffix .md`) leaves them.
_CODE_SUFFIXES = frozenset(suffix.lower() for suffix in (*PYTHON_SUFFIXES, *BRACE_SUFFIXES))


def _in_scope(here: pathlib.PurePath, wanted: set[str]) -> bool:
    """Whether a path **below the root** is one a sync reads: by its
    suffix, or as a package manifest when the sync reads code at all.

    The hidden-directory rule is about what is under the root — a
    repository means its source and not its ``.git``. It must be judged on
    the part below the root and never on the whole path: a root reached
    through a dot-segment (``~/.config/notes``, ``~/.claude/projects``, or
    the checkout this is developed in) would otherwise exclude its own
    entire tree, and the receipt would report ``files_found: 0`` with every
    file sitting on disk.
    """
    if any(part.startswith(".") or part == "__pycache__" for part in here.parts):
        return False
    if here.suffix.lower() in wanted:
        return True
    # By its path below the root, so `requirements/test.txt` is the
    # manifest `is_manifest` says it is, wherever the sync started.
    return bool(wanted & _CODE_SUFFIXES) and is_manifest(here.as_posix())


#: The namespace every key this writes begins with. A space is shared —
#: with hand-written records and with the document ``DirectorySync``, which
#: keeps its own managed identities — so the identities this owns are
#: separated by construction rather than by convention. Nothing outside a
#: sync writes this prefix, so nothing outside a sync can be replaced by
#: one.
KEYS = "scone.sync/1"


def sync_key(marker: str, path: str) -> str:
    """The identity a synced file is stored under: its marker and its path.

    Length-prefixed rather than separated, because a marker and a path are
    both text a caller chose and any separator could appear in either.
    ``map`` stores files under the same identity, so the two commands
    recognise each other's memories of a file rather than adding a second.
    """
    return f"{KEYS}:{len(marker)}:{marker}/{path}"


_key = sync_key


def default_marker(root: str | pathlib.Path) -> str:
    """The name a directory's memories are held under when none is given:
    its absolute path."""
    return str(pathlib.Path(root).resolve())


def _files(root: pathlib.Path, suffixes: Sequence[str],
           limit: int, ignore: "Ignore | None" = None) -> tuple[list[pathlib.Path], int, int, int, int, int, int]:
    """(files to read, how many there are, directories unread, links, special
    files, files the ignore rules skipped, directories they pruned).

    Walked explicitly rather than with ``rglob``, which swallows a
    ``PermissionError`` and returns what it could reach — indistinguishable
    from a smaller directory. A sync that cannot see part of a tree must
    know it, because the alternative is forgetting memory for files that
    are still on disk.
    """
    wanted = _wanted(suffixes)
    found: list[pathlib.Path] = []
    unreadable = links = special = ignored = pruned = 0
    stack = [root]
    while stack:
        here = stack.pop()
        try:
            entries = sorted(here.iterdir())
        except OSError:
            unreadable += 1
            continue
        for entry in entries:
            try:
                # A link is left alone, and counted. What it points at is
                # outside the root the caller named -- following one stores
                # content nobody asked for, under a path inside the root, so
                # nothing in the space would say where it really came from.
                if entry.is_symlink():
                    links += 1
                    continue
                directory = entry.is_dir()
                # "Not a directory" is not "a file". A named pipe matching
                # the suffixes would be opened and block until somebody
                # wrote to it, and a sync that hangs is worse than one that
                # counts wrongly: nothing reports it and nothing recovers.
                ordinary = directory or entry.is_file()
            except OSError:
                unreadable += 1
                continue
            below = entry.relative_to(root)
            if directory:
                if any(part.startswith(".") or part == "__pycache__" for part in below.parts):
                    continue
                if ignore is not None and ignore.ignored(below.as_posix(), directory=True):
                    pruned += 1
                    continue
                stack.append(entry)
            elif not ordinary:
                if _in_scope(below, wanted):
                    special += 1
            elif _in_scope(below, wanted):
                if ignore is not None and ignore.ignored(below.as_posix(), directory=False):
                    ignored += 1
                    continue
                found.append(entry)
    found.sort()
    return found[:limit], len(found), unreadable, links, special, ignored, pruned


async def sync_directory(
    engine: "MemoryEngine",
    space: str,
    root: str | pathlib.Path,
    *,
    marker: Optional[str] = None,
    suffixes: Sequence[str] = SUFFIXES,
    apply: bool = False,
    remove: bool = False,
    limit: int = MAX_FILES,
    max_bytes: int = MAX_BYTES,
    ignore: bool = True,
) -> SyncReceipt:
    """Bring ``space`` into step with ``root``, or say what that would do.
    ``ignore`` reads the tree's `.gitignore` and `.sconeignore` files and
    leaves what they exclude unread; off, the tree is read whole."""
    where = pathlib.Path(root)
    if not where.is_dir():
        raise InvalidInput(f"{root} is not a directory to sync")
    if not 1 <= limit <= MAX_FILES:
        raise InvalidInput(f"the file limit must be from 1 to {MAX_FILES}, not {limit}")
    if not 1 <= max_bytes <= 50_000_000:
        raise InvalidInput(f"the byte limit must be from 1 to 50000000, not {max_bytes}")
    if not suffixes:
        raise InvalidInput("a sync needs at least one suffix to look for")
    name = marker if marker is not None else default_marker(where)
    if not name.strip():
        raise InvalidInput("a marker cannot be blank: it is what names this directory's memories")

    rules = Ignore.load(where) if ignore else None
    reading, there, unreadable, links, special, skipped, pruned = _files(where, suffixes, limit, rules)
    known = {episode.source: episode
             for episode in await engine.episodes(space, {"sync": name})
             if episode.source}
    # The refusal comes before any write or any forget, so a wrong path
    # cannot take the space with it.
    if remove and not there and known:
        raise InvalidInput(
            f"{root} matched no files, and {len(known)} memory(ies) are held for {name!r}. "
            f"Refusing to forget them: a directory that has become empty is far more often a "
            f"wrong or unmounted path than a repository whose every file was deleted. Sync "
            f"without remove to see the plan, or delete the space if that is what you mean.")

    tally = _Tally()
    # Relative imports are followed to files this tree holds: the ones
    # walked now and the ones the marker already remembers.
    resolve = file_resolver({*(path.relative_to(where).as_posix() for path in reading), *known})
    seen: set[str] = set()
    for path in reading:
        here = path.relative_to(where).as_posix()
        seen.add(here)
        # One byte past the limit: enough to know the file is longer
        # without reading the rest of it. Reading a gigabyte to keep a
        # kilobyte is how a sync of a large tree exhausts a machine.
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            tally.cut += 1
            tally.saw(here, "cut")
        text = raw[:max_bytes].decode("utf-8", errors="replace")
        if not text.strip():
            tally.empty += 1
            tally.saw(here, "empty")
            continue
        held = known.get(here)
        if held is not None and held.content == text:
            tally.unchanged += 1
            tally.saw(here, "unchanged")
            continue
        what = "updated" if held is not None else "added"
        if apply:
            # The key carries the marker. Without it two directories synced
            # into one space share an identity for every filename they have
            # in common, and `replace` forgets the other's episode to store
            # this one -- a deletion nobody asked for and the receipt does
            # not mention. The marker's length goes in front so no marker
            # and path can be split two ways; a separator alone could be,
            # since both halves are text a caller chose.
            done = await engine.replace(space, Record(content=text, kind="file", source=here,
                                                      dedup_key=_key(name, here),
                                                      metadata={"sync": name}), resolve=resolve)
            tally.claims(done.claims_closed, done.claims_unread)
            tally.embeddings_reused += done.added.embeddings_reused
        setattr(tally, what, getattr(tally, what) + 1)
        tally.saw(here, what)

    # Two different reasons an episode's file is not in `seen`, and only
    # one of them means it is gone.
    #
    # A file the cap stopped us reading is unread. A file this run's
    # suffixes no longer select was never looked for. Neither is missing,
    # and forgetting on either would delete memory because a flag changed
    # rather than because anything left the disk.
    # Three ways a file leaves the walk without leaving the disk: the cap
    # stopped us, the suffixes no longer select it, or a directory holding
    # it could not be read. Only a walk clear of all three can say what is
    # gone.
    whole = len(reading) == there and not unreadable
    wanted = _wanted(suffixes)
    out_of_scope = [path for path in known if not _in_scope(pathlib.PurePosixPath(path), wanted)]
    # A memory for a file the ignore rules now exclude is not for a file
    # that is gone: the rules narrowed the walk, as a flag would, and a
    # narrowed walk is not a deleted file.
    ignored_memories = ([path for path in known if path not in seen and path not in set(out_of_scope)
                         and rules.ignored(path, directory=False)] if rules is not None else [])
    missing = ([path for path in known
                if path not in seen and path not in set(out_of_scope) and path not in set(ignored_memories)]
               if whole else [])
    forgotten = 0
    for gone in missing:
        tally.saw(gone, "missing")
        if apply and remove:
            await engine.forget(space, known[gone].episode_id)
            forgotten += 1
            if engine.code_graph:
                # The file is gone, so nothing it said is stated any more.
                retired = await engine.close_unstated(space, known[gone].episode_id,
                                                      reason=f"{gone} was removed", kind="source_removed")
                tally.claims(retired.closed, retired.unread)

    receipt = SyncReceipt(
        root=str(where), marker=name, space=space, applied=apply, removing=remove,
        files_found=there, files_read=len(reading), added=tally.added, updated=tally.updated,
        ignored=skipped, ignored_directories=pruned, ignore_files=rules.files if rules is not None else (),
        ignore_truncated=rules.truncated if rules is not None else False,
        ignore_unusable=rules.unusable if rules is not None else (), ignored_memories=len(ignored_memories),
        embeddings_reused=tally.embeddings_reused,
        unchanged=tally.unchanged, removed=len(missing), forgotten=forgotten,
        out_of_scope=len(out_of_scope), unreadable=unreadable, links=links,
        special=special,
        empty=tally.empty, cut=tally.cut, changes=tuple(tally.changes),
        listed_all=tally.listed_all, checked_for_missing=whole,
        claims_closed=tally.claims_closed, claims_unread=tally.claims_unread)
    if apply:
        # Only a sync that wrote can be a last success; a plan succeeded
        # at nothing and must not look like a completed sync.
        await engine._emit(space, "sync.completed", receipt.record())
    return receipt


async def last_sync(engine: "MemoryEngine", space: str, marker: str) -> Optional[dict]:
    """The record of the last sync that wrote for this marker, or None if
    none ever did. Raises :class:`NoEventLog` when nothing keeps the
    record, because a store that cannot tell must not answer "never".

    Only an applied sync is recorded: a plan succeeded at nothing.
    """
    if engine.events is None:
        raise NoEventLog(
            f"this engine has no event log, so when {marker!r} last synced is not recorded "
            f"anywhere. Attach an event log to keep it.")
    for event in await engine.events.query(space, kind="sync.completed", limit=MAX_LISTED):
        if event.payload.get("marker") == marker:
            return dict(event.payload) | {"at": event.ts}
    return None
