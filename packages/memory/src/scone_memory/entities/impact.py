"""What a change reaches: the declarations a diff touches, and what rests on them.

``graph affected`` answers "what rests on this symbol"; a pull request
touches lines in many files, and the question people bring to it is the
same one asked of the diff as a whole. This reads a unified diff -- what
``git diff`` writes -- finds what each hunk touches by re-reading the
changed file at the root, which holds the tree **after** the change
(the diff's ``b`` side, whose line numbers the hunks give), with the
declaration reader that cuts files into chunks, and asks the graph what
rests on each touched thing, once, nearest first.

What a hunk touches is named at two levels, because the graph holds
edges at two levels: a declaration (``pkg/store.py:Shelf.keep``) for a
call that was bound to it, and the file, spelt as a path and, for Python,
as the module an import names (``pkg.store``), for every import of it. A
change outside any declaration -- an import line, a constant -- touches
the file alone. A file the diff names that the root does not hold is
taken as removed, and asked about as a file. A file the graph does not
hold is said so; a bound that bites is said so; and an empty answer means
nothing *here* rests on the change, never that nothing does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal, Optional, Sequence

from ..core.errors import InvalidInput
from ..ingestion.code import code_language, declarations
from .affected import MAX_HOPS, MAX_NAME, affected

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

#: Files of one diff examined; past it the rest are counted, not read.
MAX_FILES = 200
#: Things asked of the graph for one diff, declarations and files together.
MAX_TARGETS = 400
#: Bytes of diff read.
MAX_DIFF_BYTES = 4_000_000
#: Dependants listed in one answer; the rest are counted.
MAX_LISTED = 1_000

_HEADER = re.compile(r'^diff --git (?P<a>"a/(?:[^"\\]|\\.)*"|a/.+?) (?P<b>"b/(?:[^"\\]|\\.)*"|b/.+)$')
_OCTAL = re.compile(r"\\([0-7]{3})")
_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "a": "\a", "b": "\b", "f": "\f", "r": "\r", "v": "\v"}


def _unquote(spelt: str) -> str:
    """A path as git wrote it: a name with a non-ASCII or special
    character comes quoted, with octal escapes for its bytes."""
    if not (spelt.startswith('"') and spelt.endswith('"')):
        return spelt
    inner = spelt[1:-1]
    raw = bytearray()
    i = 0
    while i < len(inner):
        if inner[i] == "\\" and i + 1 < len(inner):
            octal = _OCTAL.match(inner, i)
            if octal:
                raw.append(int(octal.group(1), 8))
                i += 4
                continue
            raw.extend(_ESCAPES.get(inner[i + 1], inner[i + 1]).encode("utf-8"))
            i += 2
            continue
        raw.extend(inner[i].encode("utf-8"))
        i += 1
    return raw.decode("utf-8", errors="replace")
_HUNK = re.compile(r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@")


@dataclass(frozen=True)
class ChangedFile:
    """One file of the diff, and the lines of its new text the hunks cover.
    A removed file has no new text and no lines."""
    path: str
    removed: bool
    #: Half-open line ranges of the file after the change, one per hunk. A
    #: hunk that only removes lines is taken to touch the line it removes
    #: before, so a deletion between two declarations touches both.
    lines: tuple[tuple[int, int], ...]


def parse_diff(text: str) -> tuple[ChangedFile, ...]:
    """The files a unified diff changes, in diff order, and where."""
    if not isinstance(text, str):
        raise InvalidInput("a diff is text")
    if len(text.encode("utf-8")) > MAX_DIFF_BYTES:
        raise InvalidInput(f"a diff is read up to {MAX_DIFF_BYTES} bytes")
    files: list[ChangedFile] = []
    path: Optional[str] = None
    removed = False
    lines: list[tuple[int, int]] = []

    def close() -> None:
        if path is not None:
            files.append(ChangedFile(path, removed, () if removed else tuple(lines)))

    for line in text.splitlines():
        header = _HEADER.match(line)
        if header:
            close()
            path, removed, lines = _unquote(header.group("b"))[2:], False, []
            continue
        if line.startswith("+++ "):
            named = _unquote(line[4:].split("\t", 1)[0])
            if named == "/dev/null":
                removed = True
            elif named.startswith("b/") and path is None:
                path = named[2:]
            continue
        if line.startswith("rename to ") and path is not None:
            path = _unquote(line[len("rename to "):])
            continue
        hunk = _HUNK.match(line)
        if hunk and path is not None:
            start = int(hunk.group("new_start"))
            count = int(hunk.group("new_count") or "1")
            if count == 0:
                start, count = max(start, 1), 2
            lines.append((start, start + count))
    close()
    if not files:
        raise InvalidInput("no file changes in the diff: it reads what `git diff` writes")
    return tuple(files)


@dataclass(frozen=True)
class Touched:
    """One thing the diff touches, and what the graph said rests on it."""
    path: str
    #: The declaration, or None for the file itself.
    name: Optional[str]
    #: How the graph was asked: the label given to ``affected``.
    asked: str
    #: ``unasked``: a name longer than the graph takes, so nothing was asked.
    status: Literal["found", "nothing", "unknown", "ambiguous", "unasked"]
    reached: int
    lines: tuple[int, int] = (0, 0)
    #: Dependants the graph's own answer left out of its list, and whether
    #: it stopped at its depth with more to follow -- a bound that bit
    #: inside one answer is a bound that bit this one.
    not_listed: int = 0
    stopped_at_depth: bool = False


@dataclass(frozen=True)
class Reaches:
    label: str
    #: The fewest hops from anything the diff touches.
    depth: int
    through: str
    #: What it rests on, as the graph was asked for it.
    via: str


@dataclass(frozen=True)
class Impact:
    files: int
    files_examined: int
    files_removed: int
    #: Files at the root the reader has no declaration language for; asked
    #: about as files only.
    files_unread: int
    targets: tuple[Touched, ...]
    targets_not_asked: int
    reached: tuple[Reaches, ...]
    not_listed: int
    by_depth: dict[int, int] = field(default_factory=dict)
    #: Any answer whose read of the graph was itself truncated.
    partial_read: bool = False
    #: Any answer that left dependants unlisted or stopped at its depth with
    #: more to follow, so the union here is not the whole of what rests on
    #: the change.
    truncated: bool = False
    #: Files the diff names outside the root, or by an absolute path; not read.
    files_outside: int = 0
    mode: str = "current"
    at: str = ""
    why: str = ""

    def record(self) -> dict[str, object]:
        return {"files": self.files, "files_examined": self.files_examined, "files_removed": self.files_removed,
                "files_unread": self.files_unread, "targets_not_asked": self.targets_not_asked,
                "targets": [[t.path, t.name, t.asked, t.status, t.reached, list(t.lines), t.not_listed, t.stopped_at_depth]
                            for t in self.targets],
                "reached": len(self.reached), "not_listed": self.not_listed,
                "by_depth": {str(k): v for k, v in sorted(self.by_depth.items())},
                "partial_read": self.partial_read, "truncated": self.truncated, "files_outside": self.files_outside,
                "mode": self.mode, "at": self.at, "why": self.why,
                "entities": [[r.label, r.depth, r.through, r.via] for r in self.reached]}


def _module_names(path: str) -> tuple[str, ...]:
    """The names an import may spell a Python file by: the dotted module,
    and for a package's __init__, the package."""
    if not path.endswith(".py"):
        return ()
    stem = path[:-3].replace("\\", "/")
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return (stem.replace("/", "."),) if stem else ()


def _touched_names(content: str, path: str, ranges: Sequence[tuple[int, int]]) -> tuple[tuple[Optional[str], tuple[int, int]], ...]:
    """The innermost declaration under each changed line, once each in
    source order, and None where a changed line is outside every one."""
    found = declarations(content, language=code_language(path))
    named: dict[Optional[str], tuple[int, int]] = {}
    for start, end in ranges:
        for line in range(start, end):
            inner = None
            for one in found:
                if one.first_line <= line <= one.last_line and (
                        inner is None or (one.last_line - one.first_line) < (inner.last_line - inner.first_line)):
                    inner = one
            if inner is None:
                named.setdefault(None, (start, end))
            else:
                named.setdefault(inner.name, (inner.first_line, inner.last_line))
    return tuple(named.items())


async def impact(engine: "MemoryEngine", space: str, diff: str, *, root: str | Path = ".",
                 max_hops: int = 4, limit: int = MAX_LISTED, max_files: int = MAX_FILES,
                 max_targets: int = MAX_TARGETS) -> Impact:
    """Everything in this graph that rests on what ``diff`` changes."""
    if not 1 <= max_hops <= MAX_HOPS:
        raise InvalidInput(f"max_hops is between 1 and {MAX_HOPS}")
    if not 1 <= limit <= MAX_LISTED:
        raise InvalidInput(f"limit is between 1 and {MAX_LISTED}")
    if not 1 <= max_files <= MAX_FILES or not 1 <= max_targets <= MAX_TARGETS:
        raise InvalidInput("max_files and max_targets are positive and within their bounds")
    changed = parse_diff(diff)
    base = Path(root)
    if not base.is_dir():
        raise InvalidInput(f"root {str(root)!r} is not a directory; it should hold the tree after the change")
    resolved = base.resolve()
    asks: list[tuple[str, Optional[str], str, tuple[int, int]]] = []
    removed = unread = outside = 0
    for file in changed[:max_files]:
        target = base / file.path
        if Path(file.path).is_absolute() or not target.resolve().is_relative_to(resolved):
            # A diff names paths under its own tree; one that points
            # elsewhere is not read, whatever it says it is.
            outside += 1
            continue
        if file.removed or not target.is_file():
            removed += 1
            names: list[tuple[Optional[str], tuple[int, int]]] = [(None, (0, 0))]
        elif code_language(file.path) is None:
            unread += 1
            names = [(None, (0, 0))]
        else:
            try:
                content = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                unread += 1
                content = ""
            names = list(_touched_names(content, file.path, file.lines)) if content else [(None, (0, 0))]
        for name, span in names:
            if name is not None:
                asks.append((file.path, name, f"{file.path}:{name}", span))
        asks.append((file.path, None, file.path, (0, 0)))
        for module in _module_names(file.path):
            asks.append((file.path, None, module, (0, 0)))
    targets: list[Touched] = []
    best: dict[str, Reaches] = {}
    by_depth: dict[int, int] = {}
    partial = truncated = False
    mode, at = "current", ""
    for path, name, asked, span in asks[:max_targets]:
        if len(asked.encode()) > MAX_NAME:
            # The graph takes a name up to its bound; a longer one is
            # listed as unasked rather than ending the whole answer.
            targets.append(Touched(path, name, asked, "unasked", 0, span))
            continue
        # The walk keeps its own bound so the shape of the answer is whole;
        # ``limit`` cuts the list this answer writes, and says so. What the
        # walk itself left out is carried up, never dropped on the floor.
        blast = await affected(engine, space, asked, max_hops=max_hops)
        partial = partial or blast.partial_read
        truncated = truncated or bool(blast.not_listed) or blast.stopped_at_depth
        mode, at = blast.mode, blast.at or at
        targets.append(Touched(path, name, asked, blast.status, len(blast.reached), span, blast.not_listed, blast.stopped_at_depth))
        for dependant in blast.reached:
            kept = best.get(dependant.label)
            if kept is None or dependant.depth < kept.depth:
                best[dependant.label] = Reaches(dependant.label, dependant.depth, dependant.through, asked)
    touched_labels = {t.asked for t in targets}
    reached = sorted((r for r in best.values() if r.label not in touched_labels), key=lambda r: (r.depth, r.label))
    for reach in reached:
        by_depth[reach.depth] = by_depth.get(reach.depth, 0) + 1
    listed = tuple(reached[:limit])
    found = sum(1 for t in targets if t.status == "found")
    unknown = sum(1 for t in targets if t.status == "unknown")
    unasked = sum(1 for t in targets if t.status == "unasked")
    why = (f"{len(changed)} file(s) in the diff, {min(len(changed), max_files)} examined, "
           f"{len(targets)} thing(s) asked about ({found} with dependants here, {unknown} not in this graph); "
           f"{len(reached)} thing(s) rest on the change within {max_hops} hop(s)")
    if outside:
        why += f"; {outside} file(s) outside the root, not read"
    if unasked:
        why += f"; {unasked} name(s) longer than the graph takes, not asked"
    if len(changed) > max_files:
        why += f"; {len(changed) - max_files} file(s) not examined (max_files)"
    if len(asks) > max_targets:
        why += f"; {len(asks) - max_targets} thing(s) not asked about (max_targets)"
    if len(reached) > limit:
        why += f"; {len(reached) - limit} not listed (limit)"
    if truncated:
        why += "; an answer behind this one left dependants unlisted or stopped at its depth, so this is not the whole of it"
    why += ("; an empty answer means nothing in this graph rests on the change, not that nothing does"
            + ("; a read behind an answer was itself truncated" if partial else ""))
    return Impact(files=len(changed), files_examined=min(len(changed), max_files), files_removed=removed,
                  files_unread=unread, targets=tuple(targets), targets_not_asked=max(0, len(asks) - max_targets),
                  reached=listed, not_listed=max(0, len(reached) - limit), by_depth=by_depth,
                  partial_read=partial, truncated=truncated, files_outside=outside, mode=mode, at=at, why=why)
