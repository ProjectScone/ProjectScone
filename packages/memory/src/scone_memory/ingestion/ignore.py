"""What a tree says not to read: its `.gitignore` files, and a
`.sconeignore` beside them.

A repository walked whole is a repository with its `node_modules`,
`build`, `dist`, `target` and `vendor` in it: thousands of files nobody
wrote, embedded, remembered and put in the graph ahead of the source
they were asked about. The tree already says what is not its own -- git
reads `.gitignore` in every directory -- and the leading code-graph tool
respects it automatically. So do `map` and `sync`: every `.gitignore`
below the root is read, its patterns hold for the directory it sits in
and everything under it, and a `.sconeignore` in any directory is read
after them.

The rules are git's, written here rather than borrowed: a blank line or
a `#` comment says nothing; `!` re-includes; a trailing `/` matches only
a directory; a pattern with a slash anywhere but its end is anchored to
its file's directory, one without matches at any depth below it; `*`
and `?` never cross a slash and `**` does; `[...]` is a class; `\\`
escapes. Within one kind of file the last matching pattern wins, and a
deeper file's patterns come after a shallower one's. A `.sconeignore`
can only exclude more: what `.gitignore` excludes stays excluded whatever
it says, and a file under an excluded directory is never re-included,
as in git.

Bounded and disclosed: at most `MAX_IGNORE_FILES` files and `MAX_RULES`
patterns are read, and when that bound bit the record and the walk's
receipt say so; a pattern that cannot be read is counted; the receipt
says how many files the rules skipped and which ignore files were read;
`--no-ignore` reads the tree whole. One thing that is git's and not
here: git matches case-insensitively where `core.ignorecase` is set (a
Mac's default file system); these rules match as written.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import posixpath
import re
from typing import Callable, Optional, Sequence

from .mcp_config import WALKED_DOT_NAMES, is_mcp_config

#: Read in every directory, in this order: the tree's own exclusions,
#: then ours.
GIT_IGNORE = ".gitignore"
OWN_IGNORE = ".sconeignore"
#: Ignore files read before the loader stops and says so.
MAX_IGNORE_FILES = 500
#: Patterns kept before the loader stops and says so.
MAX_RULES = 20_000
#: Directories never walked whatever the rules: a repository's own
#: `.git`, editor state, byte-code.
ALWAYS_SKIPPED = ("__pycache__",)


@dataclass(frozen=True)
class Rule:
    """One pattern, as written and as matched."""

    pattern: str
    negated: bool
    directory_only: bool
    #: The directory the rule's file sits in, relative to the root
    #: ("" at the root). The rule holds for that directory and below.
    base: str
    source: str
    regex: re.Pattern[str]

    def matches(self, relative: str, *, directory: bool) -> bool:
        if self.directory_only and not directory:
            return False
        if self.base:
            if not relative.startswith(self.base + "/"):
                return False
            relative = relative[len(self.base) + 1:]
        return self.regex.fullmatch(relative) is not None


def _translate(glob: str) -> str:
    """A gitignore glob as a regular expression over a slash-separated
    path: `*` and `?` stop at a slash, `**` does not, a class is a class,
    a backslash escapes what follows."""
    out: list[str] = []
    i, n = 0, len(glob)
    while i < n:
        c = glob[i]
        if c == "\\" and i + 1 < n:
            out.append(re.escape(glob[i + 1]))
            i += 2
        elif c == "*":
            if glob.startswith("**", i):
                after = glob[i + 2:i + 3]
                before = glob[i - 1:i] if i else ""
                if (before in ("", "/")) and after == "/":
                    out.append("(?:.*/)?")  # `**/` — zero or more directories
                    i += 3
                    continue
                if before == "/" and after == "":
                    out.append(".*")  # trailing `/**` — everything inside
                    i += 2
                    continue
                # `**` not beside a slash: as in git, ordinary asterisks,
                # which never cross a slash.
                out.append("[^/]*")
                i += 2
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            close = _class_end(glob, i)
            if close == -1:
                out.append(re.escape(c))
                i += 1
                continue
            body = glob[i + 1:close]
            if body.startswith("!"):
                body = "^" + body[1:]
            for posix, python in _POSIX_CLASSES.items():
                body = body.replace(posix, python)
            out.append("[" + body.replace("\\", "\\\\") + "]")
            i = close + 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


#: POSIX bracket expressions git accepts, as Python spells them.
_POSIX_CLASSES = {"[:alpha:]": "a-zA-Z", "[:digit:]": "0-9", "[:alnum:]": "a-zA-Z0-9", "[:upper:]": "A-Z",
                  "[:lower:]": "a-z", "[:space:]": " \\t\\n\\r\\f\\v", "[:punct:]": "!-/:-@\\[-`{-~",
                  "[:xdigit:]": "0-9A-Fa-f", "[:blank:]": " \\t"}


def _class_end(glob: str, start: int) -> int:
    """Where the class opened at ``start`` closes: a `]` first in the
    class (or after `!`/`^`) is a member, and a POSIX `[:name:]` inside
    is skipped over."""
    i = start + 1
    if i < len(glob) and glob[i] in "!^":
        i += 1
    if i < len(glob) and glob[i] == "]":
        i += 1
    while i < len(glob):
        if glob.startswith("[:", i):
            end = glob.find(":]", i + 2)
            if end == -1:
                return -1
            i = end + 2
            continue
        if glob[i] == "]":
            return i
        i += 1
    return -1


def _trimmed(line: str) -> str:
    """Trailing spaces are not part of a pattern unless escaped: a space
    behind an odd number of backslashes stays."""
    end = len(line)
    while end and line[end - 1] == " ":
        slashes = 0
        while end - 2 - slashes >= 0 and line[end - 2 - slashes] == "\\":
            slashes += 1
        if slashes % 2 == 1:
            break
        end -= 1
    return line[:end]


def parse_rules(text: str, *, base: str = "", source: str = GIT_IGNORE,
                unusable: Optional[list[str]] = None) -> list[Rule]:
    """The rules one ignore file holds, in the order they are written. A
    pattern that cannot be read as one is appended to ``unusable`` when
    that list is given, so the reader can say a rule was passed over."""
    rules: list[Rule] = []
    for raw in text.splitlines():
        line = _trimmed(raw.rstrip("\r"))
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        negated = line.startswith("!")
        if negated or line.startswith("\\!") or line.startswith("\\#"):
            line = line[1:] if negated else line[1:]
        directory_only = line.endswith("/") and not line.endswith("\\/")
        if directory_only:
            line = line[:-1]
        if not line:
            continue
        anchored = "/" in line.lstrip("/") or line.startswith("/")
        line = line.lstrip("/")
        body = _translate(line)
        expression = body if anchored else "(?:.*/)?" + body
        try:
            regex = re.compile(expression)
        except re.error:
            if unusable is not None:
                unusable.append(f"{source}: {raw.strip()}")
            continue
        rules.append(Rule(raw.strip(), negated, directory_only, base, source, regex))
    return rules


def _decide(rules: Sequence[Rule], relative: str, *, directory: bool) -> Optional[bool]:
    """Whether the last matching rule ignores the path, or None when none matches."""
    answer: Optional[bool] = None
    for rule in rules:
        if rule.matches(relative, directory=directory):
            answer = not rule.negated
    return answer


class Ignore:
    """Every ignore file under a root, read once, asked per path."""

    def __init__(self, git: Sequence[Rule] = (), own: Sequence[Rule] = (), *, files: Sequence[str] = (),
                 truncated: bool = False, unusable: Sequence[str] = (), links: int = 0) -> None:
        self.git = tuple(git)
        self.own = tuple(own)
        self.files = tuple(files)
        #: An ignore file or a pattern past the bound was left unread, so
        #: the tree's exclusions are not all known.
        self.truncated = truncated
        #: Patterns that could not be read as one, by file.
        self.unusable = tuple(unusable)
        #: Ignore files that were symbolic links, left unread and counted.
        self.links = links
        self._known: dict[tuple[str, bool], bool] = {}

    def ignored(self, relative: str, *, directory: bool = False) -> bool:
        """Whether a path below the root (posix, relative) is left unread:
        it, or any directory above it, is excluded. A `.sconeignore` can
        exclude what `.gitignore` allows and never the other way round."""
        relative = relative.strip("/")
        if not relative:
            return False
        key = (relative, directory)
        if key in self._known:
            return self._known[key]
        parent = posixpath.dirname(relative)
        if parent and self.ignored(parent, directory=True):
            self._known[key] = True
            return True
        answer = bool(_decide(self.git, relative, directory=directory)) or bool(
            _decide(self.own, relative, directory=directory))
        self._known[key] = answer
        return answer

    def record(self) -> dict[str, object]:
        return {"files": list(self.files), "rules": len(self.git) + len(self.own), "truncated": self.truncated,
                "unusable": list(self.unusable), "links": self.links}

    @classmethod
    def load(cls, root: str | Path, *, git_name: str = GIT_IGNORE, own_name: str = OWN_IGNORE) -> "Ignore":
        """Read the ignore files a walk from ``root`` would meet, top down,
        never descending into a directory the rules so far exclude."""
        top = Path(root)
        git: list[Rule] = []
        own: list[Rule] = []
        files: list[str] = []
        unusable: list[str] = []
        truncated = False
        links = 0
        partial = cls()
        stack = [Path("")]
        while stack:
            here = stack.pop()
            base = here.as_posix() if str(here) != "." else ""
            for name, into in ((git_name, git), (own_name, own)):
                path = top / here / name
                try:
                    if path.is_symlink():
                        links += 1
                        continue
                    if not path.is_file():
                        continue
                    # The bound is said only when a file is actually left
                    # unread, not when the last one fits exactly.
                    if len(files) >= MAX_IGNORE_FILES:
                        truncated = True
                        break
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                source = f"{base}/{name}" if base else name
                files.append(source)
                read = parse_rules(text, base=base, source=source, unusable=unusable)
                if len(git) + len(own) + len(read) > MAX_RULES:
                    read = read[: max(0, MAX_RULES - len(git) - len(own))]
                    truncated = True
                into.extend(read)
                partial = cls(git, own, files=files, truncated=truncated)
            if truncated:
                break
            try:
                entries = sorted(os.scandir(top / here), key=lambda entry: entry.name, reverse=True)
            except OSError:
                continue
            for entry in entries:
                try:
                    if entry.is_symlink() or not entry.is_dir():
                        continue
                except OSError:
                    continue
                below = here / entry.name
                relative = below.as_posix()
                if entry.name.startswith(".") or entry.name in ALWAYS_SKIPPED:
                    continue
                if partial.ignored(relative, directory=True):
                    continue
                stack.append(below)
        return cls(git, own, files=files, truncated=truncated, unusable=unusable, links=links)


@dataclass(frozen=True)
class Walk:
    """What a walk found and what it left: files in order, and the counts
    a receipt needs."""

    files: tuple[Path, ...]
    #: Files the suffixes selected that the ignore rules skipped, and the
    #: directories pruned by them (with an unknown number of files each).
    ignored_files: int
    ignored_directories: int
    #: Directories that could not be read. Each hides an unknown number.
    unreadable: int
    links: int


def walk_files(root: str | Path, *, keep: Callable[[Path], bool], ignore: Optional[Ignore] = None) -> Walk:
    """Every ordinary file under ``root`` that ``keep`` selects, in sorted
    order, skipping dot directories, byte-code, symbolic links and what
    the ignore rules exclude. Explicit rather than ``rglob``, which
    swallows a permission error and returns what it could reach."""
    top = Path(root)
    found: list[Path] = []
    ignored_files = ignored_directories = unreadable = links = 0
    stack = [top]
    while stack:
        here = stack.pop()
        try:
            entries = sorted(here.iterdir())
        except OSError:
            unreadable += 1
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    links += 1
                    continue
                directory = entry.is_dir()
                ordinary = directory or entry.is_file()
            except OSError:
                unreadable += 1
                continue
            relative = entry.relative_to(top).as_posix()
            # A dot-name is skipped whether directory or file, as `sync`
            # skips it: the two commands walk one tree the same way. The
            # exceptions are the MCP configurations, which live in dot-names
            # by every tool's convention, and the directories that hold one;
            # from those directories only the configuration is read.
            if (entry.name.startswith(".") and entry.name.lower() not in WALKED_DOT_NAMES) or entry.name in ALWAYS_SKIPPED:
                continue
            dotted = any(part.startswith(".") for part in relative.split("/"))
            if dotted and not directory and not is_mcp_config(relative):
                continue
            if directory:
                if ignore is not None and ignore.ignored(relative, directory=True):
                    ignored_directories += 1
                    continue
                stack.append(entry)
            elif ordinary and keep(entry):
                if ignore is not None and ignore.ignored(relative, directory=False):
                    ignored_files += 1
                    continue
                found.append(entry)
    found.sort()
    return Walk(tuple(found), ignored_files, ignored_directories, unreadable, links)
