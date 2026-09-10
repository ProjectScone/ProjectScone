"""Prove our sources are our own.

A reference tree may be read for its behaviour and never for its text.
This finds runs of identical code lines shared between what we ship and
what we read, so a lifted block is a failing test rather than a matter of
trust. Shared idiom is not a finding: an import block, a line of
boilerplate or a repeated bracket says nothing about authorship, while
eight consecutive lines of the same working code say everything.

Run it over the tree:

    python -m scone_memory.testing.originality --ours packages/memory/src
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

#: Suffixes worth comparing. Prose and data say nothing about authorship.
CODE_SUFFIXES = (".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".mjs")
#: Directories that hold nobody's writing: caches, dependencies, builds.
SKIP_DIRS = frozenset({
    "__pycache__", "node_modules", ".git", ".venv", "venv", "target", "build",
    "dist", ".next", ".turbo", ".mypy_cache", ".pytest_cache", ".ruff_cache",
})
#: A window every line of which is at most this long is boilerplate.
SHORT_LINE = 20
#: A window with fewer distinct lines than this is repetition, not design.
MIN_DISTINCT = 4


def normalise(source: str) -> list[tuple[int, str]]:
    """Code lines with their 1-based source line numbers: comments and
    blanks dropped, runs of whitespace collapsed, so indentation and
    spacing differences never hide a copy nor invent one."""
    out: list[tuple[int, str]] = []
    for number, raw in enumerate(source.splitlines(), start=1):
        text = " ".join(raw.split())
        if not text or text.startswith("#") or text.startswith("//"):
            continue
        out.append((number, text))
    return out


def sources(root: Path, suffixes: Sequence[str] = CODE_SUFFIXES) -> list[Path]:
    """Every code file under ``root``, caches and dependencies skipped."""
    root = Path(root)
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if SKIP_DIRS.intersection(path.parts):
            continue
        found.append(path)
    return found


def _idiom(window: Sequence[str]) -> bool:
    """Whether a window is shared idiom rather than shared design."""
    if all(line.startswith(("import ", "from ", "use ", "pub use ")) for line in window):
        return True
    if all(len(line) <= SHORT_LINE for line in window):
        return True
    return len(set(window)) < MIN_DISTINCT


@dataclass(frozen=True)
class Overlap:
    """One run of identical code in two files."""

    ours: Path
    our_line: int
    theirs: Path
    their_line: int
    lines: int
    text: tuple[str, ...]

    def report(self) -> str:
        return (f"{self.ours}:{self.our_line} matches {self.theirs}:{self.their_line} "
                f"for {self.lines} lines:\n  " + "\n  ".join(self.text[:12]))


def _read(path: Path) -> list[tuple[int, str]]:
    try:
        return normalise(path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return []


def scan(ours: Iterable[Path], theirs: Iterable[Path], window: int = 8) -> list[Overlap]:
    """Runs of ``window`` or more identical code lines shared between the
    two sets, longest first. Each of our runs is reported once, at the
    first place it was read from, so one lifted block is one finding."""
    reference: dict[Path, list[tuple[int, str]]] = {}
    index: dict[tuple[str, ...], list[tuple[Path, int]]] = {}
    for path in theirs:
        lines = _read(path)
        if len(lines) < window:
            continue
        reference[path] = lines
        for start in range(len(lines) - window + 1):
            key = tuple(text for _, text in lines[start:start + window])
            index.setdefault(key, []).append((path, start))

    found: list[Overlap] = []
    for path in ours:
        mine = _read(path)
        start = 0
        while start + window <= len(mine):
            key = tuple(text for _, text in mine[start:start + window])
            hits = None if _idiom(key) else index.get(key)
            if not hits:
                start += 1
                continue
            other, at = hits[0]
            theirs_lines = reference[other]
            length = window
            while (start + length < len(mine) and at + length < len(theirs_lines)
                   and mine[start + length][1] == theirs_lines[at + length][1]):
                length += 1
            found.append(Overlap(
                ours=path, our_line=mine[start][0], theirs=other, their_line=theirs_lines[at][0],
                lines=length, text=tuple(text for _, text in mine[start:start + length]),
            ))
            start += length
    return sorted(found, key=lambda o: (-o.lines, str(o.ours), o.our_line))


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="report code we ship that is identical to a reference tree")
    parser.add_argument("--ours", default="packages/memory/src", help="tree to check (default packages/memory/src)")
    parser.add_argument("--reference", default="reference", help="tree that was read (default reference)")
    parser.add_argument("--window", type=int, default=8, help="identical code lines that count as a copy (default 8)")
    args = parser.parse_args(argv)

    reference = Path(args.reference)
    if not reference.exists():
        print(f"no reference tree at {reference}: nothing to compare against")
        return 0
    found = scan(sources(Path(args.ours)), sources(reference), window=args.window)
    for overlap in found:
        print(overlap.report())
    print(f"{len(found)} run(s) of {args.window}+ identical code lines")
    return 1 if found else 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
