"""Cut code where its declarations are, not every so many characters.

A source file chunked by length puts a function's head in one chunk and
its body in the next, so recall answers "where is this done?" with half a
function and a citation that names no function at all. This cuts at
declaration boundaries instead: a function or a class is one span when it
fits, several line-aligned spans when it does not, and the spans of
neighbouring small declarations share one chunk up to the target.

Two ways of finding declarations, and no new dependency for either.
Python is parsed with ``ast``, so the spans are exact: the ``def`` line
with its decorators and the comment lines written directly above it,
through the last line of the body. Brace languages are read by a header
line and a brace count, which is a guess a parser would not need to make:
a brace inside a template literal that spans lines can be counted wrongly,
and the result is a chunk boundary in the wrong place, never a changed
byte. Anything else is cut the ordinary way.

Offsets here are code points, which is what Python slices, and every span
begins and ends on a line boundary. ``declaration_at`` and
``declarations_in`` also take the UTF-8 byte offsets that chunks are
stored with (``offsets="bytes"``), so a recalled chunk can say which
declaration it came from and on which lines. Nothing is rewritten:
``content[span.start:span.end]`` is the source, unchanged (invariant I1).
"""

from __future__ import annotations

import ast
import bisect
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Literal, Optional, Sequence

from .chunker import DEFAULT_TARGET, MIN_CHUNK, Span, chunk_spans

Language = Literal["python", "braces"]

#: Suffixes ``ast`` can parse, so their spans are exact.
PYTHON_SUFFIXES = (".py", ".pyi")
#: Suffixes read by a header line and a brace count.
BRACE_SUFFIXES = (
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts", ".go", ".rs",
    ".java", ".kt", ".kts", ".swift", ".scala", ".cs", ".c", ".h", ".cc", ".cpp",
    ".cxx", ".hpp", ".hh", ".m", ".mm", ".php", ".dart", ".zig",
)

#: A header may not begin with a word that opens an ordinary block.
_NOT_A_NAME = frozenset(
    ("if", "else", "for", "while", "switch", "case", "catch", "try", "do", "return",
     "match", "loop", "with", "using", "when", "guard", "defer", "go", "select",
     "new", "delete", "throw", "await", "yield", "in", "of", "is", "as", "where")
)
_MODIFIERS = (r"(?:(?:export|default|public|private|protected|internal|package|static|final|"
              r"abstract|sealed|async|unsafe|extern|inline|override|open|suspend|virtual|"
              r"const|mut|pub(?:\([^)]*\))?)\s+)*")
_NAMED = re.compile(
    _MODIFIERS + r"(?:function|func|fn|class|struct|interface|enum|trait|impl|record|object|protocol)"
    r"\s+(?:<[^>]*>\s*)?(?:[A-Za-z_$][\w$]*\s+for\s+)?([A-Za-z_$][\w$]*)")
_RECEIVER = re.compile(r"func\s*\(\s*[A-Za-z_$][\w$]*\s+[*&]?([A-Za-z_$][\w$]*)[^)]*\)\s*([A-Za-z_$][\w$]*)")
_ASSIGNED = re.compile(
    _MODIFIERS + r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+?)?=\s*"
    r"(?:async\s+)?(?:function\b|\([^)]*\)\s*(?::[^=]+?)?=>|[A-Za-z_$][\w$]*\s*=>)")
_MEMBER = re.compile(
    r"^(?:(?:public|private|protected|internal|static|final|abstract|override|open|async|"
    r"suspend|get|set|inline|operator|constructor)\s+)*"
    r"(?:[A-Za-z_$][\w$<>\[\].?]*\s+)?([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\([^;]*$")

#: A signature may run over a few lines before its body opens.
MAX_HEADER_LINES = 6
#: Scanning stops long before a machine-written file can exhaust memory.
MAX_LINES = 200_000
#: Declarations nested deeper than this are not something a person reads,
#: and naming them would cost more than the names are worth.
MAX_DEPTH = 32
#: How many files' declarations are kept, so that a recall naming several
#: chunks of one file parses it once.
MAX_PARSED = 8


@dataclass(frozen=True)
class Declaration:
    """One named thing in a source file, and exactly where it is."""

    name: str
    kind: str
    start: int
    end: int
    first_line: int
    last_line: int

    def where(self) -> str:
        """What a citation says: the name and the lines it occupies."""
        if self.first_line == self.last_line:
            return f"{self.name} (line {self.first_line})"
        return f"{self.name} (lines {self.first_line}-{self.last_line})"


@dataclass(frozen=True)
class CodeSpan(Span):
    """A span of source with the declarations it holds, outermost first.
    Empty names mean module-level code that no declaration contains."""

    names: tuple[str, ...] = ()


def code_language(source: Optional[str]) -> Optional[Language]:
    """The language a stored source name implies, or None when the name
    does not say. Nothing is guessed from the content: a file called
    notes.md holding a code block is prose that quotes code."""
    if not source:
        return None
    name = source.replace("\\", "/").rsplit("/", 1)[-1].lower()
    dot = name.rfind(".")
    if dot <= 0:
        return None
    suffix = name[dot:]
    if suffix in PYTHON_SUFFIXES:
        return "python"
    return "braces" if suffix in BRACE_SUFFIXES else None


def declarations(content: str, *, language: Optional[Language]) -> tuple[Declaration, ...]:
    """Every declaration in source order, nested ones included and named
    by everything that holds them. Empty when the source does not parse,
    holds no declaration, or is in no language this reads."""
    if not content or language is None:
        return ()
    if content.count("\n") > MAX_LINES:
        return ()
    return _found(str(content), language)


@lru_cache(maxsize=MAX_PARSED)
def _found(content: str, language: Language) -> tuple[Declaration, ...]:
    """The last few files' declarations, so that naming every chunk of one
    recall does not parse the same file over and over."""
    return _python(content) if language == "python" else _braces(content)


def code_spans(content: str, target: int = DEFAULT_TARGET, *,
               language: Optional[Language]) -> list[CodeSpan]:
    """Spans cut at declaration boundaries, falling back to the ordinary
    chunker when there is no declaration to cut at."""
    if target <= 0:
        raise ValueError("target must be positive")
    if not content:
        return []
    found = declarations(content, language=language)
    tops = _outermost(found)
    if not tops:
        return [CodeSpan(span.start, span.end) for span in chunk_spans(content, target)]
    starts = _line_starts(content)
    pieces: list[CodeSpan] = []
    whole: list[bool] = []
    for unit in _units(content, tops):
        cut = _split(content, unit, target, found, starts)
        pieces.extend(cut)
        whole.extend([len(cut) == 1] * len(cut))
    return _merge(content, pieces, whole, target)


def declarations_in(content: str, start: int, end: int, *, language: Optional[Language],
                    offsets: Literal["chars", "bytes"] = "chars") -> tuple[Declaration, ...]:
    """Every declaration a span touches, outermost first. What a recalled
    chunk is of, when it holds more than one small function."""
    where = _as_chars(content, start, end, offsets)
    if where is None:
        return ()
    first, last = where
    touching = [d for d in declarations(content, language=language) if d.start < last and first < d.end]
    return tuple(sorted(touching, key=lambda d: (d.start, -d.end)))


def declaration_at(content: str, start: int, end: int, *, language: Optional[Language],
                   offsets: Literal["chars", "bytes"] = "chars") -> Optional[Declaration]:
    """The innermost declaration that holds all of a span, or None when
    the span is module-level code or crosses more than one declaration."""
    where = _as_chars(content, start, end, offsets)
    if where is None:
        return None
    first, last = where
    holding = [d for d in declarations(content, language=language) if d.start <= first and last <= d.end]
    return max(holding, key=lambda d: (d.start, -d.end)) if holding else None


def line_span(content: str, start: int, end: int,
              offsets: Literal["chars", "bytes"] = "chars") -> tuple[int, int]:
    """The 1-based first and last line a span covers, so a citation can
    name lines rather than offsets. An empty span is its own line twice."""
    where = _as_chars(content, start, end, offsets)
    if where is None:
        return (1, 1)
    first, last = where
    begins = _endings(content, 0, first) + 1
    body = content[first:last].rstrip("\r\n")
    return (begins, begins + _endings(body, 0, len(body)))


def _endings(content: str, first: int, last: int) -> int:
    """How many lines end between two offsets, counting a carriage return
    with a newline after it as the one ending it is."""
    return (content.count("\n", first, last) + content.count("\r", first, last)
            - content.count("\r\n", first, last))


def code_context(source: Optional[str], names: Sequence[str]) -> str:
    """The line put in front of a code chunk before it is embedded, so a
    chunk carries the file and the declarations it lost when it was cut.
    Stored text is never changed by this (invariant I1)."""
    parts = [part for part in (source, ", ".join(n for n in names if n)) if part]
    return " | ".join(parts)


def _as_chars(content: str, start: int, end: int,
              offsets: Literal["chars", "bytes"]) -> Optional[tuple[int, int]]:
    """A span in code points, however it arrived, or None when it is not
    a span of this content."""
    if start < 0 or end < start:
        return None
    if offsets == "bytes" and not content.isascii():
        raw = content.encode()
        if end > len(raw):
            return None
        return (len(raw[:start].decode(errors="ignore")), len(raw[:end].decode(errors="ignore")))
    return (start, end) if end <= len(content) else None


def _line_starts(content: str) -> list[int]:
    """Where every line begins. Line endings are counted as Python's own
    parser counts them: a lone carriage return ends a line, and so does a
    carriage return with a newline after it, which is one ending and not
    two. Offsets stay offsets into the source exactly as it arrived."""
    starts = [0]
    index, length = 0, len(content)
    while index < length:
        ch = content[index]
        if ch == "\r":
            index += 2 if index + 1 < length and content[index + 1] == "\n" else 1
            starts.append(index)
        elif ch == "\n":
            index += 1
            starts.append(index)
        else:
            index += 1
    return starts


def _lines(content: str, starts: list[int]) -> list[str]:
    """Each line with its ending still on it, in one pass over the source."""
    edges = [*starts[1:], len(content)]
    return [content[start:end] for start, end in zip(starts, edges)]


def _line_end(content: str, starts: list[int], line: int) -> int:
    """Where line ``line`` (1-based) ends, its newline included."""
    return starts[line] if line < len(starts) else len(content)


def _with_comments(content: str, starts: list[int], line: int) -> int:
    """The first line of a declaration, counting the comment lines written
    directly above it with nothing blank in between."""
    while line > 1:
        above = content[starts[line - 2] : starts[line - 1]].strip()
        if not above.startswith("#") and not above.startswith("//"):
            break
        line -= 1
    return line


def _python(content: str) -> tuple[Declaration, ...]:
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, RecursionError):
        return ()
    starts = _line_starts(content)
    found: list[Declaration] = []
    # An explicit stack, because a file that parses can still be nested
    # deeper than Python will recurse: a sum of a thousand terms is a
    # thousand nodes deep and is perfectly ordinary code.
    work: list[tuple[ast.AST, str, bool, int]] = [(tree, "", False, 0)]
    while work:
        node, prefix, in_class, depth = work.pop()
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                work.append((child, prefix, in_class, depth))
                continue
            if depth >= MAX_DEPTH:
                continue
            name = prefix + child.name
            opens = min([child.lineno, *(d.lineno for d in child.decorator_list)])
            first = _with_comments(content, starts, opens)
            last = min(child.end_lineno or child.lineno, len(starts))
            kind = ("class" if isinstance(child, ast.ClassDef)
                    else "method" if in_class else "function")
            found.append(Declaration(name, kind, starts[first - 1],
                                     _line_end(content, starts, last), first, last))
            work.append((child, f"{name}.", isinstance(child, ast.ClassDef), depth + 1))
    return tuple(sorted(found, key=lambda d: (d.start, -d.end)))


def _bare(line: str, in_block: bool) -> tuple[str, bool]:
    """The line with its comments and string contents taken out, so that
    braces can be counted. Returns whether a block comment is still open."""
    kept: list[str] = []
    quote = ""
    index = 0
    while index < len(line):
        ch = line[index]
        if in_block:
            if line.startswith("*/", index):
                in_block = False
                index += 2
                continue
            index += 1
        elif quote:
            if ch == "\\":
                index += 2
                continue
            if ch == quote:
                quote = ""
            index += 1
        elif line.startswith("//", index) or ch == "#":
            break
        elif line.startswith("/*", index):
            in_block = True
            index += 2
        elif ch in "\"'`":
            quote = ch
            index += 1
        else:
            kept.append(ch)
            index += 1
    return "".join(kept), in_block


def _header(line: str) -> Optional[str]:
    """The name a line declares, or None when it declares nothing."""
    text = line.strip()
    if not text or text.startswith(("@", "*", "}")):
        return None
    receiver = _RECEIVER.match(text)
    if receiver:
        return f"{receiver.group(1)}.{receiver.group(2)}"
    for pattern in (_NAMED, _ASSIGNED):
        match = pattern.match(text)
        if match:
            return match.group(1)
    member = _MEMBER.match(text)
    if member and member.group(1) not in _NOT_A_NAME and "=" not in text.split("(")[0]:
        return member.group(1)
    return None


def _braces(content: str) -> tuple[Declaration, ...]:
    starts = _line_starts(content)
    lines = _lines(content, starts)
    bare: list[str] = []
    in_block = False
    for line in lines:
        clean, in_block = _bare(line, in_block)
        bare.append(clean)
    found: list[Declaration] = []
    # A stack rather than recursion: a thousand nested braces are still a
    # file somebody generated, and it should be read, not crashed on.
    work: list[tuple[int, int, str, int]] = [(0, len(lines), "", 0)]
    while work:
        first, limit, prefix, depth = work.pop()
        index = first
        while index < limit:
            name = _header(bare[index])
            opened = _opens(bare, index, limit)
            if name is None or opened is None:
                index += 1
                continue
            close = _closes(bare, opened, limit)
            if close is None:
                index += 1
                continue
            began = _with_comments(content, starts, index + 1)
            found.append(Declaration(prefix + name, _kind(bare[index]), starts[began - 1],
                                     _line_end(content, starts, close + 1), began, close + 1))
            if depth + 1 < MAX_DEPTH:
                work.append((opened + 1, close, f"{prefix}{name}.", depth + 1))
            index = close + 1
    return tuple(sorted(found, key=lambda d: (d.start, -d.end)))


def _kind(line: str) -> str:
    text = line.strip()
    for word in ("class", "struct", "interface", "enum", "trait", "impl", "record", "protocol"):
        if re.search(rf"\b{word}\b", text):
            return "class"
    return "function"


def _opens(bare: list[str], index: int, limit: int) -> Optional[int]:
    """The line where a signature's body opens, within a few lines of the
    header, or None when it never does (a prototype, a call, a field)."""
    for line in range(index, min(index + MAX_HEADER_LINES, limit)):
        text = bare[line]
        if "{" in text:
            return line
        if ";" in text or (line > index and not text.strip()):
            return None
    return None


def _closes(bare: list[str], opened: int, limit: int) -> Optional[int]:
    depth = 0
    for line in range(opened, limit):
        depth += bare[line].count("{") - bare[line].count("}")
        if depth <= 0:
            return line
    return None


def _outermost(found: Sequence[Declaration]) -> list[Declaration]:
    """The declarations no other declaration holds."""
    tops: list[Declaration] = []
    for declaration in found:
        if not tops or declaration.start >= tops[-1].end:
            tops.append(declaration)
    return tops


def _units(content: str, tops: Sequence[Declaration]) -> list[CodeSpan]:
    """The file as declarations and the module-level code between them."""
    units: list[CodeSpan] = []
    cursor = 0
    for declaration in tops:
        if content[cursor : declaration.start].strip():
            units.append(CodeSpan(cursor, declaration.start))
        units.append(CodeSpan(declaration.start, declaration.end, (declaration.name,)))
        cursor = declaration.end
    if content[cursor:].strip():
        units.append(CodeSpan(cursor, len(content)))
    return units


def _split(content: str, unit: CodeSpan, target: int, found: Sequence[Declaration],
           starts: list[int]) -> list[CodeSpan]:
    """One unit, cut at its own lines when it is longer than the target.
    A nested declaration is preferred as a cut, then a blank line. Which
    lines are blank is worked out once, a line at a time, rather than by
    reading back over the file for every candidate."""
    if unit.end - unit.start <= target:
        return [unit]
    inner = {d.start for d in found if unit.start < d.start < unit.end}
    places = starts[bisect.bisect_right(starts, unit.start):bisect.bisect_left(starts, unit.end)]
    above = [unit.start, *places[:-1]]
    blank = {place for start, place in zip(above, places) if not content[start:place].strip()}
    pieces: list[CodeSpan] = []
    cursor = unit.start
    while unit.end - cursor > target:
        room = places[bisect.bisect_right(places, cursor):bisect.bisect_right(places, cursor + target)]
        if not room:
            following = places[bisect.bisect_right(places, cursor):]
            if not following:
                break
            cut = following[0]
        else:
            cut = _prefer(room, inner, blank, cursor, target)
        pieces.append(CodeSpan(cursor, cut, unit.names))
        cursor = cut
    pieces.append(CodeSpan(cursor, unit.end, unit.names))
    # What is left over at the end of a declaration is the end of that
    # declaration, and a few lines of it alone say nothing. It goes back
    # to the piece it was cut from, as the ordinary chunker's tail does.
    if len(pieces) > 1 and len(content[pieces[-1].start : pieces[-1].end].strip()) < MIN_CHUNK:
        tail = pieces.pop()
        pieces[-1] = CodeSpan(pieces[-1].start, tail.end, unit.names)
    return pieces


def _prefer(room: Sequence[int], inner: set[int], blank: set[int], cursor: int, target: int) -> int:
    """The furthest cut that is worth taking: a nested declaration, then a
    blank line, then any line, and never so early that it wastes a chunk."""
    enough = cursor + target // 2
    for wanted in (inner, blank):
        liked = [place for place in room if place in wanted and place >= enough]
        if liked:
            return max(liked)
    return max(room)


def _merge(content: str, pieces: list[CodeSpan], whole: list[bool], target: int) -> list[CodeSpan]:
    """Neighbouring whole units share a chunk while they fit in one."""
    merged: list[CodeSpan] = []
    joined: list[bool] = []
    for piece, entire in zip(pieces, whole):
        if (merged and joined[-1] and entire
                and piece.end - merged[-1].start <= target):
            names = merged[-1].names + tuple(n for n in piece.names if n not in merged[-1].names)
            merged[-1] = CodeSpan(merged[-1].start, piece.end, names)
            continue
        merged.append(piece)
        joined.append(entire)
    return merged
