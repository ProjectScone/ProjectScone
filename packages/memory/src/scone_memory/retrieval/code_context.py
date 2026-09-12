"""A recalled body, with the two things that make it readable.

A chunk of code already says which declaration it came from --
``Engine.forget`` -- and that is where our answer stopped. It did not say
the declaration's **signature**, so a caller saw a body without its
parameters, and it did not say what the file **imported**, so a name in
the body could not be traced to where it came from. For the question
"how is this done here", a body without its signature and its imports is
a fragment.

The reference that has this prepends the context into the chunk text.
Ours does not: invariant I1 says ``content[span.start:span.end]`` is the
source unchanged, and a chunk that has grown a header is no longer a
quotation of the file. So the context sits **beside** the chunk, quoted
from the source with the line numbers it came from, and every line of it
can be checked against the file.

Three rules, each with a test:

- **Nothing is guessed from content.** A file called ``notes.md`` holding
  a code block is prose that quotes code, and gets no code context. The
  language comes from the stored name, as everywhere else.
- **A source confirmed gone is dropped, not answered.** The same rule
  merging and windowing learned: the fragment we happen to hold is
  deleted text.
- **A list that stopped says so.** A file with more imports than the
  bound reports ``more_imports``, because a count of what we listed must
  never read as a count of what the file imports.

No model is called and nothing is re-embedded.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Optional, Sequence

from ..core.errors import Gone, InvalidInput, NotFound, SconeError
from ..core.models import RecallItem
from ..ingestion.code import Declaration, code_language, declarations_in
from ..memory.engine import check_space

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

#: Imports listed for one file. Past this the list says it stopped.
MAX_IMPORTS = 200
#: Episodes read in one call, and the most a caller may ask for. A read
#: that fails counts against it: it is work done.
MAX_EPISODES = 100
#: Lines a signature may run to. A header longer than this is quoted to
#: here and says it was clipped.
MAX_SIGNATURE_LINES = 12

#: How the brace languages open a line that brings a name in. This reads
#: **lines**, where ``ingestion/code_graph.py`` resolves **targets** --
#: quoting the line the file wrote needs no resolution, and a resolved
#: target is not something the reader can check against the source.
_BROUGHT_IN = re.compile(
    r"^\s*(?:"
    r"import\b"                       # JS/TS/Java/Kotlin/Scala/Go
    r"|from\s+['\"]"                  # JS re-export form
    r"|#\s*include\b"                 # C/C++
    r"|(?:pub\s+)?use\b"              # Rust
    r"|using\b"                       # C#
    r"|(?:const|let|var)\s+.*\brequire\s*\("   # CommonJS
    r")")


@dataclass(frozen=True)
class Signature:
    """A declaration's header, quoted from the source."""

    name: str
    kind: str
    line: int
    #: The header as the file wrote it, from the declaring keyword
    #: through the end of its parameter list.
    text: str
    #: Whether the header ran past the line bound and was quoted to it.
    clipped: bool = False


@dataclass(frozen=True)
class Imported:
    """One line that brings a name into the file, quoted."""

    line: int
    text: str


@dataclass(frozen=True)
class ChunkContext:
    """What one recalled chunk needed and could not say for itself."""

    chunk_id: int
    language: str
    #: The declarations holding this chunk, outermost first.
    holders: tuple[Signature, ...] = ()
    imports: tuple[Imported, ...] = ()
    #: Whether the file brings in more names than were listed.
    more_imports: bool = False


@dataclass(frozen=True)
class CodeContext:
    """The context found, and every reason a chunk has none."""

    by_chunk: dict[int, ChunkContext] = field(default_factory=dict)
    #: Items whose stored name does not say a language. Not a failure.
    not_code: int = 0
    #: Items whose episode is confirmed absent; dropped rather than
    #: answered from text this space no longer holds.
    gone: int = 0
    #: Items whose episode could not be read, which is not a finding that
    #: it is absent.
    unread: int = 0
    #: Items skipped because the episode budget was spent before theirs
    #: was reached. Nobody looked at those.
    not_read: int = 0
    #: Contexts whose import list reached the bound.
    capped: int = 0
    why: str = ""

    def record(self) -> dict[str, object]:
        return {"chunks": len(self.by_chunk), "not_code": self.not_code, "gone": self.gone,
                "unread": self.unread, "not_read": self.not_read, "capped": self.capped,
                "why": self.why,
                "by_chunk": {str(key): {
                    "language": value.language, "more_imports": value.more_imports,
                    "holders": [[s.name, s.line, s.text] for s in value.holders],
                    "imports": [[i.line, i.text] for i in value.imports],
                } for key, value in self.by_chunk.items()}}


def _signature(lines: list[str], declaration: Declaration, language: str) -> Signature:
    """A declaration's header, from its keyword to the end of its
    parameters.

    The terminator is found by bracket depth rather than by masking
    strings, because the ``:`` ending a Python header and the ``{``
    opening a brace body are always at depth zero -- a colon inside a
    default argument is inside brackets by definition.
    """
    start = declaration.first_line
    close = ":" if language == "python" else "{"
    depth = 0
    taken: list[str] = []
    clipped = True
    for offset in range(MAX_SIGNATURE_LINES):
        index = start - 1 + offset
        if index >= len(lines):
            clipped = False
            break
        line = lines[index]
        taken.append(line)
        done = False
        for char in line:
            if char in "([{" and not (char == "{" and depth == 0 and close == "{"):
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif depth == 0 and char == close:
                done = True
                break
            if depth == 0 and char == close and close == "{":
                done = True
                break
        if done:
            clipped = False
            break
    return Signature(name=declaration.name, kind=declaration.kind, line=start,
                     text="\n".join(taken), clipped=clipped)


def _imports(content: str, language: str, limit: int) -> tuple[tuple[Imported, ...], bool]:
    """The lines this file brings names in on, quoted, and whether there
    are more than were listed."""
    lines = content.splitlines()
    found: list[Imported] = []
    if language == "python":
        try:
            tree = ast.parse(content)
        except (SyntaxError, ValueError, RecursionError):
            return (), False
        at: list[tuple[int, int]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                at.append((node.lineno, node.end_lineno or node.lineno))
        for first, last in sorted(at):
            if len(found) >= limit:
                return tuple(found), True
            text = "\n".join(lines[first - 1:last])
            found.append(Imported(line=first, text=text))
        return tuple(found), False
    for number, line in enumerate(lines, 1):
        if _BROUGHT_IN.match(line):
            if len(found) >= limit:
                return tuple(found), True
            found.append(Imported(line=number, text=line))
    return tuple(found), False


async def code_context(engine: "MemoryEngine", space: str, items: Sequence[RecallItem], *,
                       imports: int = MAX_IMPORTS,
                       episodes: int = MAX_EPISODES) -> CodeContext:
    """The signature and imports each recalled code chunk sits inside."""
    check_space(space)
    if type(imports) is not int or not 1 <= imports <= MAX_IMPORTS:
        raise InvalidInput(f"imports is a whole number from 1 to {MAX_IMPORTS}, not {imports!r}")
    if type(episodes) is not int or not 1 <= episodes <= MAX_EPISODES:
        raise InvalidInput(f"episodes is a whole number from 1 to {MAX_EPISODES}, not "
                           f"{episodes!r}")

    found: dict[int, ChunkContext] = {}
    read: dict[int, Optional[str]] = {}
    unavailable: set[int] = set()
    not_code = vanished = unread = unbudgeted = capped = 0
    for item in items:
        language = code_language(item.source)
        if language is None:
            not_code += 1
            continue
        if item.episode_id in unavailable:
            unread += 1
            continue
        if item.episode_id not in read:
            if len(read) + len(unavailable) >= episodes:
                unbudgeted += 1
                continue
            try:
                read[item.episode_id] = (await engine.episode(space, item.episode_id)).content
            except (Gone, NotFound):
                read[item.episode_id] = None
            except SconeError:
                unavailable.add(item.episode_id)
                unread += 1
                continue
        content = read[item.episode_id]
        if content is None:
            vanished += 1
            continue
        lines = content.splitlines()
        holders = declarations_in(content, item.start, item.end, language=language,
                                  offsets="bytes")
        brought, more = _imports(content, language, imports)
        if more:
            capped += 1
        found[item.chunk_id] = ChunkContext(
            chunk_id=item.chunk_id, language=language,
            holders=tuple(_signature(lines, holder, language) for holder in holders),
            imports=brought, more_imports=more)

    why = (f"{len(found)} chunk(s) given the signature they sit inside and the imports of "
           f"their file" if found else "no chunk had code context to give")
    if not_code:
        why += (f"; {not_code} item(s) are stored under a name that does not say a language, so "
                f"nothing was read from them -- a file of prose quoting code is prose")
    if vanished:
        why += (f"; {vanished} item(s) are no longer there and were dropped rather than "
                f"answered from text this space has deleted")
    if unread:
        why += (f"; {unread} item(s) could not be read, which is not a finding that there was "
                f"nothing to read")
    if unbudgeted:
        why += (f"; {unbudgeted} item(s) were skipped because the budget of {episodes} "
                f"episode(s) was spent before theirs was reached -- nobody looked at those")
    if capped:
        why += (f"; {capped} file(s) bring in more than {imports} names and this listed "
                f"{imports} of them, not all of them")
    return CodeContext(by_chunk=found, not_code=not_code, gone=vanished, unread=unread,
                       not_read=unbudgeted, capped=capped, why=why)
