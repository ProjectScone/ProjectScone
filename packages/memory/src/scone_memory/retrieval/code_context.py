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
import io
import re
import tokenize
from typing import TYPE_CHECKING, Optional, Sequence

from ..core.errors import Gone, InvalidInput, NotFound, SconeError
from ..core.models import RecallItem
from ..ingestion.code import Declaration, code_language, declarations_in
from ..ingestion.code_graph import masked
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

    #: The items a caller should hand on, with any whose source is
    #: confirmed gone removed -- so a receipt saying "dropped" and the
    #: passages a caller prints cannot disagree.
    items: tuple[RecallItem, ...] = ()
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
    #: Contexts with a signature longer than the line bound, quoted to
    #: there. A header cut short must not read as the whole header.
    shortened: int = 0
    why: str = ""

    def record(self) -> dict[str, object]:
        return {"chunks": len(self.by_chunk), "not_code": self.not_code, "gone": self.gone,
                "unread": self.unread, "not_read": self.not_read, "capped": self.capped,
                "shortened": self.shortened, "why": self.why,
                "items": [item.model_dump() for item in self.items],
                "by_chunk": {str(key): {
                    "language": value.language, "more_imports": value.more_imports,
                    "holders": [{"name": s.name, "line": s.line, "text": s.text,
                                 "clipped": s.clipped} for s in value.holders],
                    "imports": [[i.line, i.text] for i in value.imports],
                } for key, value in self.by_chunk.items()}}


def _python_headers(content: str) -> dict[int, tuple[int, int]]:
    """Each Python declaration's ``def``/``class`` line to the exact
    character span of its header, ending at its own colon.

    Two wrong answers came before this one, and each was wrong in a way
    the code confidently explained:

    - **Bracket depth over raw text.** Justified on the grounds that a
      colon in a default argument is inside brackets. So it is -- but a
      bracket inside a *string* is inside nothing, so
      ``def f(value="("):`` never reached depth zero at its own colon
      and ``def f(value=")"):`` went negative.
    - **``ast`` body-start minus one line.** ``ast`` reports where a
      declaration *begins*, not where its header ends, and the gap
      between them is not empty: leading comments and blank lines in the
      body are not statements, so they became part of the "signature"; a
      class whose first method carries a decorator took that decorator
      into the class header, because ``FunctionDef.lineno`` points at the
      ``def``; long leading body comments set ``clipped`` on a one-line
      header; and ``def f(): return secret`` returned its body as its
      signature -- which in a withholding answer is the leak shape
      again.

    ``tokenize`` is the tool that actually knows: strings and comments
    are single tokens, so no bracket inside either can move the depth,
    and the colon that ends the header is the first ``:`` operator at
    depth zero after the declaring keyword. Character offsets, not
    lines, so a one-line suite yields ``def f():`` and not its body.

    The header begins at ``def``, ``class`` or the ``async`` before
    them, so **a decorator is not part of the signature**. That is a
    choice rather than an oversight: a decorator is a separate
    statement, the reason a class header must not swallow its first
    method's decorator is that they belong to different declarations,
    and a rule that starts at the keyword cannot make that mistake in
    either direction. A caller wanting the decorators has the
    declaration's own line range.
    """
    # `_line_starts`, not a second table built here. The first version of
    # this function had its own `splitlines(keepends=True)` copy, and
    # when the newline semantics were corrected only the shared helper
    # was fixed -- so the bug stayed exactly where the finding was about,
    # in a duplicate four lines away. One table, one convention.
    starts = _line_starts(content)

    def offset(row: int, column: int) -> int:
        return starts[row - 1] + column

    at: dict[int, tuple[int, int]] = {}
    begin: Optional[tuple[int, int]] = None
    depth = 0
    previous_async: Optional[tuple[int, int]] = None
    try:
        for token in tokenize.generate_tokens(io.StringIO(content).readline):
            if begin is None:
                if token.type == tokenize.NAME and token.string == "async":
                    previous_async = token.start
                    continue
                if token.type == tokenize.NAME and token.string in ("def", "class"):
                    head = previous_async or token.start
                    begin, depth = (head[0], offset(*head)), 0
                elif token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT,
                                        tokenize.INDENT, tokenize.DEDENT):
                    previous_async = None
                continue
            if token.type == tokenize.OP:
                if token.string in "([{":
                    depth += 1
                elif token.string in ")]}":
                    depth -= 1
                elif token.string == ":" and depth == 0:
                    at[begin[0]] = (begin[1], offset(token.end[0], token.end[1]))
                    begin, previous_async = None, None
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        # A file `ast` accepted should tokenize, so this is unreachable in
        # practice; returning what was read keeps a partial answer honest
        # rather than claiming none.
        pass
    return at


def _header(headers: dict[int, tuple[int, int]],
            declaration: Declaration) -> Optional[tuple[int, int]]:
    """The character span of a declaration's own header, or None.

    Found by range rather than by exact key: a declaration's span starts
    at its decorators and at the comment lines written directly above
    it, so ``first_line`` is often earlier than the ``def`` the parser
    reports. Looking up ``first_line`` alone returned the comment line
    and nothing else.
    """
    inside = [line for line in headers
              if declaration.first_line <= line <= declaration.last_line]
    return headers[min(inside)] if inside else None


def _brace_header(blank: list[str], starts: list[int], start: int) -> tuple[int, bool]:
    """A brace declaration's header, through the ``{`` that opens its body.

    Scanned over the **blanked** copy, where string literals and comments
    are replaced in place, so a brace inside a string cannot move the
    depth and every line and column still indexes the real source.

    Answers a character offset rather than a line, for the same reason
    the Python path does: ``function contact() { return secret; }`` puts
    a body on the header's line, and quoting the line whole made that
    body part of the signature. The Python path was made
    character-precise and this one was left counting lines -- the same
    half-fix, one language later.
    """
    depth = 0
    for offset in range(MAX_SIGNATURE_LINES):
        index = start - 1 + offset
        if index >= len(blank):
            return starts[min(index, len(starts) - 1)], False
        line = blank[index]
        for column, char in enumerate(line):
            if char == "{" and depth == 0:
                return starts[index] + column + 1, False
            if char in "([":
                depth += 1
            elif char in ")]":
                depth = max(0, depth - 1)
            elif char == ";" and depth == 0:
                # A declaration with no body of its own: the header is
                # the whole of it, up to and including the semicolon.
                return starts[index] + column + 1, False
    last = start - 1 + MAX_SIGNATURE_LINES - 1
    return starts[min(last, len(starts) - 1)] + len(blank[min(last, len(blank) - 1)]), True


def _line_starts(content: str) -> list[int]:
    """The offset each **physical** line begins at, as the tokenizer
    counts them.

    Not ``str.splitlines()``, which was the first version of this and
    was wrong: it also breaks on U+2028, U+2029, U+0085, vertical tab
    and form feed, none of which the tokenizer treats as a physical
    newline. A string literal holding one -- ``banner = "one\u2028two"``
    is ordinary code -- shifted every row after it, and a signature came
    back as ``'two"\ndef conta'``. Two coordinate systems that disagree,
    which is the third time that shape has bitten this work.

    ``io.StringIO`` is what ``tokenize`` reads through here, and with its
    default ``newline='\n'`` it performs no translation and ends a line
    only at ``\n``. So this does too, and a ``\r`` stays part of its
    line, keeping offsets into the source exactly as it arrived.
    """
    at = [0]
    index = content.find("\n")
    while index != -1:
        at.append(index + 1)
        index = content.find("\n", index + 1)
    return at


def _looks_like_a_header(text: str, language: str) -> bool:
    """Whether a quoted span is plausibly the header it claims to be.

    A post-condition rather than trust in the arithmetic. Every
    coordinate mistake in this work has produced text that is obviously
    not a header -- a fragment of a string literal, half of two lines --
    and one cheap check turns the next one into a visibly incomplete
    answer instead of a confident quotation of the wrong bytes.
    """
    said = text.strip()
    if not said:
        return False
    if language == "python":
        return said.endswith(":") and said.split("(")[0].split()[0] in (
            "def", "class", "async") or said.startswith(("def ", "class ", "async "))
    return said.endswith(("{", ";"))


def _signature(content: str, lines: list[str], blank: list[str], starts: list[int],
               declaration: Declaration, language: str,
               headers: dict[int, tuple[int, int]]) -> Signature:
    """A declaration's header, quoted from the source."""
    if language == "python":
        span = _header(headers, declaration)
        if span is None:
            # The header's end could not be established, so the declaring
            # line is all that can honestly be quoted -- and `clipped`
            # is the field that says a header is incomplete.
            return Signature(name=declaration.name, kind=declaration.kind,
                             line=declaration.first_line,
                             text=lines[declaration.first_line - 1].rstrip("\r"),
                             clipped=True)
        text = content[span[0]:span[1]]
        held = text.split("\n")
        clipped = len(held) > MAX_SIGNATURE_LINES
        text = "\n".join(line.rstrip("\r") for line in held[:MAX_SIGNATURE_LINES])
        if not _looks_like_a_header(text, language):
            return Signature(name=declaration.name, kind=declaration.kind,
                             line=declaration.first_line,
                             text=lines[declaration.first_line - 1].rstrip("\r"),
                             clipped=True)
        return Signature(name=declaration.name, kind=declaration.kind,
                         line=content[:span[0]].count("\n") + 1,
                         text=text, clipped=clipped)
    start = declaration.first_line
    at, clipped = _brace_header(blank, starts, start)
    text = content[starts[start - 1]:at].strip("\n")
    if not _looks_like_a_header(text, language):
        return Signature(name=declaration.name, kind=declaration.kind, line=start,
                         text=(lines[start - 1].rstrip("\r") if start - 1 < len(lines)
                               else ""), clipped=True)
    return Signature(name=declaration.name, kind=declaration.kind, line=start,
                     text=text, clipped=clipped)


def _imports(content: str, language: str, limit: int) -> tuple[tuple[Imported, ...], bool]:
    """The lines this file brings names in on, quoted, and whether there
    are more than were listed.

    Physical lines, indexed by the parser's own line numbers, so the same
    convention as everything else here. This was the third copy of a
    ``splitlines()`` table in this file and had the same fault as the
    other two: a line separator inside a string literal above an import
    shifted every line after it, and the import came back misquoted.
    """
    lines = [line.rstrip("\r") for line in content.split("\n")]
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
    kept: list[RecallItem] = []
    read: dict[int, Optional[str]] = {}
    unavailable: set[int] = set()
    not_code = vanished = unread = unbudgeted = capped = shortened = 0
    for item in items:
        language = code_language(item.source)
        if language is None:
            not_code += 1
            kept.append(item)
            continue
        if item.episode_id in unavailable:
            unread += 1
            kept.append(item)
            continue
        if item.episode_id not in read:
            if len(read) + len(unavailable) >= episodes:
                unbudgeted += 1
                kept.append(item)
                continue
            try:
                read[item.episode_id] = (await engine.episode(space, item.episode_id)).content
            except (Gone, NotFound):
                read[item.episode_id] = None
            except SconeError:
                unavailable.add(item.episode_id)
                unread += 1
                kept.append(item)
                continue
        content = read[item.episode_id]
        if content is None:
            # Confirmed absent. Counting it as dropped while a caller
            # still holds the item is how stale text gets served under a
            # clean reason, so the item goes with the count.
            vanished += 1
            continue
        # Split the same way `_line_starts` counts, so a row indexes the
        # same text in all three.
        lines = content.split("\n")
        blank = masked(content, prose=False).split("\n")
        starts = _line_starts(content)
        headers = _python_headers(content) if language == "python" else {}
        holders = declarations_in(content, item.start, item.end, language=language,
                                  offsets="bytes")
        brought, more = _imports(content, language, imports)
        if more:
            capped += 1
        signatures = tuple(_signature(content, lines, blank, starts, holder, language, headers)
                           for holder in holders)
        if any(signature.clipped for signature in signatures):
            shortened += 1
        found[item.chunk_id] = ChunkContext(
            chunk_id=item.chunk_id, language=language, holders=signatures,
            imports=brought, more_imports=more)
        kept.append(item)

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
    if shortened:
        why += (f"; {shortened} signature(s) run past {MAX_SIGNATURE_LINES} lines and were "
                f"quoted to there, so those headers are incomplete")
    return CodeContext(items=tuple(kept), by_chunk=found, not_code=not_code, gone=vanished,
                       unread=unread, not_read=unbudgeted, capped=capped, shortened=shortened,
                       why=why)
