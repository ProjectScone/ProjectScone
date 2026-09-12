"""Cutting a document where it already divides itself.

``chunker.py`` prefers a paragraph break, then a sentence end, then any
whitespace, then a hard cut inside a word. It knows nothing about
headings, numbered clauses or tables, so a cut at the byte target lands
wherever the prose allows: ``Article 7.2`` can end one chunk while the
clause it names begins the next, and a table's rows can arrive without
the header row that says what their columns mean. The clause number and
the column names are usually the query terms, and the chunk holding the
answer cannot say what it is.

The leading document pipeline answers this with a chunker per document
type -- one for books, one for laws, one for papers, one for resumes --
so someone must declare what kind of document this is before it is read.
Ours reads the structure the document already carries, which needs no
model, no new dependency and no such declaration.

``structure.parse_structure`` already finds headings, fenced code and
pipe tables in byte offsets, and is used by retrieval and by source
inspection. This builds on it rather than parsing again, and adds only
what that parser deliberately leaves out: setext headings, numbered and
lettered clauses, and question/answer pairs.

Four rules, each with a test:

- **A document with no structure chunks exactly as it does today.** Cut
  positions decide what chunks exist, and stored offsets are part of the
  shared specification the Rust product reads, so this path is
  byte-identical to ``chunk_spans`` and a test compares the two.
- **A section longer than the target is still split**, and the receipt
  separates chunks beginning at a boundary from chunks beginning where
  the target fell. "Structure-aware" must not read as "every chunk is a
  section".
- **A table is never cut.** Its header is what its rows mean. A table
  longer than the target is handed back whole and counted, because a
  chunk over the target a caller can see is better than rows a caller
  cannot read.
- **Structure that is not there is not invented.** ``1984 was a year`` is
  prose, not clause 1984; a ``#`` inside a fenced block is a comment; and
  the ``---`` closing YAML front matter is not a heading underline.

Nothing here rewrites a byte of the document. This chooses cut positions
and counts them.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .chunker import DEFAULT_TARGET, Span, chunk_spans
from .structure import parse_structure

#: Structural units read from one document. Past this the rest is split
#: by size and the receipt says the count is what we read.
MAX_SECTIONS = 10_000

#: The underline under a setext heading.
_UNDERLINE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
#: The delimiter around YAML front matter, which is also a setext
#: underline -- so without this the last key of the front matter reads as
#: a heading and the document is cut at its own metadata.
_FRONT = re.compile(r"^-{3,}[ \t]*$")
#: Lines a front-matter block may run to. Past this the opening ``---``
#: was a horizontal rule, not a delimiter, and nothing is skipped.
_FRONT_LINES = 64
#: A numbered, lettered or named clause. Digits alone are not enough: a
#: line may legitimately begin with a year, so a bare run of digits must
#: be followed by ``.`` or ``)``. The lookahead requires the marker to be
#: followed by text or to end the line, so ``e.g. something`` is not
#: clause "e".
_CLAUSE = re.compile(
    r"^ {0,3}(?:"
    r"\d+(?:\.\d+)+\.?"                                    # 1.2, 3.4.5
    r"|\d+[.)]"                                            # 1. or 1)
    r"|\([A-Za-z0-9]{1,3}\)"                               # (a), (iv), (2)
    r"|[a-z][.)]"                                          # a. or b)
    r"|[IVXLC]+[.)]"                                       # IV. or XI)
    r"|§ ?\d+"                                             # § 12
    r"|(?i:Article|Section|Chapter|Clause|Rule|Schedule|Appendix|Part)[ \t]+\d+"
    r")(?=[ \t]+\S|[ \t]*$)")
#: A question or an answer opening a pair.
_PAIR = re.compile(r"^ {0,3}(?:Q|A|Question|Answer)[ \t]*[:.][ \t]*\S")


@dataclass(frozen=True)
class Unit:
    """One structural unit, in code points, as the document wrote it."""

    start: int
    end: int
    kind: str
    #: The marker or heading line, verbatim -- never normalised, so it can
    #: be checked against the source. Empty for kind ``text``, which is
    #: not something the document labelled: it is the interval between a
    #: table's last row and whatever follows, which belongs to neither.
    label: str


@dataclass(frozen=True)
class Structured:
    """The chunks, and how each of them came to start where it does."""

    spans: tuple[Span, ...] = ()
    #: Structural units read. With ``capped`` set this is what was read,
    #: not what the document holds.
    units: int = 0
    #: Chunks beginning at a structural boundary.
    at_boundary: int = 0
    #: Chunks beginning where the byte target fell: inside a unit longer
    #: than the target, or in a document with no structure at all.
    by_size: int = 0
    #: Tables handed back whole, header with rows.
    tables: int = 0
    #: Chunks longer than the target because splitting them would have
    #: broken what they are. A caller sizing a context window needs this
    #: rather than an assurance that nothing exceeds the target.
    over_target: int = 0
    #: Whether the unit bound was reached, so the rest of the document was
    #: split by size alone.
    capped: bool = False
    why: str = ""

    def record(self) -> dict[str, object]:
        return {"chunks": len(self.spans), "units": self.units, "at_boundary": self.at_boundary,
                "by_size": self.by_size, "tables": self.tables, "over_target": self.over_target,
                "capped": self.capped, "why": self.why,
                "spans": [[s.start, s.end] for s in self.spans]}


def _points(content: str, wanted: set[int]) -> dict[int, int]:
    """Byte offsets to code-point offsets, in one pass.

    ``parse_structure`` answers in UTF-8 bytes because stored spans are
    bytes; ``chunk_spans`` works in code points because that is what
    slicing a ``str`` uses. Converting only the offsets asked for keeps
    this linear in the document and constant in the mapping.
    """
    if content.isascii():
        return {byte: byte for byte in wanted}
    at: dict[int, int] = {}
    byte = 0
    for index, char in enumerate(content):
        if byte in wanted:
            at[byte] = index
        byte += len(char.encode())
    if byte in wanted:
        at[byte] = len(content)
    return at


def _after_front_matter(lines: list[str]) -> int:
    """The first line that is prose rather than front matter.

    Only skips a block that is actually closed: an opening ``---`` with no
    partner is a horizontal rule, and swallowing the document after one
    would be worse than reading its metadata as a heading.
    """
    if not lines or not _FRONT.match(lines[0].rstrip("\r\n")):
        return 0
    for index in range(1, min(len(lines), _FRONT_LINES)):
        if _FRONT.match(lines[index].rstrip("\r\n")):
            return index + 1
    return 0


def units(content: str, limit: int = MAX_SECTIONS) -> tuple[Unit, ...]:
    """The structural units this document carries, in order.

    Reads at most ``limit``; a caller needing to know whether it read all
    of them asks for one more than it wants.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not content:
        return ()
    structure = parse_structure(content)
    # Headings and tables come from the shared parser; code fences come
    # from it too, and are where a clause pattern must not fire.
    heads = {section.start: section.title for section in structure.sections if section.level > 0}
    tables = {block.start: block.end for block in structure.blocks if block.kind == "table"}
    quiet = [(block.start, block.end) for block in structure.blocks
             if block.kind in ("fenced_code", "table")]

    found: list[tuple[int, str, str]] = []
    lines = content.splitlines(keepends=True)
    begin = _after_front_matter(lines)
    byte = sum(len(line.encode()) for line in lines[:begin])
    for index in range(begin, len(lines)):
        line = lines[index]
        stripped = line.rstrip("\r\n")
        inside = any(start <= byte < end for start, end in quiet)
        if byte in heads:
            found.append((byte, "heading", stripped.strip()))
        elif byte in tables:
            found.append((byte, "table", stripped.strip()))
        elif not inside and stripped.strip():
            kind = ""
            if (index + 1 < len(lines)
                    and _UNDERLINE.match(lines[index + 1].rstrip("\r\n"))
                    and not _UNDERLINE.match(stripped)):
                kind = "heading"
            elif _CLAUSE.match(stripped):
                kind = "clause"
            elif _PAIR.match(stripped):
                kind = "pair"
            if kind:
                found.append((byte, kind, stripped.strip()))
        if len(found) >= limit:
            break
        byte += len(line.encode())

    ends = {start: tables[start] for start, _, _ in found if start in tables}
    wanted = ({start for start, _, _ in found} | set(ends.values())
              | {len(content.encode())})
    at = _points(content, wanted)
    starts = [start for start, _, _ in found]
    out: list[Unit] = []
    for position, (start, kind, label) in enumerate(found):
        after = starts[position + 1] if position + 1 < len(starts) else len(content.encode())
        end = min(ends[start], after) if start in ends else after
        out.append(Unit(at[start], _point(at, end, content), kind, label))
        if end < after:
            # A table ends where its rows end, not where the next unit
            # begins, so the prose between the two belongs to neither.
            # Until this it belonged to no chunk at all and was silently
            # unretrievable -- the only unit boundary that does not
            # abut the next one.
            between = Unit(_point(at, end, content), _point(at, after, content), "text", "")
            if content[between.start:between.end].strip():
                out.append(between)
    return tuple(out)


def _point(at: dict[int, int], byte: int, content: str) -> int:
    return at[byte] if byte in at else len(content)


def structured_spans(content: str, target: int = DEFAULT_TARGET, *,
                     units_max: int = MAX_SECTIONS) -> Structured:
    """Chunk at the document's own divisions, falling back to size."""
    if target <= 0:
        raise ValueError("target must be positive")
    if units_max <= 0:
        raise ValueError("units_max must be positive")
    read = units(content, units_max + 1)
    capped = len(read) > units_max
    read = read[:units_max]
    if not read:
        plain = chunk_spans(content, target)
        return Structured(spans=tuple(plain), by_size=len(plain),
                          why=f"no structure found, so {len(plain)} chunk(s) were split by size "
                              f"exactly as they would have been without this pass")

    tail = len(content) if not capped else read[-1].end
    spans: list[Span] = []
    at_boundary = by_size = kept_tables = over = 0

    opening = content[:read[0].start]
    if opening.strip():
        # Text before the first boundary is a unit the document did not
        # label, and splitting it by size is the honest reading.
        before = [Span(s.start, s.end) for s in chunk_spans(opening, target)]
        spans.extend(before)
        by_size += len(before)

    def only_a_marker(unit: Unit) -> bool:
        """A unit that is its own first line and nothing else.

        A heading above a table is such a unit, because the table became
        a unit of its own. The heading says what the table means, the
        same job one level up from the header row, so it belongs in the
        table's chunk rather than left at the end of the one before.
        """
        body = content[unit.start:unit.end].split("\n", 1)
        return unit.kind != "table" and (len(body) == 1 or not body[1].strip())

    group: tuple[int, int] | None = None
    # Where the group's trailing run of marker-only units begins, if its
    # last unit is one. A group usually starts with a prose section and
    # ends with the heading above a table, so it has to split before that
    # run rather than refuse to split at all.
    run: int | None = None
    for index, unit in enumerate(read):
        end = unit.end if index + 1 < len(read) else tail
        whole = unit.kind == "table"
        if end - unit.start > target or whole:
            # Only a heading running right up to the table counts: a gap
            # means prose between them, and that prose is its own unit.
            joins = whole and group is not None and run is not None and group[1] == unit.start
            if group is not None:
                if joins and run is not None:
                    if run > group[0]:
                        spans.append(Span(group[0], run))
                        at_boundary += 1
                else:
                    spans.append(Span(*group))
                    at_boundary += 1
            if whole:
                start = run if joins and run is not None else unit.start
                group, run = None, None
                # A table's header is what its rows mean. Splitting it
                # hands back rows nobody can read.
                spans.append(Span(start, end))
                at_boundary += 1
                kept_tables += 1
                over += 1 if end - start > target else 0
                continue
            group, run = None, None
            inside = [Span(unit.start + s.start, unit.start + s.end)
                      for s in chunk_spans(content[unit.start:end], target)]
            spans.extend(inside)
            at_boundary += 1
            by_size += len(inside) - 1
            continue
        if group is None:
            group = (unit.start, end)
            run = unit.start if only_a_marker(unit) else None
        elif end - group[0] <= target:
            group = (group[0], end)
            run = (run if run is not None else unit.start) if only_a_marker(unit) else None
        else:
            spans.append(Span(*group))
            at_boundary += 1
            group = (unit.start, end)
            run = unit.start if only_a_marker(unit) else None
    if group is not None:
        spans.append(Span(*group))
        at_boundary += 1

    if capped:
        rest = content[tail:]
        if rest.strip():
            after = [Span(tail + s.start, tail + s.end) for s in chunk_spans(rest, target)]
            spans.extend(after)
            by_size += len(after)
    elif spans:
        # The last chunk runs to the end, so the only text outside a chunk
        # is whitespace.
        spans[-1] = Span(spans[-1].start, len(content))

    labelled = sum(1 for unit in read if unit.kind != "text")
    why = (f"{at_boundary} chunk(s) begin at one of {labelled} structural boundary(ies)"
           if at_boundary else f"{labelled} structural boundary(ies) found, none usable as a "
                               f"chunk start")
    if by_size:
        why += (f"; {by_size} begin where the byte target fell, inside a unit longer than "
                f"{target} or outside any unit")
    if kept_tables:
        why += f"; {kept_tables} table(s) were kept whole, header with rows"
    if over:
        why += (f"; {over} chunk(s) are longer than {target} because splitting them would have "
                f"broken what they are")
    if capped:
        why += (f"; the document holds more than {units_max} structural units and this read "
                f"{units_max} of them, not all of them -- the rest was split by size")
    return Structured(spans=tuple(spans), units=labelled, at_boundary=at_boundary,
                      by_size=by_size, tables=kept_tables, over_target=over, capped=capped,
                      why=why)
