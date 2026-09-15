"""Chunking profiles: a genre's own boundaries, declared, over the structure chunker.

``structure_chunks`` reads what any document carries and packs it up to
the target. It cannot know what a genre knows: that ``Article 2`` owns the
``(a)`` below it, that ``A:`` belongs to the ``Q:`` above it, that a
references list is not part of the conclusion it follows, or that
``Section 4.2 Transfers`` is a section at all. The leading document
pipeline has a chunker per genre for this -- laws cut at article
numbering, papers keep the abstract whole, Q&A files make one chunk per
pair -- each a separate program chosen by whoever uploads the file.

Here a profile is data: a name and a tuple of ``Rule``s, each a line
pattern with a rank. The structure chunker's line scan asks the profile's
reader what a line is, so headings, tables, fences and front matter are
read exactly as they are without one. Then the units are packed as a
tree instead of a list:

- **A subtree that fits the target is never split.** An article with its
  clauses, a procedure with its steps, is one chunk when it can be.
- **A heading travels with its first child.** When a subtree is too big,
  its own text goes into the first chunk of its children if it is only a
  marker or the two fit together, so ``Article 2`` never ends a chunk
  while ``(a)`` begins the next. A marker shorter than ``MIN_CHUNK`` stays
  with a child that fits the target alone even when the two do not: that
  chunk is over by less than ``MIN_CHUNK``, and counted.
- **An ``alone`` unit never shares a chunk with a sibling**: each
  question-and-answer pair, each paper or resume section, the references.
- **Structure that is not there is not invented.** A named marker must
  be followed by a title, not a sentence (``Section 3 of this Act``), a
  section name must be the whole line (``Experience shows``), and a
  question must open a paragraph.

Nothing here rewrites a byte: the title path is not prepended to a chunk
as the reference implementation does, because stored text must stay an
exact excerpt. The receipt names the profile, how often each rule matched
and which kind of unit each chunk began at.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Mapping

from ..core.errors import InvalidInput
from .chunker import DEFAULT_TARGET, MIN_CHUNK, Span, chunk_spans
from .structure_chunks import MAX_SECTIONS, Structured, Unit, units

#: Where a rule's own ranks start, so that a Markdown heading (depth 1..6)
#: outranks any marker a profile infers from plain text.
_BELOW_HEADINGS = 6
#: A named marker must be followed by the end of the line, or by punctuation
#: or a title -- not by a lowercase word, which makes it a sentence. The
#: point in ``3.2`` is not punctuation, a lowercase word after punctuation is
#: still a sentence (``Chapter 3. of``), and so is one after a sub-reference
#: (``Article 6 (1) of``).
_TITLED = (r"(?=[ \t]*$|[ \t]*[.:\-–—](?![ \t]*[a-z\d])"
           r"|[ \t]+(?:[A-Z\"'‘“\[]|\((?![\da-z]{1,5}\)[ \t]+[a-z])))")
#: An enumerator is followed by text, or ends its line.
_THEN = r"(?=[ \t]+\S|[ \t]*$)"
_CJK_NUMBER = "[〇零一二三四五六七八九十百千0-9]+"


@dataclass(frozen=True)
class Rule:
    """One kind of line a genre divides at."""

    name: str
    pattern: re.Pattern[str]
    #: Rank among the profile's markers, 1 outermost. None ranks by first
    #: appearance under the nearest ranked unit: the United States code
    #: puts ``(a)`` above ``(1)`` and the European Union ``1.`` above ``(a)``,
    #: and both are right about their own documents.
    level: int | None = 1
    #: Never shares a chunk with a sibling.
    alone: bool = False
    #: False for a line that is counted but is not a place to cut: an answer
    #: belongs to its question.
    cuts: bool = True
    #: Only matches inside an open unit of this name.
    under: str | None = None
    #: Only matches as the first line of a paragraph.
    opens_paragraph: bool = False


@dataclass(frozen=True)
class Profile:
    name: str
    about: str
    rules: tuple[Rule, ...]

    def reader(self) -> "ProfileReader":
        return ProfileReader(self)


class ProfileReader:
    """Names each line for one pass over one document, and counts matches."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.matched: Counter[str] = Counter()
        self._open: list[tuple[int, str]] = []
        #: Enumerator styles in the order this article first used them.
        self._styles: dict[str, int] = {}
        #: The last letter enumerator read in this article.
        self._letter = ""
        #: The depth each rule rank last took as a Markdown heading.
        self._placed: dict[int, int] = {}
        ranked = [rule.level for rule in profile.rules if rule.level is not None]
        self._enumerators = _BELOW_HEADINGS + max(ranked, default=0) + 1

    def heading(self, title: str, level: int) -> tuple[str, int]:
        rule = self._rule(title, True)
        kind = "heading" if rule is None else rule.name
        if rule is not None and rule.level is not None:
            self._placed[rule.level] = level
        return self._opens(level, kind)

    def line(self, text: str, opens_paragraph: bool) -> tuple[str, int] | None:
        rule = self._rule(text, opens_paragraph)
        if rule is None or not rule.cuts:
            return None
        if rule.level is not None:
            return self._opens(self._ranked(rule.level), rule.name)
        rank = self._styles.setdefault(rule.name, len(self._styles))
        return rule.name, self._push(self._enumerators + rank, rule.name)

    def _opens(self, depth: int, name: str) -> tuple[str, int]:
        """A heading or a ranked marker: the enumerators under it start again,
        in their order and in which letter came last."""
        self._styles, self._letter = {}, ""
        return name, self._push(depth, name)

    def _ranked(self, level: int) -> int:
        """Where a plain line of this rank sits: where the rank sat as a
        heading, or just above the nearest deeper rank that did, so
        ``References`` after ``## Conclusion`` is its sibling and not its
        child. With neither, below every heading."""
        if level in self._placed:
            return self._placed[level]
        deeper = [below for below in self._placed if below > level]
        if deeper:
            return self._placed[min(deeper)] - (min(deeper) - level)
        return _BELOW_HEADINGS + level

    def _rule(self, text: str, opens_paragraph: bool) -> Rule | None:
        for rule in self.profile.rules:
            if rule.opens_paragraph and not opens_paragraph:
                continue
            if rule.under is not None and all(name != rule.under for _, name in self._open):
                continue
            if rule.pattern.match(text):
                chosen = self._letter_or_numeral(rule, text)
                self.matched[chosen.name] += 1
                return chosen
        return None

    def _letter_or_numeral(self, rule: Rule, text: str) -> Rule:
        """``(i)`` after ``(h)`` in the same article is the ninth letter, not
        the first numeral -- unless the innermost open unit is a capital, as in the
        United States code's (h)(1)(A)(i), where it is a clause."""
        if rule.name == "letter":
            self._letter = text.lstrip("(")[0]
        elif (rule.name == "roman" and self._letter and text[1:3] == chr(ord(self._letter) + 1) + ")"
              and not (self._open and self._open[-1][1] == "capital")):
            return next(other for other in self.profile.rules if other.name == "letter")
        return rule

    def _push(self, depth: int, name: str) -> int:
        while self._open and self._open[-1][0] >= depth:
            self._open.pop()
        self._open.append((depth, name))
        return depth


def _named(*names: str) -> str:
    return "|".join(names)


_SECTION_NUMBER = r"(?:(?:\d+|[IVX]+|[A-Z])\.?[ \t]+)?"

PROFILES: Mapping[str, Profile] = {profile.name: profile for profile in (
    Profile("statute", "laws, regulations and contracts: parts, chapters, articles and their clauses", (
        Rule("part", re.compile(r"(?:PART|Part|TITLE|Title|BOOK|Book|DIVISION|Division)[ \t]+"
                                r"(?:\d+[A-Z]?|[IVXLC]+|[A-Z]|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b"
                                + _TITLED), level=1),
        Rule("chapter", re.compile(r"(?:(?:CHAPTER|Chapter|SUBCHAPTER|Subchapter)[ \t]+(?:\d+[A-Z]?|[IVXLC]+|[A-Z])\b"
                                   + _TITLED + "|第" + _CJK_NUMBER + "[章编])"), level=2),
        Rule("article", re.compile(r"(?:(?:(?:ARTICLE|Article|Art\.|SECTION|Section|Sec\.|CLAUSE|Clause|Rule)[ \t]+|§§?[ \t]*)"
                                   r"\d+[A-Za-z]?(?:\.\d+)*" + _TITLED + "|第" + _CJK_NUMBER + "条)"), level=3),
        Rule("subsection", re.compile(r"\d+(?:\.\d+)+\.?" + _THEN), level=None),
        Rule("paragraph", re.compile(r"\d{1,3}[.)]" + _THEN), level=None),
        Rule("numbered", re.compile(r"\(\d{1,3}\)" + _THEN), level=None),
        Rule("roman", re.compile(r"\((?:i{1,3}|iv|vi{0,3}|ix|xi{0,3})\)" + _THEN), level=None),
        Rule("letter", re.compile(r"(?:\([a-z]{1,2}\)|[a-z]\))" + _THEN), level=None),
        Rule("capital", re.compile(r"\([A-Z]{1,2}\)" + _THEN), level=None),
    )),
    Profile("paper", "research papers: abstract and sections, with the references list kept apart", (
        Rule("abstract", re.compile(_SECTION_NUMBER + r"(?i:abstract)(?:[ \t]*[:.—-].*)?$"), alone=True),
        Rule("references", alone=True, pattern=re.compile(
            _SECTION_NUMBER + r"(?i:references|bibliography|works cited|literature cited)[ \t]*:?$")),
        Rule("section", re.compile(_SECTION_NUMBER + "(?i:" + _named(
            "introduction", "background", "related work", "preliminaries", "methods?", "methodology",
            "materials and methods", "approach", "experiments?", "experimental setup", "evaluation", "results",
            "results and discussion", "discussion", "analysis", "limitations", "conclusions?", "future work",
            "acknowledge?ments", r"appendix(?: [A-Z])?", "appendices") + r")[ \t]*:?$"), alone=True),
        Rule("subsection", re.compile(r"\d+(?:\.\d+)+\.?[ \t]+[A-Z][^.!?]*$"), level=2),
        Rule("reference", re.compile(r"(?:\[\d{1,4}\]|\d{1,4}\.)[ \t]+\S"), level=2, under="references"),
    )),
    Profile("manual", "manuals and runbooks: numbered steps kept whole under the task they belong to", (
        Rule("task", re.compile(r"(?:To|How to)[ \t]+\S.{0,100}:$"), level=1),
        Rule("procedure", re.compile(r"(?:\d+(?:\.\d+)+\.?|(?:Chapter|Section|Procedure)[ \t]+\d+(?:\.\d+)*[.:]?)[ \t]+[A-Z]"),
             level=1),
        Rule("step", re.compile(r"(?:(?i:step)[ \t]*\d+[.:)]?|\d{1,3}[.)])" + _THEN), level=2),
        Rule("substep", re.compile(r"(?:[a-z][.)]|\([a-z]\))" + _THEN), level=3),
    )),
    Profile("qa", "questions and answers: each question with its answer, one pair per chunk", (
        Rule("question", re.compile(r"(?:Q\d*|Question(?:[ \t]+\d+)?)[ \t]*[:.)][ \t]*\S"), alone=True),
        Rule("answer", re.compile(r"(?:A\d*|Answer)[ \t]*[:.)][ \t]*\S"), cuts=False),
        Rule("question", re.compile(r".{2,160}\?$"), alone=True, opens_paragraph=True),
    )),
    Profile("resume", "resumes: summary, experience, education, skills and the rest, a section per chunk", (
        Rule("section", re.compile("(?i:" + _named(
            "summary", "professional summary", "profile", "objective", "(?:work |professional )?experience",
            "employment(?: history)?", "education", "(?:technical |core )?skills", "projects", "certifications?",
            "publications", r"awards(?: and honou?rs)?", "languages", "interests",
            "volunteer(?:ing| experience)?", "references", "contact(?: information)?") + r")[ \t]*:?$"), alone=True),
    )),
)}


def profile_named(name: object) -> Profile:
    """The profile of this name, or a refusal naming the ones there are."""
    if not isinstance(name, str) or name not in PROFILES:
        raise InvalidInput(f"chunking_profile must be one of {', '.join(PROFILES)}, not {name!r}")
    return PROFILES[name]


@dataclass(frozen=True)
class Profiled(Structured):
    """Structure chunking under a profile: the counts, and what the rules found."""

    profile: str = ""
    #: How many lines each rule matched, including rules that do not cut.
    matched: dict[str, int] = field(default_factory=dict)
    #: How many chunks began at each kind of unit.
    began: dict[str, int] = field(default_factory=dict)

    def record(self) -> dict[str, object]:
        return super().record() | {"profile": self.profile, "matched": dict(self.matched),
                                   "began": dict(self.began)}


def profiled_spans(content: str, target: int = DEFAULT_TARGET, *, profile: str,
                   units_max: int = MAX_SECTIONS) -> Profiled:
    """Chunk at the boundaries this profile declares, falling back to size.

    A target that is not positive is refused by ``chunk_spans``, which
    every path reaches."""
    if units_max <= 0:
        raise ValueError("units_max must be positive")
    chosen = profile_named(profile)
    reader = chosen.reader()
    read = units(content, units_max + 1, reader=reader)
    capped = len(read) > units_max
    read = read[:units_max]
    if not read:
        plain = chunk_spans(content, target)
        return Profiled(spans=tuple(plain), by_size=len(plain), profile=chosen.name, matched=dict(reader.matched),
                        why=f"profile {chosen.name} found none of its boundaries, so {len(plain)} chunk(s) were "
                            f"split by size exactly as they would have been without it")
    return _Packer(content, target, chosen, read, capped, units_max, dict(reader.matched)).pack()


class _Packer:
    """Units packed as a tree: whole subtrees first, a heading with its first child."""

    def __init__(self, content: str, target: int, profile: Profile, read: tuple[Unit, ...], capped: bool,
                 units_max: int, matched: dict[str, int]) -> None:
        self.content, self.target, self.profile, self.read = content, target, profile, read
        self.capped, self.units_max, self.matched = capped, units_max, matched
        self.tail = read[-1].end if capped else len(content)
        self.alone = {rule.name for rule in profile.rules if rule.alone}
        self.spans: list[Span] = []
        self.began: Counter[str] = Counter()
        self.by_size = 0
        # Every table is kept whole: inside a subtree that fits, or on its own.
        self.tables = sum(1 for unit in read if unit.kind == "table")
        # Where each unit's subtree ends: the next unit no deeper than it.
        self.last = [len(read)] * len(read)
        stack: list[int] = []
        for index, unit in enumerate(read):
            while stack and read[stack[-1]].depth >= unit.depth:
                self.last[stack.pop()] = index
            stack.append(index)

    def end(self, index: int) -> int:
        after = self.last[index]
        return self.read[after].start if after < len(self.read) else self.tail

    def children(self, index: int) -> list[int]:
        found, child = [], index + 1
        while child < self.last[index]:
            found.append(child)
            child = self.last[child]
        return found

    def whole(self, start: int, end: int, kind: str) -> None:
        self.spans.append(Span(start, end))
        self.began[kind] += 1

    def split(self, start: int, end: int, kind: str) -> None:
        """The first piece begins at the unit; the rest where the target fell."""
        pieces = [Span(start + piece.start, start + piece.end)
                  for piece in chunk_spans(self.content[start:end], self.target)]
        self.whole(pieces[0].start, pieces[0].end, kind)
        self.spans.extend(pieces[1:])
        self.by_size += len(pieces) - 1

    def place(self, indices: list[int], lead: tuple[int, str] | None = None) -> None:
        """Siblings in order, packed while they fit. ``lead`` is where the
        first of them starts when its parent's own text travels with it."""
        group: tuple[int, int, str] | None = None
        for position, index in enumerate(indices):
            unit = self.read[index]
            start, kind = lead if position == 0 and lead is not None else (unit.start, unit.kind)
            end = self.end(index)
            packs = unit.kind not in self.alone and end - start <= self.target
            if packs and group is not None and end - group[0] <= self.target:
                group = (group[0], end, group[2])
                continue
            if group is not None:
                self.whole(*group)
            group = (start, end, kind) if packs else None
            if packs:
                continue
            if unit.kind == "table":
                # Over the target, and still whole: its header is what its rows mean.
                self.whole(start, end, kind)
            elif end - unit.start > self.target:
                self.descend(index, start, kind)
            else:
                # Alone, and it fits; or it fits without the heading before
                # it, which is then shorter than MIN_CHUNK and comes along.
                self.whole(start, end, kind)
        if group:
            self.whole(*group)

    def descend(self, index: int, start: int, kind: str) -> None:
        kids = self.children(index)
        if not kids:
            self.split(start, self.end(index), kind)
            return
        own = self.read[kids[0]].start
        if own - start < MIN_CHUNK or self.end(kids[0]) - start <= self.target:
            self.place(kids, (start, kind))
        else:
            self.split(start, own, kind)
            self.place(kids)

    def pack(self) -> Profiled:
        # Text before the first unit, and past the unit bound, is split by
        # size; either is empty or blank when there is none, and yields nothing.
        opening = chunk_spans(self.content[:self.read[0].start], self.target)
        self.spans.extend(opening)
        self.by_size += len(opening)
        roots, index = [], 0
        while index < len(self.read):
            roots.append(index)
            index = self.last[index]
        self.place(roots)
        rest = chunk_spans(self.content[self.tail:], self.target)
        self.spans.extend(Span(self.tail + piece.start, self.tail + piece.end) for piece in rest)
        self.by_size += len(rest)
        at_boundary = sum(self.began.values())
        # Counted from the chunks rather than from the decisions that make
        # them: a table kept whole is one way over, and a last piece shorter
        # than MIN_CHUNK that the size chunker joins to the one before it is
        # another nobody here decided.
        over = sum(1 for span in self.spans if span.end - span.start > self.target)
        labelled = sum(1 for unit in self.read if unit.kind != "text")
        name = self.profile.name
        why = f"profile {name}: {at_boundary} chunk(s) begin at one of {labelled} boundary(ies) its rules found"
        if self.by_size:
            why += (f"; {self.by_size} begin where the byte target fell, inside a unit longer than "
                    f"{self.target} or outside any unit")
        if self.tables:
            why += f"; {self.tables} table(s) were kept whole, header with rows"
        if over:
            why += (f"; {over} chunk(s) are longer than {self.target}: a table kept whole, a heading shorter "
                    f"than {MIN_CHUNK} kept with the unit under it, or a last piece shorter than {MIN_CHUNK} "
                    f"joined to the one before it")
        if self.capped:
            why += (f"; the document holds more than {self.units_max} structural units and this read "
                    f"{self.units_max} of them, not all of them -- the rest was split by size")
        return Profiled(spans=tuple(self.spans), units=labelled, at_boundary=at_boundary, by_size=self.by_size,
                        tables=self.tables, over_target=over, capped=self.capped, why=why, profile=name,
                        matched=self.matched, began=dict(self.began))
