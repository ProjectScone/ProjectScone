"""What each region of a page is, from a layout engine or from the page itself.

The reading-order pass (`ocr.layout`) says where the columns are and the
running-lines pass (`ingestion.pdf_layout`) which lines repeat at the
edges; neither says what a region *is*. A reader needs that: a heading
is where a section starts, a table is not prose, a footnote is not the
paragraph above it, and a page number is nothing at all. A layout
engine -- anything that draws labelled rectangles on a page image --
can say all of it, and this module maps such an answer onto one
vocabulary (``RegionLabel``). Without an engine it infers the part that
geometry and text can tell on their own: page numbers by their shape
and place, running headers and footers by their repetition, a title and
headings by their size or their isolation, lists by their bullets,
tables by their grid, footnotes by their place and their mark. The rest
is a paragraph, which is what a page mostly is. What nothing here can
tell -- a figure, a formula, a sidebar, a caption -- is an engine's to
say, and stays unsaid otherwise. Every label is a reading of geometry,
not a fact about the document, and the receipt says which rule gave it.
"""
from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import InvalidInput
from .tables import ENUMERATOR
from .types import REGION_LABELS, LayoutResult, OcrRegion, RegionLabel

#: A page number on its own line: "12", "- 12 -", "Page 12", "12 of 40", "12/40".
PAGE_NUMBER = re.compile(r"(?:page\s+)?[-–—]?\s*\d{1,4}(?:\s*[-–—]|\s*(?:of|/)\s*\d{1,4})?", re.IGNORECASE)
#: A line that opens a list item: a bullet, a number or a letter with its mark, then the item.
#: A list item's opening: a bullet, or an enumerator with its closing
#: punctuation -- "1.", "(a)", "(ii)", "2.1." as a definition list
#: numbers its entries -- and a word after it; the same marks the grid
#: inference refuses as a table's first column.
LIST_ITEM = re.compile(rf"^(?:[•·◦▪▫‣●○■□\-–—*]|{ENUMERATOR})\s+\S")
BULLETS = "•·◦▪▫‣●○■□-–—*"
#: A footnote's opening: its number (not the start of a longer one, so a
#: year is not a footnote) or its mark, a space, then its text.
FOOTNOTE = re.compile(r"^(?:\d{1,2}(?!\d)\s+|[*†‡§¹²³⁴⁵⁶⁷⁸⁹]\s*)\S")
#: The strip of the page a page number or a footnote lives in.
EDGE = 0.08
FOOT = 0.8
#: A title sits in the top third of the first page.
TITLE_BELOW = 0.34
#: How much taller than the page's median line a title and a heading are.
TITLE_HEIGHT = 1.4
HEADING_HEIGHT = 1.25
#: A heading set apart by space: this many median line heights above and below.
ISOLATION = 1.5
TITLE_WORDS, HEADING_WORDS, ISOLATED_WORDS = 12, 15, 10
#: Lines a page may hold before the rules decline to read it.
MAX_LINES = 20_000
STRATEGY_INFERRED: Literal['labels-v1'] = 'labels-v1'
STRATEGY_ENGINE: Literal['engine-boxes-v1'] = 'engine-boxes-v1'
Rule = Literal['page_number', 'running', 'table', 'list', 'title', 'heading', 'footnote']

#: How the labels other vocabularies use read in this one. PP-Structure's
#: (PaddleOCR) and deepdoc's (ragflow) names, and this module's own.
ENGINE_LABELS: Mapping[str, RegionLabel] = {
    **{name: name for name in REGION_LABELS},  # type: ignore[misc]
    'doc_title': 'title', 'document title': 'title', 'paragraph_title': 'heading', 'paragraph title': 'heading',
    'section_title': 'heading', 'text': 'paragraph', 'abstract': 'paragraph', 'content': 'paragraph',
    'sidebar text': 'sidebar', 'sidebar_text': 'sidebar', 'aside': 'sidebar',
    'image': 'figure', 'chart': 'figure', 'seal': 'figure', 'picture': 'figure', 'header image': 'header',
    'footer image': 'footer', 'figure_title': 'caption', 'figure title': 'caption', 'figure caption': 'caption',
    'figure_caption': 'caption', 'table_title': 'caption', 'table title': 'caption', 'table caption': 'caption',
    'table_caption': 'caption', 'chart title': 'caption', 'chart_title': 'caption',
    'vision_footnote': 'footnote', 'footnotes': 'footnote', 'page number': 'page_number', 'number': 'page_number',
    'page_number': 'page_number', 'references': 'reference', 'lists of references': 'reference', 'equation': 'formula',
    'formula_number': 'formula', 'formula number': 'formula', 'algorithm': 'code', 'table of contents': 'list',
    'table_of_contents': 'list', 'list': 'list',
}


def as_label(text: str) -> RegionLabel | None:
    """A label as another vocabulary spells it, in this one; None when
    this vocabulary has no name for it, which an engine result counts."""
    return ENGINE_LABELS.get(text.strip().lower().replace('-', '_') if '_' in text or '-' in text else text.strip().lower())


class LayoutLabels(BaseModel):
    """How a page's regions were labelled: by an engine's boxes or by the
    rules, which rules fired, how many regions carry each label, and how
    many carry none (outside every engine box) or were dropped (an engine
    label this vocabulary has no name for)."""
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    source: Literal['engine', 'inferred']
    strategy: Literal['engine-boxes-v1', 'labels-v1']
    engine: str | None = Field(default=None, min_length=1, max_length=96)
    counts: dict[str, int] = Field(default_factory=dict)
    rules: tuple[Rule, ...] = Field(default=(), max_length=7)
    unlabeled: int = Field(default=0, ge=0)
    dropped: int = Field(default=0, ge=0)
    notes: tuple[Literal['line_limit', 'table_limit'], ...] = Field(default=(), max_length=2)


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width * height if width > 0 and height > 0 else 0.0


def label_from_engine(regions: Sequence[OcrRegion], layout: LayoutResult) -> tuple[list[RegionLabel | None], LayoutLabels]:
    """Each region's label from the engine box it lies in most; None for a
    region outside every box. Boxes are in the same displayed-page frame
    as the regions, normalized, so no scaling is needed."""
    labels: list[RegionLabel | None] = []
    for region in regions:
        best, best_area = None, 0.0
        for block in layout.regions:
            area = _overlap(region.box, block.box)
            if area > best_area:
                best, best_area = block.label, area
        labels.append(best)
    counts: dict[str, int] = {}
    for label in labels:
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
    receipt = LayoutLabels(source='engine', strategy=STRATEGY_ENGINE, engine=layout.engine, counts=counts,
                           unlabeled=sum(1 for label in labels if label is None), dropped=layout.dropped)
    return labels, receipt


@dataclass(frozen=True)
class _Line:
    indices: tuple[int, ...]
    text: str
    box: tuple[float, float, float, float]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def words(self) -> int:
        return len(self.text.split())


def _lines(regions: Sequence[OcrRegion]) -> list[_Line]:
    """Regions gathered into the lines the recognizer read them in, in the
    order they came: a text-layer run is a line of its own row."""
    # A text layer's grid numbers runs by row only: two columns' runs on
    # one row are two lines, told apart by the column the reading order gave.
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for index, region in enumerate(regions):
        column = getattr(region, 'reading_column', None)
        groups.setdefault((region.block, region.paragraph, region.line, column if isinstance(column, int) else -1),
                          []).append(index)
    lines: list[_Line] = []
    for indices in groups.values():
        boxes = [regions[i].box for i in indices]
        lines.append(_Line(tuple(indices), ' '.join(regions[i].text for i in indices),
                           (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))))
    lines.sort(key=lambda line: (line.box[1], line.box[0]))
    return lines


def running_key(text: str) -> str:
    """A line as a running-line key: one space between words, every number
    the same, so "Chapter 3 · 41" matches "Chapter 3 · 42"."""
    return re.sub(r"\d+", "#", " ".join(text.split()))


#: A line at a page's edge on this many pages is a running header or footer.
RUNNING_PAGES = 3


def running_keys(pages: Mapping[int, Sequence[OcrRegion]]) -> dict[str, int]:
    """The running-line keys of a document's recognized pages, by page
    number: the text of a line within the page's top or bottom strip that
    recurs on ``RUNNING_PAGES`` pages or more, numbers made alike so
    "Chapter 3 · 41" on one page matches "Chapter 3 · 42" on the next --
    each with the first page it appears on, where it may be a title rather
    than a running line, as ``ingestion.pdf_layout`` reads a text layer."""
    pages_of: dict[str, set[int]] = {}
    for number, regions in pages.items():
        for line in _lines(regions):
            if line.box[1] < EDGE * 2 or line.box[3] > 1 - EDGE * 2:
                key = running_key(line.text)
                if key.strip("# -–—/"):
                    pages_of.setdefault(key, set()).add(number)
    return {key: min(seen) for key, seen in pages_of.items() if len(seen) >= RUNNING_PAGES}


def infer_labels(regions: Sequence[OcrRegion], *, first_page: bool = False,
                 running: Collection[str] = (), sized: bool = True) -> tuple[list[RegionLabel | None], LayoutLabels]:
    """Each region's label by the rules, and the receipt. ``running`` holds
    the running-line keys (``running_key``) found across the document's
    pages; ``sized`` says whether line heights mean anything (an OCR
    raster: yes; a text layer's character grid, where every row is one
    unit tall: no, and a heading is told by its isolation alone)."""
    from .tables import infer_tables

    lines = _lines(regions)
    notes: list[Literal['line_limit', 'table_limit']] = []
    if len(lines) > MAX_LINES:
        receipt = LayoutLabels(source='inferred', strategy=STRATEGY_INFERRED, notes=('line_limit',))
        return [None] * len(regions), receipt
    # A page the reading order split into columns is not a table for
    # having columns: the grid is looked for within each column.
    tabular: set[int] = set()
    by_column: dict[int, list[int]] = {}
    for index, region in enumerate(regions):
        column = getattr(region, 'reading_column', None)
        by_column.setdefault(column if isinstance(column, int) else 0, []).append(index)
    for indices in by_column.values():
        try:
            for table in infer_tables([regions[i] for i in indices]).tables:
                for cell in table.cells:
                    tabular.update(indices[i] for i in cell.regions)
        except InvalidInput:
            if 'table_limit' not in notes:
                notes.append('table_limit')
    heights = [line.height for line in lines if line.height > 0]
    typical = median(heights) if heights else 0.0
    keys = {running_key(text) for text in running}
    # A list item is a bulleted line, or a numbered or lettered line with
    # another such line beside it: two numbered headings a page apart are
    # not a list, and neither is a lone "1." that opens a sentence.
    marked = [position for position, line in enumerate(lines) if LIST_ITEM.match(line.text.strip())]
    items: set[int] = set()
    for position in marked:
        text = lines[position].text.strip()
        if text[0] in BULLETS:
            items.add(position)
            continue
        for other in (position - 1, position + 1):
            if other in marked and typical > 0 and abs(lines[other].box[1] - lines[position].box[1]) <= 2.5 * typical:
                items.add(position)
    if len(items) < 2:
        items = set()
    labels: list[RegionLabel | None] = [None] * len(regions)
    rules: list[Rule] = []
    titled = False

    given: list[RegionLabel | None] = []

    # Space that sets a line apart: on a raster, a good part of a line's
    # height; on a grid, whose rows are one unit tall, one blank row.
    apart = (ISOLATION if sized else 0.9) * typical

    def isolated_above(position: int) -> bool:
        if typical <= 0 or position == 0:
            return typical > 0
        return lines[position].box[1] - lines[position - 1].box[3] >= apart

    def isolated(position: int) -> bool:
        if typical <= 0:
            return False
        line = lines[position]
        below = lines[position + 1].box[1] - line.box[3] if position + 1 < len(lines) else apart
        return isolated_above(position) and below >= apart

    def labels_of(position: int) -> RegionLabel | None:
        return given[position] if 0 <= position < len(given) else None

    for position, line in enumerate(lines):
        text = line.text.strip()
        top, bottom = line.box[1], line.box[3]
        label: RegionLabel
        rule: Rule | None
        if PAGE_NUMBER.fullmatch(text) and (top < EDGE or bottom > 1 - EDGE):
            label, rule = 'page_number', 'page_number'
        elif keys and running_key(text) in keys and (top < EDGE * 2 or bottom > 1 - EDGE * 2):
            label, rule = ('header' if top < 0.5 else 'footer'), 'running'
        elif tabular and any(index in tabular for index in line.indices):
            label, rule = 'table', 'table'
        elif (top > FOOT and FOOTNOTE.match(text) and (not sized or typical <= 0 or line.height <= 0.9 * typical)
              and (sized or position == 0 or isolated_above(position) or labels_of(position - 1) == 'footnote')):
            # Before the heading rule: a footnote is short, set apart and
            # unpunctuated too, and its place and its mark are what tell it.
            # On a grid, which has no sizes, it sits under a gap or another.
            label, rule = 'footnote', 'footnote'
        elif (first_page and not titled and top < TITLE_BELOW and line.words <= TITLE_WORDS
              and ((sized and typical > 0 and line.height >= TITLE_HEIGHT * typical) or (not sized and isolated(position)))):
            label, rule, titled = 'title', 'title', True
        elif (line.words <= HEADING_WORDS and not text.endswith(('.', ',', ';', ':', '?', '!'))
              and position not in items
              and ((sized and typical > 0 and line.height >= HEADING_HEIGHT * typical)
                   or (line.words <= ISOLATED_WORDS and isolated(position) and text[:1].isalnum()))):
            # Before the list rule: a numbered heading ("3. Results") has a
            # list item's shape, and its size or the space around it is what
            # tells it, unless numbered lines sit beside it.
            label, rule = 'heading', 'heading'
        elif position in items:
            label, rule = 'list', 'list'
        else:
            label, rule = 'paragraph', None
        if rule is not None and rule not in rules:
            rules.append(rule)
        given.append(label)
        for index in line.indices:
            labels[index] = label
    counts: dict[str, int] = {}
    for each in labels:
        if each is not None:
            counts[each] = counts.get(each, 0) + 1
    receipt = LayoutLabels(source='inferred', strategy=STRATEGY_INFERRED, counts=counts, rules=tuple(rules),
                           notes=tuple(notes))
    return labels, receipt
