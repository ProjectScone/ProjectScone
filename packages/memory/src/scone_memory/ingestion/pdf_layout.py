"""Reading order for a PDF's text layer, from the grid the text layer already gives.

pypdf's layout mode puts every run of glyphs on a character grid by
its position on the page. On a two-column page that grid holds both
columns side by side, so reading it row by row interleaves them: the
stored text carries half a sentence from each column on every line,
and a chunk cut anywhere in it quotes two paragraphs at once. No
layout model is needed to see this; the grid's whitespace is the
geometry. Each run of characters on a row becomes a region with a box
estimated from its grid position, the same whitespace partitioning
that orders OCR words orders those regions, and the page is written
out column by column. Where that order would read a table column-wise
-- a page whose sides are not prose -- the page stays as extracted and
the receipt says so.

A line repeated at the top or bottom of three or more pages is a
running header or footer: it stays where it first appears and goes
after, and a page number on a line of its own goes everywhere. Every
line removed is recorded on its page, so nothing extracted is lost.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import median
from typing import Literal, Sequence

from ..ocr.labels import LayoutLabels, infer_labels
from ..ocr.layout import ReadingOrderReceipt, order_columns
from ..ocr.types import OcrRegion
from .pdf import MAX_REGIONS_PER_PAGE, PdfTextRegion

#: Spaces between two runs on a row before they are two regions. Justified
#: prose opens gaps of two or three; a column gutter or a table cell gap
#: is wider.
RUN_GAP = 4
#: What a side of a cut must look like before it is read on its own: at
#: least this many characters wide at its widest, at least half its rows a
#: single run, and the typical row at least half the widest. A table's
#: number column fails the first, the rest of a table the second, a
#: label column usually the third; a prose column with a small table in
#: it passes all three, and the table's rows stay whole inside it.
MIN_PROSE_WIDTH = 15
PROSE_ROW_SHARE = 0.5
PROSE_WIDTH_SHARE = 0.5
#: A line at a page's edge on this many pages is a running header or footer.
RUNNING_PAGES = 3
#: Filled rows at each end of a page that can be running lines.
EDGE_ROWS = 2

GRID: Literal['grid-columns-v1'] = 'grid-columns-v1'

#: A run of text: anything but the grid's own spaces. The layout grid is
#: drawn with ASCII spaces, so any other space -- the no-break space in a
#: table's "$62.0\xa0billion", a thin space in "1\u2009000" -- is text. A
#: currency sign after a space opens a run of its own: a statement sets
#: the next column's sign as close to a number as its own words sit.
_CURRENCY = "$\u20ac\u00a3\u00a5"
_RUN = re.compile(rf"[^ \n](?:[^ \n]| {{1,{RUN_GAP - 1}}}(?=[^ \n{_CURRENCY}]))*")
_PAGE_NUMBER = re.compile(r"(?:page\s+)?[-–—]?\s*\d{1,4}(?:\s*[-–—]|\s*(?:of|/)\s*\d{1,4})?", re.IGNORECASE)


def _normal(row: str) -> str:
    """A row as a running-line key: one space between words, every number
    the same, so "Chapter 3  · 41" on one page matches "Chapter 3 · 42" on
    the next."""
    return re.sub(r"\d+", "#", " ".join(row.split()))


def _edges(rows: Sequence[str]) -> list[int]:
    filled = [index for index, row in enumerate(rows) if row.strip()]
    return sorted(set(filled[:EDGE_ROWS] + filled[-EDGE_ROWS:]))


def running_rows(pages: Sequence[Sequence[str]]) -> list[dict[int, str]]:
    """Per page, the edge rows to leave out and their text.

    A page number on its own row goes on every page. A line whose words
    recur at the top or bottom of ``RUNNING_PAGES`` pages or more goes on
    every page but the first it appears on, which may be the page where
    it is a title. No page is emptied by this."""
    edges = [_edges(rows) for rows in pages]
    pages_of: dict[str, list[int]] = {}
    for number, rows in enumerate(pages):
        for index in edges[number]:
            key = _normal(rows[index])
            if key.strip("# -–—/"):
                seen = pages_of.setdefault(key, [])
                if not seen or seen[-1] != number:
                    seen.append(number)
    left_out: list[dict[int, str]] = []
    for number, rows in enumerate(pages):
        drop: dict[int, str] = {}
        filled = sum(1 for row in rows if row.strip())
        for index in edges[number]:
            text = rows[index].strip()
            recurring = pages_of.get(_normal(rows[index]), [])
            running = len(recurring) >= RUNNING_PAGES and recurring[0] != number
            if (running or _PAGE_NUMBER.fullmatch(text)) and filled - len(drop) > 1:
                drop[index] = text
        left_out.append(drop)
    return left_out


@dataclass(frozen=True)
class PageLayout:
    """A page's text in reading order, the regions that order came from
    (none when the page is left as extracted), the receipt that says what
    was found, and the running lines left out."""
    text: str
    regions: tuple[PdfTextRegion, ...]
    receipt: ReadingOrderReceipt | None
    running: tuple[str, ...]
    #: What each region is, by the rules (`ocr.labels`); None when the
    #: page was left as extracted and has no regions to label.
    labels: LayoutLabels | None = None


def _reads_as_prose(runs: Sequence[tuple[int, int, int]], indices: Sequence[int]) -> bool:
    """Whether one side of a proposed cut reads like prose: rows that are
    mostly one run each, and about as wide as each other. A table's label
    column is narrow and ragged, its number columns are several runs per
    row."""
    spans: dict[int, list[tuple[int, int]]] = {}
    for index in indices:
        row, start, end = runs[index]
        spans.setdefault(row, []).append((start, end))
    widths = [max(end for _, end in cells) - min(start for start, _ in cells) for cells in spans.values()]
    single = sum(1 for cells in spans.values() if len(cells) == 1)
    return (len(spans) >= 3 and max(widths) >= MIN_PROSE_WIDTH and single >= PROSE_ROW_SHARE * len(spans)
            and median(widths) >= PROSE_WIDTH_SHARE * max(widths))


def lay_out_page(rows: Sequence[str], offset: int, *, drop: dict[int, str] | None = None,
                 direction: Literal['ltr', 'rtl'] = 'ltr', first_page: bool = False) -> PageLayout:
    """The page's rows, as pypdf's layout mode wrote them, in reading order.

    ``offset`` is where the page's text begins in the document's UTF-8
    bytes, so the regions carry absolute spans like OCR regions do.
    ``drop`` names running rows to leave out, from ``running_rows``.
    ``first_page`` lets the rules read a title. The regions come back
    labelled by the rules (a grid has no sizes, so a heading is told by
    its isolation), and the receipt says which rules fired."""
    drop = drop or {}
    kept = [row for index, row in enumerate(rows) if index not in drop]
    running = tuple(drop[index] for index in sorted(drop))
    # Only the grid's own spaces are stripped: a row ending in a no-break
    # space keeps it, as its run does, so the spans below stay inside the text.
    as_extracted = "\n".join(kept).rstrip(' \n')
    runs = [(row, match.start(), match.end()) for row, line in enumerate(kept) for match in _RUN.finditer(line)]
    if not runs:
        return PageLayout(as_extracted, (), None, running)
    if len(runs) > MAX_REGIONS_PER_PAGE:
        receipt = ReadingOrderReceipt(strategy=GRID, direction=direction, columns=0, notes=('no_separating_gutter', 'region_limit'))
        return PageLayout(as_extracted, (), receipt, running)
    width = max(len(line) for line in kept)
    height = len(kept)
    if width < RUN_GAP:
        # A page narrower than one gutter holds no gutter: a lone numeral,
        # a colophon's "II". It stays as extracted, and says so.
        receipt = ReadingOrderReceipt(strategy=GRID, direction=direction, columns=0, notes=('no_separating_gutter',))
        return PageLayout(as_extracted, (), receipt, running)
    regions = [OcrRegion(text=kept[row][start:end], line=row,
                         box=(start / width, row / height, end / width, (row + 1) / height))
               for row, start, end in runs]
    # A grid gutter is exact: a few characters of nothing between two
    # columns, or between an equation's number and the next column. Each
    # side of a cut must read as prose, or the cut is refused and the
    # table it would have split stays whole.
    order = order_columns(regions, direction=direction, min_gap=(RUN_GAP - 0.5) / width,
                          accept=lambda indices: _reads_as_prose(runs, indices))
    receipt = order.receipt.model_copy(update={'strategy': GRID})
    if not receipt.columns:
        # A page whose order is kept carries its regions when the rules
        # find a table among them, so the grid can be carried as cells;
        # a page of prose stays as extracted, with nothing to carry.
        labelled, labels = infer_labels(regions, first_page=first_page, running=running, sized=False)
        if 'table' not in labelled:
            return PageLayout(as_extracted, (), receipt, running)
        starts = [0]
        for line in kept:
            starts.append(starts[-1] + len(line.encode('utf-8')) + 1)
        kept_regions = tuple(
            PdfTextRegion(**regions[index].model_dump(), provider_index=index, reading_column=0,
                          start=offset + starts[row] + len(kept[row][:start].encode('utf-8')),
                          end=offset + starts[row] + len(kept[row][:end].encode('utf-8'))).model_copy(update={'label': label})
            for index, ((row, start, end), label) in enumerate(zip(runs, labelled)))
        return PageLayout(as_extracted, kept_regions, receipt, running, labels)
    parts: list[str] = []
    ordered: list[PdfTextRegion] = []
    previous: tuple[int, int, int] | None = None
    for position, index in enumerate(order.indices):
        row, start, end = runs[index]
        column = order.columns[position]
        if previous is None:
            separator = ''
        elif previous[0] == row and previous[2] == column and start >= previous[1]:
            # Two runs of one row read together keep the gap between them,
            # so an equation number or a table cell sits where it was.
            separator = ' ' * (start - previous[1])
        else:
            separator = '\n'
        offset += len(separator)
        text = regions[index].text
        text_end = offset + len(text.encode('utf-8'))
        parts.extend((separator, text))
        ordered.append(PdfTextRegion(**regions[index].model_dump(), start=offset, end=text_end,
                                     provider_index=index, reading_column=column))
        offset = text_end
        previous = (row, end, column)
    labelled, labels = infer_labels(ordered, first_page=first_page, running=running, sized=False)
    return PageLayout(''.join(parts), tuple(region.model_copy(update={'label': label})
                                            for region, label in zip(ordered, labelled)), receipt, running, labels)
