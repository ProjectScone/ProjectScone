"""Exact answers from a table's own cells, every cell quoted where it lives.

A spreadsheet that was ingested is more than its text. The source
declared which cells are headers, which rows are data and which row is
its totals, and the retained manifest keeps every cell with its byte
span in the stored text. So "the total Revenue where Region is West" is
not a question for a model to answer from a passage: it is a filter
over rows, an exact sum over cells, and a list of the cells that made
it, each one checkable against the episode's content by anyone with the
bytes. Nothing is generated; nothing is rounded.

What this is honest about, on every record: which rows were matched of
how many, which cells were used and whether all of them are quoted
(``MAX_QUOTED`` caps the list, never the computation), how many totals
rows were set aside so a sheet's own sum never counts twice, how the
cell text was read as a number (thousands separators and a leading
currency sign are removed; anything else is not a number and an
aggregate over it is refused, quoting the cell), and that the meaning
of the operands -- units, what a column really holds -- was not
verified: ``verified_accuracy`` is False, as everywhere else.

Tables come from the document manifest through ``document_provenance``,
so a cell's span is the same one the provenance route shows, and the
formats that declare cells (spreadsheets, HTML and Word tables, and
delimited text) all read the same way here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import InvalidInput

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..ingestion.files import DocumentProvenance
    from ..memory.engine import MemoryEngine

#: Cells quoted on one answer; the computation covers every matched row regardless.
MAX_QUOTED = 200

Operation = Literal['count', 'sum', 'average', 'min', 'max', 'rows']
Comparison = Literal['==', '!=', 'contains', '>', '<', '>=', '<=']
Reason = Literal['table_not_found', 'table_required', 'column_not_found', 'column_required',
                 'non_numeric_cell', 'ordering_needs_numbers', 'no_rows_matched']

_NUMBER = re.compile(r'^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$')
_CURRENCY = '$€£¥'
NUMERIC_FORM = ('a cell is a number when, after removing surrounding whitespace, a leading currency sign and '
                'thousands separators, it is digits with an optional sign and decimal part')


class Condition(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    column: str = Field(min_length=1, max_length=256)
    op: Comparison
    value: str = Field(max_length=512)


class TableQueryArgs(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    operation: Operation
    #: The column aggregated; ``count`` and ``rows`` need none.
    column: Optional[str] = Field(default=None, max_length=256)
    #: The table's name or locator; needed only when the document holds more than one.
    table: Optional[str] = Field(default=None, max_length=4096)
    where: tuple[Condition, ...] = Field(default=(), max_length=8)


class TableQueryError(InvalidInput):
    """A question the table cannot answer, and the one reason why."""

    def __init__(self, reason: Reason, detail: str = '') -> None:
        super().__init__(f'{reason}: {detail}' if detail else reason)
        self.reason: Reason = reason


@dataclass(frozen=True)
class QuotedCell:
    """One cell as the record shows it: ``content[start:end]`` is ``text`` in UTF-8 bytes of the episode."""

    table: str
    row: int
    column: str
    text: str
    start: int
    end: int
    locator: str

    def record(self) -> dict[str, object]:
        return {'table': self.table, 'row': self.row, 'column': self.column, 'text': self.text,
                'start': self.start, 'end': self.end, 'locator': self.locator}


@dataclass(frozen=True)
class TableRow:
    number: int
    cells: dict[str, QuotedCell]


@dataclass(frozen=True)
class Table:
    locator: str
    name: str
    columns: tuple[str, ...]
    rows: tuple[TableRow, ...]
    #: Rows the source marked as totals, set aside so a sum never counts them.
    totals_rows: int
    #: Where the column names came from: what the source declared, or the cells' own headers.
    basis: str

    def summary(self) -> dict[str, object]:
        return {'locator': self.locator, 'name': self.name, 'columns': list(self.columns), 'rows': len(self.rows),
                'totals_rows_excluded': self.totals_rows, 'basis': self.basis}


@dataclass(frozen=True)
class TableAnswer:
    operation: str
    column: Optional[str]
    table: str
    table_name: str
    #: Exact: a terminating decimal, else ``p/q``; None for ``rows``.
    value: Optional[str]
    rows_matched: int
    rows_total: int
    cells: tuple[QuotedCell, ...]
    cells_used: int
    quotes_truncated: bool
    totals_rows_excluded: int
    conditions: tuple[Condition, ...]
    numeric_form: str
    #: Rows an ordering condition could not read as a number, and so did not match.
    rows_skipped_non_numeric: int = 0
    #: Matched rows with no cell at all in the queried column (a JSON object
    #: without the key); they are not in the number, and this says so.
    rows_without_cell: int = 0
    verified_accuracy: Literal[False] = False

    def record(self) -> dict[str, object]:
        return {'schema_version': 1, 'operation': self.operation, 'column': self.column, 'table': self.table,
                'table_name': self.table_name, 'value': self.value, 'rows_matched': self.rows_matched,
                'rows_total': self.rows_total, 'cells': [cell.record() for cell in self.cells],
                'cells_used': self.cells_used, 'quotes_truncated': self.quotes_truncated,
                'totals_rows_excluded': self.totals_rows_excluded,
                'rows_skipped_non_numeric': self.rows_skipped_non_numeric,
                'rows_without_cell': self.rows_without_cell,
                'where': [{'column': c.column, 'op': c.op, 'value': c.value} for c in self.conditions],
                'numeric_form': self.numeric_form, 'coverage': 'matched_rows_only', 'verified_accuracy': False,
                'notice': ('Exact computation over the matched rows of one table. What a column means, its units, '
                           'and whether the rows are all the rows there are, were not verified.')}


def number(text: str) -> Optional[Fraction]:
    """The exact number a cell holds, or None when it is not one by the rule in ``NUMERIC_FORM``."""
    body = text.strip()
    sign = ''
    if body and body[0] in '+-':
        sign, body = body[0], body[1:].lstrip()
    if body and body[0] in _CURRENCY:
        body = body[1:].lstrip()
    # One sign at most: "--5" and "- -5" are not numbers by the rule, and
    # an empty cell is no number at all rather than a crash.
    if not body or body[0] in '+-' or not _NUMBER.match(body):
        return None
    return Fraction(sign + body.replace(',', ''))


def render(value: Fraction) -> str:
    """A terminating decimal exactly, else ``p/q``; never a float."""
    denominator = value.denominator
    while denominator % 2 == 0:
        denominator //= 2
    while denominator % 5 == 0:
        denominator //= 5
    if denominator != 1:
        return f'{value.numerator}/{value.denominator}'
    scale = 0
    while (value.denominator * 10 ** scale) % 1 or (value * 10 ** scale).denominator != 1:
        scale += 1
    digits = value.numerator * 10 ** scale // value.denominator
    text = str(abs(digits)).rjust(scale + 1, '0')
    whole, fraction = text[:-scale] if scale else text, text[-scale:] if scale else ''
    out = f'{whole}.{fraction}' if fraction else whole
    return f'-{out}' if digits < 0 else out


@dataclass
class _Draft:
    name: str
    basis: str
    columns: list[str]
    rows: dict[int, dict[str, QuotedCell]]
    totals: set[int]


def tables_from(provenance: "DocumentProvenance") -> tuple[Table, ...]:
    """The tables a document's manifest declares, as rows keyed by column name."""
    drafts: dict[str, _Draft] = {}
    #: Per table, the column position a header name was first seen at, so
    #: two headers spelt the same at different positions stay two columns.
    positions: dict[str, dict[str, int]] = {}
    #: Per table whose header row was read from its shape (a PDF page's),
    #: the row that header sits in: a row above it is the table's title,
    #: not a row of it. An HTML reader's ``th`` on a row label says nothing
    #: of the kind, so only such pages' tables are read this way.
    first_header: dict[str, int] = {}
    for segment in provenance.segments:
        if segment.metadata.get('header_basis') != 'pdf_first_row':
            continue
        for cell in segment.table_cells:
            if cell.is_header:
                first_header[cell.table_locator] = min(first_header.get(cell.table_locator, cell.row), cell.row)
    offset = 0
    for segment in provenance.segments:
        role = segment.metadata.get('table_role')
        declared = _declared_columns(segment.metadata.get('table_columns'))
        for cell in segment.table_cells:
            if cell.row < first_header.get(cell.table_locator, 0):
                continue  # a title across the table, above its header
            draft = drafts.get(cell.table_locator)
            if draft is None:
                basis = segment.metadata.get('header_basis') or ('delimited_columns' if declared else 'cell_headers')
                if basis == 'pdf_first_row' and cell.table_locator not in first_header:
                    basis = 'cell_headers'  # a page's other table, whose shape gave no header
                draft = drafts[cell.table_locator] = _Draft(segment.metadata.get('table_name') or cell.table_locator,
                                                            basis, [], {}, set())
            column_headers = [h.text for h in cell.headers if h.association in ('column', 'explicit')]
            if cell.is_header and not column_headers:
                continue  # a header names a column; it holds no data
            if role == 'totals':
                draft.totals.add(cell.row)
                continue
            if column_headers:
                name = column_headers[0]
                first = positions.setdefault(cell.table_locator, {}).setdefault(name, cell.column)
                if first != cell.column:
                    name = f'{name} [column {cell.column + 1}]'
            elif declared and cell.column < len(declared):
                name = declared[cell.column]
            else:
                name = f'column {cell.column + 1}'
            if name not in draft.columns:
                draft.columns.append(name)
            positions.setdefault(cell.table_locator, {}).setdefault(name, cell.column)
            draft.rows.setdefault(cell.row, {})[name] = QuotedCell(cell.table_locator, cell.row, name, cell.text,
                                                                   offset + cell.start, offset + cell.end, cell.locator)
        offset += len(segment.text.encode('utf-8')) + 2
    # Columns in the table's own order, whichever cell the text reached
    # first: a wrapped word under one column can come before the row above.
    for locator, draft in drafts.items():
        draft.columns.sort(key=lambda name: positions[locator].get(name, 0))
    return tuple(Table(locator, draft.name, tuple(draft.columns),
                       tuple(TableRow(number, cells) for number, cells in sorted(draft.rows.items())),
                       len(draft.totals), draft.basis)
                 for locator, draft in drafts.items())


def _declared_columns(raw: Optional[str]) -> tuple[str, ...]:
    if not raw:
        return ()
    import json
    try:
        value = json.loads(raw)
    except ValueError:
        return ()
    return tuple(str(item) for item in value) if isinstance(value, list) else ()


def _holds(cell: Optional[QuotedCell], condition: Condition, skipped: list[QuotedCell]) -> bool:
    """Whether ``cell`` satisfies ``condition``; a cell an ordering cannot read as a number does not, and is noted."""
    if cell is None:
        return False
    left = number(cell.text)
    right = number(condition.value)
    if condition.op in ('>', '<', '>=', '<='):
        if right is None:
            raise TableQueryError('ordering_needs_numbers', f'{condition.column} {condition.op} {condition.value!r}')
        if left is None:
            skipped.append(cell)
            return False
        return {'>': left > right, '<': left < right, '>=': left >= right, '<=': left <= right}[condition.op]
    if condition.op == 'contains':
        return condition.value.strip().casefold() in cell.text.casefold()
    same = (left == right) if left is not None and right is not None else (
        cell.text.strip().casefold() == condition.value.strip().casefold())
    return same if condition.op == '==' else not same


def answer_from(tables: Sequence[Table], args: TableQueryArgs) -> TableAnswer:
    """The answer over the tables a document holds; pure, so it can be checked from a record."""
    if not tables:
        raise TableQueryError('table_not_found', 'the document declares no table')
    if args.table is not None:
        chosen = [t for t in tables if args.table in (t.name, t.locator)]
        if not chosen:
            raise TableQueryError('table_not_found', f'{args.table!r} is not a table of this document')
        table = chosen[0]
    elif len(tables) == 1:
        table = tables[0]
    else:
        raise TableQueryError('table_required', f'the document holds {len(tables)} tables; name one')
    for condition in args.where:
        if condition.column not in table.columns:
            raise TableQueryError('column_not_found', f'{condition.column!r} is not a column of {table.name}')
    if args.operation in ('count', 'rows'):
        column = None
    elif args.column is None:
        raise TableQueryError('column_required', f'{args.operation} needs a column')
    elif args.column not in table.columns:
        raise TableQueryError('column_not_found', f'{args.column!r} is not a column of {table.name}')
    else:
        column = args.column
    skipped: list[QuotedCell] = []
    matched = [row for row in table.rows if all(_holds(row.cells.get(c.column), c, skipped) for c in args.where)]
    used: list[QuotedCell] = []
    value: Optional[str]
    if args.operation == 'rows':
        used = [row.cells[name] for row in matched for name in table.columns if name in row.cells]
        value = None
    elif args.operation == 'count':
        # What was counted is evidence too: the first cell of each row, so a
        # reader can see the rows behind the number.
        used = [next(row.cells[name] for name in table.columns if name in row.cells) for row in matched if row.cells]
        value = str(len(matched))
    else:
        assert column is not None
        numbers: list[Fraction] = []
        without = 0
        for row in matched:
            cell = row.cells.get(column)
            if cell is None:
                without += 1
                continue
            parsed = number(cell.text)
            if parsed is None:
                raise TableQueryError('non_numeric_cell', f'{column} in row {row.number} holds {cell.text!r}')
            numbers.append(parsed)
            used.append(cell)
        if not numbers and args.operation != 'sum':
            raise TableQueryError('no_rows_matched', f'{args.operation} over no rows has no value')
        if args.operation == 'sum':
            value = render(sum(numbers, Fraction(0)))
        elif args.operation == 'average':
            value = render(sum(numbers, Fraction(0)) / len(numbers))
        elif args.operation == 'min':
            value = render(min(numbers))
        else:
            value = render(max(numbers))
    return TableAnswer(args.operation, column, table.locator, table.name, value, len(matched), len(table.rows),
                       tuple(used[:MAX_QUOTED]), len(used), len(used) > MAX_QUOTED, table.totals_rows,
                       args.where, NUMERIC_FORM, len({cell.locator for cell in skipped}),
                       without if args.operation in ('sum', 'average', 'min', 'max') else 0)


def verify_quotes(answer: TableAnswer, content: bytes) -> bool:
    """Whether every quoted cell is where the answer says it is in ``content``."""
    return all(content[cell.start:cell.end] == cell.text.encode('utf-8') for cell in answer.cells)


async def episode_tables(memory: "MemoryEngine", space: str, episode_id: int) -> tuple[Table, ...]:
    """The tables an ingested document declares; empty when it declares none."""
    from ..ingestion.files import document_provenance

    return tables_from(await document_provenance(memory, space, episode_id))


async def query_table(memory: "MemoryEngine", space: str, episode_id: int, args: TableQueryArgs) -> TableAnswer:
    """Answer ``args`` over the document's tables, from the retained manifest."""
    return answer_from(await episode_tables(memory, space, episode_id), args)
