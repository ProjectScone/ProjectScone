# Table querying: exact answers from a document's own cells

A spreadsheet that was ingested is more than its text. The source
declared which cells are headers, which rows are data and which row is
its totals, and the retained manifest keeps every cell with its byte
span in the stored text. So "the total Revenue where Region is West" is
not a question for a model to answer from a passage: it is a filter
over rows, an exact sum over cells, and a list of the cells that made
it, each one checkable against the episode's content by anyone with the
bytes. Nothing is generated; nothing is rounded.

```bash
curl -s -H "authorization: Bearer $KEY" http://127.0.0.1:7437/v1/episodes/12/tables
# {"episode_id": 12, "tables": [{"locator": "sheet:Sales/table:1", "name": "RevenueTable",
#   "columns": ["Region", "Revenue"], "rows": 4, "totals_rows_excluded": 1,
#   "basis": "xlsx_table_declaration"}]}

curl -s -X POST -H "authorization: Bearer $KEY" -H 'content-type: application/json' \
  http://127.0.0.1:7437/v1/episodes/12/tables/query \
  -d '{"operation": "sum", "column": "Revenue", "where": [{"column": "Region", "op": "==", "value": "West"}]}'
# {"value": "1270.5", "rows_matched": 2, "rows_total": 4,
#  "cells": [{"table": "…", "row": 1, "column": "Revenue", "text": "€20", "start": 118, "end": 121, "locator": "sheet:Sales/cell:C5"}, …],
#  "cells_used": 2, "quotes_truncated": false, "totals_rows_excluded": 1, "rows_skipped_non_numeric": 0,
#  "coverage": "matched_rows_only", "verified_accuracy": false, "notice": "…"}
```

In code: `episode_tables(memory, space, episode_id)` lists the tables;
`query_table(memory, space, episode_id, TableQueryArgs(...))` answers;
`answer_from(tables, args)` is the pure part, so a record can be
recomputed; `verify_quotes(answer, content_bytes)` checks every quoted
cell against the episode's bytes.

## What can be asked

- `operation`: `count`, `sum`, `average`, `min`, `max`, or `rows` (the
  matched rows themselves, every cell quoted).
- `column`: the column aggregated; `count` and `rows` need none.
- `where`: up to eight conditions, all of which a row must satisfy.
  `==`, `!=` and `contains` compare text after trimming and case folding,
  or as numbers when both sides read as numbers; `>`, `<`, `>=`, `<=`
  need a numeric value, and a row whose cell is not a number does not
  match — the answer counts such rows in `rows_skipped_non_numeric`
  rather than guessing.
- `table`: the table's name or locator, needed only when the document
  holds more than one; otherwise the query is refused with
  `table_required` rather than answered from the first.

## What the record is honest about

- **Exact.** Sums, averages, minima and maxima are computed on exact
  fractions; a value is a terminating decimal or `p/q`, never a float.
- **Numbers by a written rule.** A cell is a number when, after removing
  surrounding whitespace, a leading currency sign (`$ € £ ¥`) and
  thousands separators, it is digits with an optional sign and decimal
  part (`numeric_form` says so). Anything else — `n/a`, `12%`, `1.2e3` —
  is not a number, and an aggregate that meets one is refused with
  `non_numeric_cell`, quoting the cell, rather than skipping it.
- **Totals rows set aside.** A row the source marked as totals (an xlsx
  table's `totalsRowCount`) is never a data row, so a sheet's own sum
  never counts twice; `totals_rows_excluded` says how many.
- **Every cell used is quoted** with its UTF-8 byte span in the stored
  episode content, the same offsets `GET /v1/episodes/{id}/document`
  shows; at most 200 are listed (`cells_used` is the true count and
  `quotes_truncated` says when the list was cut — the computation
  always covers every matched row).
- **Not verified:** what a column means, its units, and whether the
  rows are all the rows there are. `verified_accuracy` is false on every
  record, as it is everywhere else in this framework.

## Which documents have tables

Any format whose parser declares cells: spreadsheets with declared
tables (`xlsx`; header names from the sheet's own table declaration),
HTML tables (`<th>` scope and headers attributes; a row header is a
value of its column), Word tables with repeating header rows, and
delimited text (`csv`, `tsv`: one row per segment, the header line's
labels as the columns, `basis: delimited_columns`). A document that
declares no table lists none, and a query on it is refused with
`table_not_found`. JSON leaves, OCR tables (geometry without byte
spans) and spreadsheets read through the converters (`xls`, `xlsb`) do
not carry cells yet.

No model is involved anywhere here. Reading a question in words and
turning it into these arguments is a caller's job — or an agent's: with
`SCONE_CONVERSATIONS_TOOL_TABLES=1` (`ScopedMemoryTools(...,
enable_tables=True)`) the tool loop offers `list_tables` and
`query_table`, whose answer is this record with each cell tied to the
stored chunk that holds it, prepared and revalidated like every other
tool evidence. Keeping the words-to-arguments step outside the
computation is what lets the computation be exact.
