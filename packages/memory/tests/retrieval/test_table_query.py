"""Exact answers from a table's own cells, every cell quoted where it lives.

A spreadsheet that was ingested is more than its text: the source
declared which cells are headers and which rows are data, and the
retained manifest keeps every cell with its byte span in the stored
text. So "the total Revenue where Region is West" is not a question for
a model; it is a filter, an exact sum, and a list of the cells that
made it, each one checkable against the episode's content.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.files import ingest_document
from scone_memory.retrieval import table_query as module
from scone_memory.retrieval.table_query import (Condition, TableQueryArgs, TableQueryError, episode_tables,
                                                query_table, verify_quotes)

from ..ingestion.test_xlsx_tables import cell, row, workbook

DEFINITION = ('id="1" name="RevenueTable" displayName="RevenueTable" ref="B4:C9" totalsRowCount="1">'
              '<tableColumns count="2"><tableColumn id="1" name="Region"/><tableColumn id="2" name="Revenue"/></tableColumns>')
ROWS = (row(1, cell('A1', 'Notes'))
        + row(4, cell('B4', 'Region') + cell('C4', 'Revenue'))
        + row(5, cell('B5', 'West') + cell('C5', '€20'))
        + row(6, cell('B6', 'East') + cell('C6', '35'))
        + row(7, cell('B7', 'West') + cell('C7', '1,250.50'))
        + row(8, cell('B8', 'North') + cell('C8', 'n/a'))
        + row(9, cell('B9', 'Total') + cell('C9', '1,305.50', formula='<f>SUM(C5:C8)</f>')))


def html_table() -> bytes:
    body = ''.join(f'<tr><th scope="row">District {i}</th><td>{10 * i}</td><td>{i}</td></tr>' for i in range(1, 5))
    return (f'<table><tr><th>District</th><th>Revenue</th><th>Cost</th></tr>{body}</table>').encode()


async def memory_with(raw: bytes, filename: str):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    saved = await ingest_document(memory, 's', raw, filename=filename)
    return memory, saved.added.episode_id


def args(operation: str, column: str | None = None, *where: tuple[str, str, str], table: str | None = None) -> TableQueryArgs:
    return TableQueryArgs(operation=operation, column=column, table=table,
                          where=tuple(Condition(column=c, op=o, value=v) for c, o, v in where))


async def test_a_spreadsheet_becomes_rows_with_its_declared_columns_and_its_totals_row_set_aside():
    memory, episode_id = await memory_with(workbook(ROWS, DEFINITION), 'sales.xlsx')
    (table,) = await episode_tables(memory, 's', episode_id)
    assert table.name == 'RevenueTable' and table.columns == ('Region', 'Revenue')
    assert [r.cells['Region'].text for r in table.rows] == ['West', 'East', 'West', 'North']
    assert table.totals_rows == 1 and table.basis == 'xlsx_table_declaration'
    episode = await memory.episode('s', episode_id)
    for r in table.rows:
        for quoted in r.cells.values():
            assert episode.content.encode()[quoted.start:quoted.end] == quoted.text.encode()


async def test_a_filtered_sum_is_exact_and_quotes_only_the_cells_it_used():
    memory, episode_id = await memory_with(workbook(ROWS, DEFINITION), 'sales.xlsx')
    answer = await query_table(memory, 's', episode_id, args('sum', 'Revenue', ('Region', '==', 'West')))
    assert answer.value == '1270.5' and answer.rows_matched == 2 and answer.rows_total == 4
    assert [c.text for c in answer.cells] == ['€20', '1,250.50'] and answer.cells_used == 2
    assert answer.quotes_truncated is False and answer.totals_rows_excluded == 1
    episode = await memory.episode('s', episode_id)
    assert verify_quotes(answer, episode.content.encode())
    assert answer.verified_accuracy is False and 'thousands' in answer.numeric_form


async def test_the_totals_row_never_counts_twice():
    memory, episode_id = await memory_with(workbook(ROWS, DEFINITION), 'sales.xlsx')
    answer = await query_table(memory, 's', episode_id, args('sum', 'Revenue', ('Region', '!=', 'North')))
    assert answer.value == '1305.5', 'the sheet\'s own total, computed from the rows, not read from the totals row'


async def test_count_average_min_max_and_rows():
    memory, episode_id = await memory_with(workbook(ROWS, DEFINITION), 'sales.xlsx')
    assert (await query_table(memory, 's', episode_id, args('count', None, ('Region', '==', 'West')))).value == '2'
    average = await query_table(memory, 's', episode_id, args('average', 'Revenue', ('Region', '!=', 'North')))
    assert average.value == '2611/6' and Fraction(average.value) == Fraction('1305.5') / 3, 'exact, never rounded'
    assert (await query_table(memory, 's', episode_id, args('min', 'Revenue', ('Region', '!=', 'North')))).value == '20'
    assert (await query_table(memory, 's', episode_id, args('max', 'Revenue', ('Region', '!=', 'North')))).value == '1250.5'
    listed = await query_table(memory, 's', episode_id, args('rows', None, ('Revenue', '>', '30')))
    assert listed.value is None and listed.rows_matched == 2
    assert [c.text for c in listed.cells] == ['East', '35', 'West', '1,250.50']
    assert listed.rows_skipped_non_numeric == 1, "the North row's n/a is not greater than 30, and the answer says it was passed over"


async def test_conditions_compare_numbers_as_numbers_and_text_as_text():
    memory, episode_id = await memory_with(workbook(ROWS, DEFINITION), 'sales.xlsx')
    ordered = await query_table(memory, 's', episode_id, args('count', None, ('Revenue', '>=', '35')))
    assert ordered.value == '2' and ordered.rows_skipped_non_numeric == 1
    assert (await query_table(memory, 's', episode_id, args('count', None, ('Region', 'contains', 'ST')))).value == '3'
    assert (await query_table(memory, 's', episode_id, args('count', None, ('Region', '==', ' west ')))).value == '2'
    with pytest.raises(TableQueryError) as refused:
        await query_table(memory, 's', episode_id, args('count', None, ('Region', '>', 'M')))
    assert refused.value.reason == 'ordering_needs_numbers'


async def test_a_cell_that_is_not_a_number_refuses_the_aggregate_and_is_quoted():
    memory, episode_id = await memory_with(workbook(ROWS, DEFINITION), 'sales.xlsx')
    with pytest.raises(TableQueryError) as refused:
        await query_table(memory, 's', episode_id, args('sum', 'Revenue'))
    assert refused.value.reason == 'non_numeric_cell' and 'n/a' in str(refused.value)


async def test_unknown_columns_missing_columns_and_empty_averages_are_refused():
    memory, episode_id = await memory_with(workbook(ROWS, DEFINITION), 'sales.xlsx')
    with pytest.raises(TableQueryError) as refused:
        await query_table(memory, 's', episode_id, args('sum', 'Profit'))
    assert refused.value.reason == 'column_not_found'
    with pytest.raises(TableQueryError) as refused:
        await query_table(memory, 's', episode_id, args('sum', None))
    assert refused.value.reason == 'column_required'
    with pytest.raises(TableQueryError) as refused:
        await query_table(memory, 's', episode_id, args('average', 'Revenue', ('Region', '==', 'Mars')))
    assert refused.value.reason == 'no_rows_matched'
    assert (await query_table(memory, 's', episode_id, args('sum', 'Revenue', ('Region', '==', 'Mars')))).value == '0'
    with pytest.raises(TableQueryError) as refused:
        await query_table(memory, 's', episode_id, args('count', None, table='Other'))
    assert refused.value.reason == 'table_not_found'


async def test_quoted_cells_are_capped_with_the_used_count_and_the_value_still_exact(monkeypatch):
    monkeypatch.setattr(module, 'MAX_QUOTED', 1)
    memory, episode_id = await memory_with(workbook(ROWS, DEFINITION), 'sales.xlsx')
    answer = await query_table(memory, 's', episode_id, args('sum', 'Revenue', ('Region', '==', 'West')))
    assert answer.value == '1270.5' and answer.cells_used == 2 and len(answer.cells) == 1 and answer.quotes_truncated


async def test_an_html_table_reads_its_row_headers_as_the_first_column():
    memory, episode_id = await memory_with(html_table(), 'districts.html')
    (table,) = await episode_tables(memory, 's', episode_id)
    assert table.columns == ('District', 'Revenue', 'Cost') and len(table.rows) == 4
    answer = await query_table(memory, 's', episode_id, args('sum', 'Revenue', ('Cost', '<=', '2')))
    assert answer.value == '30' and [c.text for c in answer.cells] == ['10', '20']
    episode = await memory.episode('s', episode_id)
    assert verify_quotes(answer, episode.content.encode())
    assert not verify_quotes(answer, b'x' * len(episode.content.encode())), 'a check that cannot fail checks nothing'


async def test_a_document_with_two_tables_needs_one_named():
    two = html_table() + b'<table><tr><th>Item</th><th>Qty</th></tr><tr><td>Bolt</td><td>7</td></tr></table>'
    memory, episode_id = await memory_with(two, 'two.html')
    tables = await episode_tables(memory, 's', episode_id)
    assert len(tables) == 2 and tables[1].columns == ('Item', 'Qty')
    with pytest.raises(TableQueryError) as refused:
        await query_table(memory, 's', episode_id, args('count'))
    assert refused.value.reason == 'table_required'
    named = await query_table(memory, 's', episode_id, args('sum', 'Qty', table=tables[1].locator))
    assert named.value == '7' and named.table == tables[1].locator


async def test_a_document_without_tables_says_so():
    memory, episode_id = await memory_with(b'Just prose, no table here.', 'notes.txt')
    assert await episode_tables(memory, 's', episode_id) == ()
    with pytest.raises(TableQueryError) as refused:
        await query_table(memory, 's', episode_id, args('count'))
    assert refused.value.reason == 'table_not_found'


def test_the_record_carries_the_question_the_cells_and_the_caveat():
    from scone_memory.retrieval.table_query import QuotedCell, TableAnswer
    answer = TableAnswer(operation='sum', column='Revenue', table='t', table_name='T', value='1',
                         rows_matched=1, rows_total=2, cells=(QuotedCell('t', 0, 'Revenue', '1', 5, 6, 'c'),),
                         cells_used=1, quotes_truncated=False, totals_rows_excluded=0,
                         conditions=(Condition(column='Region', op='==', value='West'),), numeric_form='plain')
    record = answer.record()
    assert record['verified_accuracy'] is False and record['coverage'] == 'matched_rows_only'
    assert record['cells'] == [{'table': 't', 'row': 0, 'column': 'Revenue', 'text': '1', 'start': 5, 'end': 6, 'locator': 'c'}]
    assert record['where'] == [{'column': 'Region', 'op': '==', 'value': 'West'}] and 'notice' in record
    assert record['rows_skipped_non_numeric'] == 0


def test_args_are_bounded():
    with pytest.raises(Exception):
        TableQueryArgs(operation='sum', where=tuple(Condition(column='c', op='==', value='v') for _ in range(9)))
    with pytest.raises(Exception):
        TableQueryArgs(operation='total')


async def test_delimited_text_queries_the_same_way():
    memory, episode_id = await memory_with(b'region,revenue\r\nWest,"1,250.50"\r\nEast,35\r\nWest,20\r\n', 'sales.csv')
    (table,) = await episode_tables(memory, 's', episode_id)
    assert table.columns == ('region', 'revenue') and table.basis == 'delimited_columns' and len(table.rows) == 3
    answer = await query_table(memory, 's', episode_id, args('sum', 'revenue', ('region', '==', 'West')))
    assert answer.value == '1270.5' and [c.text for c in answer.cells] == ['1,250.50', '20']
    episode = await memory.episode('s', episode_id)
    assert verify_quotes(answer, episode.content.encode())


async def test_a_json_document_queries_like_a_spreadsheet():
    import json

    raw = json.dumps({"rows": [{"region": "West", "revenue": "1,250.50"}, {"region": "East", "revenue": 35},
                               {"region": "West", "revenue": 20}]}).encode()
    memory, episode_id = await memory_with(raw, 'sales.json')
    (table,) = await episode_tables(memory, 's', episode_id)
    assert table.name == 'json:/rows' and table.columns == ('region', 'revenue') and table.basis == 'json_object_keys'
    answer = await query_table(memory, 's', episode_id, args('sum', 'revenue', ('region', '==', 'West')))
    assert answer.value == '1270.5' and [c.text for c in answer.cells] == ['1,250.50', '20']
    episode = await memory.episode('s', episode_id)
    assert verify_quotes(answer, episode.content.encode())


def test_an_empty_or_double_signed_cell_is_no_number_not_a_crash():
    from scone_memory.retrieval.table_query import number

    assert number("") is None and number("   ") is None and number("--5") is None and number("- -5") is None
    assert number("-5") == -5 and number("$1,270.50") == Fraction("1270.5") and number("+") is None and number("$") is None


async def test_rows_without_the_column_are_counted_and_duplicate_headers_stay_two_columns():
    from scone_memory.retrieval.table_query import TableQueryArgs, answer_from, episode_tables

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        added = await ingest_document(memory, 'alpha', b'[{"region": "West", "revenue": 20}, {"region": "East"}]', filename='sales.json')
        tables = await episode_tables(memory, 'alpha', added.added.episode_id)
        answer = answer_from(tables, TableQueryArgs(operation='sum', column='revenue'))
        assert (answer.value, answer.rows_matched, answer.cells_used, answer.rows_without_cell) == ('20', 2, 1, 1)
        assert answer.record()['rows_without_cell'] == 1
        html = b'<table><tr><th>Amount</th><th>Amount</th></tr><tr><td>10</td><td>20</td></tr></table>'
        twice = await ingest_document(memory, 'alpha', html, filename='two.html')
        tables = await episode_tables(memory, 'alpha', twice.added.episode_id)
        assert tables[0].columns == ('Amount', 'Amount [column 2]'), "two headers spelt the same at two positions are two columns"
        total = answer_from(tables, TableQueryArgs(operation='sum', column='Amount'))
        assert total.value == '10' and total.cells_used == 1
        both = answer_from(tables, TableQueryArgs(operation='sum', column='Amount [column 2]'))
        assert both.value == '20'
        blank = await ingest_document(memory, 'alpha', b'region,revenue\r\nWest,20\r\nEast,\r\n', filename='blank.csv')
        tables = await episode_tables(memory, 'alpha', blank.added.episode_id)
        with pytest.raises(TableQueryError) as refused:
            answer_from(tables, TableQueryArgs(operation='sum', column='revenue'))
        assert refused.value.reason == 'non_numeric_cell'
    finally:
        await memory.close()
