"""Production readers retain physical lines, literal TSV fields and HTML code."""
import json

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import BuiltinDocumentParser


@pytest.mark.parametrize('separator', ['\v', '\f', '\x1c', '\x85', '\u2028', '\u2029'])
async def test_unicode_and_page_separators_do_not_invent_source_lines(separator):
    raw = f'alpha{separator}beta\r\n\f\r\ngamma\rdelta'.encode()
    parsed = await BuiltinDocumentParser().parse(raw, 'source.lisp')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('line:1', f'alpha{separator}beta'), ('line:3', 'gamma'), ('line:4', 'delta')]


@pytest.mark.parametrize('separator', ['\x85', '\u2028', '\u2029'])
async def test_jsonl_unicode_separators_stay_inside_string_records(separator):
    raw = json.dumps({'quote': f'alpha{separator}beta'}, ensure_ascii=False) + '\r\n{"n":2}\n'
    parsed = await BuiltinDocumentParser().parse(raw.encode(), 'source.jsonl')
    assert [s.locator for s in parsed.segments] == ['line:1#/quote', 'line:2#/n']
    assert parsed.segments[0].text == '/quote: ' + json.dumps(f'alpha{separator}beta', ensure_ascii=False)


@pytest.mark.parametrize('separator', ['\x85', '\u2028', '\u2029', '\r'])
async def test_jsonl_non_lf_separators_do_not_turn_invalid_json_into_multiple_records(separator):
    with pytest.raises(InvalidInput, match='JSON'):
        await BuiltinDocumentParser().parse(f'1{separator}2\n'.encode(), 'source.ndjson')


async def test_tsv_quotes_are_literal_and_never_swallow_later_rows():
    raw = b'size\tlabel\n"12\tinch\n3\tfoot"\n4\t"yard" at start\n'
    parsed = await BuiltinDocumentParser().parse(raw, 'source.tsv')
    assert [s.text for s in parsed.segments] == ['size: "12\nlabel: inch', 'size: 3\nlabel: foot"',
                                               'size: 4\nlabel: "yard" at start']
    assert [(s.locator, s.metadata) for s in parsed.segments] == [
        ('row:2', {'line_start': '2', 'line_end': '2'}),
        ('row:3', {'line_start': '3', 'line_end': '3'}),
        ('row:4', {'line_start': '4', 'line_end': '4'})]


@pytest.mark.parametrize('suffix,delimiter', [('csv', ','), ('tsv', '\t')])
async def test_blank_delimited_records_are_skipped_without_losing_physical_line_locations(suffix, delimiter):
    raw = f'a{delimiter}b\r\n1{delimiter}2\r\n\r\n3{delimiter}4\r\n\r\n'
    parsed = await BuiltinDocumentParser().parse(raw.encode(), f'source.{suffix}')
    assert [s.text for s in parsed.segments] == ['a: 1\nb: 2', 'a: 3\nb: 4']
    assert [s.metadata for s in parsed.segments] == [
        {'line_start': '2', 'line_end': '2'}, {'line_start': '4', 'line_end': '4'}]
    assert [s.locator for s in parsed.segments] == ['row:2', 'row:4']


@pytest.mark.parametrize('suffix,delimiter', [('csv', ','), ('tsv', '\t')])
async def test_leading_blank_records_do_not_become_empty_headers(suffix, delimiter):
    raw = f'\r\na{delimiter}b\r\n1{delimiter}2\r\n'
    parsed = await BuiltinDocumentParser().parse(raw.encode(), f'source.{suffix}')
    assert parsed.segments[0].text == 'a: 1\nb: 2'
    assert parsed.segments[0].locator == 'row:3'
    assert parsed.segments[0].metadata == {'line_start': '3', 'line_end': '3'}


async def test_html_pre_keeps_code_indentation_across_inline_elements():
    raw = b'<p>Example:</p>\n<pre>  def f(x):\n    <b>if x:</b>\n\treturn 1\n    return 2\n</pre><p>After   code</p>'
    parsed = await BuiltinDocumentParser().parse(raw, 'source.html')
    assert [s.text for s in parsed.segments] == [
        'Example:', '  def f(x):\n    if x:\n\treturn 1\n    return 2\n', 'After code']
    assert parsed.segments[1].locator == 'line:2'


async def test_html_nonbreaking_spaces_do_not_become_breakable_ascii_spaces():
    parsed = await BuiltinDocumentParser().parse(b'<p>Total: 10&nbsp;000 EUR</p>', 'source.html')
    assert parsed.segments[0].text == 'Total: 10\u00a0000 EUR'


async def test_html_normal_text_locator_starts_at_content_after_leading_source_newlines():
    parsed = await BuiltinDocumentParser().parse(b'<div>\n \n    Start <b>here</b>\n</div>', 'source.html')
    assert parsed.segments[0].text == 'Start here'
    assert parsed.segments[0].locator == 'line:3'
