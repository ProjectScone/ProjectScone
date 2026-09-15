"""`scone doc-markdown FILE` writes a document as Markdown without opening a store."""
from __future__ import annotations

from io import StringIO
import json

from scone_memory.runtime.cli import main

PAGE = b'<h1>Install</h1><ul><li>Linux</li><li>macOS</li></ul><table><tr><th>Port</th></tr><tr><td>80</td></tr></table>'


def run(argv: list[str]) -> tuple[int, str]:
    out = StringIO()
    # An unusable store setting proves the command never builds an engine.
    code = main(argv, env={'SCONE_STORE': 'no-such-store'}, out=out)
    return code, out.getvalue()


def test_markdown_goes_to_stdout_and_the_receipt_to_stderr(tmp_path, capsys) -> None:
    page = tmp_path / 'guide.html'
    page.write_bytes(PAGE)
    code, stdout = run(['doc-markdown', str(page)])
    assert code == 0
    assert stdout == '# Install\n\n- Linux\n- macOS\n\n| Port |\n| --- |\n| 80 |\n'
    assert capsys.readouterr().err.strip() == (
        'doc-markdown: 4 blocks from 5 segments (1 heading, 2 list items, 1 table); not cut')
    notes = tmp_path / 'notes.txt'
    notes.write_bytes(b'plain words\n')
    assert run(['doc-markdown', str(notes)]) == (0, 'plain words\n')
    assert capsys.readouterr().err.strip() == (
        'doc-markdown: 1 block from 1 segment; the reader declared no headings, lists or tables; not cut')


def test_json_is_the_whole_record_and_a_low_bound_says_where_it_cut(tmp_path, capsys) -> None:
    page = tmp_path / 'guide.html'
    page.write_bytes(PAGE)
    code, stdout = run(['doc-markdown', str(page), '--json'])
    record = json.loads(stdout)
    assert code == 0 and record['markdown'].startswith('# Install') and record['file'] == str(page)
    assert record['spans'][0]['sources'][0]['locator'] == 'line:1'
    code, stdout = run(['doc-markdown', str(page), '--max-bytes', '12'])
    assert code == 0 and stdout == '# Install\n'
    assert capsys.readouterr().err.strip().endswith(
        'cut at extracted byte 9 by the 12-byte bound; 4 segments not wholly written')


def test_an_unreadable_or_unsupported_file_is_an_error_not_a_trace(tmp_path, capsys) -> None:
    assert run(['doc-markdown', str(tmp_path / 'missing.html')])[0] == 2
    assert 'cannot be read' in capsys.readouterr().err
    other = tmp_path / 'data.unknownformat'
    other.write_bytes(b'x')
    assert run(['doc-markdown', str(other)])[0] == 2
    assert capsys.readouterr().err.startswith('error: ')
    page = tmp_path / 'guide.html'
    page.write_bytes(PAGE)
    assert run(['doc-markdown', str(page), '--max-bytes', '0'])[0] == 2
    assert 'markdown byte bound' in capsys.readouterr().err


def test_the_command_default_bound_is_the_assemblers_bound() -> None:
    from scone_memory.ingestion.formats.markdown_assembly import MAX_MARKDOWN_BYTES
    from scone_memory.runtime.document_markdown import DEFAULT_MAX_BYTES

    assert DEFAULT_MAX_BYTES == MAX_MARKDOWN_BYTES
