"""Independent CLI invocations: SQLite sources, file attachments and journal."""
import io
import json
import os

import pytest

from scone_memory.runtime.cli import main


def setup(tmp_path):
    root = tmp_path / 'sources'
    root.mkdir()
    key = tmp_path / 'journal.key'
    key.write_bytes(b'k' * 32)
    key.chmod(0o600)
    arguments = ['sync-directory', str(root), '--journal', str(tmp_path / 'sync.journal'),
                 '--key-file', str(key), '--parser-revision', 'builtin-v1', '--store-id', 'local-fixture', '--json']
    env = {'SCONE_DOCUMENTS': 'sqlite', 'SCONE_VECTORS': 'sqlite', 'SCONE_SQLITE_PATH': str(tmp_path / 'memory.db'),
           'SCONE_EMBEDDER': 'hash', 'SCONE_EVENTS': 'none'}
    return root, key, arguments, env


def test_cli_persists_reconciliation_and_returns_current_retrievable_source(tmp_path):
    root, _, arguments, env = setup(tmp_path)
    (root / 'report.txt').write_text('First telescope schedule')
    out = io.StringIO()
    assert main(arguments, env=env, out=out) == 0
    first = json.loads(out.getvalue())
    assert first['complete'] and first['receipts'][0]['status'] == 'added'
    old = first['receipts'][0]['episode_id']
    (root / 'report.txt').write_text('Updated telescope schedule')
    out = io.StringIO()
    assert main(arguments, env=env, out=out) == 0
    current = json.loads(out.getvalue())
    assert current['collection_id'] == first['collection_id']
    assert current['receipts'][0]['status'] == 'updated'
    assert current['receipts'][0]['episode_id'] != old
    out = io.StringIO()
    assert main(['recall', 'telescope', '--json'], env=env, out=out) == 0
    recalled = json.loads(out.getvalue())
    assert {item['episode_id'] for item in recalled['items']} == {current['receipts'][0]['episode_id']}
    (root / 'report.txt').unlink()
    out = io.StringIO()
    assert main(arguments + ['--delete-missing'], env=env, out=out) == 0
    assert json.loads(out.getvalue())['receipts'][0]['status'] == 'deleted'


def test_cli_partial_run_has_nonzero_status_and_structured_receipt(tmp_path):
    root, _, arguments, env = setup(tmp_path)
    os.mkfifo(root / 'pipe.txt')
    out = io.StringIO()
    assert main(arguments, env=env, out=out) == 1
    result = json.loads(out.getvalue())
    assert not result['complete'] and result['issues'][0]['code'] == 'special_file'


@pytest.mark.parametrize('kind', ['public', 'symlink', 'fifo', 'wrong-size', 'inside-root'])
def test_cli_refuses_unsafe_key_file_before_journal_creation(tmp_path, kind):
    root, key, arguments, env = setup(tmp_path)
    if kind == 'public':
        key.chmod(0o644)
    elif kind == 'symlink':
        real = tmp_path / 'real.key'
        key.rename(real)
        key.symlink_to(real)
    elif kind == 'fifo':
        key.unlink()
        os.mkfifo(key, 0o600)
    elif kind == 'wrong-size':
        key.write_bytes(b'short')
    else:
        key.rename(root / 'journal.key')
        arguments[arguments.index('--key-file') + 1] = str(root / 'journal.key')
    assert main(arguments, env=env, out=io.StringIO()) == 2
    assert not (tmp_path / 'sync.journal').exists()


@pytest.mark.parametrize('arguments', [['--max-files', '0'], ['--max-total-bytes', '-1']])
def test_invalid_cli_bounds_return_input_error_without_traceback(tmp_path, arguments, capsys):
    _, _, base, env = setup(tmp_path)
    assert main(base + arguments, env=env, out=io.StringIO()) == 2
    assert 'directory scan limits are invalid' in capsys.readouterr().err


def test_programmatic_main_rejects_invalid_key_path_as_input_error(tmp_path):
    _, _, arguments, env = setup(tmp_path)
    arguments[arguments.index('--key-file') + 1] = 'bad\x00key'
    assert main(arguments, env=env, out=io.StringIO()) == 2


def test_text_diagnostics_escape_terminal_controls_in_rejected_filenames(tmp_path):
    root, _, arguments, env = setup(tmp_path)
    (root / 'bad\x1b[2J.txt').write_bytes(b'invalid name')
    arguments.remove('--json')
    out = io.StringIO()
    assert main(arguments, env=env, out=out) == 1
    assert '\x1b' not in out.getvalue()
    assert 'bad\\u001b[2J.txt' in out.getvalue()
