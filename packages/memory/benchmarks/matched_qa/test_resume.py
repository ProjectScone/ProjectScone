from __future__ import annotations

from pathlib import Path

import pytest

from .run import failed_row, previous_rows, repair_journal_tail


def test_torn_answer_remains_an_attempt_not_a_retry(tmp_path: Path) -> None:
    attempts = tmp_path / 'attempts.jsonl'
    answers = tmp_path / 'observations.jsonl'
    attempts.write_text('{"id":"q","arm":"scone"}\n')
    answers.write_bytes(b'{"id":"q","arm":"scone","ans')
    repair_journal_tail(answers)
    assert answers.read_bytes() == b''
    assert len(list(tmp_path.glob('observations.jsonl.torn-*'))) == 1
    pending = set(previous_rows(attempts)) - set(previous_rows(answers))
    assert pending == {('q', 'scone')}
    assert failed_row('q', 'scone', 'interrupted_attempt')['completed'] is False


def test_complete_tail_gets_newline_and_duplicate_journal_rejected(tmp_path: Path) -> None:
    path = tmp_path / 'attempts.jsonl'
    line = '{"id":"q","arm":"scone"}'
    path.write_text(line)
    repair_journal_tail(path)
    assert path.read_text() == line + '\n'
    assert len(previous_rows(path)) == 1
    path.write_text((line + '\n') * 2)
    with pytest.raises(ValueError, match='duplicate'):
        previous_rows(path)
