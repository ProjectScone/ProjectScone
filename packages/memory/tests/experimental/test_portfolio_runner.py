import json

import pytest

from research20.run import save
from research20.validity import run


def test_partial_portfolio_cannot_be_reported_as_twenty_completed(tmp_path):
    output = tmp_path / 'report.json'
    save(output, run(), None)
    report = json.loads(output.read_text())
    assert report['complete'] is False
    assert len(report['results']) == 5
    assert report['provider_audits'] == []
    assert 'validity.py' in report['source_sha256']
    assert not output.with_suffix('.json.pending').exists()


def test_duplicate_ids_and_failed_runs_never_count_as_complete(tmp_path):
    result = run()[0]
    with pytest.raises(ValueError, match='duplicate'):
        save(tmp_path / 'duplicates.json', [result, result], None)
    output = tmp_path / 'failed.json'
    save(output, run(), None, 'ValueError')
    report = json.loads(output.read_text())
    assert report['complete'] is False
    assert report['error'] == 'ValueError'
