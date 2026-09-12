"""Import identity and control intent survive restart without executing work."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from scone_memory.agents.workflow import WorkflowError
from scone_memory.ingestion.import_store import DocumentImportSpec, DocumentImportStore, ImportConflict


def spec(**changes):
    return DocumentImportSpec.model_validate({'attachment_id': 'a' * 64,
        'filename': 'Private-report.pdf', 'parser_revision': 'parser-1', **changes})


def test_input_and_cancel_intent_survive_restart_without_plaintext(tmp_path):
    path = tmp_path / 'imports.sqlite'
    store = DocumentImportStore(path, key=b'k' * 32)
    saved = store.register('alpha', 'import-1', spec())
    assert store.register('alpha', 'import-1', spec()) == saved
    started = store.start_attempt('alpha', 'import-1', expected_revision=0)
    assert started.attempt == 1 and started.revision == 1
    cancelled = store.request_cancel('alpha', 'import-1')
    assert cancelled.cancel_requested_at is not None and cancelled.revision == 2
    assert store.request_cancel('alpha', 'import-1') == cancelled
    assert store.register('alpha', 'import-1', spec()) == cancelled
    store.close()
    raw = path.read_bytes()
    assert b'Private-report.pdf' not in raw and b'a' * 64 not in raw and b'alpha' not in raw
    reopened = DocumentImportStore(path, key=b'k' * 32)
    assert reopened.get('alpha', 'import-1') == cancelled
    assert reopened.get('beta', 'import-1') is None
    reopened.close()


@pytest.mark.parametrize('change', [
    {'attachment_id': 'b' * 64}, {'filename': 'Other.pdf'}, {'parser_revision': 'parser-2'},
    {'pdf_ocr': {'mode': 'all_pages', 'reading_order': 'columns_ltr'}},
    {'limits': {'max_text_bytes': 1000}},
])
def test_same_identifier_cannot_rebind_input_options_or_parser(tmp_path, change):
    store = DocumentImportStore(tmp_path / 'imports.sqlite', key=b'k' * 32)
    original = store.register('alpha', 'same', spec())
    with pytest.raises(ImportConflict):
        store.register('alpha', 'same', spec(**change))
    assert store.get('alpha', 'same') == original
    store.close()


def test_cancel_cannot_be_lost_to_an_old_start_or_an_automatic_retry(tmp_path):
    store = DocumentImportStore(tmp_path / 'imports.sqlite', key=b'k' * 32)
    store.register('alpha', 'one', spec())
    cancelled = store.request_cancel('alpha', 'one')
    with pytest.raises(ImportConflict):
        store.start_attempt('alpha', 'one', expected_revision=0)
    with pytest.raises(WorkflowError, match='cancelled'):
        store.start_attempt('alpha', 'one', expected_revision=cancelled.revision)
    resumed = store.start_attempt('alpha', 'one', expected_revision=cancelled.revision, resume=True)
    assert resumed.attempt == 1 and resumed.cancel_requested_at is None
    with pytest.raises(ImportConflict):
        store.start_attempt('alpha', 'one', expected_revision=cancelled.revision, resume=True)
    store.close()


def test_competing_starts_have_one_revision_winner(tmp_path):
    path = tmp_path / 'imports.sqlite'
    store = DocumentImportStore(path, key=b'k' * 32)
    store.register('alpha', 'one', spec()); store.close()
    barrier = Barrier(2)
    def start(_):
        other = DocumentImportStore(path, key=b'k' * 32)
        try:
            barrier.wait(timeout=5)
            try:
                return other.start_attempt('alpha', 'one', expected_revision=0)
            except ImportConflict:
                return None
        finally:
            other.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(start, range(2)))
    assert sum(result is not None for result in results) == 1


def test_paging_capacity_and_cursor_are_space_bound(tmp_path):
    store = DocumentImportStore(tmp_path / 'imports.sqlite', key=b'k' * 32, max_imports=4)
    for name in ['one', 'two', 'three']:
        store.register('alpha', name, spec())
    store.register('beta', 'one', spec())
    page = store.list('alpha', limit=2)
    assert len(page.items) == 2 and page.next_after
    rest = store.list('alpha', limit=2, after=page.next_after)
    assert len(rest.items) == 1 and rest.next_after is None
    assert len({request.import_id for request in (*page.items, *rest.items)}) == 3
    with pytest.raises(WorkflowError):
        store.list('beta', after=page.next_after)
    with pytest.raises(WorkflowError, match='limit'):
        store.register('alpha', 'four', spec())
    assert store.register('alpha', 'one', spec()) is not None
    store.close()


@pytest.mark.parametrize('identifier', [True, 1, '', '../source', 'x' * 129])
def test_strict_import_identifiers_reject_coercion_and_paths(tmp_path, identifier):
    store = DocumentImportStore(tmp_path / 'imports.sqlite', key=b'k' * 32)
    with pytest.raises((WorkflowError, ValueError)):
        store.register('alpha', identifier, spec())
    store.close()


def test_wrong_key_and_ocr_on_another_format_are_refused(tmp_path):
    path = tmp_path / 'imports.sqlite'
    store = DocumentImportStore(path, key=b'k' * 32); store.close()
    with pytest.raises(WorkflowError):
        DocumentImportStore(path, key=b'x' * 32)
    with pytest.raises(ValueError):
        spec(filename='report.csv', pdf_ocr={'mode': 'all_pages', 'reading_order': 'provider'})


def test_an_attempted_import_requires_explicit_resume_even_without_cancel(tmp_path):
    store = DocumentImportStore(tmp_path / 'imports.sqlite', key=b'k' * 32)
    store.register('alpha', 'one', spec())
    started = store.start_attempt('alpha', 'one', expected_revision=0)
    with pytest.raises(WorkflowError, match='resume_required'):
        store.start_attempt('alpha', 'one', expected_revision=started.revision)
    resumed = store.start_attempt('alpha', 'one', expected_revision=started.revision, resume=True)
    assert resumed.attempt == 2 and resumed.revision == 2
    store.close()
