"""The directory journal is private, authenticated, exclusive and crash safe."""
import os

import pytest

from scone_memory.core.errors import InvalidInput


def journal(tmp_path, **changes):
    from scone_memory.ingestion.source_journal import SourceJournal
    root = tmp_path / 'documents'
    root.mkdir(exist_ok=True)
    options = dict(path=tmp_path / 'sync.journal', root=root, space='alpha', store_id='local-catalog', key=b'k' * 32)
    return SourceJournal(**(options | changes))


def test_journal_reopens_exact_encrypted_source_transition(tmp_path):
    from scone_memory.ingestion.source_journal import SourceEntry, SourceRevision
    j = journal(tmp_path)
    with j.locked():
        state = j.load()
        revision = SourceRevision(original_sha256='a' * 64, manifest_sha256='b' * 64, parser_revision='parser-v1')
        state.entries['confidential-report.txt'] = SourceEntry(state='replace', pending=revision)
        j.save(state)
    raw = (tmp_path / 'sync.journal').read_bytes()
    assert b'confidential-report' not in raw and b'parser-v1' not in raw
    assert (tmp_path / 'sync.journal').stat().st_mode & 0o777 == 0o600
    with journal(tmp_path).locked() as reopened:
        assert reopened.load() == state


@pytest.mark.parametrize('changes', [{'key': b'x' * 32}, {'space': 'beta'}, {'store_id': 'another-catalog'}])
def test_foreign_binding_and_key_refuse_without_changing_journal(tmp_path, changes):
    with journal(tmp_path).locked() as j:
        j.save(j.load())
    before = (tmp_path / 'sync.journal').read_bytes()
    with pytest.raises(InvalidInput), journal(tmp_path, **changes).locked() as j:
        j.load()
    assert (tmp_path / 'sync.journal').read_bytes() == before


def test_different_root_refused(tmp_path):
    with journal(tmp_path).locked() as j:
        j.save(j.load())
    other = tmp_path / 'other'
    other.mkdir()
    with pytest.raises(InvalidInput), journal(tmp_path, root=other).locked() as j:
        j.load()


@pytest.mark.parametrize('damage', ['truncate', 'flip', 'empty'])
def test_damaged_journal_is_never_treated_as_new(tmp_path, damage):
    with journal(tmp_path).locked() as j:
        j.save(j.load())
    path = tmp_path / 'sync.journal'
    raw = path.read_bytes()
    damaged = raw[:-1] if damage == 'truncate' else raw[:-1] + bytes([raw[-1] ^ 1]) if damage == 'flip' else b''
    path.write_bytes(damaged)
    with pytest.raises(InvalidInput), journal(tmp_path).locked() as j:
        j.load()
    assert path.read_bytes() == damaged


def test_lock_excludes_other_objects_and_reentrance(tmp_path):
    with journal(tmp_path).locked() as first:
        with pytest.raises(InvalidInput), journal(tmp_path).locked():
            pass
        with pytest.raises(InvalidInput), first.locked():
            pass
        first.save(first.load())
    with journal(tmp_path).locked() as next_owner:
        next_owner.load()


@pytest.mark.parametrize('target', ['sync.journal', 'sync.journal.lock'])
@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'public', 'fifo'])
def test_unsafe_files_are_refused_without_following_or_blocking(tmp_path, target, kind):
    destination = tmp_path / 'untouched'
    destination.write_bytes(b'private material')
    destination.chmod(0o600)
    path = tmp_path / target
    if kind == 'symlink':
        path.symlink_to(destination)
    elif kind == 'hardlink':
        os.link(destination, path)
    elif kind == 'fifo':
        os.mkfifo(path, 0o600)
    else:
        path.write_bytes(b'public')
        path.chmod(0o644)
    with pytest.raises(InvalidInput), journal(tmp_path).locked() as j:
        j.load()
    assert destination.read_bytes() == b'private material'


def test_failed_atomic_replace_preserves_previous_state(tmp_path, monkeypatch):
    with journal(tmp_path).locked() as j:
        state = j.load()
        j.save(state)
        before = (tmp_path / 'sync.journal').read_bytes()
        def fail(*args, **kwargs):
            raise OSError('simulated interruption')
        monkeypatch.setattr(os, 'replace', fail)
        with pytest.raises(InvalidInput):
            j.save(state)
        assert (tmp_path / 'sync.journal').read_bytes() == before
        assert sorted(path.name for path in tmp_path.iterdir()) == ['documents', 'sync.journal', 'sync.journal.lock']
    with journal(tmp_path).locked() as j:
        assert j.load() == state


def test_save_requires_loaded_collection_and_valid_state(tmp_path):
    from scone_memory.ingestion.source_journal import SourceState
    j = journal(tmp_path)
    with pytest.raises(InvalidInput):
        j.load()
    with j.locked():
        state = j.load()
        with pytest.raises(InvalidInput):
            j.save(SourceState(collection_id='f' * 32))
        state.entries['../escape.txt'] = object()
        with pytest.raises(InvalidInput):
            j.save(state)
    assert not (tmp_path / 'sync.journal').exists()


def test_disappearing_journal_does_not_start_another_collection_mid_run(tmp_path):
    with journal(tmp_path).locked() as j:
        j.save(j.load())
        (tmp_path / 'sync.journal').unlink()
        with pytest.raises(InvalidInput):
            j.load()


def test_public_parent_is_refused(tmp_path):
    tmp_path.chmod(0o777)
    try:
        with pytest.raises(InvalidInput), journal(tmp_path).locked():
            pass
    finally:
        tmp_path.chmod(0o700)


@pytest.mark.parametrize('change', ['revision', 'entry'])
def test_unchecked_nested_copies_cannot_overwrite_valid_journal(tmp_path, change):
    from scone_memory.ingestion.source_journal import SourceEntry, SourceRevision
    with journal(tmp_path).locked() as j:
        state = j.load()
        revision = SourceRevision(original_sha256='a' * 64, manifest_sha256='b' * 64,
                                  parser_revision='v1', episode_id=1)
        entry = SourceEntry(state='active', current=revision)
        state.entries['report.txt'] = entry
        j.save(state)
        before = (tmp_path / 'sync.journal').read_bytes()
        broken = (entry.model_copy(update={'current': revision.model_copy(update={'original_sha256': 'invalid'})})
                  if change == 'revision' else entry.model_copy(update={'state': 'unknown'}))
        state.entries['report.txt'] = broken
        with pytest.raises(InvalidInput):
            j.save(state)
        assert (tmp_path / 'sync.journal').read_bytes() == before
    with journal(tmp_path).locked() as j:
        assert j.load().entries['report.txt'] == entry
