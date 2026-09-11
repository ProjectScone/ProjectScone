"""Directory absence is credible only after a bounded, complete stable scan."""
import os

import pytest

from scone_memory.core.errors import InvalidInput


def test_snapshot_reads_nested_files_deterministically_and_skips_unsupported(tmp_path):
    from scone_memory.ingestion.source_scan import DirectoryScanner
    (tmp_path / 'nested').mkdir()
    (tmp_path / 'z.txt').write_bytes(b'last')
    (tmp_path / 'nested' / 'a.md').write_bytes(b'first')
    (tmp_path / 'photo.unknown').write_bytes(b'not a document')
    scanner = DirectoryScanner(tmp_path)
    first = scanner.scan()
    assert first.complete and first.skipped == 1
    assert [item.path for item in first.files] == ['nested/a.md', 'z.txt']
    assert [scanner.read(item) for item in first.files] == [b'first', b'last']
    assert first.same_inventory(scanner.scan())
    (tmp_path / 'z.txt').write_bytes(b'edited')
    assert not first.same_inventory(scanner.scan())
    with pytest.raises(InvalidInput):
        scanner.read(first.files[1])
    (tmp_path / 'z.txt').unlink()
    assert [item.path for item in scanner.scan().files] == ['nested/a.md']


@pytest.mark.parametrize('kind', ['symlink-file', 'symlink-directory', 'fifo'])
def test_special_entries_make_inventory_incomplete(tmp_path, kind):
    from scone_memory.ingestion.source_scan import DirectoryScanner
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'valid.txt').write_bytes(b'valid')
    external = tmp_path / 'outside'
    external.mkdir()
    (external / 'secret.txt').write_bytes(b'not in source root')
    if kind == 'fifo':
        os.mkfifo(root / 'pipe.txt')
    else:
        (root / 'link').symlink_to(external if kind.endswith('directory') else external / 'secret.txt')
    snapshot = DirectoryScanner(root).scan()
    assert not snapshot.complete
    assert [item.path for item in snapshot.files] == ['valid.txt']
    assert len(snapshot.issues) == 1


@pytest.mark.parametrize('limits', [dict(max_files=1), dict(max_entries=1), dict(max_total_bytes=6), dict(max_file_bytes=3)])
def test_limits_never_produce_complete_partial_inventory(tmp_path, limits):
    from scone_memory.ingestion.source_scan import DirectoryScanner, ScanLimits
    (tmp_path / 'one.txt').write_bytes(b'first')
    (tmp_path / 'two.txt').write_bytes(b'second')
    snapshot = DirectoryScanner(tmp_path, limits=ScanLimits(**limits)).scan()
    assert not snapshot.complete and snapshot.issues


def test_depth_budget_and_invalid_names_are_disclosed(tmp_path):
    from scone_memory.ingestion.source_scan import DirectoryScanner, ScanLimits
    (tmp_path / 'a').mkdir()
    (tmp_path / 'a' / 'b').mkdir()
    (tmp_path / 'a' / 'b' / 'secret.txt').write_bytes(b'deep')
    (tmp_path / 'bad\nname.txt').write_bytes(b'invalid')
    snapshot = DirectoryScanner(tmp_path, limits=ScanLimits(max_depth=1)).scan()
    assert not snapshot.complete
    assert {issue.code for issue in snapshot.issues} == {'depth_limit', 'invalid_path'}


def test_root_replacement_and_parent_symlink_are_refused(tmp_path):
    from scone_memory.ingestion.source_scan import DirectoryScanner
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'nested').mkdir()
    (root / 'nested' / 'a.txt').write_bytes(b'first')
    scanner = DirectoryScanner(root)
    original = scanner.scan()
    (root / 'nested').rename(root / 'old')
    (root / 'nested').symlink_to(root / 'old', target_is_directory=True)
    with pytest.raises(InvalidInput):
        scanner.read(original.files[0])
    root.rename(tmp_path / 'old-root')
    root.mkdir()
    assert not scanner.scan().complete


def test_mutation_during_read_is_disclosed_and_not_retained(tmp_path, monkeypatch):
    from scone_memory.ingestion.source_scan import DirectoryScanner
    path = tmp_path / 'a.txt'
    path.write_bytes(b'original')
    original_read = os.read
    changed = False
    def change(fd, size):
        nonlocal changed
        result = original_read(fd, size)
        if not changed and result:
            changed = True
            path.write_bytes(b'changed!')
        return result
    monkeypatch.setattr(os, 'read', change)
    snapshot = DirectoryScanner(tmp_path).scan()
    assert not snapshot.complete and not snapshot.files


def test_custom_suffixes_are_explicit_and_validated(tmp_path):
    from scone_memory.ingestion.source_scan import DirectoryScanner
    (tmp_path / 'file.custom').write_bytes(b'custom')
    snapshot = DirectoryScanner(tmp_path, extensions=frozenset({'.custom'})).scan()
    assert snapshot.complete and snapshot.files[0].path == 'file.custom'
    for suffixes in [frozenset({'txt'}), frozenset({'.TXT'}), frozenset({'.a/b'}), frozenset()]:
        with pytest.raises(InvalidInput):
            DirectoryScanner(tmp_path, extensions=suffixes)


@pytest.mark.parametrize('changes', [{'timeout_seconds': float('nan')}, {'max_files': 10001}, {'max_entries': True}])
def test_unchecked_limit_copies_cannot_disable_scan_bounds(tmp_path, changes):
    from scone_memory.ingestion.source_scan import DirectoryScanner, ScanLimits
    with pytest.raises(InvalidInput):
        DirectoryScanner(tmp_path, limits=ScanLimits().model_copy(update=changes))
