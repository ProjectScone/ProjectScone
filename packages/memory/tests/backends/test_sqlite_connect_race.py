"""Opening an existing catalog must wait before reading a writable snapshot."""
import sqlite3
import threading
import time

from scone_memory.backends.sqlite import connect


def test_open_serializes_derived_schema_with_a_committing_writer(tmp_path, monkeypatch):
    path = tmp_path / 'catalog.db'
    connect(path).close()
    original_connect = sqlite3.connect
    writer = original_connect(path, check_same_thread=False)
    started = threading.Event()
    locked = threading.Event()
    failures = []

    def write():
        try:
            assert started.wait(2)
            writer.execute('BEGIN IMMEDIATE')
            writer.execute("INSERT OR REPLACE INTO meta VALUES ('concurrent_writer', 'committed')")
            locked.set()
            time.sleep(0.1)
            writer.commit()
        except BaseException as error:
            failures.append(error)
            locked.set()

    class Connection(sqlite3.Connection):
        def executescript(self, script):
            if 'CREATE INDEX IF NOT EXISTS chunks_window' in script:
                started.set()
                assert locked.wait(2)
            return super().executescript(script)

    def observed(*args, **kwargs):
        return original_connect(*args, **kwargs, factory=Connection)

    monkeypatch.setattr(sqlite3, 'connect', observed)
    thread = threading.Thread(target=write)
    thread.start()
    opened = None
    try:
        opened = connect(path)
        assert opened.execute("SELECT value FROM meta WHERE key='concurrent_writer'").fetchone()[0] == 'committed'
        assert not opened.in_transaction
    finally:
        started.set()
        thread.join(3)
        writer.close()
        if opened is not None:
            opened.close()
    assert not failures
