"""A benchmark's vector cache at the width the models emit: float32, never evicted.

``SqliteEmbeddingCache`` packs doubles and evicts by use, which suits an
engine. A LongMemEval-M run holds millions of chunk vectors: as doubles they
took 5.9 GB for 100 items on a disk with 12 GB free. Every model this harness
runs (fastembed ONNX, the hosted copy) produces float32, so storing float32
drops nothing the model emitted and halves the file. Nothing is evicted: a
cache that dropped vectors would make a rerun embed again and misreport what
it cost.

It is keyed exactly as ``CachedEmbedder`` keys (``cache_key``), so it drops
in wherever that wrapper takes a cache.
"""
from __future__ import annotations

import sqlite3
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path

_BATCH = 900  # below SQLite's default bound on host parameters in one statement


class Float32Cache:
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, '
                           'vector BLOB NOT NULL)')
        self.asked = self.found = self.stored = 0

    def take(self, keys: Sequence[str], dim: int) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        unique = list(dict.fromkeys(keys))
        for start in range(0, len(unique), _BATCH):
            chunk = unique[start:start + _BATCH]
            rows = self._conn.execute(
                f'SELECT key, dim, vector FROM vectors WHERE key IN ({",".join("?" * len(chunk))})', chunk)
            for key, width, blob in rows:
                if width != dim:
                    raise ValueError(f'cached vector for {key[:16]} is {width} wide, asked for {dim}')
                values = array('f')
                values.frombytes(bytes(blob))
                found[key] = [float(v) for v in values]
        self.asked += len(keys)
        self.found += sum(1 for key in keys if key in found)
        return found

    def keep(self, vectors: Mapping[str, Sequence[float]], dim: int) -> None:
        rows = []
        for key, vector in vectors.items():
            if len(vector) != dim:
                raise ValueError(f'vector for {key[:16]} is {len(vector)} wide, not {dim}')
            rows.append((key, dim, array('f', (float(v) for v in vector)).tobytes()))
        with self._conn:
            self._conn.executemany('INSERT OR REPLACE INTO vectors (key, dim, vector) VALUES (?, ?, ?)', rows)
        self.stored += len(rows)

    def count(self) -> int:
        return int(self._conn.execute('SELECT count(*) FROM vectors').fetchone()[0])

    def record(self) -> dict[str, object]:
        return {'kind': 'float32 sqlite, never evicted', 'asked': self.asked, 'found': self.found,
                'stored': self.stored}

    def close(self) -> None:
        self._conn.close()


def migrate_doubles(source: str | Path, target: Float32Cache, *, batch: int = 20_000) -> int:
    """Copies a ``SqliteEmbeddingCache`` file (packed doubles) into ``target`` as float32; returns rows copied."""
    reader = sqlite3.connect(str(source))
    copied = 0
    rows = reader.execute('SELECT key, dim, vector FROM vectors')
    while True:
        chunk = rows.fetchmany(batch)
        if not chunk:
            break
        converted: dict[int, dict[str, list[float]]] = {}
        for key, dim, blob in chunk:
            values = array('d')
            values.frombytes(bytes(blob))
            converted.setdefault(int(dim), {})[key] = list(values)
        for dim, vectors in converted.items():
            target.keep(vectors, dim)
        copied += len(chunk)
    reader.close()
    return copied
