"""Vectors kept by the text they came from, so an unchanged chunk is never
embedded twice.

A file that changes on one line is stored again whole: `sync`, `map` and
`replace` cut it into chunks and embed every chunk, though all but one
are the same text as before, and the embedder is the expensive part of
ingestion -- with a local model, seconds per file. The reference
framework's ingestion pipeline keeps a cache of transformed nodes for
this. Here the cache is of vectors alone, keyed by what was embedded:
the embedder's id and width and the exact text it was given, prefix
included. Two chunks with the same text under the same embedder have
the same vector by definition, so a hit is the vector the embedder would
have returned, not an approximation of it; and a different embedder, a
different width or a different contextual prefix is a different key.

Bounded: a cache holds at most ``max_entries`` vectors and drops the
least recently used past that, and its record says how many it dropped.
Disclosed: every stored record says how many of its chunks came from the
cache (``Added.embeddings_reused``), and a cache says what it holds.
Nothing here is evidence -- a vector read back is checked for width and
finiteness, and a row that fails is treated as absent and removed -- and
nothing here may refuse a write: a cache that fails (a full disk, a
damaged file, a locked database) is a miss, counted in its record, and
the embedder answers as if there were no cache. A rebuild of the stored
vectors (``reembed_vectors``) clears the cache first, since a model can
change behind an id that did not.
"""

from __future__ import annotations

from array import array
from collections import OrderedDict
import hashlib
import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence

from ..core.errors import InvalidInput

#: The key's version. Changing what is embedded for a text (a prefix, a
#: normalisation) without changing the embedder's id must change this.
KEY_VERSION = "embedding-chunk-v1"
#: Vectors a file cache holds by default before it drops the least
#: recently used. At 768 doubles a vector, this is about 120 MB on disk.
MAX_ENTRIES = 20_000
#: Vectors an in-memory cache holds by default. A Python list of floats
#: is about four times the packed size, so the bound is a quarter of the
#: file's for about the same memory.
MAX_MEMORY_ENTRIES = 5_000
#: Rows a file cache writes before it evicts, so one large batch cannot
#: leave the file far past its bound.
_KEEP_SLICE = 1_000

_log = logging.getLogger(__name__)


def cache_key(embedder_id: str, dim: int, text: str) -> str:
    """What names a vector: the embedder, its width and the exact text."""
    return hashlib.sha256(json.dumps([KEY_VERSION, embedder_id, dim, text],
                                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sound(vector: Sequence[float], dim: int) -> bool:
    """A vector of the width, finite throughout. Integer components are
    accepted as the pipeline's own validator accepts them."""
    return (len(vector) == dim and dim > 0
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vector))


class EmbeddingCache(Protocol):
    """Where vectors wait to be reused. ``take`` returns the vectors it
    holds for the keys given, and counts each key asked as reused; ``keep``
    stores vectors the embedder just returned; ``clear`` empties it;
    ``failed`` is told of a failure the pipeline swallowed; ``record``
    says what it holds and what it did; ``close`` releases what it holds
    open."""

    def take(self, keys: Sequence[str], dim: int) -> dict[str, list[float]]: ...

    def keep(self, vectors: Mapping[str, Sequence[float]], dim: int) -> None: ...

    def clear(self) -> None: ...

    def failed(self, doing: str, error: BaseException) -> None: ...

    def record(self) -> dict[str, object]: ...

    async def close(self) -> None: ...


class _Counted:
    """What both caches count: served, kept, dropped and failed."""

    name = ""
    reused = kept = evicted = failures = 0
    last_failure: Optional[str] = None
    closed = False

    def failed(self, doing: str, error: BaseException) -> None:
        self.failures += 1
        self.last_failure = f"{doing}: {type(error).__name__}: {error}"
        _log.warning("embedding cache %s failed while %s: %s: %s", self.name, doing, type(error).__name__, error)

    def _counts(self) -> dict[str, object]:
        return {"reused": self.reused, "kept": self.kept, "evicted": self.evicted, "failures": self.failures,
                "last_failure": self.last_failure}


def _bound(max_entries: int) -> int:
    if type(max_entries) is not int or max_entries < 1:
        raise InvalidInput("an embedding cache holds at least one vector")
    return max_entries


class InMemoryEmbeddingCache(_Counted):
    """For one process: the least recently used vector goes first."""

    name = "memory"

    def __init__(self, max_entries: int = MAX_MEMORY_ENTRIES) -> None:
        self.max_entries = _bound(max_entries)
        self._held: OrderedDict[str, list[float]] = OrderedDict()

    def take(self, keys: Sequence[str], dim: int) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for key in keys:
            vector = self._held.get(key)
            if vector is None:
                continue
            if not _sound(vector, dim):
                del self._held[key]
                continue
            self._held.move_to_end(key)
            found[key] = list(vector)
        # Counted per key asked, so two chunks of one text are two served.
        self.reused += sum(1 for key in keys if key in found)
        return found

    def keep(self, vectors: Mapping[str, Sequence[float]], dim: int) -> None:
        for key, vector in vectors.items():
            if not _sound(vector, dim):
                raise InvalidInput("an embedding cache keeps only finite vectors of the embedder's width")
            self._held[key] = [float(v) for v in vector]
            self._held.move_to_end(key)
            self.kept += 1
            # Evicted as it goes: the bound holds through a keep, not
            # only after one.
            while len(self._held) > self.max_entries:
                self._held.popitem(last=False)
                self.evicted += 1

    def clear(self) -> None:
        self._held.clear()

    def record(self) -> dict[str, object]:
        return {"store": self.name, "entries": len(self._held), "max_entries": self.max_entries, **self._counts()}

    async def close(self) -> None:
        self.closed = True
        self._held.clear()


class SqliteEmbeddingCache(_Counted):
    """For every process that opens the same file: a `sync` run tomorrow
    reuses what today's embedded. One table, vectors as packed doubles,
    a use counter that orders eviction. Reading updates the counter, so
    the file must be writable; on one that is not, every take fails and
    is counted as a miss."""

    name = "sqlite"

    def __init__(self, path: str | Path, max_entries: int = MAX_ENTRIES) -> None:
        self.max_entries = _bound(max_entries)
        self.path = str(Path(str(path)).expanduser()) if str(path) != ":memory:" else ":memory:"
        try:
            self._conn = sqlite3.connect(self.path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("CREATE TABLE IF NOT EXISTS vectors ("
                               "key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL, used INTEGER NOT NULL)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS vectors_used ON vectors(used)")
            self._conn.commit()
        except sqlite3.Error as error:
            raise InvalidInput(f"SCONE_EMBEDDING_CACHE: cannot open {self.path}: {error}") from None

    def _tick(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(used), 0) FROM vectors").fetchone()
        return int(row[0]) + 1

    def take(self, keys: Sequence[str], dim: int) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        if not keys:
            return found
        now = self._tick()
        damaged: list[str] = []
        for key in keys:
            row = self._conn.execute("SELECT dim, vector FROM vectors WHERE key = ?", (key,)).fetchone()
            if row is None:
                continue
            width, blob = int(row[0]), row[1]
            values = array("d")
            try:
                values.frombytes(bytes(blob))
            except ValueError:
                damaged.append(key)
                continue
            vector = values.tolist()
            if width != dim or not _sound(vector, dim):
                damaged.append(key)
                continue
            found[key] = vector
        if damaged:
            self._conn.executemany("DELETE FROM vectors WHERE key = ?", [(key,) for key in damaged])
        if found:
            self._conn.executemany("UPDATE vectors SET used = ? WHERE key = ?", [(now, key) for key in found])
        self._conn.commit()
        self.reused += sum(1 for key in keys if key in found)
        return found

    def keep(self, vectors: Mapping[str, Sequence[float]], dim: int) -> None:
        if not vectors:
            return
        now = self._tick()
        rows = []
        for key, vector in vectors.items():
            if not _sound(vector, dim):
                raise InvalidInput("an embedding cache keeps only finite vectors of the embedder's width")
            rows.append((key, dim, array("d", (float(v) for v in vector)).tobytes(), now))
        # Written a slice at a time and evicted after each, so the file
        # never holds more than a slice past its bound.
        for start in range(0, len(rows), _KEEP_SLICE):
            self._conn.executemany("INSERT OR REPLACE INTO vectors (key, dim, vector, used) VALUES (?, ?, ?, ?)",
                                   rows[start:start + _KEEP_SLICE])
            self.kept += len(rows[start:start + _KEEP_SLICE])
            held = int(self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
            over = held - self.max_entries
            if over > 0:
                self._conn.execute("DELETE FROM vectors WHERE key IN "
                                   "(SELECT key FROM vectors ORDER BY used ASC, key ASC LIMIT ?)", (over,))
                self.evicted += over
            self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM vectors")
        self._conn.commit()

    def record(self) -> dict[str, object]:
        held = int(self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
        return {"store": self.name, "path": self.path, "entries": held, "max_entries": self.max_entries, **self._counts()}

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._conn.close()


def build_embedding_cache(setting: str | None, *, max_entries: Optional[int] = None) -> EmbeddingCache | None:
    """``SCONE_EMBEDDING_CACHE``: unset (or ``none``, ``off``) for none,
    ``memory`` for one that lives with the process, otherwise the path of
    a file shared by every process that opens it. A path that cannot be
    opened is refused by name."""
    if setting is None or not setting.strip() or setting.strip().lower() in ("none", "off", "0", "false"):
        return None
    if setting.strip().lower() == "memory":
        return InMemoryEmbeddingCache(max_entries if max_entries is not None else MAX_MEMORY_ENTRIES)
    return SqliteEmbeddingCache(setting.strip(), max_entries if max_entries is not None else MAX_ENTRIES)
