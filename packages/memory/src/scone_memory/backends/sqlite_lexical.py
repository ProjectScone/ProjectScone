"""The SQLite text lane over our own tokens, so the two stores agree by construction.

SQLite's built-in tokenizer keeps an unspaced run -- a Japanese or Thai
phrase -- as one token, so a part of it cannot be found, while the
in-memory lane cuts such runs into character grams and finds them. Two
stores that answer the same query differently are a bug a reader
cannot see. This index is the fix: a derived, disposable table holding
each chunk's terms exactly as ``retrieval.lexical.tokenize`` makes them
(diacritics folded, as the in-memory lane folds them), searched through
an FTS5 shadow that only splits on the spaces between those terms. The
lane then ranks the same tokens in both stores.

Like the fact-search postings, it is versioned by the tokenizer and the
Unicode data it ran under, rebuilt whole when either changes, and kept
current by triggers that mark a chunk dirty on write and by a
synchronisation pass at query time, bounded per call. The original
``chunks_fts`` table stays in the schema untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import re
import sqlite3
from typing import Iterator, Sequence
import unicodedata
from uuid import uuid4

from ..retrieval.lexical import TOKENIZER_VERSION, fold_diacritics, tokenize

_VERSION_KEY = "chunk_lexical_version"
_VERSION = f"1;tokenizer={TOKENIZER_VERSION};unicode={unicodedata.unidata_version};fold=diacritics"
#: An apostrophe inside a token ("don't") would split it for the shadow
#: tokenizer; a modifier-letter apostrophe is a letter to it, and the
#: query side makes the same swap.
_APOSTROPHE = "ʼ"
_DDL = (
    "CREATE TABLE IF NOT EXISTS chunk_lexical (chunk_id INTEGER PRIMARY KEY, space TEXT NOT NULL, terms TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS chunk_lexical_space ON chunk_lexical(space)",
    "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_lexical_fts USING fts5(terms, content='chunk_lexical', content_rowid='chunk_id',"
    " tokenize='unicode61 remove_diacritics 0')",
    "CREATE TABLE IF NOT EXISTS chunk_lexical_dirty (chunk_id INTEGER PRIMARY KEY, space TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS chunk_lexical_dirty_space ON chunk_lexical_dirty(space, chunk_id)",
    """CREATE TRIGGER IF NOT EXISTS chunk_lexical_insert AFTER INSERT ON chunks BEGIN
        INSERT OR REPLACE INTO chunk_lexical_dirty(chunk_id, space) VALUES (NEW.id, NEW.space);
        END""",
    """CREATE TRIGGER IF NOT EXISTS chunk_lexical_delete AFTER DELETE ON chunks BEGIN
        DELETE FROM chunk_lexical WHERE chunk_id = OLD.id;
        DELETE FROM chunk_lexical_dirty WHERE chunk_id = OLD.id;
        END""",
    """CREATE TRIGGER IF NOT EXISTS chunk_lexical_ai AFTER INSERT ON chunk_lexical BEGIN
        INSERT INTO chunk_lexical_fts(rowid, terms) VALUES (NEW.chunk_id, NEW.terms);
        END""",
    """CREATE TRIGGER IF NOT EXISTS chunk_lexical_ad AFTER DELETE ON chunk_lexical BEGIN
        INSERT INTO chunk_lexical_fts(chunk_lexical_fts, rowid, terms) VALUES ('delete', OLD.chunk_id, OLD.terms);
        END""",
)
_NAMED = re.compile(r"CREATE (?:VIRTUAL )?(TABLE|INDEX|TRIGGER) IF NOT EXISTS (\w+)", re.I)


def term(token: str) -> str:
    """One of our tokens as the shadow index holds it."""
    return fold_diacritics(token).replace("'", _APOSTROPHE)


def terms_of(text: str) -> str:
    return " ".join(term(token) for token in tokenize(text))


def lexical_match(query: str, prefixes: Sequence[str] = ()) -> str | None:
    """The FTS5 expression for the query's terms and any prefixes, OR-joined and quoted; None when empty."""
    pieces = ['"' + term(token).replace('"', '""') + '"' for token in tokenize(query)]
    pieces += ['"' + term(prefix).replace('"', '""') + '"*' for prefix in prefixes if prefix]
    return " OR ".join(pieces) if pieces else None


@contextmanager
def _savepoint(conn: sqlite3.Connection) -> Iterator[None]:
    name = "chunk_lexical_" + uuid4().hex
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise
    else:
        conn.execute(f"RELEASE {name}")


def _normalized(sql: str) -> str:
    return " ".join(sql.casefold().replace("if not exists ", "").split()).rstrip(";")


def _derived_objects(conn: sqlite3.Connection) -> tuple[bool, list[sqlite3.Row]]:
    expected: dict[tuple[str, str], str] = {}
    for statement in _DDL:
        found = _NAMED.match(statement.strip())
        assert found is not None
        expected[(found.group(1).lower(), found.group(2).casefold())] = _normalized(statement)
    rows = conn.execute("SELECT type, name, sql FROM sqlite_master WHERE name COLLATE NOCASE IN (SELECT value FROM json_each(?))",
                        (json.dumps([name for _, name in expected]),)).fetchall()
    complete = len(rows) == len(expected) and all(
        expected.get((row["type"], row["name"].casefold())) == _normalized(row["sql"] or "") for row in rows)
    return complete, rows


def initialize_lexical(conn: sqlite3.Connection) -> None:
    """Trust the version marker only when every derived object matches; else rebuild whole.

    A rebuild does not tokenise anything here: every chunk is marked dirty
    and the next query of each space brings that space up to date, so an
    old database opens at once and pays as it is read."""
    with _savepoint(conn):
        conn.execute("UPDATE meta SET value=value WHERE 0")
        complete, objects = _derived_objects(conn)
        row = conn.execute("SELECT value FROM meta WHERE key=?", (_VERSION_KEY,)).fetchone()
        if complete and row is not None and row[0] == _VERSION:
            return
        if not complete:
            priority = {"trigger": 0, "index": 1, "view": 2, "table": 3}
            for existing in sorted(objects, key=lambda item: priority[item["type"]]):
                conn.execute(f"DROP {existing['type']} IF EXISTS {existing['name']}")
        for statement in _DDL:
            conn.execute(statement)
        conn.execute("DELETE FROM chunk_lexical")
        conn.execute("DELETE FROM chunk_lexical_dirty")
        conn.execute("INSERT INTO chunk_lexical_dirty(chunk_id, space) SELECT id, space FROM chunks")
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (_VERSION_KEY, _VERSION))


def synchronize_lexical(conn: sqlite3.Connection, space: str) -> int:
    """Bring ``space``'s lexical rows up to date; the number of chunks re-read."""
    done = 0
    with _savepoint(conn):
        while True:
            rows = conn.execute("""SELECT c.id, c.text FROM chunk_lexical_dirty AS d INDEXED BY chunk_lexical_dirty_space
                JOIN chunks AS c ON c.id = d.chunk_id AND c.space = d.space
                WHERE d.space = ? ORDER BY d.chunk_id LIMIT 256""", (space,)).fetchall()
            if not rows:
                return done
            for row in rows:
                conn.execute("INSERT OR REPLACE INTO chunk_lexical(chunk_id, space, terms) VALUES (?, ?, ?)",
                             (row["id"], space, terms_of(row["text"])))
                conn.execute("DELETE FROM chunk_lexical_dirty WHERE chunk_id = ?", (row["id"],))
                done += 1
