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
current atomically with new chunk writes. Triggers mark external writes
dirty, and a bounded query-time pass backfills old or externally written
rows. The original
``chunks_fts`` table stays in the schema untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
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
    """The FTS5 expression for the query's terms and any prefixes, OR-joined
    and quoted; None when empty. A term a prefix covers is left to the
    prefix phrase, so bm25 counts it once, as the in-memory lane does."""
    stems = [term(prefix) for prefix in prefixes if prefix]
    tokens = [term(token) for token in tokenize(query)]
    pieces = ['"' + token.replace('"', '""') + '"' for token in tokens if not any(token.startswith(stem) for stem in stems)]
    pieces += ['"' + stem.replace('"', '""') + '"*' for stem in stems]
    return " OR ".join(pieces) if pieces else None


#: The idf FTS5's bm25() gives a phrase more than half the rows hold, so a
#: weight can be moved from one idf to another as bm25() itself would weigh it.
_FTS5_MIN_IDF = 1e-6


def _fts5_idf(rows: int, hits: int) -> float:
    """The idf FTS5's bm25() gives a phrase ``hits`` of ``rows`` rows hold."""
    idf = math.log((rows - hits + 0.5) / (hits + 0.5))
    return idf if idf > 0 else _FTS5_MIN_IDF


def _hits(conn: sqlite3.Connection, phrase: str) -> int:
    return int(conn.execute("SELECT count(*) FROM chunk_lexical_fts WHERE chunk_lexical_fts MATCH ?", (phrase,)).fetchone()[0])


#: Query words one family search moves to their own idf at most. Each reads
#: two tables into the join, beside the search's four, and SQLite joins at
#: most 64; the rarest words are moved first, and the rest keep their
#: family's weight and are counted, so the lane can say so.
MAX_EXACT_FORMS = 24


def exact_form_rank(conn: sqlite3.Connection, query: str,
                    prefixes: Sequence[str]) -> tuple[str, str, str, list[object], int]:
    """The common table expressions, the rank expression, the joins it reads
    and their parameters, in the order they appear, for a family search that
    weighs the query's own words, and the number of words past
    ``MAX_EXACT_FORMS`` left at their family's weight.

    bm25() scores a prefix phrase at the family's idf and cannot weigh one
    row's phrase differently from another's. A row holding one of the
    query's own words in a family has that phrase's part moved to the idf
    of the rarest such word it holds: the family phrase's part of that
    row's bm25() is read, and the difference between the two idfs, as a
    multiple of it, is added. The family is still one phrase, counted once,
    as ``Bm25.search`` counts it with ``exact_forms``; a row holding only
    relatives keeps its rank.

    The part is read only for rows holding the word, as the bm25() of the
    family AND the word less the bm25() of the word alone (each phrase keeps
    its idf over the whole index in any expression), each into a
    materialized table. Joined as a subquery instead, SQLite looked the
    prefix up again for every matching row, seconds a query on 45,000
    chunks; read for every row the family holds and tested there for the
    word, it cost three times the lane's own time.
    """
    stems = [term(prefix) for prefix in prefixes if prefix]
    tokens = [term(token) for token in tokenize(query)]
    tables: list[str] = []
    rank, joins, phrases, multiples = "bm25(chunk_lexical_fts)", "", [], []
    rows = int(conn.execute("SELECT count(*) FROM chunk_lexical").fetchone()[0])
    # Every word that would move, as (idf, family, word). A word no row holds
    # would take the highest idf and move nothing, and a word as common as
    # its family (no row holds a relative without it) moves nothing either;
    # neither costs a scan. A repeated word is weighed once.
    movable: list[tuple[float, int, str]] = []
    family_idfs: list[float] = []
    for number, stem in enumerate(stems):
        family_idf = _fts5_idf(rows, _hits(conn, '"' + stem.replace('"', '""') + '"*'))
        family_idfs.append(family_idf)
        for form in dict.fromkeys(token for token in tokens if token.startswith(stem)):
            hits = _hits(conn, '"' + form.replace('"', '""') + '"')
            idf = _fts5_idf(rows, hits)
            if hits and idf > family_idf:
                movable.append((idf, number, form))
    # The rarest words first, across families, up to the bound; the sort is
    # stable, so words as rare keep the query's family order.
    movable.sort(key=lambda move: -move[0])
    kept, cut = movable[:MAX_EXACT_FORMS], max(0, len(movable) - MAX_EXACT_FORMS)
    for number, stem in enumerate(stems):
        family = '"' + stem.replace('"', '""') + '"*'
        family_idf = family_idfs[number]
        # The rarest form first: coalesce takes the first form a row holds.
        moves: list[str] = []
        for idf, _, form in (move for move in kept if move[1] == number):
            word = '"' + form.replace('"', '""') + '"'
            alias = f"family{number}_{len(moves)}"
            for name, phrase in ((alias, f"{family} AND {word}"), (f"{alias}_word", word)):
                tables.append(f"{name} AS MATERIALIZED (SELECT rowid AS chunk_id, bm25(chunk_lexical_fts) AS rank"
                              " FROM chunk_lexical_fts WHERE chunk_lexical_fts MATCH ?)")
                joins += f" LEFT JOIN {name} ON {name}.chunk_id = c.id"
                phrases.append(phrase)
            moves.append(f"? * ({alias}.rank - {alias}_word.rank)")
            multiples.append(idf / family_idf - 1)
        if moves:
            rank += " + coalesce(" + ", ".join(moves) + ", 0.0)"
    return ("WITH " + ", ".join(tables) + " " if tables else ""), rank, joins, [*phrases, *multiples], cut


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


#: Chunks one synchronisation pass re-reads at most. A query after a
#: rebuild or a bulk write pays for this many and no more; what is still
#: behind is counted and returned, so the lane can say it is behind.
MAX_SYNC_ROWS = 2_048


def index_lexical_chunk(conn: sqlite3.Connection, chunk_id: int, space: str, text: str) -> None:
    """Index one chunk inside the caller's transaction before publishing it."""
    conn.execute("INSERT OR REPLACE INTO chunk_lexical(chunk_id, space, terms) VALUES (?, ?, ?)",
                 (chunk_id, space, terms_of(text)))
    conn.execute("DELETE FROM chunk_lexical_dirty WHERE chunk_id = ?", (chunk_id,))


def synchronize_lexical(conn: sqlite3.Connection, space: str) -> tuple[int, int]:
    """Bring ``space``'s lexical rows up to date, up to ``MAX_SYNC_ROWS`` of
    them; the number of chunks re-read and the number still behind."""
    done = 0
    with _savepoint(conn):
        while done < MAX_SYNC_ROWS:
            rows = conn.execute("""SELECT c.id, c.text FROM chunk_lexical_dirty AS d INDEXED BY chunk_lexical_dirty_space
                JOIN chunks AS c ON c.id = d.chunk_id AND c.space = d.space
                WHERE d.space = ? ORDER BY d.chunk_id LIMIT ?""", (space, min(256, MAX_SYNC_ROWS - done))).fetchall()
            if not rows:
                break
            for row in rows:
                index_lexical_chunk(conn, row['id'], space, row['text'])
                done += 1
        behind = conn.execute("SELECT count(*) FROM chunk_lexical_dirty WHERE space = ?", (space,)).fetchone()[0]
    return done, int(behind)
