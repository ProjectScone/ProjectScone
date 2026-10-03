"""What a schema file says: the tables it makes, their columns, and what rests on what.

A code graph that knows a project's files and packages still stops at
the database: a table is what half the functions read and write, and a
migration that drops a column reaches every one of them. The reference
graph introspects a live database; this reads the schema as it is
written down -- ``CREATE TABLE`` statements and the views and foreign
keys around them in a ``.sql`` file -- which is what a repository holds,
what a diff changes, and what nobody has to connect to.

Claims, each quoted from the line that makes it: the file defines a
table; a table defines each of its columns; a table with a foreign key
depends on the table it references, and a view depends on the tables
it selects from. ``depends_on`` is the predicate a manifest's
dependencies already use, so ``graph affected orders`` lists the tables
and views that rest on ``orders`` beside the code that imports the
module. Nothing here binds code to a table: a query is a string, and a
string that names a table is a guess this graph does not make.
"""

from __future__ import annotations

import re
from typing import Optional

from .code import MAX_LINES
from .code_graph import DEFINES, CodeClaim
from .manifests import DEPENDS_ON, _Claims

SCHEMA_SUFFIXES = (".sql", ".ddl")
_IDENT = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)(?:\.(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*))?'
_CREATE_TABLE = re.compile(r"\bcreate\s+(?:(?:temp|temporary|unlogged)\s+)?table\s+(?:if\s+not\s+exists\s+)?(" + _IDENT + r")\s*\(", re.I)
_CREATE_VIEW = re.compile(r"\bcreate\s+(?:or\s+replace\s+)?(?:(?:temp|temporary|materialized)\s+)?view\s+(?:if\s+not\s+exists\s+)?("
                          + _IDENT + r")\b", re.I)
_ALTER_FK = re.compile(r"\balter\s+table\s+(?:only\s+)?(?:if\s+exists\s+)?(" + _IDENT + r")\b", re.I)
_REFERENCES = re.compile(r"\breferences\s+(" + _IDENT + r")", re.I)
_PART = re.compile(r'"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([A-Za-z_][\w$]*)')
#: An item of a table body that is a constraint, not a column: the word
#: alone is not enough, since a column may be called key or check; what
#: follows it is what tells them apart.
_CONSTRAINT = re.compile(r"^\s*(?:constraint\b|primary\s+key\b|foreign\s+key\b|unique\s*(?:\(|key\b|index\b)|check\s*\("
                         r"|(?:index|key)\s+" + _IDENT + r"\s*\(|(?:index|key)\s*\(|exclude\b|like\s+" + _IDENT + r"\s*$)", re.I)
#: Where one statement ends when no semicolon says so: a T-SQL batch
#: separator, or the next statement beginning at the start of a line.
_BREAK = re.compile(r"^\s*go\s*$|^\s*(?:create|alter|drop|insert|update|delete|grant|revoke|set|use|begin|commit)\b", re.I | re.M)
_TOKEN = re.compile(r'"[^"]*"|`[^`]*`|\[[^\]]*\]|[A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)*|\(|\)|,|;|\S')
#: After FROM, the words that end the list of sources.
_CLAUSE_END = frozenset({"where", "group", "order", "having", "limit", "union", "except", "intersect", "join", "on", "using",
                         "left", "right", "inner", "outer", "cross", "natural", "full", "window", "fetch", "offset", "for",
                         "returning", "into", "values", "lateral", "with", "as"})


def is_schema(path: str) -> bool:
    """Known by its suffix: ``.sql`` or ``.ddl``, wherever it sits."""
    return bool(path) and path.replace("\\", "/").lower().rsplit("/", 1)[-1].endswith(SCHEMA_SUFFIXES)


def _name(raw: str) -> str:
    """A name as the graph spells it: quotes and brackets gone, case kept."""
    return ".".join(next(part for part in found.groups() if part is not None) for found in _PART.finditer(raw))


def _blank(content: str) -> str:
    """The text with every comment turned to spaces and every string
    literal's contents too, so lines and offsets stay where they were, a
    table named in a comment is not a table, and a comma, a paren, a
    semicolon or the word REFERENCES inside a string is not syntax. Quoted
    identifiers keep their text, since a name is what they hold; a
    backslash inside a string escapes the next character."""
    out: list[str] = []
    i = 0
    n = len(content)
    while i < n:
        if content.startswith("--", i):
            end = content.find("\n", i)
            end = n if end < 0 else end
            out.append(" " * (end - i))
            i = end
        elif content.startswith("/*", i):
            end = content.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append(re.sub(r"[^\n]", " ", content[i:end]))
            i = end
        elif content[i] == "'":
            j = i + 1
            while j < n:
                if content[j] == "\\":
                    j += 2
                    continue
                if content[j] == "'":
                    if j + 1 < n and content[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            end = min(j + 1, n)
            out.append("'" + re.sub(r"[^\n]", " ", content[i + 1:end - 1]) + ("'" if end - 1 > i and content[end - 1] == "'" else ""))
            i = end
        elif content[i] in '"`[':
            quote = {'"': '"', "`": "`", "[": "]"}[content[i]]
            end = content.find(quote, i + 1)
            end = n if end < 0 else end + 1
            out.append(content[i:end])
            i = end
        else:
            out.append(content[i])
            i += 1
    return "".join(out)


def _statement_end(plain: str, start: int) -> int:
    """Where the statement beginning at ``start`` ends: the first
    semicolon, a GO line, or the next statement starting a line."""
    semicolon = plain.find(";", start)
    breaks = _BREAK.search(plain, start)
    candidates = [end for end in (semicolon, breaks.start() if breaks else -1) if end >= 0]
    return min(candidates) if candidates else len(plain)


def _sources(statement: str) -> list[tuple[str, int]]:
    """The tables a SELECT reads, with their offsets: names after a FROM
    that belongs to a SELECT at the same nesting (so EXTRACT(MONTH FROM
    x) names nothing), the comma list after it with aliases skipped, and
    each JOIN's table; a name followed by a paren is a function, a name a
    WITH clause defined is not a table, and a keyword is neither."""
    named: set[str] = set()
    for cte in re.finditer(r"(?:\bwith\s+(?:recursive\s+)?|,\s*)(" + _IDENT + r")\s*(?:\([^)]*\)\s*)?as\s*\(", statement, re.I):
        named.add(_name(cte.group(1)).lower())
    tokens = [(m.group(0), m.start()) for m in _TOKEN.finditer(statement)]
    found: list[tuple[str, int]] = []
    selected = [False]
    i = 0

    def table(index: int) -> tuple[str, Optional[tuple[str, int]]]:
        """("stop", None) at the end of a list, ("skip", None) for a name
        that is not a table but is followed by an alias like one (a WITH
        name), ("name", (name, offset)) for a table."""
        if index >= len(tokens):
            return "stop", None
        word, at = tokens[index]
        lowered = word.lower()
        if lowered in _CLAUSE_END or word in "(),;" or not _PART.match(word):
            return "stop", None
        if index + 1 < len(tokens) and tokens[index + 1][0] == "(":
            return "stop", None
        name = _name(word)
        if not name or name.lower() in named:
            return "skip", None
        return "name", (name, at)

    while i < len(tokens):
        word, at = tokens[i]
        lowered = word.lower()
        if word == "(":
            selected.append(False)
        elif word == ")":
            if len(selected) > 1:
                selected.pop()
        elif lowered == "select":
            selected[-1] = True
        elif lowered == "join":
            kind, source = table(i + 1)
            if kind == "name" and source is not None:
                found.append(source)
        elif lowered == "from" and selected[-1]:
            j = i + 1
            while True:
                kind, source = table(j)
                if kind == "stop":
                    break
                if kind == "name" and source is not None:
                    found.append(source)
                j += 1
                # an alias, with or without AS, then a comma or the end of the list
                if j < len(tokens) and tokens[j][0].lower() == "as":
                    j += 1
                if j < len(tokens) and tokens[j][0] != "," and _PART.match(tokens[j][0]) and tokens[j][0].lower() not in _CLAUSE_END:
                    j += 1
                if j < len(tokens) and tokens[j][0] == ",":
                    j += 1
                    continue
                break
        i += 1
    return found


def _line_at(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def _body(content: str, start: int) -> tuple[str, int]:
    """The parenthesised body beginning after ``start`` (the opening
    paren), and where it ends; empty when the parens never close."""
    depth = 0
    for i in range(start, len(content)):
        if content[i] == "(":
            depth += 1
        elif content[i] == ")":
            depth -= 1
            if depth == 0:
                return content[start + 1:i], i
    return "", len(content)


def _items(body: str) -> list[tuple[int, str]]:
    """The comma-separated items of a table body at depth zero, each with
    its offset into the body."""
    items: list[tuple[int, str]] = []
    depth = 0
    at = 0
    for i, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            items.append((at, body[at:i]))
            at = i + 1
    items.append((at, body[at:]))
    return [(offset, text) for offset, text in items if text.strip()]


def schema_claims(content: str, path: str) -> tuple[CodeClaim, ...]:
    """What a schema file defines and what rests on what, as claims about
    it; empty for a file that is not a schema or longer than a source
    file is read."""
    if not content or not is_schema(path) or content.count("\n") > MAX_LINES:
        return ()
    text = _Claims(content, path)
    plain = _blank(content)
    for found in _CREATE_TABLE.finditer(plain):
        table = _name(found.group(1))
        text.say(path, DEFINES, f"{path}:{table}", _line_at(plain, found.start()))
        body, _ = _body(plain, found.end() - 1)
        base = found.end()
        for offset, item in _items(body):
            line = _line_at(plain, base + offset + (len(item) - len(item.lstrip())))
            if _CONSTRAINT.match(item):
                referenced = _REFERENCES.search(item)
                if referenced:
                    text.say(f"{path}:{table}", DEPENDS_ON, f"{path}:{_name(referenced.group(1))}", line)
                continue
            column = re.match(r"\s*(" + _IDENT + r")", item)
            if not column:
                continue
            text.say(f"{path}:{table}", DEFINES, f"{path}:{table}.{_name(column.group(1))}", line)
            referenced = _REFERENCES.search(item)
            if referenced:
                text.say(f"{path}:{table}", DEPENDS_ON, f"{path}:{_name(referenced.group(1))}", line)
    for found in _CREATE_VIEW.finditer(plain):
        view = _name(found.group(1))
        text.say(path, DEFINES, f"{path}:{view}", _line_at(plain, found.start()))
        statement = plain[found.end():_statement_end(plain, found.end())]
        seen: set[str] = set()
        for name, at in _sources(statement):
            if name.lower() in seen or name == view:
                continue
            seen.add(name.lower())
            text.say(f"{path}:{view}", DEPENDS_ON, f"{path}:{name}", _line_at(plain, found.end() + at))
    for found in _ALTER_FK.finditer(plain):
        table = _name(found.group(1))
        statement = plain[found.end():_statement_end(plain, found.end())]
        for referenced in _REFERENCES.finditer(statement):
            text.say(f"{path}:{table}", DEPENDS_ON, f"{path}:{_name(referenced.group(1))}",
                     _line_at(plain, found.end() + referenced.start()))
    return tuple(text.found)


def schema_or_none(content: str, path: str) -> Optional[tuple[CodeClaim, ...]]:
    """The claims when the path is a schema, else None, for a dispatcher."""
    return schema_claims(content, path) if is_schema(path) else None
