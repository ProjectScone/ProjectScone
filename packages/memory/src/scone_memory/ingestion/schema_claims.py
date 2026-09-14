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
_FROM = re.compile(r"\bfrom\s+(" + _IDENT + r"(?:\s*,\s*" + _IDENT + r")*)|\bjoin\s+(" + _IDENT + r")", re.I)
_CONSTRAINT = re.compile(r"^\s*(?:constraint\b|primary\s+key\b|foreign\s+key\b|unique\b|check\b|index\b|key\b|exclude\b|like\b)", re.I)


def is_schema(path: str) -> bool:
    """Known by its suffix: ``.sql`` or ``.ddl``, wherever it sits."""
    return bool(path) and path.replace("\\", "/").lower().rsplit("/", 1)[-1].endswith(SCHEMA_SUFFIXES)


def _name(raw: str) -> str:
    """A name as the graph spells it: quotes and brackets gone, case kept."""
    return ".".join(next(part for part in found.groups() if part is not None) for found in _PART.finditer(raw))


def _blank_comments(content: str) -> str:
    """The text with every comment turned to spaces, so lines and offsets
    stay where they were and a table named in a comment is not a table."""
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
        elif content[i] in "'\"`":
            quote = content[i]
            end = content.find(quote, i + 1)
            end = n if end < 0 else end + 1
            out.append(content[i:end])
            i = end
        else:
            out.append(content[i])
            i += 1
    return "".join(out)


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
    plain = _blank_comments(content)
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
        line = _line_at(plain, found.start())
        text.say(path, DEFINES, f"{path}:{view}", line)
        end = plain.find(";", found.end())
        statement = plain[found.end():end if end >= 0 else len(plain)]
        seen: set[str] = set()
        for source in _FROM.finditer(statement):
            listed = source.group(1) or source.group(2) or ""
            for raw in re.split(r"\s*,\s*", listed):
                name = _name(raw)
                if not name or name.lower() in seen or name == view:
                    continue
                seen.add(name.lower())
                text.say(f"{path}:{view}", DEPENDS_ON, f"{path}:{name}", _line_at(plain, found.end() + source.start()))
    for found in _ALTER_FK.finditer(plain):
        table = _name(found.group(1))
        end = plain.find(";", found.end())
        statement = plain[found.end():end if end >= 0 else len(plain)]
        for referenced in _REFERENCES.finditer(statement):
            text.say(f"{path}:{table}", DEPENDS_ON, f"{path}:{_name(referenced.group(1))}",
                     _line_at(plain, found.end() + referenced.start()))
    return tuple(text.found)


def schema_or_none(content: str, path: str) -> Optional[tuple[CodeClaim, ...]]:
    """The claims when the path is a schema, else None, for a dispatcher."""
    return schema_claims(content, path) if is_schema(path) else None
