"""What a Ruby, Lua, shell, Perl or fish file says, read from its syntax tree.

The grammars in ``code_tree`` cut these files at their definitions; until
now they said nothing to the graph, so a Ruby service was a set of chunks
with no ``defines`` and no ``imports`` -- invisible to ``graph affected``
and to every question about what depends on what. This reads the same
tree for claims, under the same rule as the other readers: a claim is
quoted from the line it was read on, and the reader claims only what the
tree settles.

- ``defines``: every named definition, qualified by what holds it, as
  ``code_tree.tree_declarations`` already names them for chunking. The
  file defines its top-level things; a module or class defines what it
  holds.
- ``imports``: what the file loads, as the language spells it -- Ruby's
  ``require``/``require_relative``/``load`` with a literal string, Lua's
  ``require("x")``, shell's ``source x`` and ``. x``, Perl's ``use`` and
  ``require`` (pragmas, the lowercase ones, are not modules and are not
  claimed), fish's ``source x``. A load whose target is not a literal
  (``require name``, ``source "$HOME/x.sh"``) is not a claim: the tree
  cannot say what it names. A load inside a function body runs when the
  function is called, not when the file loads, and is not claimed either;
  nor is one nested more than ``_LOAD_DEPTH`` levels under a top-level
  statement.
- ``notes``, ``flags``, ``cites``: the same tagged comments the other
  readers take (``WHY:``, ``NOTE:``, ``TODO:``, ``ADR-12``), from the
  tree's comment nodes, attached to the declaration they sit in.

Calls are not claimed. Binding a call needs the type of a receiver or the
resolution of a bare name across files, which none of these grammars
supplies; the brace reader's history (six shapes of false edge) says what
guessing costs. Without the grammar pack nothing here runs, and a file
keeps the reading it had.
"""
from __future__ import annotations

import re
from typing import Any, Iterator, Optional

from .code import MAX_LINES, _line_starts
from .code_graph import CITES, DEFINES, FLAGS, IMPORTS, MAX_CLAIMS, NOTES, CodeClaim, _cited, _holder
from .code_tree import available, tree_declarations

#: A tagged comment: the tag, then what it says; comment markers stripped
#: first. A note needs its colon and a flag needs a word boundary, as the
#: line readers have them, so `Notes live here` is prose and not a note.
_TAGGED = re.compile(r"^(?:(WHY|NOTE|RATIONALE)\s*:|(TODO|FIXME|HACK|XXX)\b\s*:?)\s*(.+?)\s*$", re.IGNORECASE)
_MARKERS = re.compile(r"^[\s#/\-*\[=]+|[\s*/\-\]]+$")
#: Ruby methods that load a file by a literal name.
_RUBY_LOADS = frozenset({"require", "require_relative", "load"})
#: Shell words that read another file into the current one.
_SHELL_SOURCES = frozenset({"source", "."})
#: How far under a top-level statement a load is looked for.
_LOAD_DEPTH = 5


def _text(node: Any, raw: bytes) -> str:
    return raw[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _literal(node: Any, raw: bytes) -> Optional[str]:
    """The content of a string literal node, or None when it is not one
    or holds interpolation."""
    if node is None or node.type not in {"string", "string_literal"}:
        return None
    text = _text(node, raw).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        inner = text[1:-1]
        return None if ("#{" in inner or not inner) else inner
    return None


def _named_children(node: Any) -> list[Any]:
    return list(node.named_children)


def _first_named(node: Any, kind: str) -> Optional[Any]:
    return next((child for child in node.named_children if child.type == kind), None)


def _loads(grammar: str, root: Any, raw: bytes) -> Iterator[tuple[str, int]]:
    """What the file loads, with the line, per grammar, from top-level
    statements and what sits up to ``_LOAD_DEPTH`` levels under them,
    function bodies left out."""
    def statements() -> Iterator[Any]:
        # A load sits in a top-level statement or a few levels under it
        # (Lua's `local x = require("x")` is four deep); nothing inside a
        # function body is a load of the file's own.
        stack: list[tuple[Any, int]] = [(top, 0) for top in reversed(root.named_children)]
        while stack:
            node, depth = stack.pop()
            yield node
            if depth < _LOAD_DEPTH and not node.type.endswith(("function_definition", "function_declaration", "method", "subroutine_declaration_statement")):
                stack.extend((child, depth + 1) for child in reversed(node.named_children))
    for node in statements():
        if grammar == "ruby" and node.type == "call":
            method = node.child_by_field_name("method")
            arguments = node.child_by_field_name("arguments")
            if method is not None and _text(method, raw) in _RUBY_LOADS and arguments is not None:
                target = _literal(next(iter(arguments.named_children), None), raw)
                if target:
                    yield target, node.start_point[0] + 1
        elif grammar == "lua" and node.type == "function_call":
            callee = node.child_by_field_name("name")
            arguments = node.child_by_field_name("arguments")
            if callee is not None and _text(callee, raw) == "require" and arguments is not None:
                target = _literal(next(iter(arguments.named_children), None), raw)
                if target:
                    yield target, node.start_point[0] + 1
        elif grammar in {"bash", "fish"} and node.type == "command":
            name = node.child_by_field_name("name")
            argument = node.child_by_field_name("argument")
            if name is not None and _text(name, raw) in _SHELL_SOURCES and argument is not None:
                target = _text(argument, raw).strip().strip("\"'")
                # A path with an expansion in it (`"$HOME/x.sh"`, `$lib`) is
                # not a literal the tree can name.
                if target and "$" not in target and "`" not in target:
                    yield target, node.start_point[0] + 1
        elif grammar == "perl" and node.type == "use_statement":
            package = _first_named(node, "package")
            if package is not None:
                name = _text(package, raw)
                if name and not name.islower():  # `use strict`, `use warnings`: pragmas, not modules
                    yield name, node.start_point[0] + 1
        elif grammar == "perl" and node.type == "require_expression":
            word = _first_named(node, "bareword")
            if word is not None:
                yield _text(word, raw), node.start_point[0] + 1


def _comments(root: Any, raw: bytes) -> Iterator[tuple[str, int]]:
    """Every comment line with its own line number: a block comment
    (Lua's ``--[[ ]]``, Ruby's ``=begin``) is one node over several lines,
    and a claim quoted from its first line would not show what it cites."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "comment":
            for offset, text in enumerate(_text(node, raw).splitlines()):
                yield text, node.start_point[0] + 1 + offset
            continue
        stack.extend(reversed(node.named_children))


def tree_claims(content: str, path: str, *, grammar: str) -> tuple[CodeClaim, ...]:
    """The claims a file in a grammar-read language makes: what it defines,
    what it loads, and the tagged comments it carries. Empty without the
    grammar pack, for a file longer than a source file is read, or for a
    grammar the pack cannot load."""
    if not available() or not content or content.count("\n") > MAX_LINES:
        return ()
    declared = tree_declarations(content, grammar)
    from tree_sitter_language_pack import get_parser

    try:
        parser = get_parser(grammar)  # type: ignore[arg-type]
    except Exception:
        return ()
    raw = content.encode("utf-8")
    root = parser.parse(raw).root_node
    lines = content.splitlines()
    starts = _line_starts(content)
    # A claim's span is in bytes of the UTF-8 source, as every reader
    # records it; the line starts are offsets into the text.
    byte_starts, at = [], 0
    for index, begins in enumerate(starts):
        byte_starts.append(at)
        ends = starts[index + 1] if index + 1 < len(starts) else len(content)
        at += len(content[begins:ends].encode("utf-8"))
    found: list[CodeClaim] = []

    def say(subject: str, predicate: str, obj: str, line: int) -> None:
        if len(found) >= MAX_CLAIMS or not 1 <= line <= len(lines):
            return
        begins = byte_starts[line - 1]
        text = lines[line - 1]
        found.append(CodeClaim(subject, predicate, obj, text.strip(), line, begins, begins + len(text.encode("utf-8"))))

    declared_names = {item.name for item in declared}
    for item in declared:
        # The holder is what encloses the definition (`Shop.Cart` holds
        # `Shop.Cart.add`) when that is itself a declaration of the file; a
        # dotted name whose prefix is a table or variable (`M.add` under
        # `local M = {}`) is held by the file, so nothing names an entity
        # nothing defines.
        holder, _, _ = item.name.rpartition(".")
        say(f"{path}:{holder}" if holder in declared_names else path, DEFINES, f"{path}:{item.name}", item.first_line)
    for target, line in _loads(grammar, root, raw):
        say(path, IMPORTS, target, line)
    for comment, line in _comments(root, raw):
        body = _MARKERS.sub("", comment)
        tagged = _TAGGED.match(body)
        if tagged:
            note, flag, said = tagged.groups()
            say(_holder(declared, path, line), NOTES if note else FLAGS, said, line)
        for document in _cited(body):
            say(_holder(declared, path, line), CITES, document, line)
    return tuple(found)
