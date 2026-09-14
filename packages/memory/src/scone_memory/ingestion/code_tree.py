"""Declarations in languages the line readers do not know, read from the syntax tree.

The Python reader walks Python's own syntax tree and the brace reader
finds the headers a brace family shares. Ruby, PHP, Lua, Swift, Scala,
shell and their kin are neither, and until now a file in one of them was
prose that happened to contain code: no declaration names on its chunks,
no cuts at its definitions. A tree-sitter grammar gives each of them an
exact tree, and the grammars agree on one convention this reads: a
definition node carries its name in a field called ``name``. That is
all this depends on -- no per-language node lists to keep, no guessing
from text. A node with no ``name`` field is not a declaration here, and
a language whose grammar names things another way is not read yet and
is left as prose, as before.

The grammars come from an optional package; without it nothing here
runs and every file keeps its old reading. The reader is bounded like
the others: files over ``MAX_LINES`` are not parsed, and declarations
deeper than ``MAX_DEPTH`` are not named.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

#: Source suffixes read here, each with its grammar's name in the pack.
#: The Python and brace families keep their own readers (PHP, Swift, Scala
#: and Kotlin are brace-family files), so nothing here changes how an
#: existing space is cut; only files that were prose gain declarations.
TREE_SUFFIXES: dict[str, str] = {
    ".rb": "ruby", ".rake": "ruby",
    ".lua": "lua",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".pl": "perl", ".pm": "perl",
    ".fish": "fish",
}
#: Node types that hold a definition of what they name; a grammar's
#: ``_definition``, ``_declaration`` and ``_item`` nodes, and the few
#: bare words some grammars use (Ruby's ``class``, ``module``, ``method``).
_DEFINITION_TYPES = frozenset({"class", "module", "method", "singleton_method", "function", "object", "trait",
                               "struct", "enum", "interface", "protocol", "extension", "impl",
                               "package_statement", "subroutine_declaration_statement"})
_DEFINITION_ENDINGS = ("_definition", "_declaration", "_item", "_specification")
MAX_LINES = 200_000
MAX_DEPTH = 32
MAX_PARSED = 8


def grammar_for(suffix: str) -> Optional[str]:
    """The grammar a suffix names here, or None."""
    return TREE_SUFFIXES.get(suffix)


@lru_cache(maxsize=1)
def available() -> bool:
    """Whether the grammar pack is importable; checked once."""
    try:
        import tree_sitter_language_pack  # noqa: F401
    except ImportError:
        return False
    return True


def _kind(node_type: str) -> str:
    for ending in _DEFINITION_ENDINGS:
        if node_type.endswith(ending):
            return node_type[:-len(ending)] or node_type
    return node_type


def _is_definition(node_type: str) -> bool:
    return node_type in _DEFINITION_TYPES or node_type.endswith(_DEFINITION_ENDINGS)


@lru_cache(maxsize=MAX_PARSED)
def tree_declarations(content: str, grammar: str) -> tuple[Any, ...]:
    """Every named definition in source order, qualified by what holds it.

    Returns ``Declaration`` rows from ``ingestion.code``; typed loosely to
    keep the import one-directional."""
    from .code import Declaration

    if not available() or content.count("\n") > MAX_LINES:
        return ()
    from tree_sitter_language_pack import get_parser

    if grammar not in TREE_SUFFIXES.values():
        return ()  # only bundled grammars this reader names; the pack would otherwise look elsewhere
    try:
        parser = get_parser(grammar)  # type: ignore[arg-type]
    except Exception:  # a grammar the pack cannot load is a file that stays prose
        return ()
    raw = content.encode("utf-8")
    tree = parser.parse(raw)
    found: list[Declaration] = []

    def walk(node: Any, holders: tuple[str, ...], depth: int) -> None:
        if depth > MAX_DEPTH:
            return
        named = holders
        if _is_definition(node.type):
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = raw[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace").strip()
                if name:
                    named = (*holders, name)
                    found.append(Declaration(".".join(named), _kind(node.type), node.start_byte, node.end_byte,
                                             node.start_point[0] + 1, node.end_point[0] + 1))
        for child in node.named_children:
            walk(child, named, depth + 1)

    walk(tree.root_node, (), 0)
    return tuple(found)
