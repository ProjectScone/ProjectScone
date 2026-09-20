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
    ".ex": "elixir", ".exs": "elixir",
    ".proto": "proto",
    ".sol": "solidity",
    ".tf": "hcl", ".tfvars": "hcl", ".hcl": "hcl",
    ".erl": "erlang", ".hrl": "erlang",
    ".ps1": "powershell", ".psm1": "powershell",
    ".jl": "julia",
    ".r": "r", ".R": "r",
    ".hs": "haskell",
    ".clj": "clojure", ".cljs": "clojure", ".cljc": "clojure", ".edn": "clojure",
}
#: The words that open a definition in Clojure, where a definition is a
#: list like any other and the first symbol tells them apart.
_CLOJURE_DEFINES: dict[str, str] = {
    "ns": "namespace", "def": "value", "defn": "function", "defn-": "function", "defmacro": "macro",
    "defmulti": "function", "defmethod": "function", "defprotocol": "protocol", "defrecord": "record",
    "deftype": "type", "definterface": "interface", "defonce": "value",
}
#: Haskell declaration nodes, and what each declares. A `signature` is
#: left out: it says what a function's type is, not that it exists, and
#: the function's own equation declares it.
_HASKELL_DEFINES: dict[str, str] = {"data_type": "data", "newtype": "newtype", "type_synonym": "type",
                                    "class": "class", "function": "function", "bind": "value"}
#: The words that open a definition in Elixir, and what each defines.
#: Elixir has no definition node: ``defmodule`` and ``def`` are calls
#: like ``IO.puts``, and only the word tells them apart.
_ELIXIR_DEFINES: dict[str, str] = {"defmodule": "module", "def": "function", "defp": "function",
                                   "defmacro": "macro", "defmacrop": "macro"}
#: Where Protobuf keeps a definition's name: in a child node named for
#: the kind, not in a ``name`` field.
_PROTO_NAMES: dict[str, str] = {"message": "message_name", "enum": "enum_name",
                                "service": "service_name", "rpc": "rpc_name"}
#: A name Elixir builds at compile time is no name this reader can read.
_UNREADABLE = frozenset({"unquote", "unquote_splicing"})
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


def _text_of(node: Any, raw: bytes) -> str:
    return raw[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()


def _elixir_declared(node: Any, raw: bytes) -> Optional[tuple[str, str]]:
    """What an Elixir call defines, if it defines anything.

    ``defmodule Billing.Invoice do`` and ``def total(items) do`` are both
    ``call`` nodes: the word is the call's own identifier, and the name
    is the first argument -- an alias for a module, and for a function
    either a call (``total(items)``) or a bare identifier (``rate``).
    """
    if node.type != "call" or not node.named_children:
        return None
    word = node.named_children[0]
    kind = _ELIXIR_DEFINES.get(_text_of(word, raw)) if word.type == "identifier" else None
    if kind is None:
        return None
    arguments = next((child for child in node.named_children if child.type == "arguments"), None)
    first = next(iter(arguments.named_children), None) if arguments is not None else None
    if first is None:
        return None
    if kind == "module":
        return (_text_of(first, raw), kind) if first.type == "alias" else None
    if first.type == "identifier":
        return _text_of(first, raw), kind
    if first.type == "call" and first.named_children:
        named = first.named_children[0]
        name = _text_of(named, raw)
        # `def unquote(name)(args)` names something only the compiler knows.
        return (name, kind) if named.type == "identifier" and name not in _UNREADABLE else None
    return None


def _hcl_declared(node: Any, raw: bytes) -> Optional[tuple[str, str]]:
    """What a Terraform block declares: its labels, under its type.

    ``resource "aws_s3_bucket" "logs"`` is the bucket everything else
    writes ``aws_s3_bucket.logs`` to reach, so the labels joined by a dot
    are the name and the block's own word is the kind. A block with no
    labels -- ``terraform``, ``locals``, a nested ``lifecycle`` -- names
    nothing and declares nothing.
    """
    if node.type != "block" or not node.named_children:
        return None
    word = node.named_children[0]
    if word.type != "identifier":
        return None
    labels = [_quoted(child, raw) for child in node.named_children if child.type == "string_lit"]
    named = [label for label in labels if label]
    return (".".join(named), _text_of(word, raw)) if named else None


def _erlang_declared(node: Any, raw: bytes) -> Optional[tuple[str, str]]:
    """Erlang's module attribute names the module; a function declaration
    is named by the atom its first clause opens with."""
    if node.type == "module_attribute":
        atom = next((c for c in node.named_children if c.type == "atom"), None)
        return (_text_of(atom, raw), "module") if atom is not None else None
    if node.type == "fun_decl":
        clause = next((c for c in node.named_children if c.type == "function_clause"), None)
        atom = next((c for c in clause.named_children if c.type == "atom"), None) if clause is not None else None
        return (_text_of(atom, raw), "function") if atom is not None else None
    return None


def _first(node: Any, kind: str) -> Optional[Any]:
    return next((child for child in node.named_children if child.type == kind), None)


def _julia_declared(node: Any, raw: bytes) -> Optional[tuple[str, str]]:
    """Julia keeps a function's name inside its signature and a struct's
    inside its type head, and neither node has a name field. A short
    definition (``rate(x) = x * 0.2``) is an assignment whose left side
    is the call being defined."""
    if node.type == "module_definition":
        name = _first(node, "identifier")
        return (_text_of(name, raw), "module") if name is not None else None
    if node.type in {"function_definition", "macro_definition"}:
        signature = _first(node, "signature")
        called = _first(signature, "call_expression") if signature is not None else None
        name = _first(called, "identifier") if called is not None else None
        return (_text_of(name, raw), node.type[:-len("_definition")]) if name is not None else None
    if node.type == "struct_definition":
        head = _first(node, "type_head")
        name = _first(head, "identifier") if head is not None else None
        return (_text_of(name, raw), "struct") if name is not None else None
    if node.type == "assignment":
        called = next(iter(node.named_children), None)
        name = _first(called, "identifier") if called is not None and called.type == "call_expression" else None
        return (_text_of(name, raw), "function") if name is not None else None
    return None


def _r_declared(node: Any, raw: bytes) -> Optional[tuple[str, str]]:
    """R names a function by what it is assigned to.

    Its own ``function_definition`` node carries the keyword ``function``
    in the name field, so the generic rule would declare every function
    in a file as ``function``. The name is the identifier on the other
    side of the arrow, and a binding whose right side is not a function
    is a value, not a declaration.
    """
    if node.type != "binary_operator" or len(node.named_children) < 2:
        return None
    operator = node.child_by_field_name("operator")
    if operator is None or _text_of(operator, raw) not in {"<-", "<<-", "="}:
        return None
    left, right = node.named_children[0], node.named_children[1]
    if left.type != "identifier" or right.type != "function_definition":
        return None
    return _text_of(left, raw), "function"


def dotted_module(node: Any, raw: bytes) -> str:
    """A Haskell module name, whose parts the tree keeps one by one:
    ``Data.Map`` arrives as ``Data`` and ``Map``."""
    parts = [_text_of(part, raw) for part in node.named_children if part.type == "module_id"]
    return ".".join(part for part in parts if part) or _text_of(node, raw)


def _haskell_declared(node: Any, raw: bytes) -> Optional[tuple[str, str]]:
    """Haskell's header declares the module; a data type and a class keep
    their name in a ``name`` child, and a function equation in the
    ``variable`` it opens with."""
    if node.type == "header":
        module = _first(node, "module")
        return (dotted_module(module, raw), "module") if module is not None else None
    kind = _HASKELL_DEFINES.get(node.type)
    if kind is None:
        return None
    named = _first(node, "name") or _first(node, "variable")
    return (_text_of(named, raw), kind) if named is not None else None


def _clojure_declared(node: Any, raw: bytes) -> Optional[tuple[str, str]]:
    """Clojure writes every definition as a list whose first symbol is
    the word that defines: ``(defn total [items] ...)``. The name is the
    symbol after it, and a list opening with anything else is a call."""
    if node.type != "list_lit" or len(node.named_children) < 2:
        return None
    word, named = node.named_children[0], node.named_children[1]
    if word.type != "sym_lit" or named.type != "sym_lit":
        return None
    kind = _CLOJURE_DEFINES.get(_text_of(word, raw))
    return (_text_of(named, raw), kind) if kind is not None else None


def _quoted(node: Any, raw: bytes) -> str:
    """The text inside a quoted literal, without its quotes."""
    return _text_of(node, raw).strip().strip('"\'')


def _declared(node: Any, raw: bytes, grammar: str) -> Optional[tuple[str, str]]:
    """The name and kind this node declares in this grammar, or None.

    Each grammar is read by a rule taken from its own tree. The generic
    rule -- a definition-shaped node type with a ``name`` field -- reads
    Ruby, Lua, Perl, shell, fish and Solidity. It does not read Elixir
    (no definition node) or Protobuf (the name is a child, not a field),
    and it misreads others badly enough that they are left out: R's
    ``function_definition`` has the keyword ``function`` in its name
    field, and Julia's functions have no name field at all.
    """
    if grammar == "elixir":
        return _elixir_declared(node, raw)
    if grammar == "hcl":
        return _hcl_declared(node, raw)
    if grammar == "erlang":
        return _erlang_declared(node, raw)
    if grammar == "julia":
        return _julia_declared(node, raw)
    if grammar == "r":
        return _r_declared(node, raw)
    if grammar == "haskell":
        return _haskell_declared(node, raw)
    if grammar == "clojure":
        return _clojure_declared(node, raw)
    if grammar == "powershell":
        if node.type != "function_statement":
            return None
        named = next((c for c in node.named_children if c.type == "function_name"), None)
        return (_text_of(named, raw), "function") if named is not None else None
    if grammar == "proto":
        child = _PROTO_NAMES.get(node.type)
        if child is None:
            return None
        named = next((c for c in node.named_children if c.type == child), None)
        return (_text_of(named, raw), node.type) if named is not None else None
    if not _is_definition(node.type):
        return None
    named = node.child_by_field_name("name")
    return (_text_of(named, raw), _kind(node.type)) if named is not None else None


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
        declared = _declared(node, raw, grammar)
        if declared is not None and declared[0]:
            name, kind = declared
            named = (*holders, name)
            found.append(Declaration(".".join(named), kind, node.start_byte, node.end_byte,
                                     node.start_point[0] + 1, node.end_point[0] + 1))
        for child in node.named_children:
            walk(child, named, depth + 1)

    walk(tree.root_node, (), 0)
    # One declaration per qualified name. A language that writes a
    # function as one clause per shape (Elixir) or one method per type
    # (Julia) has a definition node for each, and all of them declare
    # the same name: the first is where it is declared, and the rest
    # would be the same edge drawn again.
    once: dict[str, Any] = {}
    for declaration in found:
        once.setdefault(declaration.name, declaration)
    return tuple(once.values())
