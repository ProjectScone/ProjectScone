"""Calls and declarations read from a syntax tree, for the brace family.

The line reader in ``code_graph.py`` cannot bind a call and does not try:
binding one needs **lexical scope**, and a reader that sees masked lines
has none. Six shapes of false edge over three rounds of review said so,
and the brace call graph built on names was withdrawn.

This is the same capability with a parser under it. It claims only what a
syntax tree can settle, which is less than it sounds. Measured over 2,627
call sites in 60 files of this project's web application:

- **44% go through a receiver** (``x.y()``) and need the *type* of the
  thing on the left, which no grammar supplies. Left alone.
- **28% name something from outside the file** -- globals, JSX, values
  from a default import. Left alone.
- **15% name an import** and **5% name a declaration of the file**. These
  are what it records.
- **5% name a local or a parameter**, and are correctly *not* edges.
  That row is exactly what the line reader got wrong.

So a parser buys a fifth of the call sites. The honest claim is "sound
where it speaks", not "a call graph for TypeScript", and the module is
written to make the silence deliberate rather than accidental.

The parser is an optional extra (``scone-memory[code-graph]``). Without
it ``available()`` is False, nothing here runs, and the graph is exactly
what it was.
"""

from __future__ import annotations

from typing import Optional

from .code_graph import CALLS, DEFINES, CodeClaim, _named

try:  # pragma: no cover - exercised by whether the extra is installed
    from tree_sitter import Language, Node, Parser
    import tree_sitter_typescript as _typescript

    _TSX = Language(_typescript.language_tsx())
    _TS = Language(_typescript.language_typescript())
except Exception:  # pragma: no cover - the extra is simply absent
    _TSX = _TS = None  # type: ignore[assignment]

#: Extensions this reader knows a grammar for. A file outside it keeps the
#: line reader's answer rather than getting a worse one from the wrong
#: grammar.
GRAMMARS = {".ts": "ts", ".mts": "ts", ".cts": "ts", ".tsx": "tsx",
            ".js": "tsx", ".jsx": "tsx", ".mjs": "tsx", ".cjs": "tsx"}
#: Nodes that bind a name in the scope **around** them.
_BINDS = frozenset({"function_declaration", "generator_function_declaration",
                    "class_declaration", "abstract_class_declaration", "method_definition"})
#: Nodes that open a scope of their own.
_OPENS = _BINDS | frozenset({"arrow_function", "function_expression", "statement_block",
                             "program", "for_statement", "for_in_statement", "catch_clause",
                             "class_body"})
#: A declaration whose value is a function: `const keep = (p) => p`. Most
#: modern TypeScript writes functions this way and the line reader reports
#: none of them -- 80 in the 60 files measured above.
_FUNCTION_VALUES = frozenset({"arrow_function", "function_expression"})


def available() -> bool:
    """Whether the code-graph extra is installed and its grammar loaded."""
    return _TSX is not None


def _text(node: "Node", source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _bound(node: "Node", source: bytes) -> Optional[str]:
    """The name ``node`` binds in the scope around it, if any."""
    if node.type in _BINDS:
        name = node.child_by_field_name("name")
        return _text(name, source) if name is not None else None
    if node.type == "variable_declarator":
        name = node.child_by_field_name("name")
        return _text(name, source) if name is not None and name.type == "identifier" else None
    if node.type in ("required_parameter", "optional_parameter"):
        pattern = node.child_by_field_name("pattern")
        return _text(pattern, source) if pattern is not None and pattern.type == "identifier" else None
    if node.type == "identifier" and node.parent is not None \
            and node.parent.type == "formal_parameters":
        return _text(node, source)
    return None


def _line(node: "Node", source: bytes) -> int:
    return source[:node.start_byte].count(b"\n") + 1


def _imports(tree: "Node", source: bytes, path: str) -> dict[str, str]:
    """What each imported name is, as ``local name -> file:name``.

    Only a named import is followed. A default or namespace binding names
    a value whose members are reached through it, and a call through one
    of those is a call through a receiver.
    """
    found: dict[str, str] = {}
    for node in tree.children:
        if node.type != "import_statement":
            continue
        source_node = node.child_by_field_name("source")
        if source_node is None:
            continue
        written = _text(source_node, source).strip("\"'")
        # Relative, and nothing else. `motion/react-mini` contains a
        # slash and is not a path, so testing for one admitted every
        # scoped package subpath as a file of this graph.
        if not written.startswith("."):
            continue
        where = _named(written, path, None)
        if where is None:
            continue
        for clause in (one for one in node.children if one.type == "import_clause"):
            for named in (one for one in clause.children if one.type == "named_imports"):
                for item in (one for one in named.children if one.type == "import_specifier"):
                    original = item.child_by_field_name("name")
                    alias = item.child_by_field_name("alias") or original
                    if original is not None and alias is not None:
                        found[_text(alias, source)] = f"{where}:{_text(original, source)}"
    return found


def _declare(node: "Node", source: bytes, path: str, inside: Optional[str],
             say) -> Optional[str]:
    """Record what ``node`` declares and return its qualified name."""
    name = _bound(node, source)
    if name is None:
        return None
    if node.type == "variable_declarator":
        value = node.child_by_field_name("value")
        if value is None or value.type not in _FUNCTION_VALUES:
            return None
    elif node.type not in _BINDS:
        return None
    whole = f"{path}:{inside}.{name}" if inside else f"{path}:{name}"
    say(f"{path}:{inside}" if inside else path, DEFINES, whole, _line(node, source))
    return whole


def _walk(node: "Node", source: bytes, path: str, stack: list[dict[str, Optional[str]]],
          imports: dict[str, str], holder: Optional[str], inside: Optional[str], say) -> None:
    opened = node.type in _OPENS
    if opened:
        stack.append({})
    # Everything declared directly here is bound before anything in it is
    # read, so a function may call one declared below it -- which is what
    # these languages do.
    for child in node.children:
        name = _bound(child, source)
        if name is not None:
            stack[-1][name] = _declare(child, source, path, inside, say)
        for group in (one for one in child.children
                      if one.type in ("variable_declarator", "required_parameter",
                                      "optional_parameter", "identifier")):
            bound = _bound(group, source)
            if bound is not None:
                stack[-1][bound] = _declare(group, source, path, inside, say)

    if node.type == "call_expression" and holder is not None:
        callee = node.child_by_field_name("function")
        if callee is not None and callee.type == "identifier":
            name = _text(callee, source)
            # Innermost first: a parameter or local shadows a declaration
            # of the file, which shadows an import. A name found in any
            # scope but the outermost is a value, not a declaration, and
            # a call to it is not an edge this reader can name.
            target: Optional[str] = None
            for depth, scope in enumerate(reversed(stack)):
                if name in scope:
                    target = scope[name] if depth == len(stack) - 1 else None
                    break
            else:
                target = imports.get(name)
            if target is not None and target != holder:
                say(holder, CALLS, target, _line(node, source))

    for child in node.children:
        within = inside
        mine = holder
        if child.type in _BINDS or (child.type == "variable_declarator"
                                    and (value := child.child_by_field_name("value")) is not None
                                    and value.type in _FUNCTION_VALUES):
            name = _bound(child, source)
            if name is not None:
                mine = f"{path}:{inside}.{name}" if inside else f"{path}:{name}"
                if child.type in ("class_declaration", "abstract_class_declaration"):
                    within = f"{inside}.{name}" if inside else name
        _walk(child, source, path, stack, imports, mine, within, say)
    if opened:
        stack.pop()


def syntax_claims(content: str, path: str) -> tuple[CodeClaim, ...]:
    """Declarations and calls this file's syntax settles, or nothing.

    Returns an empty tuple when the extra is absent or the file's
    extension has no grammar here, so a caller can always ask.
    """
    suffix = path[path.rfind("."):].lower() if "." in path.rsplit("/", 1)[-1] else ""
    if not available() or suffix not in GRAMMARS:
        return ()
    source = content.encode()
    lines = content.split("\n")
    found: list[CodeClaim] = []

    def say(subject: str, predicate: str, obj: str, number: int) -> None:
        if not 1 <= number <= len(lines):
            return
        quote = lines[number - 1].strip()
        begins = len("\n".join(lines[:number - 1]).encode()) + (1 if number > 1 else 0)
        found.append(CodeClaim(subject, predicate, obj, quote, number, begins,
                               begins + len(lines[number - 1].encode())))

    tree = Parser(_TSX if GRAMMARS[suffix] == "tsx" else _TS).parse(source)
    _walk(tree.root_node, source, path, [], _imports(tree.root_node, source, path),
          None, None, say)
    return tuple(found)
