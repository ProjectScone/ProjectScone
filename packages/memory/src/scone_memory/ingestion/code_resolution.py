"""Binding a call across files, when exactly one declaration can answer.

A file on its own cannot say what ``thing.method()`` refers to: the type
of ``thing`` is stated nowhere in it. `code_graph` therefore refuses the
call and `unresolved_calls` reports it, which is honest but final --
measured over this package, 59.3% of the distinct call names the graph
cannot place are that shape, and no rule read off one file's syntax
reaches them.

A **corpus** answers a useful share of it without any type information at
all. If exactly one declaration in everything that was read is named
``method``, that is what the call meant. If two are, nothing is. That
single-definition rule is the entire safety property, and it is the bar
the leading code-graph tools hold themselves to as well: an ambiguous
name must fabricate nothing.

Measured over this package before it was built: of 831 distinct unplaced
names, **31.9% have exactly one declaration**, 32.3% have several and are
refused, and 35.9% have none because they belong to somebody else's
library. Those three are counted separately, because a reader who sees
"refused as ambiguous" added to "not ours" learns nothing from either.

**These are candidates, and this module's output is never written to the
ledger.** That was the intention, and sampling the result is what changed
it. Of twelve inferred edges drawn at random from this package, five
bound `.items()` -- the dictionary method -- to a function of ours that
happened to be the only one named `items`, and one bound a call to a
function nested inside another function, which no other file can reach.

Two computed guards remove that class: a name that is an attribute of any
builtin type cannot be told from that type's own method on an unknown
receiver, and a declaration nested more than one level deep is not
reachable from outside. With both, a fresh sample still contained
attribute reads bound as calls (`.kwargs` to a `RecallScope.kwargs`),
because the receiver's type is unknown *by construction* -- that is the
premise of the whole problem, not an oversight in the rule.

So the count is reported and the edges are not asserted. An edge nobody
can check is worse than no edge, and "roughly four in five are right" is
not a standard this graph should adopt when every other edge in it was
read from a line. The safe variant -- resolving `self.method()` through
an inheritance edge we actually read -- was measured at **2 call sites**
in this package, because the code favours composition, so it does not pay
for itself either.

What remains genuinely useful is the **shape** of the blind spot: how
much of it one more careful read of this corpus could even help with,
against how much belongs to somebody else entirely.

This is deliberately a pure function over what a caller has already
gathered: it performs no I/O, so the same rule can be driven from a
directory read, an incremental ingest, or a test, and it can be reasoned
about without a store.
"""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING, Iterable, Optional

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


#: The spellings a module stem is tried with, the older languages first so
#: a stem two languages share (`store.py` beside `store.ts`) keeps the
#: answer it had; `index.*` and Rust's `mod.rs` are tried for a directory.
_SUFFIXES = ("py", "ts", "tsx", "js", "jsx", "go", "rs", "mts", "cts", "mjs", "cjs", "java", "kt", "kts", "scala",
             "cs", "swift", "dart", "zig", "php", "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "m", "mm")


@dataclass(frozen=True)
class Resolution:
    """What a corpus could and could not settle.

    ``ambiguous`` and ``unknown`` are kept apart on purpose: the first is
    a name this corpus knows too many of, the second one it has never
    heard of. Only the first would be helped by reading more carefully;
    only the second by reading more.
    """

    edges: tuple[tuple[str, str], ...] = ()
    ambiguous: int = 0
    unknown: int = 0


#: A declaration nested deeper than `Class.method` is inside a function,
#: and nothing outside that function can call it. Nested classes are named
#: the same way and are rarer, so this errs towards fewer candidates.
REACHABLE_DEPTH = 1

#: Names that are attributes of a builtin type. `d.items()` on an unknown
#: receiver cannot be told from a call to a function of ours named
#: `items`, and in a random sample that single collision accounted for
#: five of twelve candidates. Computed rather than listed by hand, so it
#: does not drift from the language.
_BUILTIN_ATTRIBUTES = frozenset().union(*(
    dir(kind) for kind in
    (dict, list, str, set, tuple, bytes, bytearray, int, float, frozenset, object)))


def unmistakable(simple: str) -> bool:
    """Whether a declaration's name could only mean ours."""
    return simple not in _BUILTIN_ATTRIBUTES


def resolve_across_files(
    sites: Iterable[tuple[str, str]],
    declarations: Mapping[str, Sequence[str]],
) -> Resolution:
    """Bind each ``(caller, spelling)`` the graph could not place.

    ``declarations`` maps a declaration's **simple** name to every full
    name the corpus declares it under. Only the last segment of a
    spelling names the declaration -- in ``thing.rank`` the ``thing`` is
    a value, not a place to look.
    """
    found: dict[tuple[str, str], None] = {}
    ambiguous = unknown = 0
    for caller, spelling in sites:
        simple = spelling.rsplit(".", 1)[-1]
        candidates = declarations.get(simple) or ()
        if not candidates:
            unknown += 1
        elif len(candidates) > 1:
            ambiguous += 1
        elif candidates[0] != caller:
            # An edge from a thing to itself says nothing, and the line
            # reader refuses one for the same reason.
            found[(caller, candidates[0])] = None
    return Resolution(edges=tuple(sorted(found)), ambiguous=ambiguous, unknown=unknown)


def file_resolver(paths: Iterable[str], published: Optional[Mapping[str, str]] = None,
                  known: Iterable[str] = ()) -> "_FileResolver":
    """How a relative import is followed: only to a file the walk actually
    read, and never guessed at otherwise.

    A relative import names a file however the language spells it --
    Python by module and level, the brace family by a path with the
    extension left off, either of them possibly a directory's index -- so
    the candidates are tried in that order against ``paths``, the files a
    walk saw, and a name that leads anywhere else resolves to None. One
    function of the paths, shared by ``map``, ``sync`` and any batch of
    files remembered together, so the three cannot come to differ.
    """
    seen = {path.replace("\\", "/") for path in paths if isinstance(path, str)}
    return _FileResolver(seen, dict(published or {}), {path.replace("\\", "/") for path in known})


class _FileResolver:
    """One set of files, followed two ways: an import by module and level,
    a document's link by path or title (`doc_graph.link_target`)."""

    def __init__(self, seen: set[str], published: Optional[dict[str, str]] = None,
                 known: Optional[set[str]] = None) -> None:
        #: What the manifests in the space publish: a package's name, as an
        #: import spells it, to the directory that publishes it; and the
        #: files other repositories in the space mapped, which a package
        #: import may reach.
        self.published = published or {}
        self.known = known or set()
        self.seen = seen

    def links(self, path: str, target: str) -> list[str]:
        from .doc_graph import link_targets

        return link_targets(self.seen, path, target)

    def link(self, path: str, target: str) -> Optional[str]:
        from .doc_graph import link_target

        return link_target(self.seen, path, target)

    def __call__(self, path: str, level: int, module: str) -> Optional[str]:
        seen = self.seen
        here = posixpath.dirname(path)
        for _ in range(level - 1):
            here = posixpath.dirname(here)
        stem = posixpath.join(here, *module.split(".")) if module else here
        stems = [stem, posixpath.normpath(posixpath.join(posixpath.dirname(path), module))
                 if module.startswith(".") else stem]
        # An import that names a file outright (`./lib/x.h`, `./page.css`)
        # is that file when the walk saw it.
        if module.startswith(".") and stems[1] in seen:
            return stems[1]
        for base in dict.fromkeys(stems):
            for suffix in _SUFFIXES:
                if f"{base}.{suffix}" in seen:
                    return f"{base}.{suffix}"
                if f"{base}/index.{suffix}" in seen:
                    return f"{base}/index.{suffix}"
            if f"{base}/mod.rs" in seen:
                return f"{base}/mod.rs"
        for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
            if candidate in seen:
                return candidate
        return None


    def package(self, module: str, language: str) -> Optional[str]:
        """The file a module of a published package is (see ``package_item``)."""
        found, _ = self.package_item(module, language)
        return found

    def package_item(self, module: str, language: str) -> tuple[Optional[str], list[str]]:
        """The file a module of a published package is, and what of the
        module path was left over -- the item the import names in that file
        (`crate_name::store::Shelf` is `src/store.rs` and `Shelf`) -- when a
        manifest in the space publishes the package and the file was read
        here or in another repository; None otherwise. ``language`` says how a module
        maps to a path: `python` (`pkg.a.b` under `src/pkg`, `pkg` or
        `lib/pkg`), `js` (`@scope/name/sub` under the package's directory,
        `src` or `lib`, an index for the package itself), `rust`
        (`crate_name::a::b` under `src`, `mod.rs` for a directory, `lib.rs`
        for the crate), `go` (an import path below the module's path, a
        directory of Go files)."""
        if not self.published or not module:
            return None, []
        files = self.seen | self.known

        def under(base: str, rest: str) -> str:
            # A package published at the mapped root has no directory in front.
            return f"{base}/{rest}" if base and rest else base or rest

        if language == "python":
            parts = module.split(".")
            root = self.published.get(parts[0])
            if root is None:
                return None, []
            tail = "/".join(parts)
            for base in (under(root, "src"), root, under(root, "lib")):
                for candidate in (under(base, f"{tail}.py"), under(base, f"{tail}/__init__.py")):
                    if candidate in files:
                        return candidate, []
            return None, []
        if language == "js":
            segments = module.split("/")
            name = "/".join(segments[:2]) if module.startswith("@") else segments[0]
            root = self.published.get(name)
            if root is None:
                return None, []
            rest = "/".join(segments[2:] if module.startswith("@") else segments[1:])
            for base in (root, under(root, "src"), under(root, "lib"), under(root, "dist")):
                stems = [under(base, rest), under(base, f"{rest}/index")] if rest else [under(base, "index")]
                for stem in stems:
                    if rest and stem == under(base, rest) and stem in files:
                        return stem, []
                    for suffix in ("ts", "tsx", "mts", "cts", "js", "jsx", "mjs", "cjs"):
                        if f"{stem}.{suffix}" in files:
                            return f"{stem}.{suffix}", []
            return None, []
        if language == "rust":
            parts = module.split("::")
            # A crate at the mapped root publishes from "", which is a root too.
            root = self.published.get(parts[0].replace("_", "-"))
            if root is None:
                root = self.published.get(parts[0])
            if root is None:
                return None, []
            base = under(root, "src")
            for length in range(len(parts) - 1, 0, -1):
                stem = under(base, "/".join(parts[1:length + 1]))
                for candidate in (f"{stem}.rs", f"{stem}/mod.rs"):
                    if candidate in files:
                        # What follows the module is an item in it: `crate_name::a::b::Thing`.
                        return candidate, parts[length + 1:]
            for candidate in (under(base, "lib.rs"), under(base, "main.rs")):
                if candidate in files:
                    return candidate, parts[1:]
            return None, []
        if language == "go":
            for name, root in sorted(self.published.items(), key=lambda item: -len(item[0])):
                if module == name or module.startswith(name + "/"):
                    below = module[len(name):].strip("/")
                    directory = under(root, below)
                    prefix = directory + "/" if directory else ""
                    if any(path.startswith(prefix) and path.endswith(".go") and "/" not in path[len(prefix):]
                           for path in files):
                        return directory, []
                    return None, []
            return None, []
        return None, []


if TYPE_CHECKING:
    from .code_graph import Resolve
