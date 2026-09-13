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


def file_resolver(paths: Iterable[str]) -> "Resolve":
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
    seen = {path.replace("\\", "/") for path in paths}

    def resolve(path: str, level: int, module: str) -> Optional[str]:
        here = posixpath.dirname(path)
        for _ in range(level - 1):
            if not here:
                return None
            here = posixpath.dirname(here)
        stem = posixpath.join(here, *module.split(".")) if module else here
        stems = [stem, posixpath.normpath(posixpath.join(posixpath.dirname(path), module))
                 if module.startswith(".") else stem]
        for base in dict.fromkeys(stems):
            for suffix in ("py", "ts", "tsx", "js", "jsx", "go", "rs"):
                if f"{base}.{suffix}" in seen:
                    return f"{base}.{suffix}"
                if f"{base}/index.{suffix}" in seen:
                    return f"{base}/index.{suffix}"
        for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
            if candidate in seen:
                return candidate
        return None

    return resolve


if TYPE_CHECKING:
    from .code_graph import Resolve
