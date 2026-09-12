"""Code as claims: what a file defines, imports and calls.

A codebase is already a graph — this reads it without a model and without
leaving the machine, and says it in the only language this framework has:
claims, each carrying the line it was read from. Once they are claims they
are everything else too: cited, dated, recalled, traversed, and subject to
the same vocabulary as anything else a space knows.

What it will not do is guess. A call to something this file cannot see is
not recorded as a call to a name that might mean anything; it is left out
and counted, because a graph with edges nobody can check is worse than a
smaller graph.
"""

from __future__ import annotations

import pytest

from scone_memory.ingestion.code_graph import code_claims

MODULE = '''"""The planner."""

from __future__ import annotations

import json
from pkg.store import Shelf, keep


def plan(question: str) -> str:
    """Work out what is being asked."""
    return tidy(question)


def tidy(text: str) -> str:
    return text.strip()


class Engine:
    def recall(self, query: str) -> list[str]:
        found = self.rank(query)
        return json.dumps(found)

    def rank(self, query: str) -> list[str]:
        return keep(query)
'''


def claims(source: str = MODULE, path: str = "app/planner.py"):
    return code_claims(source, path, language="python")


def triples(found) -> set[tuple[str, str, str]]:
    return {(claim.subject, claim.predicate, claim.object) for claim in found}


def test_a_file_defines_what_it_holds():
    found = triples(claims())
    assert ("app/planner.py", "defines", "app/planner.py:plan") in found
    assert ("app/planner.py", "defines", "app/planner.py:Engine") in found
    assert ("app/planner.py:Engine", "defines", "app/planner.py:Engine.recall") in found, \
        "a class defines its methods, rather than the file defining them directly"


def test_a_file_imports_what_it_names():
    found = triples(claims())
    assert ("app/planner.py", "imports", "json") in found
    assert ("app/planner.py", "imports", "pkg.store") in found


def test_a_call_this_file_can_see_is_recorded():
    found = triples(claims())
    assert ("app/planner.py:plan", "calls", "app/planner.py:tidy") in found
    assert ("app/planner.py:Engine.recall", "calls", "app/planner.py:Engine.rank") in found, \
        "self.rank is the method of the class it is written in"


def test_a_call_this_file_cannot_see_is_left_out_and_counted():
    """keep() comes from another module and json.dumps from a package. A
    graph with edges nobody can check is worse than a smaller graph."""
    found = claims()
    assert not [claim for claim in found if claim.predicate == "calls" and "keep" in claim.object]
    assert not [claim for claim in found if claim.predicate == "calls" and "dumps" in claim.object]


def test_every_claim_carries_the_line_it_was_read_from():
    for claim in claims():
        assert claim.quote and "\n" not in claim.quote
        assert claim.first_line >= 1
        assert MODULE.encode()[claim.start:claim.end].decode().strip() == claim.quote


def test_a_file_that_does_not_parse_says_nothing_rather_than_guessing():
    assert code_claims("def broken(:\n", "app/broken.py", language="python") == ()


def test_a_language_without_a_parser_says_what_is_written_and_no_more():
    """Its declarations and imports are written down and can be read; what
    a call refers to is not, so none is claimed."""
    found = code_claims("import 'x'\nfunction f() {\n  g()\n}\n", "app/x.ts", language="braces")
    assert {(c.predicate, c.object) for c in found} == {("defines", "app/x.ts:f"), ("imports", "x")}


def test_a_language_this_does_not_read_at_all_says_nothing():
    assert code_claims("print 'hello'", "app/x.rb", language=None) == ()


RELATIVE = '''from . import shared
from .code import declarations
from ..entities.read import load
'''


def test_a_relative_import_names_its_file_even_when_nobody_can_resolve_it():
    """This rule used to be the opposite, and the reason it gave was
    half right: "the file alone cannot know whether .code is a module or
    a package, or where the package root is. Guessing would put an edge
    in the graph that nobody can check."

    The package root half is wrong -- a relative import is relative to
    the importing file, which is the one thing this function does know,
    so `from ..entities import read` inside `pkg/ingestion/graph.py`
    names `pkg/entities/read` by arithmetic and by Python's own rules.
    The module-or-package half is right, and is handled where it can be:
    `read.py` is named, and a graph that holds `read/__init__.py` instead
    matches the two at read time rather than at write time.

    "An edge nobody can check" was the cost of leaving it out, not of
    putting it in. Left out, the edge is unrecoverable: `engine.remember`
    and the episodes route see one file each and never supply a resolver,
    so **every** edge between two files of a Python package was dropped.
    Measured over 57 files of this package: 0 cross-file import edges
    that way, and 44 once named -- all 44 naming a file that exists on
    disk exactly as named.
    """
    found = code_claims(RELATIVE, "pkg/ingestion/graph.py", language="python")
    imports = {claim.object for claim in found if claim.predicate == "imports"}
    assert imports == {"pkg/ingestion/shared.py", "pkg/ingestion/code.py",
                       "pkg/entities/read.py"}, imports


def test_a_relative_import_resolves_against_the_files_that_were_seen():
    """Resolution belongs to whoever walked the tree, because that is who
    knows what is there."""
    seen = {"pkg/ingestion/shared.py", "pkg/ingestion/code.py", "pkg/entities/read.py"}
    found = code_claims(RELATIVE, "pkg/ingestion/graph.py", language="python",
                        resolve=lambda path, level, module: _by_path(path, level, module, seen))
    imports = {claim.object for claim in found if claim.predicate == "imports"}
    assert imports == {"pkg/ingestion/shared.py", "pkg/ingestion/code.py", "pkg/entities/read.py"}


def test_a_relative_import_of_something_not_there_is_left_out():
    found = code_claims(RELATIVE, "pkg/ingestion/graph.py", language="python",
                        resolve=lambda path, level, module: _by_path(path, level, module,
                                                                     {"pkg/ingestion/code.py"}))
    imports = {claim.object for claim in found if claim.predicate == "imports"}
    assert imports == {"pkg/ingestion/code.py"}, "only the one that is really there"


def _by_path(path: str, level: int, module: str, seen: set[str]):
    """The resolution the mapper does: by path, against what it saw."""
    import posixpath

    here = posixpath.dirname(path)
    for _ in range(level - 1):
        here = posixpath.dirname(here)
    stem = posixpath.join(here, *module.split(".")) if module else here
    for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
        if candidate in seen:
            return candidate
    return None


def test_naming_a_relative_import_does_not_depend_on_what_arrived_first():
    """The property that makes naming safe where resolving would not be.

    Resolving against the files a space has already seen would make the
    graph depend on ingestion order: a file imported before the file it
    imports would stay unresolved for good. Arithmetic on the importing
    file's own path cannot do that, because it consults nothing.

    Checked on this package as well as here: 532 relations over 60 source
    files, identical sets under forward, reversed and shuffled ingestion.
    """
    files = {
        "pkg/api/routes.py": "from ..core.errors import Bad\n\n\ndef put(p):\n    return Bad(p)\n",
        "pkg/core/errors.py": "class Bad(Exception):\n    pass\n",
        "pkg/api/__init__.py": "from .routes import put\n",
    }
    orders = ([*files], [*reversed([*files])], ["pkg/core/errors.py", "pkg/api/__init__.py",
                                               "pkg/api/routes.py"])
    seen = []
    for order in orders:
        claims: set[tuple[str, str, str]] = set()
        for path in order:
            claims |= {(claim.subject, claim.predicate, claim.object)
                       for claim in code_claims(files[path], path, language="python")}
        seen.append(claims)
    assert seen[0] == seen[1] == seen[2], [sorted(one - seen[0]) for one in seen[1:]]
    assert ("pkg/api/routes.py", "imports", "pkg/core/errors.py") in seen[0], sorted(seen[0])


def test_a_brace_relative_import_names_a_file_spelt_like_the_one_importing_it():
    """TypeScript had no cross-file edge at all.

    Measured over 58 files of this project's own web application: 39
    import edges, every one of them naming an external package, and **0
    files with a dependant**. `./store` could be store.ts, store.tsx,
    store.js, store.jsx or store/index.ts, and the extractor left it out
    rather than choose.

    It names the file the way the importing file is spelt -- a `.ts` file
    importing `./store` means `store.ts` far more often than anything
    else, and a project mixing extensions across one import is rare. The
    guess is a *candidate*: which spelling the graph actually holds is
    settled at read time, where every file is visible, exactly as a
    Python package's `__init__.py` is.
    """
    source = ("import {keep} from './store';\n"
              "import {Api} from '../core/api';\n"
              "import React from 'react';\n"
              "export function put(p: string) { return keep(p); }\n")
    found = code_claims(source, "web/memory/page.tsx", language="braces")
    imports = {claim.object for claim in found if claim.predicate == "imports"}
    # `.ts`, not the importer's own `.tsx`: that is what the compiler
    # tries first, confirmed with `tsc --traceResolution` (7.0.2).
    assert "web/memory/store.ts" in imports, imports
    assert "web/core/api.ts" in imports, imports
    # An external package still names itself and gains no path.
    assert "react" in imports, imports


def test_a_brace_import_that_already_names_its_file_keeps_that_name():
    """A web application imports stylesheets and images by relative path,
    and those paths carry their own extension. Appending the importing
    file's extension to them produced `source-content.css.tsx` and
    `scone-mark-small.png.tsx` -- names for files that cannot exist.

    Measured before this: of 40 distinct import targets naming a path in
    this project's web application, 24 landed on nothing, and nearly all
    of them were this.
    """
    source = ("import './page.css';\n"
              "import mark from '../assets/logo.png';\n"
              "import {keep} from './store';\n")
    found = code_claims(source, "web/memory/page.tsx", language="braces")
    imports = {claim.object for claim in found if claim.predicate == "imports"}
    assert "web/memory/page.css" in imports, imports
    assert "web/assets/logo.png" in imports, imports
    # No extension of its own: it takes the family's first spelling.
    assert "web/memory/store.ts" in imports, imports


def test_from_dot_import_names_its_module_without_a_resolver_too():
    """`from .code import x` and `from . import code` name a module the
    same way and were treated differently: the first went through the
    import list, the second only bound its alias when a resolver existed.
    So `core.Base` stayed an unbound word and the base class it named
    could not be placed in a file."""
    source = ("from . import core\n"
              "from .shared import Helper\n\n\n"
              "class Shelf(core.Base):\n"
              "    pass\n")
    found = code_claims(source, "pkg/ingestion/graph.py", language="python")
    imports = {claim.object for claim in found if claim.predicate == "imports"}
    inherits = {(claim.subject, claim.object) for claim in found if claim.predicate == "inherits"}
    assert "pkg/ingestion/core.py" in imports, imports
    assert ("pkg/ingestion/graph.py:Shelf", "pkg/ingestion/core.py:Base") in inherits, inherits


def test_a_brace_language_records_the_calls_it_can_see():
    """There was no call graph outside Python at all.

    Measured over 58 files of this project's web application: 263 `calls`
    relations for the equivalent Python corpus and **0** for TypeScript,
    in something the reference survey marks done. `calls` is the backbone
    of a code graph and it existed for one language.

    The rule is the one the Python reader already follows: a bare name is
    a call to this file's own declaration when it has one. Anything else
    -- a function from another module, a method on a value whose type
    nobody wrote down -- is left out rather than pointed at a name that
    might mean anything.

    Coarser than Python in one way that the assertions below state
    rather than hide: the brace declaration reader reports a class and
    not its methods, so a call written in a method is attributed to the
    class holding it.
    """
    source = ("function helper(p) { return p; }\n"
              "export class Shelf {\n"
              "  keep(p) { return helper(p); }\n"
              "  put(p) { return this.keep(p); }\n"
              "}\n"
              "export function drive(p) {\n"
              "  return elsewhere(p) + helper(p);\n"
              "}\n")
    found = code_claims(source, "web/shelf.ts", language="braces")
    calls = {(claim.subject, claim.object) for claim in found if claim.predicate == "calls"}
    assert ("web/shelf.ts:Shelf", "web/shelf.ts:helper") in calls, calls
    assert ("web/shelf.ts:drive", "web/shelf.ts:helper") in calls, calls
    # `elsewhere` is not declared here and is not guessed at.
    assert not any("elsewhere" in one for pair in calls for one in pair), calls


def test_a_brace_call_is_not_read_out_of_a_string_or_a_comment():
    """The masked copy of the source is what the declaration reader
    already uses; the call reader has to use it too, or `// helper(p)`
    and `"helper(p)"` become edges."""
    source = ("function helper(p) { return p; }\n"
              "export function drive(p) {\n"
              "  // helper(p) used to be called here\n"
              "  const said = 'helper(p)';\n"
              "  return said;\n"
              "}\n")
    found = code_claims(source, "web/shelf.ts", language="braces")
    calls = {(claim.subject, claim.object) for claim in found if claim.predicate == "calls"}
    assert not calls, calls


def test_an_unknown_suffix_is_part_of_the_name_not_an_extension():
    """`./foo.bar` means `foo.bar.ts`, confirmed against
    `tsc --traceResolution` (TypeScript 7.0.2). Treating `.bar` as an
    extension to keep sent the import to an unrelated `foo.ts`."""
    found = code_claims("import {x} from './foo.bar';\n", "web/page.ts", language="braces")
    imports = {claim.object for claim in found if claim.predicate == "imports"}
    assert imports == {"web/foo.bar.ts"}, imports


def test_a_call_in_a_method_belongs_to_the_method_and_not_also_its_class():
    """Spans nest, so a line inside a method sits inside its class too.
    The innermost declaration holding a call is the one that made it, or
    every call in a class would be claimed twice at two granularities."""
    source = ("function helper(p) { return p; }\n"
              "export class Shelf {\n"
              "  keep(p) {\n"
              "    return helper(p);\n"
              "  }\n"
              "}\n")
    found = code_claims(source, "web/shelf.ts", language="braces")
    calls = {(claim.subject, claim.object) for claim in found if claim.predicate == "calls"}
    assert calls == {("web/shelf.ts:Shelf.keep", "web/shelf.ts:helper")}, calls


def test_an_import_that_writes_its_own_extension_is_not_given_a_second():
    """`import {x} from './personas.ts'` is ordinary in modern ESM and
    under `moduleResolution: bundler`. Appending the family's extension
    to it produced `personas.ts.ts`.

    Found by measuring this project's web application rather than by
    reading: five of six import targets that landed on nothing were this,
    and `./foo.bar` -- where the suffix belongs to no language -- must
    still become `foo.bar.ts`.
    """
    found = code_claims("import {a} from './personas.ts';\n"
                        "import {b} from './wire.js';\n"
                        "import {c} from './foo.bar';\n", "web/page.ts", language="braces")
    imports = {claim.object for claim in found if claim.predicate == "imports"}
    assert imports == {"web/personas.ts", "web/wire.js", "web/foo.bar.ts"}, imports


def test_a_declaration_header_is_not_a_call_to_itself():
    """`gamma() {` matches the call pattern exactly, so a class was
    claimed to call each of its own methods at the line where they are
    written. That is where a thing is declared, not where it is used."""
    source = ("class Beta {\n"
              "  gamma() {\n"
              "    return 1;\n"
              "  }\n"
              "}\n")
    found = code_claims(source, "web/app.ts", language="braces")
    calls = {(claim.subject, claim.object) for claim in found if claim.predicate == "calls"}
    assert not calls, calls
