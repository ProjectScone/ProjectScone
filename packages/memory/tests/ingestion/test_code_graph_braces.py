"""A codebase that is not Python: what can still be read for certain.

Without a parser for the language there is no call graph worth having —
resolving a call means knowing what a name refers to, and guessing that
is how a graph fills with edges nobody can check. What a header line says
is a different matter: a declaration and an import are written down, and
the brace scanner already finds the first.

So these languages get what can be read and not what would have to be
inferred, and the docs say which is which rather than leaving somebody to
discover that calls are missing.
"""

from __future__ import annotations

import pytest

from scone_memory.ingestion.code_graph import code_claims

TYPESCRIPT = '''import { thing } from "./thing"
import other from "../other/mod"
import "side-effect"

export function alpha(n: number): number {
  return helper(n)
}

class Beta {
  gamma() {
    return 1
  }
}
'''

GO = '''package main

import (
\t"fmt"
\t"example.com/pkg/store"
)

func Handle(w int) int {
\treturn fmt.Sprint(w)
}
'''


def claims(source, path):
    return code_claims(source, path, language="braces")


def triples(found):
    return {(claim.subject, claim.predicate, claim.object) for claim in found}


def test_a_brace_file_says_what_it_declares():
    found = triples(claims(TYPESCRIPT, "web/app.ts"))
    assert ("web/app.ts", "defines", "web/app.ts:alpha") in found
    assert ("web/app.ts", "defines", "web/app.ts:Beta") in found
    assert ("web/app.ts:Beta", "defines", "web/app.ts:Beta.gamma") in found


def test_a_brace_file_says_what_it_imports():
    found = triples(claims(TYPESCRIPT, "web/app.ts"))
    assert ("web/app.ts", "imports", "side-effect") in found


def test_a_relative_import_waits_for_somebody_who_knows_the_tree():
    found = triples(claims(TYPESCRIPT, "web/app.ts"))
    assert not [one for one in found if one[1] == "imports" and one[2].startswith(".")]
    resolved = code_claims(TYPESCRIPT, "web/app.ts", language="braces",
                           resolve=lambda path, level, module: "web/thing.ts" if module == "./thing" else None)
    assert ("web/app.ts", "imports", "web/thing.ts") in triples(resolved)


def test_go_imports_are_read_from_the_block_they_are_written_in():
    found = triples(claims(GO, "cmd/main.go"))
    assert ("cmd/main.go", "imports", "fmt") in found
    assert ("cmd/main.go", "imports", "example.com/pkg/store") in found
    assert ("cmd/main.go", "defines", "cmd/main.go:Handle") in found


def test_no_calls_are_claimed_for_a_language_this_cannot_parse():
    """helper(n) is a call. Without a parser, saying so would mean guessing
    what helper refers to, and an edge nobody can check is worse than none."""
    assert not [claim for claim in claims(TYPESCRIPT, "web/app.ts") if claim.predicate == "calls"]


def test_every_brace_claim_is_quoted_from_its_own_line():
    for claim in claims(TYPESCRIPT, "web/app.ts"):
        assert claim.quote and claim.quote in TYPESCRIPT
        assert TYPESCRIPT.encode()[claim.start:claim.end].decode().strip() == claim.quote
