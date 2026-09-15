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


def test_a_call_to_something_this_file_never_declared_is_still_left_out():
    """`helper(n)` is a call, and this file does not say what `helper`
    is. Saying so would mean guessing, and an edge nobody can check is
    worse than none.

    This test used to be named for a stronger claim -- that no calls are
    claimed at all for a language without a parser -- and that is no
    longer true: a call to a declaration the file does make is recorded.
    The rule it actually guards, then and now, is that a name this file
    never declared is left alone.
    """
    calls = [claim for claim in claims(TYPESCRIPT, "web/app.ts") if claim.predicate == "calls"]
    assert not calls, calls


def test_every_brace_claim_is_quoted_from_its_own_line():
    for claim in claims(TYPESCRIPT, "web/app.ts"):
        assert claim.quote and claim.quote in TYPESCRIPT
        assert TYPESCRIPT.encode()[claim.start:claim.end].decode().strip() == claim.quote


def resolver(*paths):
    from scone_memory.ingestion.code_resolution import file_resolver

    return file_resolver(paths)


def imported(source, path, resolve=None):
    return {one[2] for one in triples(code_claims(source, path, language="braces", resolve=resolve))
            if one[1] == "imports"}


def test_jvm_languages_say_what_they_import_and_a_walked_tree_names_the_file():
    java = ("package com.acme.app;\n\nimport com.acme.store.Shelf;\nimport static com.acme.util.Text.trim;\n"
            "import java.util.*;\n\npublic class Main {}\n")
    here = "src/main/java/com/acme/app/Main.java"
    assert imported(java, here) == {"com.acme.store.Shelf", "com.acme.util.Text.trim", "java.util"}, \
        "a wildcard names its package; a static import its member"
    walked = imported(java, here, resolver("src/main/java/com/acme/store/Shelf.java",
                                           "src/main/java/com/acme/util/Text.java"))
    assert walked == {"src/main/java/com/acme/store/Shelf.java", "src/main/java/com/acme/util/Text.java", "java.util"}, \
        "the package line says where the tree is rooted; a package nobody walked keeps its name"
    kotlin = "package a.b\nimport a.b.c.Store as S\nimport a.b.d.*\n"
    assert imported(kotlin, "a/b/X.kt") == {"a.b.c.Store", "a.b.d"}, "an alias is not the name"
    scala = "package a.b\nimport a.b.{Store, Shelf => S}\nimport a.b.c.{D, E}\n"
    assert imported(scala, "a/b/X.scala") == {"a.b.Store", "a.b.Shelf", "a.b.c.D", "a.b.c.E"}, "a group names each"


def test_rust_use_paths_are_kept_whole_and_a_walked_crate_names_the_module_file():
    source = ("use std::{fs, io::{self, Read}};\nuse crate::store::Shelf;\npub(crate) use super::util::*;\n"
              "use self::inner::Thing as T;\nmod inner;\n")
    here = "crates/core/src/app/main.rs"
    assert imported(source, here) == {"std::fs", "std::io", "std::io::Read", "crate::store::Shelf", "super::util",
                                      "self::inner::Thing"}, "every path in a group, none cut to its crate"
    walked = imported(source, here, resolver("crates/core/src/store.rs", "crates/core/src/util/mod.rs",
                                             "crates/core/src/app/inner.rs"))
    assert {"crates/core/src/store.rs", "crates/core/src/util/mod.rs", "crates/core/src/app/inner.rs"} <= walked, \
        "crate:: is the nearest src, super:: the parent module, self:: this one; an item's file is the longest prefix"
    assert "std::fs" in walked and "crate::store::Shelf" not in walked, "a crate nobody walked keeps its name"


def test_c_family_includes_name_a_header_file_or_a_system_header():
    source = '#include "lib/x.h"\n#include <stdio.h>\n#import "A.h"\n# include "../y.hpp"\n'
    assert imported(source, "src/main.c") == {"src/lib/x.h", "stdio.h", "src/A.h", "y.hpp"}
    assert imported(source, "src/main.c", resolver("src/lib/x.h")) == {"src/lib/x.h", "stdio.h"}, \
        "with the tree walked, a header nobody saw is left out rather than guessed at"


def test_zig_dart_csharp_swift_and_php_imports_are_read_as_each_language_writes_them():
    zig = 'const std = @import("std");\nconst shelf = @import("store/shelf.zig");\n'
    assert imported(zig, "src/main.zig") == {"std", "src/store/shelf.zig"}
    dart = ("import 'package:flutter/material.dart';\nimport 'dart:io';\nimport 'src/util.dart';\n"
            "export 'src/api.dart';\npart 'main.g.dart';\n")
    assert imported(dart, "lib/main.dart") == {"package:flutter/material.dart", "dart:io", "lib/src/util.dart",
                                               "lib/src/api.dart", "lib/main.g.dart"}, "a scheme is a package, else a file"
    csharp = ("global using System;\nusing System.Text;\nusing static System.Math;\nusing Alias = Foo.Bar;\n"
              "using var x = new Foo();\nusing (var s = Open()) {}\n")
    assert imported(csharp, "App/Main.cs") == {"System", "System.Text", "System.Math", "Foo.Bar"}, \
        "a using declaration or statement is not an import"
    swift = "import Foundation\n@testable import App\nimport struct Foo.Bar\n"
    assert imported(swift, "Sources/App/Main.swift") == {"Foundation", "App", "Foo.Bar"}
    php = ("<?php\nnamespace App\\Http;\nuse App\\Models\\User;\nuse App\\{Console\\Kernel as CK, Support};\n"
           "use function App\\helper;\nrequire_once __DIR__ . '/../bootstrap.php';\nrequire 'lib/x.php';\n")
    assert imported(php, "app/Http/Kernel.php") == {"App\\Models\\User", "App\\Console\\Kernel", "App\\Support",
                                                    "App\\helper", "app/bootstrap.php", "app/Http/lib/x.php"}, \
        "a namespace is kept whole and a required file is a path from this file"


def test_a_grouped_import_names_every_path_in_it():
    from scone_memory.ingestion.code_graph import _spread

    assert _spread("a::{b, c::{d, self}, e as f, g::*}", "::") == ["a::b", "a::c::d", "a::c", "a::e", "a::g"]
    assert _spread("a.b.{C, D => E}", ".") == ["a.b.C", "a.b.D"]
    assert _spread("a::{b, c", "::") == [], "a group that never closes names nothing rather than a garbled name"
    assert _spread("std::io::*", "::") == ["std::io"]


def test_an_import_in_a_comment_or_a_string_is_not_one_and_a_trailing_comment_hides_none():
    zig = 'const std = @import("std");\n// TODO: switch to @import("new.zig")\n/* @import("old.zig") */\n'
    assert imported(zig, "src/main.zig") == {"std"}
    rust = 'let s = "\nuse std::fs;\n";\n/*\nuse std::io;\n*/\nuse std::env; // read once\n'
    assert imported(rust, "src/main.rs") == {"std::env"}
    java = "/*\npackage old;\n*/\npackage com.acme;\n// import com.acme.Gone;\nimport com.acme.Kept; // used\n"
    assert imported(java, "src/com/acme/Main.java", resolver("src/com/acme/Kept.java", "old/place/Kept.java")) == \
        {"src/com/acme/Kept.java"}, "the package line is the real one, and a trailing comment hides no import"
    assert imported("import UIKit // for UIColor\n", "A.swift") == {"UIKit"}


def test_a_rust_file_and_a_module_file_root_self_and_super_differently():
    plain = "use self::bar::Thing;\nuse super::util::Tool;\nuse super::super::top::Item;\n"
    walked = imported(plain, "src/app/foo.rs",
                      resolver("src/app/foo/bar.rs", "src/app/util.rs", "src/top.rs", "src/util.rs", "src/app/bar.rs"))
    assert walked == {"src/app/foo/bar.rs", "src/app/util.rs", "src/top.rs"}, \
        "foo.rs keeps its own modules under foo/ and its siblings beside it"
    owner = "use self::bar::Thing;\nuse super::util::Tool;\n"
    walked = imported(owner, "src/app/mod.rs", resolver("src/app/bar.rs", "src/util.rs", "src/app/util.rs"))
    assert walked == {"src/app/bar.rs", "src/util.rs"}, "mod.rs owns its directory, and super is the parent"


def test_a_grouped_import_that_wraps_lines_is_read_whole(monkeypatch):
    from scone_memory.ingestion import code_graph as code_graph_module

    rust = "use std::{\n    collections::HashMap,\n    fs,\n};\nuse crate::x;\n"
    assert imported(rust, "src/main.rs") == {"std::collections::HashMap", "std::fs", "crate::x"}
    scala = "package a.b\nimport a.b.{\n  C,\n  D\n}\n"
    assert imported(scala, "a/b/X.scala") == {"a.b.C", "a.b.D"}
    [wrapped] = [c for c in code_claims(rust, "src/main.rs", language="braces") if c.object == "std::fs"]
    assert wrapped.first_line == 1, "a wrapped statement is claimed on the line it begins"
    long = "use std::{\n" + "    fs,\n" * 70 + "};\n"
    assert imported(long, "src/main.rs") == set(), "a group open past the bound is not guessed at"
    monkeypatch.setattr(code_graph_module, "MAX_STATEMENT_LINES", 100)
    assert imported(long, "src/main.rs") == {"std::fs"}, "the bound is what stopped it"


def test_php_lists_traits_and_closures_and_the_other_spellings_the_review_named():
    php = ("<?php\nnamespace App;\nuse App\\Models\\User, App\\Models\\Post;\nclass X {\n    use HasFactory, Notifiable;\n}\n"
           "$f = function () use ($x) { return $x; };\n")
    assert imported(php, "app/X.php") == {"App\\Models\\User", "App\\Models\\Post"}, \
        "a flat list is each name; a trait use inside a class and a closure's use are not imports"
    braced = "<?php\nnamespace App {\n    use App\\Support;\n    class Y {\n        use Traits;\n    }\n}\n"
    assert imported(braced, "app/Y.php") == {"App\\Support"}, "a braced namespace still holds imports"
    assert imported("import scala.collection.mutable._\n", "A.scala") == {"scala.collection.mutable"}
    assert imported("using IntList = System.Collections.Generic.List<int>;\n", "A.cs") == \
        {"System.Collections.Generic.List"}
    assert imported('#include "/opt/vendor/api.h"\n#include "lib/x.h"\n', "src/a.c") == {"src/lib/x.h"}, \
        "an absolute include is nobody's file here"


def test_the_second_review_s_cases_a_lifetime_a_raw_name_two_statements_a_comment_before_an_include():
    lifetime = 'const NAME: &\'static str = "app";\nuse std::fs;\nfn f<\'a>(x: &\'a str) {}\nuse std::env;\n'
    assert imported(lifetime, "src/main.rs") == {"std::fs", "std::env"}, "a lifetime's apostrophe opens no string"
    assert imported("use crate::r#type::Foo;\n", "src/main.rs") == {"crate::r#type::Foo"}
    two = "use std::{\n    fs,\n}; use crate::x;\n"
    assert imported(two, "src/main.rs") == {"std::fs", "crate::x"}, "a statement sharing the closing line is read too"
    assert imported("import a.b.C; import d.e.F;\n", "A.java") == {"a.b.C", "d.e.F"}
    assert imported('/* note */ #include "lib/x.h"\n', "src/main.c") == {"src/lib/x.h"}
    zig = 'const a = @import("a.zig"); // was @import("b.zig")\n'
    assert imported(zig, "src/main.zig") == {"src/a.zig"}, "a commented-out import on the same line is nothing"
    assert imported("mod tests {\n    use super::*;\n}\nuse self::*;\n", "src/app/foo.rs",
                    resolver("src/app/util.rs")) == set(), "a wildcard on the module itself names nothing"
    walked = imported("use self::common::setup;\n", "tests/x.rs", resolver("tests/common/mod.rs", "tests/x/common.rs"))
    assert walked == {"tests/common/mod.rs"}, "an integration test is a crate root and owns its directory"
