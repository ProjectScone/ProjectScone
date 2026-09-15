"""Calls and declarations read from a grammar, for the brace languages beyond TypeScript."""
import pytest

from scone_memory.ingestion import code_grammar as grammar_module
from scone_memory.ingestion.code_graph import _brace_claims, code_claims
from scone_memory.ingestion.code_grammar import GRAMMARS, SUFFIXES, available, grammar_claims

pytestmark = pytest.mark.skipif(not available(), reason="the code-languages extra is not installed")


def triples(found):
    return {(claim.subject, claim.predicate, claim.object) for claim in found}


def calls(found):
    return {(claim.subject.split(":", 1)[1], claim.object.split(":", 1)[1]) for claim in found if claim.predicate == "calls"}


def defines(found):
    return {claim.object.split(":", 1)[1] for claim in found if claim.predicate == "defines"}


def test_a_call_binds_to_a_declaration_of_the_file_and_nothing_a_parameter_or_a_local_shadows():
    go = ("package p\nfunc top(p int) { x := 1; g(); o.h(); helper(); p(); x() }\nfunc helper() {}\n"
          "type K struct{}\nfunc (k *K) m() { top(2); helper() }\ntype I interface { M(x int) error }\n")
    found = grammar_claims(go, "src/main.go")
    assert defines(found) == {"top", "helper", "K", "K.m", "I", "I.M"}, "a method is named by its receiver's type"
    assert calls(found) == {("top", "helper"), ("K.m", "top"), ("K.m", "helper")}, \
        "g is nobody's, o.h() needs a type, p and x are values"
    java = ("import a.b.C;\nclass K { void m(int p) { int x = 1; g(); o.h(); s(); p(); x(); this.s(); }\n"
            "  static void s() { m(); } class Inner { void i() { s(); } } }\n")
    found = grammar_claims(java, "src/A.java")
    assert defines(found) == {"K", "K.m", "K.s", "K.Inner", "K.Inner.i"}
    assert calls(found) == {("K.m", "K.s"), ("K.s", "K.m"), ("K.Inner.i", "K.s")}, \
        "a bare call inside a class names the class's method; a call through this or a receiver is left alone"


def test_rust_impls_mods_closures_and_paths():
    rust = ("use crate::a::f;\nfn top(p: i32) { let x = 1; g(); o.h(); T::s(); helper(); p(); x(); println!(\"x\"); }\n"
            "fn helper() {}\nstruct K;\nimpl K { fn m(&self) { top(2); Self::new() } fn n() { let c = |q| { helper(); q() }; } }\n"
            "impl<'a> Db<'a> { fn open() { helper() } }\n"
            "mod inner { fn q() { super::helper() } fn r() { q() } }\n#[cfg(test)]\nmod tests { #[test] fn t() { top(1) } }\n")
    found = grammar_claims(rust, "src/main.rs")
    assert defines(found) == {"top", "helper", "K", "K.m", "K.n", "Db.open", "inner", "inner.q", "inner.r", "tests",
                              "tests.t"}, "an impl's methods belong to the type, a mod's functions to the mod"
    assert calls(found) == {("top", "helper"), ("K.m", "top"), ("K.n", "helper"), ("Db.open", "helper"),
                            ("inner.r", "inner.q"), ("tests.t", "top")}, \
        "a path call, a macro, a closure parameter and a call through super are not bound"


def test_the_c_family_names_out_of_class_definitions_typedefs_and_nothing_by_namespace():
    cpp = ("namespace n {\nclass K {\n  void m(int p) { int x = 1; helper(); o.h(); K::s(); x(); }\n  static void s();\n};\n"
           "void K::s() { helper(); }\n}\nint helper() { return 0; }\ntypedef struct { int a; } Foo;\n")
    found = grammar_claims(cpp, "src/a.cpp")
    assert defines(found) == {"K", "K.m", "K.s", "helper", "Foo"}, "a namespace holds nothing by its own name"
    assert calls(found) == {("K.m", "helper"), ("K.s", "helper")}
    c = "int helper(int p) { return p; }\nint top(int p) { int x = 1; helper(x); s.h(); g(); return 0; }\n"
    assert calls(grammar_claims(c, "src/a.c")) == {("top", "helper")}
    assert defines(grammar_claims("class H { public: void hm(); };\n", "src/h.h")) == {"H", "H.hm"}, \
        "a header may hold C++"


def test_csharp_swift_scala_and_php_shapes():
    csharp = ("namespace N;\nclass K { void M(int p) { int x = 1; G(); o.H(); S(); } "
              "static void S() { void Local() { S(); } Local(); } }\n")
    found = grammar_claims(csharp, "src/A.cs")
    assert defines(found) == {"K", "K.M", "K.S", "K.S.Local"} and calls(found) == {("K.M", "K.S"), ("K.S.Local", "K.S")}, \
        "a local function is nobody's target but may call out"
    swift = ("func top(p: Int) { let x = 1; g(); o.h(); helper(); x() }\nfunc helper() {}\n"
             "class K { init() { helper() } func m() { top(p: 2) } }\n")
    found = grammar_claims(swift, "src/A.swift")
    assert defines(found) == {"top", "helper", "K", "K.init", "K.m"}
    assert calls(found) == {("top", "helper"), ("K.init", "helper"), ("K.m", "top")}
    scala = "object O { def top(p: Int): Unit = { val x = 1; g(); o.h(); helper(); x() }; def helper() = 1 }\n"
    assert calls(grammar_claims(scala, "src/A.scala")) == {("O.top", "O.helper")}
    php = "<?php\nfunction top($p) { $x = 1; g(); $o->h(); helper(); }\nfunction helper() {}\nclass K { function m() { top(1); } }\n"
    found = grammar_claims(php, "src/a.php")
    assert defines(found) == {"top", "helper", "K", "K.m"} and calls(found) == {("top", "helper"), ("K.m", "top")}


def test_the_grammar_decides_declarations_and_calls_and_the_line_reader_keeps_the_rest(monkeypatch):
    go = "package p\nimport \"fmt\"\nfunc f() {\n\tgo func() {\n\t\thelper()\n\t}()\n}\nfunc helper() { fmt.Println() }\n"
    line = triples(_brace_claims(go, "src/main.go", None))
    assert any(one[1] == "defines" and one[2] == "src/main.go:f.func" for one in line), \
        "the line reader reads a closure as a declaration"
    merged = triples(code_claims(go, "src/main.go", language="braces"))
    assert not any(one[2] == "src/main.go:f.func" for one in merged), "the grammar decides what is declared"
    assert ("src/main.go", "imports", "fmt") in merged, "imports stay with the line reader"
    assert ("src/main.go:f", "calls", "src/main.go:helper") in merged
    monkeypatch.setattr(grammar_module, "available", lambda: False)
    assert grammar_claims(go, "src/main.go") == (), "without the extra nothing here runs"
    assert any(one[2] == "src/main.go:f.func" for one in triples(code_claims(go, "src/main.go", language="braces"))), \
        "and the line reader's answer stands whole"


def test_bounds_and_unknown_suffixes_give_nothing(monkeypatch):
    assert grammar_claims("fn f() {}\n", "src/notes.txt") == () and grammar_claims("x = 1\n", "a.py") == ()
    assert set(SUFFIXES.values()) <= set(GRAMMARS)
    monkeypatch.setattr(grammar_module, "MAX_LINES", 2)
    assert grammar_claims("fn f() {}\nfn g() {}\nfn h() {}\n", "src/main.rs") == (), "a file past the bound is not read"
    monkeypatch.setattr(grammar_module, "MAX_LINES", 200_000)
    monkeypatch.setattr(grammar_module, "MAX_DEPTH", 1)
    assert calls(grammar_claims("fn f() { g() }\nfn g() {}\n", "src/main.rs")) == set(), "past the depth nothing is walked"


def test_a_bare_call_never_names_a_sibling_method_where_the_language_needs_self_or_a_receiver():
    rust = "fn shared() {}\nstruct K;\nimpl K { fn shared(&self) {} fn m(&self) { shared(); } }\n"
    assert calls(grammar_claims(rust, "src/main.rs")) == {("K.m", "shared")}, "a bare call in an impl is the free fn"
    php = "<?php\nfunction helper() {}\nclass K { function helper() {} function m() { helper(); } }\n"
    assert calls(grammar_claims(php, "src/a.php")) == {("K.m", "helper")}, "a bare call in a PHP class is the function"
    for order in ("method first", "function first"):
        method = "type Cache struct{}\nfunc (c *Cache) Get(k string) string { return \"\" }\n"
        free = "func Get(k string) string { return \"\" }\n"
        go = "package p\n" + (method + free if order == "method first" else free + method) + "func use() { Get(\"x\") }\n"
        assert calls(grammar_claims(go, "src/c.go")) == {("use", "Get")}, f"a method is never bare ({order})"
    cpp = "struct K { static void s(); };\nvoid K::s() {}\nvoid s() {}\nvoid other() { s(); }\n"
    assert calls(grammar_claims(cpp, "src/a.cpp")) == {("other", "s")}, "K::s defined outside is not the bare s"
    inside = "struct K { static void s(); void m() { s(); } };\nvoid s() {}\n"
    assert calls(grammar_claims(inside, "src/b.cpp")) == {("K.m", "K.s")}, "inside the class the sibling wins"


def test_every_name_a_declaration_tags_is_bound_not_only_the_first():
    go = "package p\nfunc helper() {}\nfunc top(a, helper func()) { helper() }\nfunc other() { var b, helper func(); helper() }\n"
    assert calls(grammar_claims(go, "src/main.go")) == set(), "the second parameter and the second var shadow the function"
    c = "void helper(void) {}\nvoid top(void) { void (*a)(void), (*helper)(void); helper(); }\n"
    assert calls(grammar_claims(c, "src/a.c")) == set(), "the second declarator shadows the function"
    csharp = "class K { void helper() {} void M() { Func<int, int> f = helper => helper(); } }\n"
    assert calls(grammar_claims(csharp, "src/A.cs")) == set(), "a lambda's untyped parameter shadows the method"
    scala = "object O { def p() = 1; class K(p: () => Int) { def m() = p() } }\n"
    assert calls(grammar_claims(scala, "src/A.scala")) == set(), "a class parameter shadows the object's method"


def test_past_the_depth_bound_the_grammar_says_nothing_and_the_line_reader_stands(monkeypatch):
    rust = "fn f() { g() }\nfn g() {}\n"
    monkeypatch.setattr(grammar_module, "MAX_DEPTH", 1)
    assert grammar_claims(rust, "src/main.rs") == (), "a partly read file decides nothing"
    assert defines(code_claims(rust, "src/main.rs", language="braces")) == {"f", "g"}, "the line reader's declarations stand"


def test_a_method_defined_outside_its_class_sees_the_class_s_members_where_the_language_says_so():
    cpp = ("struct K { void s(); void t(); };\nvoid K::s() {}\nvoid K::t() { s(); helper(); }\nvoid s() {}\n"
           "void helper() {}\nvoid other() { s(); }\n")
    assert calls(grammar_claims(cpp, "src/a.cpp")) == {("K.t", "K.s"), ("K.t", "helper"), ("other", "s")}, \
        "inside K::t a bare s is the member; outside, the free function"
    apart = "void K::s() {}\nvoid K::t() { s(); }\n"
    assert calls(grammar_claims(apart, "src/b.cc")) == {("K.t", "K.s")}, "two definitions of one class, its body elsewhere"
    go = "package p\ntype K struct{}\nfunc (k K) Get() {}\nfunc (k K) Set() { Get() }\n"
    assert calls(grammar_claims(go, "src/k.go")) == set(), "a Go method never sees a sibling bare"


def test_destructors_operators_trait_parameters_and_a_deep_expression():
    cpp = ("struct K {\n  void m() { helper(); }\n  ~K() { helper(); }\n  bool operator==(const K&) const { return helper(); }\n};\n"
           "K::~K() {}\nint helper() { return 0; }\n")
    found = grammar_claims(cpp, "src/a.cpp")
    assert {"K.~K", "K.operator=="} <= defines(found)
    assert calls(found) == {("K.m", "helper"), ("K.~K", "helper"), ("K.operator==", "helper")}, \
        "a destructor's call is the destructor's, not the class's"
    scala = "object O { trait T(helper: () => Int) { def m() = helper() }; def helper() = 1; def caller() = helper() }\n"
    assert calls(grammar_claims(scala, "src/A.scala")) == {("O.caller", "O.helper")}, \
        "a trait's parameter shadows inside the trait alone"
    java = "class K { String s = " + " + ".join(['"a"'] * 1500) + "; void m() { n(); } void n() {} }\n"
    assert grammar_claims(java, "src/K.java") == (), "an expression nested past the bound cuts the file, without an error"
    assert defines(code_claims(java, "src/K.java", language="braces")) >= {"K"}, "and the line reader stands"


def resolver(*paths, published=None, known=()):
    from scone_memory.ingestion.code_resolution import file_resolver

    return file_resolver(paths, published, known)


def test_a_bare_call_binds_to_what_an_import_brought_in_within_and_across_repositories():
    rust = ("use crate::util::helper;\nuse crate::util::other as o;\nuse crate::store::*;\nuse crate::api;\n"
            "fn f(p: i32) { helper(); o(); stray(); api(); }\nfn g() { let helper = 1; helper() }\n")
    found = code_claims(rust, "src/main.rs", language="braces", resolve=resolver("src/util.rs", "src/store.rs", "src/api.rs"))
    assert calls(found) == {("f", "helper"), ("f", "other")}, \
        "the import's item by its name or alias; a glob, a module and a shadowed name bind nothing"
    assert {c.object for c in found if c.predicate == "calls"} == {"src/util.rs:helper", "src/util.rs:other"}
    assert code_claims(rust, "src/main.rs", language="braces", resolve=None) and \
        calls(code_claims(rust, "src/main.rs", language="braces", resolve=None)) == set(), \
        "nobody walked the tree: the imports stay names and bind nothing"
    java = "package app;\nimport static lib.Text.trim;\nimport lib.Store;\nclass Main { void run() { trim(); } }\n"
    found = code_claims(java, "app/Main.java", language="braces", resolve=resolver("lib/Text.java", "lib/Store.java"))
    assert {c.object for c in found if c.predicate == "calls"} == {"lib/Text.java:Text.trim"}, "a static import binds its member"
    crossing = ("use smfs_core::daemon::protocol::request;\nfn run() { request() }\n")
    walked = resolver("smfs/src/main.rs", published={"smfs-core": "smfs-core"},
                      known=["smfs-core/src/daemon/protocol.rs", "smfs-core/src/lib.rs"])
    found = code_claims(crossing, "smfs/src/main.rs", language="braces", resolve=walked)
    assert ("smfs/src/main.rs", "imports", "smfs-core/src/daemon/protocol.rs") in triples(found)
    assert {c.object for c in found if c.predicate == "calls"} == {"smfs-core/src/daemon/protocol.rs:request"}, \
        "an import of what another repository publishes binds the call to that repository's file"


def test_a_use_inside_a_rust_mod_binds_there_alone_and_a_kotlin_alias_is_the_bound_name():
    from scone_memory.ingestion.code_graph import _brace_claims

    rust = ("use crate::stuff::*;\nfn top() { thing() }\n#[cfg(test)]\nmod tests {\n    use crate::other::thing;\n"
            "    fn t() { thing() }\n}\n")
    found = code_claims(rust, "src/main.rs", language="braces", resolve=resolver("src/stuff.rs", "src/other.rs"))
    assert {c.object for c in found if c.predicate == "calls"} == {"src/other.rs:thing"}
    assert calls(found) == {("tests.t", "thing")}, "top's thing is the glob's, unbound; only tests' use binds"
    bound: list = []
    _brace_claims("package app\nimport app.shapes.Circle as Shape\nimport app.shapes.Square\n", "app/Draw.kt",
                  resolver("app/shapes/Circle.kt", "app/shapes/Square.kt"), bound)
    assert set(bound) == {("", "Shape", "app/shapes/Circle.kt:Circle"), ("", "Square", "app/shapes/Square.kt:Square")}
