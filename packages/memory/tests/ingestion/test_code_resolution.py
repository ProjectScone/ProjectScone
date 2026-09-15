"""Binding a call across files, when exactly one declaration can answer.

A file on its own cannot say what `thing.method()` refers to: the type of
`thing` is stated nowhere in it. Measured over this package, that is
59.3% of the distinct call names the graph cannot place, and no rule read
off one file's syntax reaches them.

A **corpus** can answer a useful share of it. If exactly one declaration
anywhere in what was read is named `method`, that is what the call meant
-- and if two are, nothing is. The single-definition rule is the whole
safety property, and it is the reference's bar too: an ambiguous name
fabricates nothing.

Measured before building: of 831 distinct unplaced names in this package,
**31.9% have exactly one declaration**, 32.3% have several and are
refused, and 35.9% have none because they belong to somebody else's
library.

These edges are **inferred**, not read. A call bound here was matched by
name against the corpus, not quoted from the source that made it, and the
ledger records that difference so a reader can weigh it.
"""

from __future__ import annotations

from scone_memory.ingestion.code_resolution import resolve_across_files


def test_a_name_with_one_declaration_anywhere_is_bound():
    found = resolve_across_files(
        [("app/a.py:use", "thing.rank")],
        {"rank": ("app/shelf.py:Shelf.rank",)},
    )
    assert found.edges == (("app/a.py:use", "app/shelf.py:Shelf.rank"),)
    assert (found.ambiguous, found.unknown) == (0, 0)


def test_a_name_with_two_declarations_binds_to_neither():
    """The rule that keeps this honest. Two plausible answers is not half
    an edge; it is no edge, and it is counted so the reader can see how
    much was refused rather than missed."""
    found = resolve_across_files(
        [("app/a.py:use", "thing.rank")],
        {"rank": ("app/shelf.py:Shelf.rank", "app/pile.py:Pile.rank")},
    )
    assert found.edges == ()
    assert (found.ambiguous, found.unknown) == (1, 0)


def test_a_name_declared_nowhere_here_is_counted_apart():
    """Somebody else's library. Different from ambiguous, and a reader
    who sees the two added together learns nothing from either."""
    found = resolve_across_files([("app/a.py:use", "session.commit")], {})
    assert found.edges == ()
    assert (found.ambiguous, found.unknown) == (0, 1)


def test_only_the_last_segment_names_the_declaration():
    """`thing.rank` is a call to something named `rank`; `thing` is a
    value, not a place to look."""
    found = resolve_across_files(
        [("app/a.py:use", "self.inner.rank")],
        {"rank": ("app/shelf.py:Shelf.rank",)},
    )
    assert found.edges == (("app/a.py:use", "app/shelf.py:Shelf.rank"),)


def test_a_call_is_never_bound_to_its_own_caller():
    """An edge from a thing to itself says nothing, and the line reader
    refuses it for the same reason."""
    found = resolve_across_files(
        [("app/shelf.py:Shelf.rank", "other.rank")],
        {"rank": ("app/shelf.py:Shelf.rank",)},
    )
    assert found.edges == ()
    assert (found.ambiguous, found.unknown) == (0, 0)


def test_the_same_edge_found_twice_is_one_edge():
    """A method called in a loop is one relationship, not twenty."""
    found = resolve_across_files(
        [("app/a.py:use", "thing.rank"), ("app/a.py:use", "other.rank")],
        {"rank": ("app/shelf.py:Shelf.rank",)},
    )
    assert found.edges == (("app/a.py:use", "app/shelf.py:Shelf.rank"),)


def test_edges_come_back_in_a_settled_order():
    """A receipt that reorders between runs cannot be compared."""
    found = resolve_across_files(
        [("app/z.py:z", "thing.rank"), ("app/a.py:a", "thing.hold")],
        {"rank": ("app/shelf.py:Shelf.rank",), "hold": ("app/shelf.py:Shelf.hold",)},
    )
    assert found.edges == (("app/a.py:a", "app/shelf.py:Shelf.hold"),
                           ("app/z.py:z", "app/shelf.py:Shelf.rank"))


def test_nothing_in_gives_nothing_out():
    found = resolve_across_files([], {"rank": ("app/shelf.py:Shelf.rank",)})
    assert found.edges == () and found.ambiguous == 0 and found.unknown == 0


def test_the_walk_names_a_file_written_outright_a_rust_mod_file_and_the_newer_suffixes():
    from scone_memory.ingestion.code_resolution import file_resolver

    walked = file_resolver(["src/lib/x.h", "src/util/mod.rs", "web/a.mjs", "lib/b.dart", "src/store.rs"])
    assert walked("src/main.c", 1, "./lib/x.h") == "src/lib/x.h", "a header named outright is that file"
    assert walked("src/main.c", 1, "./lib/y.h") is None
    assert walked("src/main.rs", 1, "util") == "src/util/mod.rs" and walked("src/main.rs", 1, "store") == "src/store.rs"
    assert walked("web/app.js", 1, "./a") == "web/a.mjs" and walked("lib/main.dart", 1, "./b") == "lib/b.dart"


def test_a_published_package_s_module_is_its_file_in_the_repository_that_publishes_it():
    from scone_memory.ingestion.code_resolution import file_resolver
    from scone_memory.ingestion.repositories import published_by

    published = published_by([("lib/pyproject.toml", "defines", "libpkg"), ("ui/package.json", "defines", "@acme/ui"),
                              ("core/Cargo.toml", "defines", "acme-core"), ("svc/go.mod", "defines", "example.com/svc"),
                              ("lib/pyproject.toml", "depends_on", "requests"), ("lib/src/a.py", "defines", "lib/src/a.py:f")])
    assert published == {"libpkg": "lib", "@acme/ui": "ui", "acme-core": "core", "example.com/svc": "svc"}, \
        "only a manifest's own package, not what it depends on nor what a file defines"
    walked = file_resolver(["app/main.py"], published,
                           known=["lib/src/libpkg/__init__.py", "lib/src/libpkg/util.py", "ui/src/button.tsx",
                                  "ui/src/index.ts", "core/src/lib.rs", "core/src/store/mod.rs", "svc/pkg/store/store.go"])
    assert walked.package("libpkg.util", "python") == "lib/src/libpkg/util.py"
    assert walked.package("libpkg", "python") == "lib/src/libpkg/__init__.py"
    assert walked.package("libpkg.missing", "python") is None and walked.package("requests", "python") is None, \
        "a module never mapped, or a package nobody here publishes, stays a name"
    assert walked.package("@acme/ui/button", "js") == "ui/src/button.tsx" and walked.package("@acme/ui", "js") == "ui/src/index.ts"
    assert walked.package("acme_core::store::Shelf", "rust") == "core/src/store/mod.rs"
    assert walked.package("acme_core", "rust") == "core/src/lib.rs"
    assert walked.package("example.com/svc/pkg/store", "go") == "svc/pkg/store"
    assert walked.package("example.com/svc/pkg/none", "go") is None and walked.package("example.com/other", "go") is None
    assert file_resolver(["app/main.py"]).package("libpkg.util", "python") is None, "nothing published, nothing followed"


def test_a_package_published_at_the_mapped_root_has_no_directory_in_front():
    from scone_memory.ingestion.code_resolution import file_resolver
    from scone_memory.ingestion.repositories import published_by

    published = published_by([("pyproject.toml", "defines", "mypkg"), ("package.json", "defines", "@acme/ui"),
                              ("Cargo.toml", "defines", "acme-core"), ("go.mod", "defines", "example.com/svc")])
    assert published == {"mypkg": "", "@acme/ui": "", "acme-core": "", "example.com/svc": ""}
    walked = file_resolver(["mypkg/util.py", "index.ts", "src/lib.rs", "pkg/store/store.go", "main.py"], published)
    assert walked.package("mypkg.util", "python") == "mypkg/util.py"
    assert walked.package("@acme/ui", "js") == "index.ts" and walked.package("acme_core", "rust") == "src/lib.rs"
    assert walked.package("example.com/svc/pkg/store", "go") == "pkg/store"
