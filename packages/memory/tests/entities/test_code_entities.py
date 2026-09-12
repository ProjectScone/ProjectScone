"""A code graph that keeps classes and drops functions is not a code graph.

Objects become entities through `classify_object`, which was written for
prose about people and organisations: `works_at`, `lives_in`,
`married_to`, `founded`. A name reaches the graph by looking like a name,
and `name_shaped` requires a capital letter.

Python functions and modules are conventionally lowercase. So
`class Shelf` became an entity and `def put`, `import json` and
`import pkg.store` did not -- every function, every module and every call
edge to one silently left the graph, while the classes stayed. Five facts
extracted from two files produced two relations.

The predicate is what settles it. An object of `defines`, `imports`,
`calls`, `inherits` or `mixes_in` is a code symbol by construction; it is
never a measurement or a date, and its case is a convention of its
language rather than evidence about what it is.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine

pytestmark = pytest.mark.asyncio

STORE = ('import json\n\n\nclass Shelf:\n    """Holds papers."""\n\n'
         '    def keep(self, paper: str) -> str:\n'
         '        return json.dumps({"paper": paper})\n')
API = ('from pkg.store import Shelf\n\n\ndef put(paper: str) -> str:\n'
       '    shelf = Shelf()\n    return shelf.keep(paper)\n')


async def graphed(**sources):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    for name, text in sources.items():
        await engine.remember("default", text, source=name.replace("__", "/") + ".py")
    return engine


async def projected(engine):
    projection, _ = await engine.entities.projection("default", mode="all",
                                                     when="2099-01-01T00:00:00Z")
    label = {entity.entity_id: entity.label for entity in projection.entities}
    return (sorted(label.values()),
            sorted(f"{label.get(r.subject_id, '?')} {r.predicate} {label.get(r.object_id, '?')}"
                   for r in projection.relations))


async def test_a_lowercase_function_is_in_the_graph():
    engine = await graphed(pkg__store=STORE, pkg__api=API)
    try:
        labels, relations = await projected(engine)
    finally:
        await engine.close()
    assert "pkg/api.py:put" in labels, labels
    assert "pkg/api.py defines pkg/api.py:put" in relations, relations


async def test_an_imported_module_is_in_the_graph():
    """Both the standard library and a module of this codebase: an import
    edge that reaches no entity is an edge the graph cannot walk."""
    engine = await graphed(pkg__store=STORE, pkg__api=API)
    try:
        labels, relations = await projected(engine)
    finally:
        await engine.close()
    assert "json" in labels, labels
    assert "pkg.store" in labels, labels
    assert "pkg/store.py imports json" in relations, relations
    assert "pkg/api.py imports pkg.store" in relations, relations


async def test_every_code_fact_of_this_fixture_reaches_the_graph():
    """Five facts from two files gave two relations, so three edges were
    lost between extraction and the graph.

    Asserted as the five specific edges, not as `relations == facts`.
    That equality holds for this fixture and is **not** the invariant: on
    a real corpus the projection carries more relations than there are
    extracted facts, because distillation contributes its own. I wrote
    the count version first and it passed -- a true assertion about a
    fixture standing in for a false claim about the system.
    """
    engine = await graphed(pkg__store=STORE, pkg__api=API)
    try:
        labels, relations = await projected(engine)
    finally:
        await engine.close()
    for edge in ("pkg/store.py imports json",
                 "pkg/store.py defines pkg/store.py:Shelf",
                 "pkg/store.py:Shelf defines pkg/store.py:Shelf.keep",
                 "pkg/api.py imports pkg.store",
                 "pkg/api.py defines pkg/api.py:put"):
        assert edge in relations, (edge, relations)


async def test_a_symbol_that_is_never_a_subject_is_the_one_that_was_lost():
    """Why the loss was concentrated rather than total, which is the
    difference between what I first claimed and what the measurement
    said.

    `subject_anchor` admits any object that is also a subject somewhere,
    whatever its case -- so a function that calls something was already
    in the graph. What fell out were the leaves: an external module, and
    a definition that never appears as a subject.
    """
    from scone_memory.entities import classify
    from scone_memory.entities.classify import ClassificationContext, classify_object

    without = frozenset()
    real, classify.CODE_PREDICATES = classify.CODE_PREDICATES, without
    try:
        plain = ClassificationContext()
        anchored = ClassificationContext(anchors=frozenset({"pkg/api.py:put"}))
        assert classify_object("pkg/api.py:put", "defines", plain).basis == "common_value"
        assert classify_object("pkg/api.py:put", "defines", anchored).basis == "subject_anchor"
        assert classify_object("json", "imports", plain).basis == "common_value"
    finally:
        classify.CODE_PREDICATES = real
    # And with the predicate rule, a leaf needs no anchor to be an entity.
    assert classify_object("json", "imports", ClassificationContext()).basis == "code_symbol"


async def test_prose_is_not_reclassified_by_a_code_predicate():
    """`classify_object` is shared with prose, so this must not turn every
    description into an entity. A code symbol is one token; a phrase is
    not."""
    from scone_memory.entities.classify import ClassificationContext, classify_object

    context = ClassificationContext()
    assert classify_object("pkg/api.py:put", "defines", context).object_class == "entity"
    assert classify_object("json", "imports", context).object_class == "entity"
    assert classify_object("shelf.keep", "calls", context).object_class == "entity"
    # A phrase is not a code symbol, whatever the predicate says.
    said = classify_object("a quorum of three members", "defines", context)
    assert said.object_class == "literal", said
