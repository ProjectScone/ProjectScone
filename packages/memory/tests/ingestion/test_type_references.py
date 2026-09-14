"""The types a declaration's signature names, as edges.

A call edge says what a function runs; nothing said what it takes or
returns. ``def put(shelf: Shelf) -> Receipt`` rests on Shelf and Receipt as
surely as on anything it calls, and a reader asking "what uses Receipt?"
found only its constructors. The reference records type references with a
noise list. Here a Python function's parameters, return and annotated
locals, and a class's annotated attributes, give ``uses_type`` edges to the
classes this file declares and the names it imported from this project,
through generics and quoted forward references alike. Standard-library and
built-in names are left out, and so is a local function named in an
annotation, since a function is not a type.
"""

from __future__ import annotations

from scone_memory.ingestion.code_graph import USES_TYPE, code_claims

SOURCE = '''
from __future__ import annotations

import datetime
from typing import Optional

import pkg.models as models
import numpy
from pkg.receipts import Receipt, make_receipt
from pkg.stamps import Stamp


class Shelf:
    label: "Label"
    added: datetime.datetime

    class Label:
        text: str


def helper() -> None:
    pass


def put(shelf: Shelf, *extra: Stamp, when: Optional[datetime.date] = None) -> list[Receipt]:
    kept: dict[str, models.Paper] = {}
    return []


def later(shelf: "Shelf", ready: helper, grid: numpy.ndarray) -> Optional["Shelf.Label"]:
    def inner(paper: models.Paper) -> None:
        note: Receipt = make_receipt()
    return None
'''

PATH = "pkg/shelf.py"


def edges() -> set[tuple[str, str]]:
    return {(claim.subject, claim.object) for claim in code_claims(SOURCE, PATH, language="python")
            if claim.predicate == USES_TYPE}


def test_a_signature_names_the_classes_it_takes_and_returns():
    found = edges()
    assert (f"{PATH}:put", f"{PATH}:Shelf") in found
    assert (f"{PATH}:put", "pkg.receipts.Receipt") in found, "named where it was imported from, as calls are"


def test_generics_varargs_and_annotated_locals_are_read():
    targets = {target for subject, target in edges() if subject == f"{PATH}:put"}
    assert {"pkg.receipts.Receipt", "pkg.models.Paper", "pkg.stamps.Stamp"} <= targets, targets


def test_quoted_forward_references_and_nested_classes_are_read():
    targets = {target for subject, target in edges() if subject == f"{PATH}:later"}
    assert {f"{PATH}:Shelf", f"{PATH}:Shelf.Label"} <= targets


def test_a_class_names_the_types_of_its_annotated_attributes():
    assert (f"{PATH}:Shelf", f"{PATH}:Shelf.Label") in edges()


def test_standard_library_builtins_and_functions_are_not_types_here():
    targets = {target for _, target in edges()}
    assert not any("datetime" in target or "typing" in target or target.endswith(":str") for target in targets)
    assert f"{PATH}:helper" not in targets, "a function named in an annotation is not a type"
    assert "pkg.receipts.make_receipt" not in targets, "an imported function that is never an annotation is no type"


def test_each_edge_is_recorded_once_per_declaration():
    claims = [claim for claim in code_claims(SOURCE, PATH, language="python") if claim.predicate == USES_TYPE]
    assert len(claims) == len({(claim.subject, claim.object) for claim in claims})


def test_a_nested_function_names_its_own_types_under_the_name_it_is_defined_by():
    found = edges()
    assert (f"{PATH}:later.inner", "pkg.models.Paper") in found
    assert (f"{PATH}:later.inner", "pkg.receipts.Receipt") in found
    assert not any(subject == f"{PATH}:later" and target in ("pkg.models.Paper", "pkg.receipts.Receipt")
                   for subject, target in found), "the enclosing function does not take its inner function's types"
    defined = {claim.object for claim in code_claims(SOURCE, PATH, language="python") if claim.predicate == "defines"}
    assert {subject for subject, _ in found} <= defined, "every edge starts at a declaration the graph defines"


async def test_a_type_edge_becomes_a_relation_between_two_entities_in_the_graph():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.entities.read import load_projection
    from scone_memory.ingestion.code_graph import record_claims

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        added = await engine.remember("default", SOURCE, source=PATH)
        await record_claims(engine, "default", episode_id=added.episode_id, content=SOURCE, path=PATH,
                            when="2026-01-01T00:00:00Z")
        projection, _ = await load_projection(engine, "default", mode="current")
    finally:
        await engine.close()
    labels = {entity.entity_id: entity.label for entity in projection.entities}
    typed = {(labels[relation.subject_id], labels[relation.object_id]) for relation in projection.relations
             if relation.predicate == USES_TYPE}
    assert (f"{PATH}:put", "pkg.receipts.Receipt") in typed
    assert (f"{PATH}:later", "numpy.ndarray") in typed, "a lowercase type is a thing when a type names it"
    assert not any(attribute.predicate == USES_TYPE for attribute in projection.attributes), "never read as a value"
