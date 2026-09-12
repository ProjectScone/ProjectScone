"""Withholding what a caller must not receive, on the way out.

`capture/redact.py` scrubs secrets on the way **in**, for the agent feed.
Memory arrives by many other doors -- `remember`, `sync`, document
import -- and none of them scrub, so a space accumulates whatever people
put in it. For a framework whose whole business is remembering what
people said, retrieval being unable to withhold anything is the sharper
gap of the two the reference survey found.

The rule that makes this honest rather than dangerous: **a report of what
was withheld must never read as a statement that the rest is clean.**
Pattern matching is a net. A caller who believes "0 withheld" means "no
personal data here" is worse off than one who was told nothing.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.withhold import KINDS, withhold

pytestmark = pytest.mark.asyncio


async def memory(*texts):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                HashEmbedder()).open()
    for text in texts:
        await engine.remember("default", text)
    return engine


async def found(engine, query):
    return (await engine.recall("default", query, limit=5)).items


async def test_an_address_is_withheld_and_the_report_names_what_kind():
    engine = await memory("Ana's address is ana.alves@meridian-health.example for the rota.")
    try:
        held = withhold(await found(engine, "ana address rota"))
    finally:
        await engine.close()
    assert held.withheld == 1, held.record()
    assert "ana.alves@meridian-health.example" not in held.items[0].text
    assert "email" in held.items[0].text, held.items[0].text
    assert held.by_kind.get("email") == 1, held.by_kind


async def test_a_secret_that_never_passed_the_capture_hook_is_still_withheld():
    """It arrived through `remember`, which does not scrub. The only place
    left to catch it is on the way out."""
    engine = await memory("The deploy key is AKIAIOSFODNN7EXAMPLE, do not share it.")
    try:
        held = withhold(await found(engine, "deploy key"))
    finally:
        await engine.close()
    assert held.by_kind.get("secret") == 1, held.by_kind
    assert "AKIAIOSFODNN7EXAMPLE" not in held.items[0].text


async def test_a_long_number_that_is_not_a_card_number_is_left_alone():
    """A run of sixteen digits is an order number more often than a card.
    Redacting it would damage the answer to protect nothing, so the digits
    are checked rather than merely counted."""
    engine = await memory("Order 1234567890123456 shipped on Tuesday.")
    try:
        held = withhold(await found(engine, "order shipped tuesday"))
    finally:
        await engine.close()
    assert held.withheld == 0, held.record()
    assert "1234567890123456" in held.items[0].text


async def test_a_number_that_passes_the_check_is_withheld():
    engine = await memory("She paid with 4111111111111111 in March.")
    try:
        held = withhold(await found(engine, "paid march"))
    finally:
        await engine.close()
    assert held.by_kind.get("card") == 1, held.by_kind
    assert "4111111111111111" not in held.items[0].text


async def test_nothing_withheld_is_not_a_statement_that_nothing_is_there():
    """The load-bearing sentence. A caller who reads "0 withheld" as "this
    is clean" has been misled by a report that was technically true."""
    engine = await memory("The crane was repainted in May.")
    try:
        held = withhold(await found(engine, "crane repainted"))
    finally:
        await engine.close()
    assert held.withheld == 0
    assert "not a finding" in held.why, held.why
    assert "net" in held.why or "patterns" in held.why, held.why


async def test_the_caller_chooses_which_kinds_are_withheld():
    engine = await memory("Write to ana.alves@meridian-health.example about AKIAIOSFODNN7EXAMPLE.")
    try:
        items = await found(engine, "write about")
        only = withhold(items, kinds=("secret",))
    finally:
        await engine.close()
    assert only.by_kind.get("secret") == 1 and "email" not in only.by_kind, only.by_kind
    assert "ana.alves@meridian-health.example" in only.items[0].text, \
        "a kind the caller did not ask for is not withheld behind their back"
    assert "secret" in only.kinds_applied and "email" not in only.kinds_applied


async def test_a_kind_nobody_defined_is_refused():
    engine = await memory("Anything at all.")
    try:
        items = await found(engine, "anything")
        with pytest.raises(InvalidInput):
            withhold(items, kinds=("astrology",))
        assert set(KINDS) >= {"email", "secret", "card", "phone", "ip"}
    finally:
        await engine.close()


async def test_withholding_keeps_the_caller_s_ranking_and_identities():
    engine = await memory("Mail ana.alves@meridian-health.example now.", "An unrelated note.")
    try:
        items = await found(engine, "mail unrelated note")
        held = withhold(items)
    finally:
        await engine.close()
    assert [i.chunk_id for i in held.items] == [i.chunk_id for i in items]
    assert [i.episode_id for i in held.items] == [i.episode_id for i in items]


async def test_an_address_is_withheld_from_every_surface_an_item_carries():
    """A passage is not the only place an item says something.

    An address put in as the source, a tag and a metadata value came back
    untouched while the report said one match and nothing unscanned --
    which reads as "this answer was covered" and was not. Scrubbing the
    prose and handing the same address back in the next field is not
    withholding, it is moving it.
    """
    address = "ana.alves@meridian-health.example"
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                HashEmbedder()).open()
    try:
        await engine.remember("default", f"Write to {address} about the rota.",
                              source=address, tags=(address,),
                              metadata={"contact": address, "owner": address})
        held = withhold(await found(engine, "write rota"))
    finally:
        await engine.close()
    [one] = [item for item in held.items if "rota" in item.text]
    assert address not in one.text
    assert address != one.source and address not in (one.source or "")
    assert not [tag for tag in one.tags if address in tag], one.tags
    assert not [value for value in one.metadata.values() if address in value], one.metadata
    # Five places held it -- the prose, the source, the tag and two
    # metadata values -- so five is the count. A count of one would say
    # the other four were clean. (A metadata *key* cannot hold an address:
    # keys are validated to [a-z][a-z0-9_]{0,31} on the way in.)
    assert held.by_kind["email"] == 5, held.record()
    assert "text" in held.surfaces and "metadata" in held.surfaces, held.surfaces


async def test_the_report_names_the_surfaces_it_scanned():
    """`unscanned: 0` is a claim about coverage, so what was covered has to
    be on the record beside it rather than left to be assumed."""
    engine = await memory("Nothing of interest here.")
    try:
        held = withhold(await found(engine, "nothing interest"))
    finally:
        await engine.close()
    assert held.surfaces == ("text", "source", "tags", "metadata"), held.surfaces
    assert "text, source, tags, metadata" in held.why, held.why
