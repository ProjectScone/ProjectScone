"""A recalled function body, with the two things that make it readable.

A chunk of code says which declaration it came from -- `Engine.forget` --
and that is where our answer stopped. It did not say the declaration's
signature, so a caller saw a body without its parameters, and it did not
say what the file imported, so a name in the body could not be traced to
where it came from. For "how is this done here", a body without its
signature and its imports is a fragment.

The reference that has this prepends the context into the chunk text.
Ours must not: invariant I1 says `content[span.start:span.end]` is the
source unchanged. So the context sits beside the chunk, quoted from the
source with line numbers, and a caller can check every line of it
against the file.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.code_context import MAX_IMPORTS, code_context

pytestmark = pytest.mark.asyncio

MODULE = '''"""A shelf of papers."""

from __future__ import annotations

import json
from pathlib import Path

from .base import Store


class Shelf(Store):
    """Holds papers."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def keep(
        self,
        paper: str,
        *,
        tag: str = "unsorted",
    ) -> Path:
        """Write a paper to the shelf and return where it went.

        The body is deliberately long enough that a chunk of it does not
        also contain the signature above, which is the whole point of
        this fixture: the caller gets the body and needs the head.
        """
        target = self._root / f"{tag}.json"
        payload = json.dumps({"paper": paper, "tag": tag})
        target.write_text(payload, encoding="utf-8")
        for _ in range(3):
            target.touch()
        return target
'''


async def memory(content=MODULE, source="pkg/shelf.py", target=200):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                               chunk_target=target).open()
    await engine.remember("default", content, source=source)
    return engine


async def test_a_recalled_body_is_given_its_signature_and_its_imports():
    engine = await memory()
    try:
        found = await engine.recall("default", "write a paper to the shelf payload dumps", limit=3)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    assert context.by_chunk, context.record()
    holding = [c for c in context.by_chunk.values()
               if any("keep" in s.name for s in c.holders)]
    assert holding, {k: [s.name for s in v.holders] for k, v in context.by_chunk.items()}
    one = holding[0]
    signature = next(s for s in one.holders if "keep" in s.name)
    # The whole signature, not its first line: `def keep(` alone says
    # nothing about the parameters, which is what a caller wants.
    assert "def keep(" in signature.text, signature.text
    assert "tag: str = \"unsorted\"" in signature.text, signature.text
    assert signature.text in MODULE, "the signature is quoted, never assembled"
    assert [i.text for i in one.imports] == [
        "from __future__ import annotations", "import json",
        "from pathlib import Path", "from .base import Store"], [i.text for i in one.imports]
    for line in one.imports:
        assert MODULE.splitlines()[line.line - 1] == line.text, (line.line, line.text)


async def test_the_holders_run_outermost_first():
    engine = await memory()
    try:
        found = await engine.recall("default", "write a paper to the shelf payload dumps", limit=3)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    chains = [[s.name for s in c.holders] for c in context.by_chunk.values() if c.holders]
    assert any(chain == ["Shelf", "Shelf.keep"] for chain in chains), chains


async def test_prose_is_not_given_code_context():
    """A file called notes.md holding a code block is prose that quotes
    code. Nothing is guessed from the content."""
    engine = await memory("Some notes about json and Path.\n\n" + MODULE, source="notes.md")
    try:
        found = await engine.recall("default", "notes json path", limit=3)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    assert context.by_chunk == {}, context.record()
    assert context.not_code == len(found.items), context.record()
    assert "does not say" in context.why, context.why


async def test_a_file_with_more_imports_than_the_bound_says_so():
    """A count a reader could mistake for the file's imports must not be
    a count of the ones we listed."""
    lines = "".join(f"import module{n}\n" for n in range(40))
    engine = await memory(lines + "\n\ndef work():\n    return module1\n", source="many.py")
    try:
        found = await engine.recall("default", "work return module", limit=3)
        context = await code_context(engine, "default", found.items, imports=5)
    finally:
        await engine.close()
    assert context.by_chunk, context.record()
    one = next(iter(context.by_chunk.values()))
    assert len(one.imports) == 5 and one.more_imports is True, one
    assert context.capped == len(context.by_chunk), context.record()
    assert "not all of them" in context.why, context.why
    assert MAX_IMPORTS > 5


async def test_a_source_that_is_gone_is_not_answered_from_stale_text():
    engine = await memory()
    try:
        found = await engine.recall("default", "write a paper shelf", limit=2)
        await engine.forget("default", found.items[0].episode_id)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    assert context.by_chunk == {} and context.gone > 0, context.record()
    assert "no longer there" in context.why, context.why


async def test_the_episode_budget_is_checked_and_bounds_the_reads():
    engine = await memory()
    try:
        found = await engine.recall("default", "write a paper shelf", limit=2)
        for bad in (0, -1, 1.5, True, "2"):
            with pytest.raises(InvalidInput):
                await code_context(engine, "default", found.items, episodes=bad)
        for bad in (0, -1, 1.5, True):
            with pytest.raises(InvalidInput):
                await code_context(engine, "default", found.items, imports=bad)
    finally:
        await engine.close()


async def test_the_cli_can_ask_for_it_and_does_not_by_default():
    """A feature nobody can reach is not a feature. `--code-context` is
    the reachable surface, and it stays opt-in like `--merge` and
    `--window` beside it."""
    import io

    from scone_memory.runtime.cli import build_parser, run

    engine = await memory()
    try:
        asked = ["recall", "write a paper to the shelf payload dumps", "--code-context",
                 "--limit", "3"]
        out = io.StringIO()
        code = await run(build_parser().parse_args(asked), engine, io.StringIO(""), out)
        shown = out.getvalue()
        assert code == 0, shown
        assert "inside Shelf.keep" in shown, shown
        assert "from pathlib import Path" in shown, shown
        plain = io.StringIO()
        code = await run(build_parser().parse_args(asked[:2] + ["--limit", "3"]),
                         engine, io.StringIO(""), plain)
        assert code == 0 and "from pathlib import Path" not in plain.getvalue(), \
            "code context stays opt-in"
    finally:
        await engine.close()


async def test_a_bracket_inside_a_string_does_not_extend_the_signature():
    """The reasoning in the first version of this module was wrong, and
    its comment said so confidently.

    It scanned for a `:` at bracket depth zero, justified because a colon
    in a default argument is inside brackets. So it is -- but a bracket
    inside a *string* is inside nothing: `def f(value="("):` never
    reaches depth zero at its own colon and kept reading the body, and
    `def f(value=")"):` drove the depth negative. The parser already
    knows where a body starts.
    """
    source = (
        'import json\n'
        '\n'
        '\n'
        'def opener(value="(", tag="x"):\n'
        '    """Doc."""\n'
        '    return json.dumps({"value": value, "tag": tag}) + "a longer body so the chunk"\n'
        '\n'
        '\n'
        'def closer(value=")"):\n'
        '    return value + "another body long enough to be its own chunk here"\n')
    engine = await memory(source, source="brackets.py", target=60)
    try:
        found = await engine.recall("default", "json dumps value tag return", limit=6)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    names = {}
    for one in context.by_chunk.values():
        for holder in one.holders:
            names[holder.name] = holder
    assert "opener" in names or "closer" in names, context.record()
    for name, holder in names.items():
        assert holder.text.count("\n") == 0, (name, holder.text)
        assert holder.text.rstrip().endswith(":"), (name, holder.text)
        assert not holder.clipped, (name, holder.text)
        assert holder.text in source, "the signature is quoted, never assembled"


async def test_a_brace_inside_a_string_does_not_extend_the_signature():
    source = (
        'import {thing} from "./thing";\n'
        '\n'
        'class Shelf {\n'
        '  keep(tag = "{") {\n'
        '    return thing(tag) + "a body long enough that the chunk is not the header";\n'
        '  }\n'
        '}\n')
    engine = await memory(source, source="shelf.ts", target=60)
    try:
        found = await engine.recall("default", "keep thing tag return body", limit=6)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    holders = [h for one in context.by_chunk.values() for h in one.holders]
    assert holders, context.record()
    for holder in holders:
        assert holder.text.count("\n") <= 1, holder.text
        assert not holder.clipped, holder.text


async def test_a_signature_longer_than_the_bound_says_it_was_cut():
    """A bound that bit with nothing saying so is the fault this codebase
    keeps making, and `clipped` existed while reaching no reader."""
    parameters = "".join(f"    argument_{n}: str = \"value\",\n" for n in range(20))
    source = ("import json\n\n\ndef wide(\n" + parameters + ") -> str:\n"
              "    return json.dumps({})\n")
    engine = await memory(source, source="wide.py", target=120)
    try:
        found = await engine.recall("default", "argument value json dumps return", limit=8)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    holders = [h for one in context.by_chunk.values() for h in one.holders]
    assert holders and any(h.clipped for h in holders), context.record()
    assert context.shortened >= 1, context.record()
    assert "incomplete" in context.why, context.why
    # The receipt has to carry it, not only the object.
    said = context.record()
    assert any(h["clipped"] for one in said["by_chunk"].values() for h in one["holders"]), said


async def test_a_forgotten_source_is_dropped_from_the_items_as_well():
    """Counting an item as dropped while handing it back is how stale
    text gets served under a clean reason. The first version of this
    removed the context entry and left the item, and the test only looked
    at `by_chunk`."""
    engine = await memory()
    try:
        found = await engine.recall("default", "write a paper shelf", limit=3)
        await engine.forget("default", found.items[0].episode_id)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    assert context.gone == len(found.items), context.record()
    assert context.items == (), "an item whose source is gone is not handed on"
    assert [item["chunk_id"] for item in context.record()["items"]] == []


async def test_the_cli_refuses_withholding_beside_anything_that_quotes_the_source_again():
    """The hole a review found in the CLI after I had fixed the HTTP
    route: `--code-context` and `--merge` both re-read the episode from
    the store *after* withholding and print source verbatim, so a
    withheld address in a default argument came back in the signature,
    and `--parts` answered without withholding at all."""
    import io

    from scone_memory.runtime.cli import build_parser, run

    address = "alice@example.test"
    source = ("import json\n\n\ndef contact(email=\"" + address + "\"):\n"
              "    return json.dumps({\"email\": email}) + \" a body long enough to be a chunk\"\n")
    engine = await memory(source, source="contact.py", target=60)
    try:
        for clash in ("--code-context", "--merge", "--parts"):
            out = io.StringIO()
            asked = ["recall", "contact email json dumps", "--withhold", "email", clash]
            with pytest.raises(InvalidInput) as raised:
                await run(build_parser().parse_args(asked), engine, io.StringIO(""), out)
            assert clash in str(raised.value), (clash, str(raised.value))
            assert address not in out.getvalue(), (clash, out.getvalue())
        # Each of them is fine on its own.
        for alone in (["--withhold", "email"], ["--code-context"]):
            out = io.StringIO()
            code = await run(build_parser().parse_args(
                ["recall", "contact email json dumps"] + alone), engine, io.StringIO(""), out)
            assert code == 0, out.getvalue()
        held = io.StringIO()
        await run(build_parser().parse_args(
            ["recall", "contact email json dumps", "--withhold", "email"]),
            engine, io.StringIO(""), held)
        assert address not in held.getvalue(), held.getvalue()
    finally:
        await engine.close()


async def test_the_cli_withholds_from_facts_and_history_too():
    """Fixed on the HTTP route and not here, which left the same address
    reachable through the other door."""
    import io

    from scone_memory.runtime.cli import build_parser, run

    address = "ana.alves@meridian-health.example"
    engine = await memory("Notes about the rota and the yard.", source="notes.txt")
    try:
        await engine.assert_fact("default", subject="Ana", predicate="email", object=address)
        out = io.StringIO()
        code = await run(build_parser().parse_args(
            ["recall", "Ana rota email yard", "--withhold", "email"]),
            engine, io.StringIO(""), out)
        shown = out.getvalue()
        assert code == 0, shown
        assert "fact" in shown, "this test needs a fact in the answer to be about anything"
        assert address not in shown, shown
    finally:
        await engine.close()


async def test_a_decorated_and_commented_declaration_still_quotes_its_signature():
    """A declaration's span starts at its decorators and at the comment
    lines written directly above it, so `first_line` is earlier than the
    `def` the parser reports. Keying the header map on `first_line`
    returned the comment line alone -- caught while writing the fix for
    the bracket bug, not by a review."""
    source = (
        'import json\n'
        '\n'
        '\n'
        '# Keeps a paper on the shelf under a tag.\n'
        '@staticmethod\n'
        'def keep(paper: str, tag: str = "unsorted") -> str:\n'
        '    """Doc."""\n'
        '    return json.dumps({"paper": paper, "tag": tag}) + " and a body long enough"\n')
    engine = await memory(source, source="decorated.py", target=60)
    try:
        found = await engine.recall("default", "paper tag json dumps body", limit=6)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    holders = [h for one in context.by_chunk.values() for h in one.holders]
    assert holders, context.record()
    signature = holders[0]
    assert "def keep(paper: str" in signature.text, signature.text
    assert signature.text in source, "quoted, never assembled"
    assert not signature.clipped, signature.text


async def test_the_json_widened_receipt_does_not_return_what_withholding_removed():
    """Found by re-reading my own order of operations after a review found
    the same class of hole three times.

    `--window` runs before `--withhold` so the widened text is scrubbed,
    which is right -- but `Widened.record()` carries its own copy of the
    passages from *before* withholding, and the JSON output emitted it
    whole. The address came back under `widened.items`.
    """
    import io
    import json as jsonlib

    from scone_memory.runtime.cli import build_parser, run

    address = "ana.alves@meridian-health.example"
    engine = await memory(f"Write to {address} about the crane survey and the rota.",
                          source="note.txt", target=40)
    try:
        out = io.StringIO()
        code = await run(build_parser().parse_args(
            ["recall", "write crane survey rota", "--window", "200",
             "--withhold", "email", "--json"]), engine, io.StringIO(""), out)
        shown = out.getvalue()
        assert code == 0, shown
        assert address not in shown, shown
        said = jsonlib.loads(shown)
        assert "widened" in said and "items" not in said["widened"], said["widened"]
        assert "items_not_repeated" in said["widened"], said["widened"]
        assert said["withheld"]["by_kind"].get("email"), said["withheld"]
        # The counts are the useful part of a receipt and they stay.
        assert said["widened"]["widened"] == 1, said["widened"]
    finally:
        await engine.close()


async def test_the_signature_ends_at_its_own_colon_and_not_at_the_body():
    """Four boundary shapes a review named after my second attempt, all of
    which `ast` body-start-minus-one gets wrong.

    `ast` reports where a declaration *begins*, not where its header
    ends. Leading comments and blank lines in a body are not statements;
    a class whose first method carries a decorator takes that decorator
    into the class header, because `FunctionDef.lineno` points at the
    `def`; and a one-line suite returns its body as its signature --
    which in a withholding answer is the leak shape again.
    """
    source = (
        'import json\n'
        '\n'
        '\n'
        'class Holder:\n'
        '    # A comment that is not a statement, so ast does not see it.\n'
        '\n'
        '    @staticmethod\n'
        '    def keep(paper: str) -> str:\n'
        '        # Another non-statement, before anything runs.\n'
        '\n'
        '        return json.dumps({"paper": paper}) + " with a body long enough to split"\n'
        '\n'
        '\n'
        'def terse(): return json.dumps({"secret": "alice@example.test"})\n')
    engine = await memory(source, source="shapes.py", target=60)
    try:
        found = await engine.recall("default", "paper json dumps secret keep holder", limit=8)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    got = {}
    for one in context.by_chunk.values():
        for holder in one.holders:
            got[holder.name] = holder
    assert got, context.record()

    if "Holder" in got:
        text = got["Holder"].text
        assert text == "class Holder:", text
        assert "@staticmethod" not in text, "the first method's decorator is not the class header"
        assert "comment" not in text, "a body comment is not part of the header"
        assert not got["Holder"].clipped, text
    if "Holder.keep" in got:
        text = got["Holder.keep"].text
        assert text.strip() == "def keep(paper: str) -> str:", text
        assert "Another non-statement" not in text, text
        assert not got["Holder.keep"].clipped, text
    if "terse" in got:
        text = got["terse"].text
        # The header ends at the colon. Returning the line whole would
        # hand back the body -- and the body is where the address is.
        assert text == "def terse():", text
        assert "alice@example.test" not in text, "the body is not the signature"

    for name, holder in got.items():
        assert holder.text in source, (name, "quoted, never assembled")


async def test_an_async_signature_keeps_its_keyword():
    source = (
        'import json\n'
        '\n'
        '\n'
        'async def fetch(url: str, *, retries: int = 3) -> str:\n'
        '    return json.dumps({"url": url, "retries": retries}) + " a long enough body here"\n')
    engine = await memory(source, source="fetching.py", target=50)
    try:
        found = await engine.recall("default", "url retries json dumps body", limit=6)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    holders = [h for one in context.by_chunk.values() for h in one.holders]
    assert holders, context.record()
    text = holders[0].text
    assert text.startswith("async def fetch("), text
    assert text.rstrip().endswith(":"), text
    assert text in source


async def test_a_colon_inside_a_dict_default_does_not_end_the_signature():
    source = (
        'import json\n'
        '\n'
        '\n'
        'def keep(tags: dict = {"a": 1, "b": 2}, label: str = "x:y") -> str:\n'
        '    return json.dumps(tags) + label + " and a body long enough to be its own chunk"\n')
    engine = await memory(source, source="colons.py", target=50)
    try:
        found = await engine.recall("default", "tags label json dumps body chunk", limit=6)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    holders = [h for one in context.by_chunk.values() for h in one.holders]
    assert holders, context.record()
    text = holders[0].text
    assert text == 'def keep(tags: dict = {"a": 1, "b": 2}, label: str = "x:y") -> str:', text


async def test_a_one_line_brace_body_is_not_part_of_the_signature():
    """The tenth instance of the same half-fix: I made the Python path
    character-precise so `def f(): return secret` stopped returning its
    body, and left the brace path counting whole lines -- so
    `function contact() { return secret; }` still did."""
    # The body is on the header's line *and* long enough that the line is
    # its own chunk, so `contact` is unavoidably the holder. My first
    # version of this fixture put a short body on that line; the chunk
    # was never recalled, `contact` was never a holder, and the
    # assertion sat behind `if "contact" in got:` and never ran. A
    # mutation proof is what exposed that -- the test passed with the fix
    # removed, which is the only reason I looked.
    source = (
        'import {thing} from "./thing";\n'
        '\n'
        'function contact() { return "alice@example.test" + " padded out with a good deal '
        'more text so that this single line is comfortably a chunk of its own"; }\n'
        '\n'
        'function other(tag) {\n'
        '  return thing(tag) + " a body long enough that the chunk is not the header";\n'
        '}\n')
    engine = await memory(source, source="contact.ts", target=60)
    try:
        found = await engine.recall("default", "contact padded text single line chunk", limit=8)
        context = await code_context(engine, "default", found.items)
    finally:
        await engine.close()
    got = {h.name: h for one in context.by_chunk.values() for h in one.holders}
    assert "contact" in got, ["the fixture must make contact a holder", sorted(got)]
    assert got["contact"].text == "function contact() {", got["contact"].text
    assert "alice@example.test" not in got["contact"].text, "the body is not the signature"
    assert not got["contact"].clipped, got["contact"].text
    for name, holder in got.items():
        assert holder.text in source, (name, "quoted, never assembled")


async def test_no_stage_receipt_repeats_the_passages_even_with_no_policy():
    """The rule, not one instance of it.

    I first dropped the receipt's copy of the text only when a
    withholding policy was active, which treats a symptom: a review then
    deleted an episode between widening's read and code context's read,
    and the widened receipt printed the deleted passage while the answer
    itself correctly returned nothing. Every stage replaces the answer's
    items with its own output, so a receipt's copy is always redundant
    with `items` and always older than it -- and is dropped from every
    stage, with no flag involved.
    """
    import io
    import json as jsonlib

    from scone_memory.runtime.cli import build_parser, run

    engine = await memory()
    try:
        out = io.StringIO()
        code = await run(build_parser().parse_args(
            ["recall", "write a paper to the shelf payload", "--window", "80",
             "--code-context", "--json"]), engine, io.StringIO(""), out)
        shown = out.getvalue()
        assert code == 0, shown
        said = jsonlib.loads(shown)
        for stage in ("widened", "code_context"):
            assert stage in said, (stage, sorted(said))
            assert "items" not in said[stage], (stage, said[stage])
            assert "items_not_repeated" in said[stage], (stage, said[stage])
        # and the answer still carries them exactly once
        assert said["items"], said.keys()
    finally:
        await engine.close()


async def test_a_line_separator_in_a_string_does_not_move_the_imports_either():
    """The same fault in a third place. `str.splitlines()` breaks on
    U+2028, U+2029, U+0085, vertical tab and form feed; the parser's line
    numbers do not. A banner constant holding one shifted every line
    after it, so a signature -- and, in a third copy of the same table,
    an import -- was quoted from the wrong place.

    Each of the three copies had to be fixed separately, which is the
    argument for one table rather than three.
    """
    for separator in ("\u2028", "\u0085", "\v"):
        for newline in ("\n", "\r\n"):
            source = newline.join([
                'banner = "one' + separator + 'two"',
                'import json',
                'from pathlib import Path',
                '',
                '',
                'def contact(tag: str = "x") -> str:',
                '    return json.dumps({"tag": tag, "where": str(Path("."))}) + banner',
                ''])
            engine = await memory(source, source="banners.py", target=60)
            try:
                found = await engine.recall("default", "tag json dumps where path banner",
                                            limit=6)
                context = await code_context(engine, "default", found.items)
            finally:
                await engine.close()
            one = next(iter(context.by_chunk.values()), None)
            assert one is not None, (repr(separator), repr(newline), context.record())
            assert [i.text for i in one.imports] == ["import json", "from pathlib import Path"], \
                (repr(separator), repr(newline), [i.text for i in one.imports])
            for holder in one.holders:
                assert holder.text == 'def contact(tag: str = "x") -> str:', \
                    (repr(separator), repr(newline), holder.text)


async def test_code_context_describes_only_passages_that_survived_every_stage():
    """Code context quotes the file -- a signature, an import line -- so a
    stage running *after* it can find that source deleted and leave the
    answer empty while the context still prints the text. Dropping the
    receipt's copy of the items cannot fix that: the signature is a
    separate copy of the same source.

    So it runs last, and the invariant is structural: every chunk the
    context describes is a chunk the answer returns.
    """
    import io
    import json as jsonlib

    from scone_memory.runtime.cli import build_parser, run

    engine = await memory(target=90)
    try:
        out = io.StringIO()
        code = await run(build_parser().parse_args(
            ["recall", "write a paper to the shelf payload target", "--merge",
             "--code-context", "--json", "--limit", "6"]), engine, io.StringIO(""), out)
        shown = out.getvalue()
        assert code == 0, shown
        said = jsonlib.loads(shown)
        answered = {item["chunk_id"] for item in said["items"]}
        described = {int(key) for key in said["code_context"]["by_chunk"]}
        assert described <= answered, (sorted(described), sorted(answered))
    finally:
        await engine.close()
