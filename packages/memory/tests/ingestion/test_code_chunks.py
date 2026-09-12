"""Code cut at its own declarations, over unchanged source bytes."""

from scone_memory.ingestion.chunker import chunk_spans
from scone_memory.ingestion.code import (
    MAX_DEPTH,
    code_language,
    code_spans,
    declaration_at,
    declarations,
)

MODULE = '''"""A module that does a thing."""

from __future__ import annotations

LIMIT = 12


def plan(question: str) -> str:
    """Work out what is being asked."""
    return question.strip()


class Engine:
    """Holds the parts."""

    def recall(self, query: str) -> list[str]:
        """Answer from memory."""

        def rank(item: str) -> int:
            return len(item)

        return sorted([query], key=rank)
'''


def test_a_declaration_is_named_by_everything_that_holds_it():
    named = {d.name: d for d in declarations(MODULE, language="python")}
    assert set(named) == {"plan", "Engine", "Engine.recall", "Engine.recall.rank"}
    assert named["plan"].kind == "function"
    assert named["Engine"].kind == "class"
    assert named["Engine.recall"].kind == "method"


def test_a_declaration_span_is_exactly_its_own_source():
    [plan] = [d for d in declarations(MODULE, language="python") if d.name == "plan"]
    text = MODULE[plan.start : plan.end]
    assert text.startswith("def plan(question: str) -> str:")
    assert text.rstrip().endswith("return question.strip()")
    assert MODULE.count(text) == 1
    assert plan.first_line == MODULE[: plan.start].count("\n") + 1
    assert plan.last_line == plan.first_line + text.rstrip().count("\n")


def test_a_decorator_belongs_to_what_it_decorates():
    source = "import functools\n\n\n@functools.cache\ndef once(x: int) -> int:\n    return x\n"
    [only] = [d for d in declarations(source, language="python") if d.name == "once"]
    assert source[only.start : only.end].startswith("@functools.cache\ndef once")


def test_every_span_starts_and_ends_on_a_line_and_covers_the_source():
    spans = code_spans(MODULE, target=200, language="python")
    seen = [0] * len(MODULE)
    for span in spans:
        assert span.start == 0 or MODULE[span.start - 1] == "\n"
        assert span.end == len(MODULE) or MODULE[span.end - 1] == "\n"
        for i in range(span.start, span.end):
            seen[i] += 1
    assert all(n == 1 for i, n in enumerate(seen) if not MODULE[i].isspace())
    assert all(n <= 1 for n in seen)


def test_a_function_smaller_than_the_target_is_never_cut_in_half():
    head = "".join(f"from pkg.module_{i} import thing_{i}\n" for i in range(12))
    body = ("def plan(question: str) -> str:\n"
            + "".join(f'    step_{i} = "{i}"\n' for i in range(8)) + "    return question\n")
    source = head + body
    target = len(head) + 40
    inside = {s.start for s in chunk_spans(source, target)} - {0}
    assert any(place > len(head) for place in inside), "the ordinary chunker cuts inside the function"
    spans = code_spans(source, target, language="python")
    [holds] = [source[s.start : s.end] for s in spans if "def plan" in source[s.start : s.end]]
    assert holds.startswith("def plan") and holds.rstrip().endswith("return question")


def test_a_span_says_which_declarations_it_holds():
    spans = code_spans(MODULE, target=200, language="python")
    assert {n for s in spans for n in s.names} == {"plan", "Engine"}


def test_a_declaration_longer_than_the_target_is_split_at_its_own_lines():
    body = "\n".join(f"    step_{i} = {i}" for i in range(200))
    source = f"def long_one() -> None:\n{body}\n"
    spans = code_spans(source, target=400, language="python")
    assert len(spans) > 1
    assert all(s.names == ("long_one",) for s in spans), [s.names for s in spans]
    assert all(source[s.start : s.end].endswith("\n") for s in spans)
    assert "".join(source[s.start : s.end] for s in spans) == source


def test_small_declarations_share_a_chunk_up_to_the_target():
    source = "".join(f"def f{i}() -> int:\n    return {i}\n\n\n" for i in range(40))
    spans = code_spans(source, target=400, language="python")
    assert 1 < len(spans) < 40
    assert all(len(source[s.start : s.end]) <= 400 for s in spans)
    assert all(len(s.names) > 1 for s in spans), "small neighbours share a chunk"


def test_what_a_recalled_span_came_from():
    spans = code_spans(MODULE, target=200, language="python")
    inner = MODULE.index("def rank(")
    found = declaration_at(MODULE, inner, inner + 10, language="python")
    assert found is not None and found.name == "Engine.recall.rank"
    top = MODULE.index("LIMIT = 12")
    assert declaration_at(MODULE, top, top + 5, language="python") is None
    assert spans, "spans are what a chunk is cut from"


def test_a_span_is_named_by_the_declaration_that_holds_all_of_it():
    start = MODULE.index("class Engine:")
    found = declaration_at(MODULE, start, len(MODULE), language="python")
    assert found is not None and found.name == "Engine"


def test_byte_offsets_name_the_same_declaration_when_the_source_is_not_ascii():
    source = 'X = "café ☕"\n\n\ndef greet() -> str:\n    return "buenos días"\n'
    where = source.encode().index(b"return")
    found = declaration_at(source, where, where + 6, language="python", offsets="bytes")
    assert found is not None and found.name == "greet"


def test_source_that_does_not_parse_is_cut_the_ordinary_way():
    broken = "def half(:\n  this is not python at all\n" + "filler line\n" * 80
    assert declarations(broken, language="python") == ()
    assert [(s.start, s.end) for s in code_spans(broken, target=300, language="python")] == [
        (s.start, s.end) for s in chunk_spans(broken, 300)
    ]


def test_a_language_is_taken_from_the_name_it_was_stored_under():
    assert code_language("src/scone_memory/retrieval/temporal.py") == "python"
    assert code_language("Webapp/src/api.ts") == "braces"
    assert code_language("crates/scone-core/src/lib.rs") == "braces"
    assert code_language("notes/2026-09-11.md") is None
    assert code_language(None) is None


def test_braces_declarations_end_where_their_body_closes():
    source = (
        "import { thing } from './thing'\n\n"
        "export function alpha(n: number): number {\n"
        "  if (n > 0) {\n    return n\n  }\n  return 0\n}\n\n"
        "class Beta {\n  gamma() {\n    return 1\n  }\n}\n"
    )
    named = {d.name: d for d in declarations(source, language="braces")}
    assert set(named) >= {"alpha", "Beta", "Beta.gamma"}
    assert source[named["alpha"].start : named["alpha"].end].rstrip().endswith("}")
    assert "class Beta" not in source[named["alpha"].start : named["alpha"].end]


def test_braces_code_is_cut_at_its_declarations():
    source = "".join(f"func handler{i}(w http.ResponseWriter) {{\n\treturn\n}}\n\n" for i in range(10))
    spans = code_spans(source, target=200, language="braces")
    assert len(spans) > 1
    assert all(s.start == 0 or source[s.start - 1] == "\n" for s in spans)
    assert "".join(source[s.start : s.end] for s in spans).replace("\n", "") == source.replace("\n", "")


class _Counting(str):
    """A string that says how much of itself has been copied, so that work
    can be asserted on rather than timed."""

    copied = 0

    def __getitem__(self, item):
        got = str.__getitem__(self, item)
        _Counting.copied += len(got)
        return got


def test_a_lone_carriage_return_is_a_line_like_any_other():
    """Python reads \\r as a line ending, so a line index that counts only
    \\n has every declaration in the wrong place."""
    source = "def plan():\r    return 1\r\r\rdef widen():\r    return 2\r"
    named = {d.name: d for d in declarations(source, language="python")}
    assert set(named) == {"plan", "widen"}
    assert source[named["plan"].start : named["plan"].end].startswith("def plan():")
    assert source[named["widen"].start : named["widen"].end].startswith("def widen():")
    assert named["widen"].first_line == 5


def test_windows_line_endings_keep_their_own_offsets():
    source = "def plan():\r\n    return 1\r\n\r\n\r\ndef widen():\r\n    return 2\r\n"
    named = {d.name: d for d in declarations(source, language="python")}
    assert set(named) == {"plan", "widen"}
    assert source[named["widen"].start : named["widen"].end] == "def widen():\r\n    return 2\r\n"
    spans = code_spans(source, target=20, language="python")
    assert all(source[s.start : s.end] for s in spans)
    assert "".join(source[s.start : s.end] for s in spans).replace("\r\n", "") == source.replace("\r\n", "")


def test_a_deeply_nested_expression_does_not_exhaust_the_stack():
    """ast parses a 1200-term sum happily; walking it must too."""
    source = "def total() -> int:\n    return " + " + ".join(str(i) for i in range(1200)) + "\n"
    assert [d.name for d in declarations(source, language="python")] == ["total"]


def test_deeply_nested_declarations_are_read_to_a_bounded_depth():
    source = "".join(f"function f{i}() {{\n" for i in range(1050)) + "}\n" * 1050
    found = declarations(source, language="braces")
    assert found and found[0].name == "f0"
    assert max(d.name.count(".") for d in found) < 100, "nesting is read to a bounded depth"


def test_python_nesting_is_read_to_a_bounded_depth_too():
    source = "".join("    " * i + f"def f{i}():\n" for i in range(40)) + "    " * 40 + "pass\n"
    found = declarations(source, language="python")
    assert found and found[0].name == "f0"
    assert max(d.name.count(".") for d in found) < MAX_DEPTH


def test_source_nested_deeper_than_python_itself_reads_is_cut_the_ordinary_way():
    """Python refuses more than a hundred levels of indentation. What its
    own parser will not read, this does not pretend to read either."""
    source = "".join("    " * i + f"def f{i}():\n" for i in range(300)) + "    " * 300 + "pass\n"
    assert declarations(source, language="python") == ()
    assert [(s.start, s.end) for s in code_spans(source, target=300, language="python")] == [
        (s.start, s.end) for s in chunk_spans(source, 300)
    ]


def test_cutting_a_long_file_does_not_copy_it_again_for_every_line():
    body = "".join(f"    step_{i} = {i}\n" for i in range(4000))
    source = _Counting("def one() -> None:\n" + body)
    _Counting.copied = 0
    code_spans(source, target=700, language="python")
    assert _Counting.copied < 20 * len(source), f"{_Counting.copied} copied of {len(source)}"
