"""Cutting a document where it already divides itself.

Our chunker prefers a paragraph break, then a sentence end, then any
whitespace, then a hard cut inside a word. It knows nothing about
headings or numbered clauses, so a 700-character cut lands wherever the
prose allows: "Article 7.2" can end one chunk while the clause it names
begins the next. The number is the query term and the chunk that answers
cannot say what it is.

ragflow ships a chunker per document type for this. Ours reads the
structure the document already carries, which needs no model, no new
dependency and no choice made at ingestion about what kind of document
this is.

Two rules keep it honest rather than dangerous:

- **A document with no structure must chunk exactly as it does today.**
  Chunk boundaries decide what chunks exist and stored offsets are part
  of the shared specification, so a silent change here is a change to
  every future span.
- **A section longer than the target still gets split**, and the receipt
  says the cut was inside a unit rather than at a boundary. Otherwise
  "structure-aware" reads as a promise it cannot keep.
"""

from __future__ import annotations

import pytest

from scone_memory.ingestion.chunker import DEFAULT_TARGET, chunk_spans
from scone_memory.ingestion.structure_chunks import MAX_SECTIONS, structured_spans, units

pytestmark = pytest.mark.asyncio

PROSE = (
    "The survey of the harbour crane was booked for the third of May. "
    "It found rust on the jib and a slew ring that needed grease. "
    "The yard did both in the same week, and the crane returned to service on the first of June. "
    "Nobody recorded who signed the handover, which became the subject of a later complaint. "
    "The complaint was settled in August after the yard produced its own record of the work. "
    "That record named a fitter who had left the company in April, so it settled nothing. ")


def text_of(content, spans):
    return [content[s.start:s.end] for s in spans]


def test_a_document_with_no_structure_chunks_exactly_as_before():
    """The safety property. If this fails, every future stored offset
    moved, and nothing in the feature is worth that."""
    content = PROSE * 6
    plain = chunk_spans(content)
    found = structured_spans(content)
    assert [(s.start, s.end) for s in found.spans] == [(s.start, s.end) for s in plain]
    assert found.units == 0 and found.at_boundary == 0
    assert found.by_size == len(plain), found.record()
    assert "no structure" in found.why, found.why


def test_a_heading_begins_its_chunk_instead_of_ending_the_one_before():
    content = ("# Handover\n\n" + PROSE + "\n\n## The complaint\n\n" + PROSE
               + "\n\n## The settlement\n\n" + PROSE)
    found = structured_spans(content)
    starts = text_of(content, found.spans)
    assert starts[0].startswith("# Handover"), starts[0][:40]
    assert any(chunk.startswith("## The complaint") for chunk in starts), [c[:24] for c in starts]
    assert any(chunk.startswith("## The settlement") for chunk in starts), [c[:24] for c in starts]
    for chunk in starts:
        assert not chunk.rstrip().endswith("## The complaint"), "a heading trailing its chunk"


def test_a_clause_keeps_its_number():
    """The failure this feature exists for: the number is the query term."""
    content = "".join(f"{n}.{n} Clause heading\n\n{PROSE}\n\n" for n in range(1, 6))
    found = structured_spans(content)
    for chunk in text_of(content, found.spans):
        assert not chunk.rstrip().endswith("Clause heading"), chunk[-60:]
    numbered = [c for c in text_of(content, found.spans) if c.lstrip()[:3] in
                ("1.1", "2.2", "3.3", "4.4", "5.5")]
    assert len(numbered) == 5, [c[:12] for c in text_of(content, found.spans)]
    assert found.units == 5, found.record()


def test_a_year_at_the_start_of_a_line_is_not_a_clause():
    """Structure that is not there must not be invented: `1984 was a year`
    is prose, and reading it as a numbered clause would cut the document
    at a sentence that merely begins with digits."""
    content = "1984 was a year of two audits.\n\n" + PROSE * 3
    assert units(content) == ()
    assert structured_spans(content).units == 0


def test_a_hash_inside_a_fenced_code_block_is_not_a_heading():
    """The mistake the code graph made once already: a comment read as
    code. Here a shell comment would read as a heading and cut the
    example in half."""
    content = ("# Deploying\n\n" + PROSE + "\n\n```bash\n# rebuild the image first\n"
               "docker build .\n# then restart\ndocker compose up -d\n```\n\n" + PROSE)
    found = units(content)
    assert [s.label for s in found] == ["# Deploying"], [s.label for s in found]
    for chunk in text_of(content, structured_spans(content).spans):
        if "docker build" in chunk:
            assert "# rebuild the image first" in chunk, "the example was cut at its comment"


def test_a_section_longer_than_the_target_is_split_and_says_so():
    """A bound that reads as a promise is the fault this codebase keeps
    making: "structure-aware" must not imply "every chunk is a section"."""
    content = "# One very long section\n\n" + PROSE * 8
    found = structured_spans(content, DEFAULT_TARGET)
    assert len(found.spans) > 1, "this fixture is only interesting if the section splits"
    assert found.at_boundary == 1, found.record()
    assert found.by_size == len(found.spans) - 1, found.record()
    assert "inside a unit longer than" in found.why, found.why


def test_small_sections_are_packed_rather_than_each_becoming_a_chunk():
    """One chunk per heading would make a document of short sections into
    a document of tiny chunks, which retrieves worse than the prose did."""
    content = "".join(f"## Note {n}\n\nA short remark about item {n}.\n\n" for n in range(1, 13))
    found = structured_spans(content)
    assert found.units == 12, found.record()
    assert len(found.spans) < 12, [len(found.spans), found.record()]
    assert all(chunk.lstrip().startswith("## Note") for chunk in text_of(content, found.spans))


def test_the_spans_never_overlap_and_drop_only_whitespace():
    """Whatever the structure, the chunks have to account for the
    document: gaps are the whitespace the chunker skips and nothing else."""
    content = ("# Title\n\n" + PROSE + "\n\n2.1 Numbered\n\n" + PROSE * 4
               + "\n\nQ: And a question?\n\nA: An answer.\n\n" + PROSE)
    found = structured_spans(content)
    end = 0
    for span in found.spans:
        assert span.start >= end, (span, end)
        assert content[end:span.start].strip() == "", repr(content[end:span.start])
        assert span.end > span.start
        end = span.end
    assert content[end:].strip() == "", repr(content[end:])


def test_more_sections_than_we_read_is_reported_as_a_bound():
    """A count of what we read must never read as a count of the
    document."""
    content = "".join(f"## Note {n}\n\nRemark {n}.\n\n" for n in range(1, 40))
    found = structured_spans(content, units_max=8)
    assert found.capped is True, found.record()
    assert found.units == 8, found.record()
    assert "not all of them" in found.why, found.why
    assert MAX_SECTIONS > 8


def test_front_matter_is_not_a_heading():
    """The same class of mistake as the fenced comment, found by reading
    the rule rather than by a corpus: YAML front matter ends with `---`,
    which is also a setext underline, so the last key of the front matter
    reads as a heading and the document is cut at its own metadata."""
    content = ("---\ntitle: The handover\nauthor: the yard\n---\n\n# Handover\n\n" + PROSE)
    found = units(content)
    assert [s.label for s in found] == ["# Handover"], [s.label for s in found]


def test_a_rule_between_paragraphs_is_not_a_heading():
    """`---` after a blank line is a horizontal rule. Nothing precedes it
    to be the heading, and reading the paragraph above as one would cut a
    document at every divider."""
    content = PROSE + "\n\n---\n\n" + PROSE
    assert units(content) == ()


async def test_the_engine_cuts_at_headings_only_when_asked():
    """A feature nobody can reach is not a feature, and a default that
    changes every stored span is not an option. Both halves, one test.
    """
    import pytest

    from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex,
                              MemoryEngine)

    content = ("# Handover\n\n" + PROSE + "\n\n## The complaint\n\n" + PROSE
               + "\n\n## The settlement\n\n" + PROSE)

    async def chunks(**options):
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                    HashEmbedder(), **options).open()
        try:
            await engine.remember("default", content, source="handover.txt")
            found = await engine.recall("default", "complaint settlement yard rust", limit=10)
            return [item.text for item in found.items]
        finally:
            await engine.close()

    asked = await chunks(structure_aware=True)
    assert any(text.startswith("## The complaint") for text in asked), [t[:30] for t in asked]
    assert any(text.startswith("## The settlement") for text in asked), [t[:30] for t in asked]

    default = await chunks()
    assert not any(text.startswith("## The complaint") for text in default), \
        "the default must not have changed"


def test_the_dispatch_sends_code_to_declarations_and_prose_to_structure():
    """Structure awareness must not take a source that is code away from
    the declaration-aware path, which is the better cut for code."""
    from scone_memory.ingestion.batch import IngestionRuntime, spans_for
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex

    def runtime(**options):
        return IngestionRuntime(
            documents=InMemoryDocumentStore(), vectors=InMemoryVectorIndex(),
            embedder=HashEmbedder(), clock=lambda: "2026-09-12T00:00:00Z",
            chunk_target=DEFAULT_TARGET, embed_text=lambda episode, text: text,
            emit=None, **options)  # type: ignore[arg-type]

    prose = "# Title\n\n" + PROSE * 2 + "\n\n## Next\n\n" + PROSE * 2
    plain = spans_for(runtime(), prose, "notes.txt")
    structured = spans_for(runtime(structure_aware=True), prose, "notes.txt")
    assert [(s.start, s.end) for s in plain] != [(s.start, s.end) for s in structured]
    assert prose[structured[0].start:].startswith("# Title")

    code = "import os\n\n\ndef alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
    assert ([(s.start, s.end) for s in spans_for(runtime(structure_aware=True), code, "m.py")]
            == [(s.start, s.end) for s in spans_for(runtime(), code, "m.py")]), \
        "code must still be cut at its declarations"


def test_a_table_is_never_cut_and_keeps_its_header():
    """A table's header is what its rows mean, and the generic chunker
    cuts at 700 characters wherever the prose allows -- so rows arrive
    without the columns they belong to. `parse_structure` already
    identifies pipe tables; this only has to refuse to cut one."""
    rows = "".join(f"| crane {n} | {2000 + n} | inspected |\n" for n in range(1, 40))
    content = ("## Fleet\n\n| asset | year | state |\n| --- | --- | --- |\n" + rows
               + "\n\n" + PROSE)
    found = structured_spans(content)
    holding = [content[s.start:s.end] for s in found.spans if "crane 20" in content[s.start:s.end]]
    assert len(holding) == 1, [len(holding), found.record()]
    assert "| asset | year | state |" in holding[0], "the rows lost their header"
    assert "crane 1 " in holding[0] and "crane 39" in holding[0], "the table was cut"
    assert found.tables == 1, found.record()
    # A chunk over the target is a fact the caller needs, not one to hide.
    assert found.over_target == 1 and "longer than" in found.why, found.why


def test_a_numbered_line_inside_a_fenced_block_is_not_a_clause():
    """The heading case is already handled by `parse_structure`, which
    suppresses heading recognition inside a fence -- so the fence test
    above passes whether or not this module does anything. Clause and
    pair patterns are this module's own, and they need the fence too: a
    numbered shell transcript would otherwise cut the example at every
    step.
    """
    content = ("# Deploying\n\n" + PROSE + "\n\n```console\n1. first run the migration\n"
               "2. then restart the workers\nQ: what if it fails?\n```\n\n" + PROSE)
    labels = [u.label for u in units(content)]
    assert labels == ["# Deploying"], labels
    holding = [c for c in text_of(content, structured_spans(content).spans)
               if "first run the migration" in c]
    assert len(holding) == 1 and "then restart the workers" in holding[0], holding


def test_prose_after_a_table_is_not_dropped():
    """A table's unit ended at the table, and the next unit began at the
    next heading, so everything between them belonged to no chunk at all
    and was silently unretrievable.

    The invariant test above should have caught this and did not: it
    asserted that the gaps between spans are whitespace, but its fixture
    never put prose between a table and the next heading. A corpus that
    does not produce the awkward shape proves nothing about it, so this
    fixture is the shape, built by hand.
    """
    content = ("# Inventory\n\n| Item | Count |\n| --- | --- |\n| A | 3 |\n\n"
               "The warehouse closes Friday.\n\n# Notes\n\nKeep this note.\n")
    found = structured_spans(content, 50)
    covered = "".join(content[s.start:s.end] for s in found.spans)
    assert "The warehouse closes Friday." in covered, found.record()
    assert "Keep this note." in covered, found.record()
    assert "| A | 3 |" in covered, found.record()


def test_every_non_whitespace_byte_of_a_mixed_document_is_in_some_chunk():
    """The invariant, on a document that actually has every awkward
    junction in it: prose before the first unit, a table between two
    headings, prose after a table, a clause, a pair, and a trailing
    paragraph with no heading of its own."""
    content = (
        "Opening remarks with no heading at all.\n\n"
        "# Inventory\n\n"
        "| Item | Count |\n| --- | --- |\n| A | 3 |\n| B | 4 |\n\n"
        "The warehouse closes Friday.\n\n"
        "## Detail\n\n2.1 The clause\n\n" + PROSE + "\n\n"
        "Q: And a question?\n\nA: An answer.\n\n"
        "| Second | Table |\n| --- | --- |\n| x | y |\n\n"
        "Closing prose after the last table.\n")
    for target in (50, 200, DEFAULT_TARGET):
        found = structured_spans(content, target)
        end = 0
        for span in found.spans:
            assert span.start >= end, (target, span, end)
            assert content[end:span.start].strip() == "", (target, repr(content[end:span.start]))
            end = span.end
        assert content[end:].strip() == "", (target, repr(content[end:]))


async def test_what_the_engine_actually_stored_covers_the_document():
    """Spans covering the text is not the same claim as the stored chunks
    covering it: the spans are code points, the store keeps UTF-8 byte
    offsets, and retaining the original episode does not make omitted
    prose retrievable. So this asserts on what came back out of the
    store, in bytes, including a multi-byte character to make the two
    coordinate systems disagree if the conversion is wrong.
    """
    from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex,
                              MemoryEngine)

    content = (
        "Opening remarks with no heading, about a café.\n\n"
        "# Inventory\n\n| Item | Count |\n| --- | --- |\n| A | 3 |\n\n"
        "The warehouse closes Friday, the naïve assumption being that nobody minds.\n\n"
        "## Detail\n\n2.1 The clause\n\n" + PROSE + "\n\n"
        "| Second | Table |\n| --- | --- |\n| x | y |\n\n"
        "Closing prose after the last table.\n")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                chunk_target=120, structure_aware=True).open()
    try:
        added = await engine.remember("default", content, source="mixed.md")
        stored = await engine.documents.chunks_of("default", added.episode_id)
        raw = content.encode()
        for chunk in stored:
            assert raw[chunk.start:chunk.end] == chunk.text.encode(), (chunk.start, chunk.end)
        end = 0
        for chunk in sorted(stored, key=lambda c: c.start):
            assert chunk.start >= end, (chunk.start, end)
            assert raw[end:chunk.start].decode().strip() == "", repr(raw[end:chunk.start])
            end = chunk.end
        assert raw[end:].decode().strip() == "", repr(raw[end:])
    finally:
        await engine.close()
    for phrase in ("café", "| A | 3 |", "naïve assumption", "The clause", "Closing prose"):
        assert any(phrase in chunk.text for chunk in stored), phrase


def test_a_heading_stays_with_the_table_it_names():
    """Found by measuring the feature on a real corpus rather than by
    reasoning about it.

    Making a table its own unit separated it from the heading directly
    above, which is the same harm one level up: the header row says what
    the columns mean and the heading says what the table means. Four of
    the six residual cases in our own docs were exactly this.
    """
    content = ("## Coverage\n\n| Reader | Evidence | Limits |\n| --- | --- | --- |\n"
               "| pdf | page spans | no OCR |\n| csv | row spans | none |\n\n" + PROSE)
    found = structured_spans(content)
    holding = [content[s.start:s.end] for s in found.spans if "| pdf | page spans" in
               content[s.start:s.end]]
    assert len(holding) == 1, found.record()
    assert holding[0].lstrip().startswith("## Coverage"), holding[0][:40]
    assert found.tables == 1, found.record()


def test_two_headings_above_a_table_both_stay_with_it():
    """A heading with no body of its own before a subheading is not an
    orphan -- but neither of them should be cut from the table the pair
    introduces."""
    content = ("# Storage\n\n## Implemented adapters\n\n| Interface | Implementations |\n"
               "| --- | --- |\n| store | memory, sqlite |\n\n" + PROSE)
    found = structured_spans(content)
    holding = [content[s.start:s.end] for s in found.spans
               if "| store | memory, sqlite |" in content[s.start:s.end]]
    assert len(holding) == 1, found.record()
    assert holding[0].lstrip().startswith("# Storage"), holding[0][:60]


def test_a_heading_above_a_table_leaves_the_prose_before_it_behind():
    """The shape the corpus actually has, which my first fix missed.

    I attached a heading to its table only when the whole packed group
    was headings. In a real document the group usually begins with a
    short prose section and ends with the heading, so the rule never
    fired -- three residual cases in our own docs, all of this shape.
    The group has to split before its trailing heading, not refuse to
    split at all.
    """
    content = ("## Background\n\nA short paragraph about adapters.\n\n"
               "## Implemented adapters\n\n| Interface | Implementations |\n|---|---|\n"
               "| store | memory, sqlite |\n\n" + PROSE)
    found = structured_spans(content)
    holding = [content[s.start:s.end] for s in found.spans
               if "| store | memory, sqlite |" in content[s.start:s.end]]
    assert len(holding) == 1, found.record()
    assert holding[0].lstrip().startswith("## Implemented adapters"), holding[0][:60]
    # and the paragraph before it is still in some chunk
    covered = "".join(content[s.start:s.end] for s in found.spans)
    assert "A short paragraph about adapters." in covered
