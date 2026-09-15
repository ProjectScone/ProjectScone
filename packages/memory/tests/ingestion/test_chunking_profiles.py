"""A genre's own boundaries, declared, over the structure chunker.

Plain structure chunking reads what any document carries -- headings,
tables, numbered lines -- and packs them up to the target. It cannot
know that ``Article 2`` owns the ``(a)`` below it, that ``A:`` belongs to
the ``Q:`` above it, or that a references list is not part of the
conclusion it follows. A profile says so: a named set of boundary rules
for one kind of document, chosen by whoever stores it.

The properties each test holds a profile to:

- **A clause never leaves its heading.** A heading line never ends a
  chunk while its first clause begins the next, and an article that fits
  the target is one chunk.
- **A question and its answer are one unit.** ``A:`` is counted, never
  cut at, and each pair begins its own chunk.
- **Structure that is not there is not invented.** ``Section 3 of this
  Act applies`` is a sentence, ``Experience shows`` is prose and a
  rhetorical question inside a paragraph is not a new question.
- **A document the profile finds nothing in chunks as it did before.**
"""
from __future__ import annotations

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.chunker import DEFAULT_TARGET, MIN_CHUNK, chunk_spans
from scone_memory.ingestion.chunking_profiles import PROFILES, profile_named, profiled_spans
from scone_memory.ingestion.structure_chunks import structured_spans, units

SENTENCE = ("The controller shall keep the record in writing, including in electronic form, "
            "and shall make it available to the supervisory authority on request. ")

STATUTE = (
    "DATA PROTECTION ACT\n\n"
    "PART 1\nGENERAL PROVISIONS\n\n"
    "Article 1\nSubject matter\n\n"
    "1. This Act lays down rules relating to the protection of natural persons with regard to the "
    "processing of personal data held by public bodies and by the undertakings they contract.\n"
    "2. This Act protects the fundamental rights and freedoms of natural persons, and in particular "
    "their right to the protection of personal data held about them by anyone.\n"
    "3. Nothing in this Act restricts the movement of personal data between the members where the "
    "processing is lawful. " + SENTENCE + "\n\n"
    "Article 2\nDefinitions\n\n"
    "(a) 'personal data' means any information relating to an identified or identifiable natural "
    "person, directly or indirectly, by reference to an identifier such as a name. " + SENTENCE + "\n"
    "(b) 'processing' means any operation performed on personal data, whether or not by automated "
    "means, such as collection, recording, organisation or storage. " + SENTENCE + "\n"
    "(c) 'controller' means the natural or legal person, public authority or other body which "
    "determines the purposes and means of the processing.\n\n"
    "Section 4.2 Transfers\n"
    "(a) A transfer of personal data to a third country may take place only where an adequate "
    "level of protection is ensured. " + SENTENCE * 5 + "\n"
    "(b) Each transfer shall be recorded. " + SENTENCE + "\n\n"
    "§ 3 Records of processing\n"
    "Each controller shall keep a record of the processing it is responsible for. " + SENTENCE + "\n"
    "Section 3 of this Act applies to every controller established in the territory.\n")

QA = "".join(
    f"Q: {question}\n"
    f"A: {answer} " + SENTENCE * 3 + "\n\n"
    for question, answer in [
        ("How do I reset my password?", "Open Settings, choose Security and follow the reset link."),
        ("Can I change the email address on my account?", "Yes, from the Profile page."),
        ("What happens to my data when I close my account?", "It is deleted after thirty days."),
        ("Who can see my records?", "Only the people you share them with."),
    ])


def texts(content, spans):
    return [content[s.start:s.end] for s in spans]


def chunk_holding(spans, offset):
    return next(index for index, span in enumerate(spans) if span.start <= offset < span.end)


def heading_splits(content, spans, heading, first_child):
    """Whether the heading and the clause right under it land in different chunks."""
    return chunk_holding(spans, content.index(heading)) != chunk_holding(spans, content.index(first_child))


def boundary_share(content, spans, profile):
    reader = profile_named(profile).reader()
    starts = {unit.start for unit in units(content, reader=reader) if unit.kind not in ("text", "table")}
    begun = sum(1 for span in spans if span.start in starts or content[:span.start].strip() == "")
    return begun / len(spans)


def test_the_registry_names_each_genre_and_every_rule_has_a_name():
    assert {"statute", "paper", "manual", "qa", "resume"} <= set(PROFILES)
    for name, profile in PROFILES.items():
        assert profile.name == name and profile.about
        assert profile.rules and all(rule.name for rule in profile.rules)


def test_an_unknown_profile_is_refused():
    with pytest.raises(InvalidInput, match="statute"):
        profile_named("sonnet")
    with pytest.raises(InvalidInput):
        profile_named(["statute"])  # type: ignore[arg-type]


def test_statute_markers_are_read_with_their_rank():
    found = units(STATUTE, reader=profile_named("statute").reader())
    kinds = {unit.label.split("\n")[0]: unit.kind for unit in found}
    assert kinds["PART 1"] == "part"
    assert kinds["Article 2"] == "article"
    assert kinds["Section 4.2 Transfers"] == "article"
    assert kinds["§ 3 Records of processing"] == "article"
    assert any(kind == "letter" for label, kind in kinds.items() if label.startswith("(a)"))
    depth = {unit.label: unit.depth for unit in found}
    assert depth["PART 1"] < depth["Article 2"] < depth[next(l for l in depth if l.startswith("(b)"))]


def test_a_cross_reference_at_the_start_of_a_line_is_not_an_article():
    found = units(STATUTE, reader=profile_named("statute").reader())
    assert not any(unit.label.startswith("Section 3 of") for unit in found)
    # Plain structure reads it as a clause; the profile asks for a title.
    assert any(unit.label.startswith("Section 3 of") for unit in units(STATUTE))


def test_plain_structure_splits_a_heading_from_its_clause_on_this_fixture():
    """The junction the profile exists for must be in the fixture, or the
    next test proves nothing."""
    plain = structured_spans(STATUTE).spans
    assert heading_splits(STATUTE, plain, "Article 2", "(a) 'personal data'")


def test_a_clause_never_splits_from_its_heading():
    found = profiled_spans(STATUTE, profile="statute")
    assert not heading_splits(STATUTE, found.spans, "Article 2", "(a) 'personal data'")
    assert not heading_splits(STATUTE, found.spans, "Section 4.2", "(a) A transfer")
    assert not heading_splits(STATUTE, found.spans, "§ 3", "Each controller shall keep")
    assert not heading_splits(STATUTE, found.spans, "Article 1", "1. This Act")


def test_an_article_that_fits_the_target_is_one_chunk():
    article = STATUTE[STATUTE.index("§ 3"):]
    assert len(article) <= DEFAULT_TARGET
    found = profiled_spans(STATUTE, profile="statute")
    assert any(chunk.strip() == article.strip() or article.strip() in chunk for chunk in texts(STATUTE, found.spans))
    assert chunk_holding(found.spans, STATUTE.index("§ 3")) == chunk_holding(found.spans, len(STATUTE) - 2)


def test_a_clause_longer_than_the_target_is_split_by_size_and_says_so():
    found = profiled_spans(STATUTE, profile="statute")
    chunks = texts(STATUTE, found.spans)
    section = next(index for index, chunk in enumerate(chunks) if chunk.startswith("Section 4.2 Transfers"))
    assert "(a) A transfer" in chunks[section] and not chunks[section + 1].startswith("(")
    assert chunks[section + 2].startswith("(b) Each transfer")
    # The title before the first unit, and the second half of the long clause.
    assert found.by_size == 2, found.record()
    assert "inside a unit longer than" in found.why, found.why


def test_small_clauses_are_packed_while_they_fit():
    content = "Article 8 Fees\n" + "".join(f"({letter}) A fee of {n} pounds applies to form {letter}.\n"
                                          for n, letter in enumerate("abcdefghjklmnopqrstu")) + SENTENCE * 5
    found = profiled_spans(content, profile="statute")
    assert len(content) > DEFAULT_TARGET and found.record()["matched"]["letter"] == 20
    assert len(found.spans) < 5, [chunk[:12] for chunk in texts(content, found.spans)]


def test_the_receipt_names_the_profile_and_what_its_rules_matched():
    found = profiled_spans(STATUTE, profile="statute")
    record = found.record()
    assert record["profile"] == "statute"
    assert record["matched"]["article"] == 4
    assert record["matched"]["part"] == 1
    assert record["matched"]["letter"] == 5
    assert record["matched"]["paragraph"] == 3
    assert sum(record["began"].values()) == found.at_boundary
    assert found.at_boundary + found.by_size == len(found.spans)


def test_more_chunks_start_at_a_boundary_than_under_plain_structure():
    profiled = profiled_spans(STATUTE, profile="statute").spans
    plain = structured_spans(STATUTE).spans
    assert boundary_share(STATUTE, profiled, "statute") >= boundary_share(STATUTE, plain, "statute")


def test_a_letter_after_h_is_a_letter_and_after_a_clause_a_numeral():
    content = ("Article 9\n(a) first " + "x " * 10 + "\n(i) nested numeral\n(ii) second numeral\n"
               "(b) b\n(c) c\n(d) d\n(e) e\n(f) f\n(g) g\n(h) h\n(i) the ninth letter\n")
    found = units(content, reader=profile_named("statute").reader())
    by_label = [(unit.label, unit.kind, unit.depth) for unit in found]
    nested = next(item for item in by_label if item[0] == "(i) nested numeral")
    ninth = next(item for item in by_label if item[0] == "(i) the ninth letter")
    letter_a = next(item for item in by_label if item[0].startswith("(a)"))
    assert nested[1] == "roman" and nested[2] > letter_a[2]
    assert ninth[1] == "letter" and ninth[2] == letter_a[2]
    first = units("Article 9\n1. x\n(i) a numeral before any letter\n(ii) and another\n", reader=profile_named("statute").reader())
    assert [unit.kind for unit in first] == ["article", "paragraph", "roman", "roman"]


def test_a_numeral_under_a_capital_after_h_is_a_numeral():
    """The United States code nests (h)(1)(A)(i): a clause under a
    subparagraph, not the letter after (h), however long ago (h) was read."""
    body = " ".join(["word"] * 40)
    content = "".join(f"{marker} {body}\n" for marker in ["Section 5 Definitions", "(g)", "(h)", "(1)", "(A)", "(i)", "(ii)", "(iii)"])
    reader = profile_named("statute").reader()
    found = units(content, reader=reader)
    assert [unit.kind for unit in found] == ["article", "letter", "letter", "numbered", "capital", "roman", "roman", "roman"]
    capital, first, second = found[4], found[5], found[6]
    assert capital.depth < first.depth == second.depth
    assert reader.matched == {"article": 1, "letter": 2, "numbered": 1, "capital": 1, "roman": 3}
    spans = profiled_spans(content, 1300, profile="statute").spans
    assert chunk_holding(spans, content.index("(A)")) == chunk_holding(spans, content.index("(i)"))


def test_a_letter_after_h_is_still_a_letter_below_a_numbered_paragraph():
    content = "Section 5 Definitions\n(h) x\n(1) y\n(2) z\n(i) the ninth letter\n"
    found = units(content, reader=profile_named("statute").reader())
    assert [(unit.kind, unit.depth) for unit in found][-1] == ("letter", found[1].depth)


def test_the_letter_h_is_forgotten_at_the_next_article():
    for heading in ["Article 2 Scope", "## Article 2 Scope"]:
        content = "Article 1 Terms\n(g) x\n(h) y\n\n" + heading + "\n\n1. z\n(i) a numeral under a paragraph\n"
        found = units(content, reader=profile_named("statute").reader())
        assert found[-1].kind == "roman" and found[-1].depth > found[-2].depth, (heading, [(u.label, u.kind) for u in found])


def test_enumerators_nest_in_the_order_the_document_uses_them():
    """The United States code puts (a) above (1); the European Union puts 1. above (a)."""
    us = units("§ 552 Public information\n(a) Each agency\n(1) shall publish\n", reader=profile_named("statute").reader())
    eu = units("Article 5 Principles\n1. Data shall be\n(a) processed lawfully\n", reader=profile_named("statute").reader())
    assert us[1].depth < us[2].depth and eu[1].depth < eu[2].depth
    assert us[1].kind == "letter" and eu[2].kind == "letter"


def pairs_split(content, spans):
    """Questions whose answer begins in a later chunk."""
    split = 0
    for question in (index for index in range(len(content)) if content.startswith("Q:", index)):
        answer = content.index("\nA:", question) + 1
        split += chunk_holding(spans, question) != chunk_holding(spans, answer)
    return split


def test_plain_structure_separates_a_question_from_its_answer_on_this_fixture():
    assert pairs_split(QA, structured_spans(QA).spans) >= 1
    assert pairs_split(QA, profiled_spans(QA, profile="qa").spans) == 0


def test_a_question_and_its_answer_are_one_chunk_and_each_pair_its_own():
    found = profiled_spans(QA, profile="qa")
    chunks = texts(QA, found.spans)
    assert len(chunks) == 4, [c[:30] for c in chunks]
    for chunk in chunks:
        assert chunk.startswith("Q:") and chunk.count("Q:") == 1 and "\nA:" in chunk
    assert found.record()["matched"] == {"question": 4, "answer": 4}
    assert found.record()["began"] == {"question": 4}


def test_a_small_pair_is_not_packed_with_the_next():
    content = "Q: One?\nA: Yes.\n\nQ: Two?\nA: No.\n"
    chunks = texts(content, profiled_spans(content, profile="qa").spans)
    assert [chunk.strip() for chunk in chunks] == ["Q: One?\nA: Yes.", "Q: Two?\nA: No."]


def test_a_question_line_opening_a_paragraph_is_a_question_and_one_inside_an_answer_is_not():
    content = ("How do I cancel?\nFrom the billing page. Why would you want to?\nMany people do.\n\n"
               "Can I come back later?\nYes, at any time.\n")
    found = units(content, reader=profile_named("qa").reader())
    assert [unit.label for unit in found] == ["How do I cancel?", "Can I come back later?"]


def test_a_heading_or_a_rule_ends_a_paragraph_so_the_next_line_may_be_a_question():
    for content in ["## Billing\nHow do I pay?\nBy card.\n", "Billing\n=======\nHow do I pay?\nBy card.\n",
                    "Intro text.\n\n---\nHow do I pay?\nBy card.\n"]:
        found = units(content, reader=profile_named("qa").reader())
        assert "How do I pay?" in [unit.label for unit in found], (content, [unit.label for unit in found])


def test_a_long_answer_is_split_with_its_question_at_the_start():
    content = "Q: Tell me everything?\nA: " + SENTENCE * 12 + "\n\nQ: Short?\nA: Yes.\n"
    found = profiled_spans(content, profile="qa")
    chunks = texts(content, found.spans)
    assert chunks[0].startswith("Q: Tell me everything?")
    assert found.by_size >= 1 and chunks[-1].startswith("Q: Short?")


def test_a_long_introduction_does_not_drag_the_first_pair_over_the_target():
    """A heading's own text travels with its first child only when that is
    a marker or when the two fit together; otherwise the pair would be the
    thing split."""
    content = ("# FAQ\n\n" + SENTENCE * 2 + "\n\nQ: First?\nA: " + SENTENCE * 2 + "\n\n" + SENTENCE
               + "\n\nQ: Second?\nA: Fine.\n")
    chunks = texts(content, profiled_spans(content, profile="qa").spans)
    first = next(chunk for chunk in chunks if "Q: First?" in chunk)
    assert first.startswith("Q: First?") and first.rstrip().endswith(SENTENCE.strip())
    assert chunks[0].startswith("# FAQ") and "Q: First?" not in chunks[0]


def test_a_short_heading_does_not_split_a_pair_that_fits_the_target_alone():
    """A marker travels with its first child even when the two come to more
    than the target: a chunk over by less than MIN_CHUNK, counted, rather
    than a pair that fits cut in two or a heading left as its own chunk."""
    words = "Open Settings, choose Security and follow the reset link that arrives. "
    head = "## Account and sign-in\n\nQuestions about signing in, passwords and the addresses we write to.\n\n"
    pair = "Q: How do I reset my password?\nA: " + words * 4 + "\n\n" + words * 4 + "It is quick and safe.\n\n"
    content = head + pair + "Q: Can I change my email?\nA: Yes, from the Profile page. " + words * 3 + "\n\n"
    assert len(head) < MIN_CHUNK and len(pair) <= DEFAULT_TARGET < len(head) + len(pair)
    found = profiled_spans(content, profile="qa")
    chunks = texts(content, found.spans)
    assert chunks[0] == head + pair.rstrip() or chunks[0] == head + pair, [len(chunk) for chunk in chunks]
    assert chunks[1].startswith("Q: Can I change")
    assert found.by_size == 0 and found.over_target == 1, found.record()
    assert "a heading shorter than 120 kept with the unit under it" in found.why, found.why


def test_a_heading_with_a_short_body_packs_with_its_first_clause_when_both_fit():
    body = "This article applies to every record kept under this Act and to every copy of one. " * 2
    content = ("Article 7 Scope\n" + body + "\n(a) first clause " + "a " * 40 + "\n"
               "(b) second clause " + SENTENCE * 5 + "\n")
    chunks = texts(content, profiled_spans(content, profile="statute").spans)
    assert chunks[0].startswith("Article 7 Scope") and "(a) first clause" in chunks[0]


def test_paper_sections_begin_chunks_and_references_stay_apart():
    content = ("Deep Harbours\nA. Author, B. Author\n\n"
               "Abstract\nWe measure harbour cranes. " + SENTENCE + "\n\n"
               "1 Introduction\n" + SENTENCE * 2 + "\n\n"
               "2. Methods\n" + SENTENCE + "\n\n2.1 Data collection\n" + SENTENCE + "\n\n"
               "Results were mixed across the three yards.\n\n"
               "Conclusion\nShort.\n\n"
               "References\n[1] Smith, J. Cranes. 2020.\n[2] Jones, K. Rust. 2021.\n\n"
               "## Notes\nWritten in the yard.\n")
    found = profiled_spans(content, profile="paper")
    chunks = texts(content, found.spans)
    assert any(chunk.startswith("Abstract") for chunk in chunks)
    assert any(chunk.startswith("1 Introduction") for chunk in chunks)
    assert any(chunk.startswith("2. Methods") for chunk in chunks)
    references = next(chunk for chunk in chunks if "[1] Smith" in chunk)
    assert references.startswith("References") and "Short." not in references and "Notes" not in references
    conclusion = next(chunk for chunk in chunks if "Short." in chunk)
    assert conclusion.startswith("Conclusion")
    matched = found.record()["matched"]
    assert matched["references"] == 1 and matched["reference"] == 2 and matched["section"] == 3
    assert matched["abstract"] == 1 and matched["subsection"] == 1
    assert "Results were mixed" not in [unit.label for unit in units(content, reader=profile_named("paper").reader())]


def test_a_reference_entry_is_only_read_inside_references():
    content = ("Introduction\n[1] showed that cranes rust.\n\nReferences\n[1] Smith, J. Cranes.\n\n"
               "Appendix A\n[2] is a footnote, not an entry.\n")
    found = units(content, reader=profile_named("paper").reader())
    assert [unit.kind for unit in found] == ["section", "references", "reference", "section"]


def test_a_markdown_heading_is_named_by_the_rule_its_title_matches():
    content = "# Paper\n\n## Abstract\n\n" + SENTENCE + "\n\n## References\n\n[1] Smith.\n"
    found = units(content, reader=profile_named("paper").reader())
    assert [(unit.kind, unit.depth) for unit in found][:3] == [("heading", 1), ("abstract", 2), ("references", 2)]
    assert profile_named("paper").reader().heading("Abstract", 2) == ("abstract", 2)
    reader = profile_named("paper").reader()
    units(content, reader=reader)
    assert reader.matched == {"abstract": 1, "references": 1, "reference": 1}


def test_a_plain_references_line_after_markdown_sections_is_their_sibling():
    """A rule sits where it sat as a heading, so mixing heading styles does
    not put the references list inside the section before it."""
    words = "We measured the effect on recall across three corpora. "
    content = ("## Results\n\n" + words * 8 + "\n\n## Conclusion\n\n" + words * 3 + "\n\nReferences\n\n"
               "[1] A. Author. A paper. 2020.\n[2] B. Author. Another paper. 2021.\n")
    found = units(content, reader=profile_named("paper").reader())
    depth = {unit.label: unit.depth for unit in found}
    assert depth["References"] == depth["## Conclusion"] < depth["[1] A. Author. A paper. 2020."]
    profiled = profiled_spans(content, profile="paper")
    chunks = texts(content, profiled.spans)
    references = next(chunk for chunk in chunks if "[1] A. Author" in chunk)
    assert references.startswith("References") and "Conclusion" not in references
    assert profiled.record()["began"] == {"section": 2, "references": 1}


def test_a_plain_marker_ranked_above_a_markdown_one_is_not_its_child():
    content = "## Article 5 Principles\n\n1. Data shall be kept.\n\nChapter 3 Rights\n\nArticle 6 Access\n\n1. Each person.\n"
    found = units(content, reader=profile_named("statute").reader())
    depth = {unit.label: unit.depth for unit in found}
    assert depth["Chapter 3 Rights"] < depth["## Article 5 Principles"] == depth["Article 6 Access"], depth


def test_a_setext_heading_is_ranked_by_its_underline():
    content = "Experience\n==========\nAnalyst.\n\nNotes\n-----\nMore.\n"
    found = units(content, reader=profile_named("resume").reader())
    assert [(unit.kind, unit.depth) for unit in found] == [("section", 1), ("heading", 2)]


def test_each_statute_marker_reads_as_its_kind():
    reader = profile_named("statute").reader
    for line, kind in [("TITLE II", "part"), ("Division 3 Offences", "part"), ("CHAPTER IV", "chapter"),
                       ("Subchapter 2 - Records", "chapter"), ("\u7b2c\u4e09\u7ae0 \u603b\u5219", "chapter"),
                       ("\u7b2c\u5341\u4e8c\u6761 \u4e2a\u4eba\u4fe1\u606f", "article"), ("Art. 5 Principles", "article"),
                       ("Sec. 12. Penalties", "article"), ("§§ 4 Scope", "article"), ("Clause 7: Payment", "article"),
                       ("Rule 3", "article"), ("4.2 Transfers", "subsection"), ("12) Twelve", "paragraph"),
                       ("(3) three", "numbered"), ("(iv) four", "roman"), ("b) bee", "letter"), ("(B) Bee", "capital")]:
        found = reader().line(line, True)
        assert found is not None and found[0] == kind, (line, found)
    for prose in ["Section 3 of this Act applies", "Part of the fee is refundable", "Chapter and verse",
                  "1984 was a year", "e.g. something", "Article twelve says"]:
        assert reader().line(prose, True) is None, prose


def test_a_dotted_or_parenthesised_cross_reference_is_not_a_marker():
    """A decimal point is not the punctuation after a marker, and a
    sub-reference followed by a lowercase word is a sentence."""
    reader = profile_named("statute").reader
    for prose in ["Section 3.2 of this Agreement applies", "§ 12.1 of the Code", "Part 2.3 of the Schedule",
                  "Article 6 (1) of this Regulation shall apply", "Chapter 3. of", "Article 4 (a) and (b) apply"]:
        assert reader().line(prose, True) is None, prose
    for line, kind in [("Section 3.2. Transfers", "article"), ("Section 1.—Short title", "article"),
                       ("§ 552. (a) Each agency shall publish", "article"), ("Section 5 (a) Each agency", "article"),
                       ("Section 12 (Repealed)", "article"), ("PART 2. GENERAL", "part"), ("Article 3.2", "article")]:
        found = reader().line(line, True)
        assert found is not None and found[0] == kind, (line, found)


def test_each_manual_heading_reads_as_its_kind():
    reader = profile_named("manual").reader
    for line, kind in [("3.2 Replacing the filter", "procedure"), ("Chapter 4 Maintenance", "procedure"),
                       ("Procedure 2: Draining", "procedure"), ("How to clean the tank:", "task"),
                       ("Step 3) Refit", "step"), ("step 4 refit", "step"), ("(c) check", "substep"), ("d. dry", "substep")]:
        found = reader().line(line, True)
        assert found is not None and found[0] == kind, line
    assert reader().line("3.2 metres of hose", True) is None


def test_enumerator_order_starts_again_under_each_article():
    for heading in ["Article 5 Principles", "## Article 5 Principles"]:
        content = "§ 552 Public information\n(a) Each agency\n(1) shall publish\n\n" + heading + "\n\n1. Data shall be\n(a) processed lawfully\n"
        found = units(content, reader=profile_named("statute").reader())
        eu = [unit for unit in found if unit.start > content.index("Article 5")]
        assert eu[-2].depth < eu[-1].depth, [(unit.label, unit.depth) for unit in found]


def test_manual_steps_stay_whole_under_their_heading():
    steps = "".join(f"{n}. {verb} the filter housing and check the seal before you go on. {SENTENCE}\n"
                    for n, verb in enumerate(["Unplug", "Open", "Remove", "Rinse", "Refit"], start=1))
    content = ("## Replacing the filter\n\n" + steps + "\n## Cleaning the tank\n\n"
               "To drain the tank:\n1. Open the valve.\n2. Wait.\n")
    found = profiled_spans(content, profile="manual")
    chunks = texts(content, found.spans)
    assert chunks[0].startswith("## Replacing the filter") and "1. Unplug" in chunks[0]
    for chunk in chunks:
        for n in range(1, 6):
            marker = f"{n}. " + ["Unplug", "Open", "Remove", "Rinse", "Refit"][n - 1]
            if marker in chunk:
                assert chunk.count("check the seal") >= 1
                start = chunk.index(marker)
                assert "go on." in chunk[start:], "a step was split"
    assert chunks[-1].startswith("## Cleaning the tank") and "2. Wait." in chunks[-1]
    assert found.record()["matched"] == {"step": 7, "task": 1}


def test_a_step_word_is_a_step_and_a_bullet_is_not():
    content = "Step 1: Unplug it.\n- a bullet\nStep 2: Open it.\n"
    found = units(content, reader=profile_named("manual").reader())
    assert [unit.label for unit in found] == ["Step 1: Unplug it.", "Step 2: Open it."]


def test_resume_sections_begin_chunks_and_prose_is_not_a_section():
    content = ("Ada Lovelace\nada@example.org\n\n"
               "Experience\nAnalyst, Engines Ltd, 1842-1843. Experience shows that notes outlive engines.\n\n"
               "Education\nPrivate tutors.\n\n"
               "SKILLS:\nMathematics, translation.\n")
    found = profiled_spans(content, profile="resume")
    chunks = texts(content, found.spans)
    assert [chunk.split("\n")[0] for chunk in chunks] == ["Ada Lovelace", "Experience", "Education", "SKILLS:"]
    assert found.record()["matched"] == {"section": 3}


def test_a_document_the_profile_finds_nothing_in_chunks_exactly_as_before():
    content = ("The survey of the harbour crane was booked for the third of May, and it found rust. " * 30)
    for name in PROFILES:
        found = profiled_spans(content, profile=name)
        assert [(s.start, s.end) for s in found.spans] == [(s.start, s.end) for s in chunk_spans(content)]
        assert found.units == 0 and found.by_size == len(found.spans)
        assert "none of its boundaries" in found.why, found.why


def test_the_spans_cover_the_document_and_drop_only_whitespace():
    for content, name in [(STATUTE, "statute"), (QA, "qa")]:
        end = 0
        found = profiled_spans(content, profile=name)
        for span in found.spans:
            assert span.start >= end and span.end > span.start
            assert content[end:span.start].strip() == ""
            end = span.end
        assert content[end:].strip() == ""
        # Nothing on these fixtures has to run over the target to stay whole.
        assert found.over_target == 0 and all(s.end - s.start <= DEFAULT_TARGET for s in found.spans)


def test_a_chunk_over_the_target_is_counted_whatever_put_it_there():
    """The size chunker joins a last piece shorter than MIN_CHUNK to the one
    before it, so a split answer can come back longer than the target.
    The receipt counts chunks, not intentions."""
    content = "Q: Long?\nA: " + "word " * 150 + "\n\nThe end of it.\n\nQ: Next?\nA: Yes.\n"
    found = profiled_spans(content, profile="qa")
    longer = [span for span in found.spans if span.end - span.start > DEFAULT_TARGET]
    assert len(longer) == 1 and found.over_target == 1 and found.tables == 0
    assert "1 chunk(s) are longer than 700" in found.why


def test_nonsense_bounds_are_refused():
    with pytest.raises(ValueError):
        profiled_spans(QA, 0, profile="qa")
    with pytest.raises(ValueError):
        profiled_spans(QA, profile="qa", units_max=0)


def test_a_table_is_kept_whole_with_the_heading_above_it():
    table = "| term | meaning |\n| --- | --- |\n" + "".join(f"| t{n} | {SENTENCE} |\n" for n in range(8))
    content = "Article 4 Terms\n" + table + "\nArticle 5 Other\n" + SENTENCE + "\n"
    found = profiled_spans(content, profile="statute")
    chunks = texts(content, found.spans)
    assert chunks[0].startswith("Article 4 Terms") and chunks[0].rstrip().endswith("|")
    assert found.tables == 1 and found.over_target == 1
    assert "1 table(s) were kept whole" in found.why and "chunk(s) are longer than" in found.why


def test_a_small_table_is_whole_and_not_over_the_target():
    content = ("Article 4 Terms\n(a) Short clause before the table.\n| term | meaning |\n| --- | --- |\n| t | a term |\n"
               "After the table, a note that belongs to Article 4.\nArticle 5 Other\n" + SENTENCE * 6 + "\n")
    found = profiled_spans(content, profile="statute")
    chunks = texts(content, found.spans)
    assert found.tables == 1 and found.over_target == 0 and "chunk(s) are longer than" not in found.why
    assert any(chunk.startswith("Article 5 Other") for chunk in chunks), [c[:20] for c in chunks]
    article_4 = next(chunk for chunk in chunks if chunk.startswith("Article 4"))
    assert "| t | a term |" in article_4 and "belongs to Article 4" in article_4


def test_the_unit_bound_is_reported_when_it_bites():
    found = profiled_spans(QA, profile="qa", units_max=2)
    assert found.capped is True
    assert "more than 2" in found.why
    chunks = texts(QA, found.spans)
    assert chunks[0].startswith("Q: How do I reset") and chunks[1].startswith("Q: Can I change")
    assert found.by_size >= 1 and "What happens" in "".join(chunks[2:])
    uncapped = profiled_spans(QA, profile="qa")
    assert uncapped.capped is False
    small = "Q: One?\nA: Yes.\n\nQ: Two?\nA: No.\n"
    assert [chunk.strip() for chunk in texts(small, profiled_spans(small, profile="qa", units_max=1).spans)] == [
        "Q: One?\nA: Yes.", "Q: Two?\nA: No."]
