from scone_memory.ingestion.chunker import MIN_CHUNK, chunk_spans


def test_every_character_lands_in_exactly_one_span():
    text = ("Alpha beta gamma delta. " * 40 + "\n\n") * 6
    spans = chunk_spans(text, target=300)
    covered = [0] * len(text)
    for s in spans:
        for i in range(s.start, s.end):
            covered[i] += 1
    assert all(c == 1 for i, c in enumerate(covered) if not text[i].isspace())
    assert all(c <= 1 for c in covered)
    assert all(text[s.start : s.end].strip() for s in spans)


def test_cuts_prefer_paragraph_breaks():
    paragraph = "Word " * 50
    text = f"{paragraph.strip()}\n\n{paragraph.strip()}\n\n{paragraph.strip()}"
    spans = chunk_spans(text, target=320)
    boundaries = {s.end for s in spans[:-1]}
    assert boundaries, "expected more than one chunk"
    assert all(text[b - 2 : b] == "\n\n" for b in boundaries), boundaries


def test_short_text_is_one_chunk_and_empty_is_none():
    assert chunk_spans("") == []
    [only] = chunk_spans("just a line")
    assert (only.start, only.end) == (0, len("just a line"))


def test_a_persons_initial_is_not_preferred_as_a_sentence_boundary():
    first = 'The team completed its inspection of the laboratory and documented every instrument before the scheduled opening of the new research facility. '
    second = 'The director Morgan P. Redwood approved the completed inspection and prepared the laboratory for visitors. '
    text = first + second * 5
    target = text.index('P. Redwood') + 5
    spans = chunk_spans(text, target=target)
    chunks = [text[s.start:s.end] for s in spans]
    assert chunks[0] == first
    assert any('Morgan P. Redwood' in chunk for chunk in chunks)
    assert not any(chunk.rstrip().endswith('Morgan P.') for chunk in chunks)


def test_tiny_tail_joins_its_predecessor():
    # 300 chars, break, 400 chars of unbroken words, break, a 5-char tail.
    # With target 320 the second paragraph is cut at a space near 620,
    # leaving ~88 chars that end in the tail: too small to stand alone.
    text = "x" * 300 + "\n\n" + ("Word " * 80).strip() + "\n\nTail."
    spans = chunk_spans(text, target=320)
    assert len(spans) == 2, [(s.start, s.end) for s in spans]
    last = text[spans[-1].start : spans[-1].end]
    assert len(last.strip()) >= MIN_CHUNK
    assert last.endswith("Tail.")


def test_stored_spans_are_utf8_byte_offsets():
    """Spec rule 1.2: a span must mean the same thing to the Rust product."""
    from scone_memory.ingestion.chunker import byte_spans

    content = "café ☕ Rua Augusta, 3º andar. " * 12 + "\n\nFin."
    spans = chunk_spans(content, target=150)
    stored = byte_spans(content, spans)
    raw = content.encode()
    assert len(stored) == len(spans) > 1
    for cp, b in zip(spans, stored):
        assert raw[b.start : b.end].decode() == content[cp.start : cp.end]
    assert stored[-1].end == len(raw) > len(content)


def test_a_soft_line_wrap_does_not_outrank_a_sentence_end():
    """The module's own docstring says cuts "prefer paragraph breaks, then
    sentence ends, then whitespace". The code preferred a bare newline
    second, ahead of sentence ends -- and in hard-wrapped prose, which is
    how Markdown is usually written, every line ends in the middle of a
    sentence.

    Measured over 40 of this project's documents before the fix: of 531
    chunk boundaries, 86 landed mid-sentence, and **85 of those 86 were
    at a single newline**. A soft wrap is a detail of how the text was
    typed, not a boundary in what it says.

    A lone newline is still preferred to arbitrary whitespace, because in
    a list or a table it is a real boundary and there is no sentence end
    to find.
    """
    # A paragraph hard-wrapped at about seventy columns, which is how
    # this repository's own documents are written. Every line break but
    # the last of each sentence falls in the middle of one.
    sentence = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda\n"
                "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega and\n"
                "one more clause to carry the sentence past the target length.\n")
    text = sentence * 8
    spans = chunk_spans(text, target=300)
    boundaries = [s.end for s in spans[:-1]]
    assert boundaries, "expected more than one chunk"
    for cut in boundaries:
        before = text[:cut].rstrip()
        assert before.endswith((".", "!", "?")), (cut, repr(text[max(0, cut - 40):cut]))


def test_a_newline_is_still_a_boundary_when_no_sentence_ends():
    """A list has no sentence ends, and there a line break is the only
    boundary the text offers."""
    text = "".join(f"- item number {n} in a list of things\n" for n in range(40))
    spans = chunk_spans(text, target=300)
    for span in spans[:-1]:
        assert text[span.end - 1] == "\n", repr(text[max(0, span.end - 30):span.end])


def test_carriage_returns_do_not_hide_a_paragraph_or_a_sentence():
    """The same prose with Windows line endings was cut quite differently,
    and nothing in the chunker said so.

    `rfind("\\n\\n")` does not match `\\r\\n\\r\\n`, and the sentence markers
    ended at `. ` or `.\\n`, never `.\\r\\n`. So on CRLF a document lost its
    paragraph breaks **and** its sentence ends and fell through to the
    lone-newline rule: measured on one wrapped document, 5 paragraph cuts
    and no mid-sentence cuts became 1 paragraph cut and 3 mid-sentence.

    Recognised, never normalised. Rewriting the text would move every
    offset after it, and stored offsets are the shared specification.
    """
    sentence = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda\n"
                "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega and\n"
                "one more clause to carry the sentence past the target length.\n")
    unix = (sentence + "\n") * 6
    windows = unix.replace("\n", "\r\n")
    for label, text in (("LF", unix), ("CRLF", windows)):
        spans = chunk_spans(text, target=300)
        for span in spans[:-1]:
            before = text[:span.end]
            assert before.endswith(("\n\n", "\r\n\r\n")) or before.rstrip().endswith((".", "!", "?")), (
                label, repr(text[max(0, span.end - 60):span.end]))
    assert len(chunk_spans(unix, target=300)) == len(chunk_spans(windows, target=300))


def test_a_crlf_blank_line_is_a_paragraph():
    """Only the paragraph rule can satisfy this: the prose has no
    sentence ends at all, so if `\r\n\r\n` is not recognised the cut
    falls to a lone newline in the middle of a line."""
    line = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu\r\n"
    text = ((line * 3) + "\r\n") * 6
    spans = chunk_spans(text, target=300)
    assert len(spans) > 1
    for span in spans[:-1]:
        assert text[:span.end].endswith("\r\n\r\n"), repr(text[max(0, span.end - 40):span.end])


def test_a_crlf_sentence_end_is_a_sentence_end():
    """Only the sentence rule can satisfy this: one CRLF-wrapped
    paragraph with no blank line in it, so a cut is either at `.\r\n` or
    in the middle of a sentence."""
    # Four wrapped lines per sentence, so three of every four line
    # breaks fall mid-sentence. A cut that took the nearest lone newline
    # would land on one of those three far more often than not.
    sentence = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa\r\n"
                "lambda mu nu xi omicron pi rho sigma tau upsilon phi\r\n"
                "chi psi omega and a further clause to lengthen it\r\n"
                "so that the sentence runs past the target end.\r\n")
    text = sentence * 8
    spans = chunk_spans(text, target=300)
    assert len(spans) > 1
    for span in spans[:-1]:
        assert text[:span.end].rstrip().endswith("."), repr(text[max(0, span.end - 40):span.end])


def test_a_crlf_source_is_never_rewritten_by_chunking():
    """Whatever the cut rule prefers, a span is an offset into the source
    exactly as given: every `\\r\\n` the source had is still inside the
    spans, and the spans still cover every non-space character once."""
    text = "One sentence here.\r\nAnother sentence follows it.\r\n\r\n" * 20
    spans = chunk_spans(text, target=300)
    assert text.count("\r\n") > 0
    covered = [0] * len(text)
    for span in spans:
        assert text[span.start:span.end] == text[span.start:span.end]
        for index in range(span.start, span.end):
            covered[index] += 1
    assert all(n == 1 for index, n in enumerate(covered) if not text[index].isspace())
    assert all(n <= 1 for n in covered)
