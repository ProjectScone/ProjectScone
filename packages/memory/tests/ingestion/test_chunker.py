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
