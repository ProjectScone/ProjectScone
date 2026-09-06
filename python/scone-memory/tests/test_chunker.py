from scone_memory.chunker import MIN_CHUNK, chunk_spans


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
