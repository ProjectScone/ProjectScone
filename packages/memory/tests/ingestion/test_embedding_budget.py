"""What goes in front of a chunk must not push the chunk out of the embedder's window.

A model embeds at most so many tokens and silently drops the rest. The
context put in front of a chunk -- the headings above it, a table's
header row, the source and date -- comes first, so a long prefix cuts
the end of the chunk it was meant to explain, and nothing says so. With
``embedding_budget`` the context is shortened until the input fits: the
outermost headings go first, then the heading line, then the table
context. The chunk itself is never cut, and a chunk too long on its own
is counted. Tokens are estimated without the model's tokenizer, a rule
chosen to err high: on this project's documents it never counted fewer
tokens than BGE's tokenizer did.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.embedding_budget import BUDGET_VERSION, estimated_tokens, fitted

def test_tokens_are_estimated_per_word_part_digit_run_mark_and_unspaced_character():
    assert estimated_tokens("the crane") == 1 + 2 + 2, "a token per four letters, and the two markers a model adds"
    assert estimated_tokens("港口起重机") == 5 + 2
    assert estimated_tokens("OcrPdfOptions") == 1 + 1 + 2 + 2, "a camel-case name counts by its parts"
    assert estimated_tokens("invoice 20931, paid.") == 2 + 3 + 1 + 1 + 1 + 2
    assert estimated_tokens("") == 2


def wrap(text: str) -> str:
    return f"Source: notes.md\n{text}"


def test_what_fits_is_left_as_it_is():
    made, outcome = fitted(body="The jib had rust.", heading="Harbour > Crane", table=None, wrap=wrap, budget=64)
    assert made == wrap("Harbour > Crane\nThe jib had rust.") and outcome == "fits"


def test_the_outermost_headings_go_first_and_the_chunk_is_kept_whole():
    body = "The jib had rust and the slew ring needed grease before the June inspection."
    heading = "Operations manual > Harbour equipment > Cranes > Annual survey"
    budget = estimated_tokens(wrap(f"Cranes > Annual survey\n{body}"))
    made, outcome = fitted(body=body, heading=heading, table=None, wrap=wrap, budget=budget)
    assert made == wrap(f"Cranes > Annual survey\n{body}") and outcome == "shortened"


def test_then_the_heading_line_then_the_table_context_and_never_the_chunk():
    body = "Row 7: crane 4, rust, 2025-05-03."
    table = "Table context: Asset | Finding | Date\n\n" + body
    no_heading = estimated_tokens(wrap(table))
    made, outcome = fitted(body=body, heading="Survey", table=table, wrap=wrap, budget=no_heading)
    assert made == wrap(table) and outcome == "shortened"
    made, outcome = fitted(body=body, heading="Survey", table=table, wrap=wrap, budget=estimated_tokens(wrap(body)))
    assert made == wrap(body) and outcome == "dropped"


def test_a_heading_that_cannot_fit_in_any_length_is_dropped():
    body = "The jib had rust and the slew ring needed grease."
    made, outcome = fitted(body=body, heading="Survey", table=None, wrap=wrap, budget=estimated_tokens(wrap(body)))
    assert made == wrap(body) and outcome == "dropped"


def test_a_chunk_too_long_on_its_own_is_embedded_bare_and_counted():
    body = "word " * 100
    made, outcome = fitted(body=body, heading="Survey", table=None, wrap=wrap, budget=20)
    assert made == wrap(body) and outcome == "body_over"


class Windowed(HashEmbedder):
    max_input_tokens = 60

    def __init__(self):
        super().__init__()
        self.seen: list[str] = []

    async def embed(self, texts):
        self.seen.extend(texts)
        return await super().embed(texts)


DOC = ("# Operations manual\n\n## Harbour equipment and every crane the port operates\n\n"
       "### Annual survey of the cranes\n\n" + "The jib had rust and the slew ring needed grease before June. " * 8)


async def test_the_engine_fits_each_input_and_says_what_it_shortened():
    embedder = Windowed()
    # The source and date prefix takes its share of the window too.
    embedder.max_input_tokens = 70
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder, heading_context=True,
                                embedding_budget=True, chunk_target=150, contextual_embeddings=True).open()
    try:
        added = await engine.remember("default", DOC, source="manual.md")
    finally:
        await engine.close()
    assert all(estimated_tokens(text) <= 70 for text in embedder.seen)
    assert all("manual.md" in text for text in embedder.seen), "the source prefix is counted inside the budget"
    budget = added.embedding_context["budget"]
    assert budget["tokens"] == 70 and budget["method"] == BUDGET_VERSION
    assert budget["shortened"] + budget["dropped"] >= 1 and budget["body_over"] == 0
    assert budget["fits"] + budget["shortened"] + budget["dropped"] == added.chunks


async def test_without_the_budget_inputs_are_as_they_were():
    embedder = Windowed()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder, heading_context=True,
                                chunk_target=150).open()
    try:
        added = await engine.remember("default", DOC, source="manual.md")
    finally:
        await engine.close()
    assert any(estimated_tokens(text) > 60 for text in embedder.seen)
    assert "budget" not in (added.embedding_context or {})


async def test_a_budget_needs_an_embedder_that_declares_its_window():
    with pytest.raises(InvalidInput, match="window"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), embedding_budget=True)


async def test_the_budget_is_part_of_the_vector_writer_identity():
    from scone_memory.memory.vector_identity import writer_of

    plain = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), Windowed(), heading_context=True)
    budgeted = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), Windowed(), heading_context=True,
                            embedding_budget=True)
    assert writer_of(budgeted) == writer_of(plain) + ";budget=" + BUDGET_VERSION


def test_the_setting_is_read_from_the_environment():
    from scone_memory.runtime.config import ENGINE_SETTINGS, Settings

    assert "embedding_budget" in ENGINE_SETTINGS
    assert Settings.from_env({"SCONE_EMBEDDING_BUDGET": "1"}).embedding_budget is True
    assert Settings.from_env({}).embedding_budget is False


def test_the_local_bge_models_declare_their_window():
    from scone_memory.embedders.local import MAX_INPUT_TOKENS, MODELS

    assert MAX_INPUT_TOKENS["bge-small-en-v1.5"] == 512 and MAX_INPUT_TOKENS["bge-base-en-v1.5"] == 512
    assert set(MAX_INPUT_TOKENS) <= set(MODELS)


async def test_table_context_is_given_up_before_the_chunk():
    from types import SimpleNamespace

    from scone_memory.core.ports import NewEpisode
    from scone_memory.ingestion.batch import embedding_inputs

    content = "Row 7: crane 4, rust, 2025-05-03."
    header = "Table context: Asset tag | Finding recorded by the surveyor | Date of the inspection visit\n\n"

    async def with_headers(new, spans):
        return [header + content]

    runtime = SimpleNamespace(context_inputs=with_headers, heading_context=False, chunk_target=700,
                              embedding_budget=estimated_tokens(content), embed_text=lambda new, text: text,
                              count_tokens=None)
    new = NewEpisode(space="default", kind="file", content=content, source="sheet.csv", tags=(), metadata={},
                     created_at="2026-09-14T00:00:00Z", ingested_at="2026-09-14T00:00:00Z", content_hash="0" * 64)
    outcomes: dict[str, object] = {}
    [made] = await embedding_inputs(runtime, new, [(0, len(content.encode()))], [content], outcomes=outcomes)
    assert made == content and outcomes == {"dropped": 1}


class Counted(Windowed):
    """An embedder that counts its own tokens: here, one per space-separated word."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


async def test_an_embedder_that_counts_its_own_tokens_is_asked_instead_of_the_estimate():
    from scone_memory.ingestion.embedding_budget import TOKENIZER_VERSION
    from scone_memory.memory.vector_identity import writer_of

    embedder = Counted()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder, heading_context=True,
                                embedding_budget=True, chunk_target=150).open()
    try:
        added = await engine.remember("default", DOC, source="manual.md")
        identity = writer_of(engine)
    finally:
        await engine.close()
    # By words, every input fits 60 whole; by the estimate, three did not.
    assert added.embedding_context["budget"]["method"] == TOKENIZER_VERSION
    assert added.embedding_context["budget"]["fits"] == added.chunks
    assert identity.endswith(";budget=" + TOKENIZER_VERSION)


def test_fitting_counts_with_the_counter_it_is_given():
    body = "one two three four five"
    made, outcome = fitted(body=body, heading="a > b > c", table=None, wrap=lambda text: text, budget=6,
                           count=lambda text: len(text.split()))
    assert made == "c\none two three four five" and outcome == "shortened"


def test_the_local_model_counts_past_its_own_window(monkeypatch):
    """Needs the model already on disk; nothing is downloaded."""
    import os
    from pathlib import Path

    cache = Path(os.environ.get("SCONE_EMBED_CACHE", Path.home() / ".scone-memory" / "fastembed"))
    if not any(cache.glob("models--*bge-small-en-v1.5*")):
        pytest.skip("bge-small-en-v1.5 is not cached locally")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from scone_memory.embedders.local import LocalEmbedder

    try:
        embedder = LocalEmbedder("bge-small-en-v1.5", cache_dir=str(cache))
    except Exception as error:  # noqa: BLE001 - an unusable cache is a skip, not a failure
        pytest.skip(f"the cached model could not be opened offline: {error}")
    assert embedder.count_tokens("the crane survey found rust") == 7
    assert embedder.count_tokens("crane " * 700) == 702, "counted past the 512 the model reads"
    assert embedder.max_input_tokens == 512
