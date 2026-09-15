"""Questions a corpus can answer, written by a local model and anchored to its own words.

Retrieval on a corpus of one's own has no benchmark: LongMemEval-S
measures conversations, the code bench measures functions. The reference
framework makes a dataset by asking a model for questions about each
node, and the pairs are then taken on trust. Here a question is kept
only with the sentence that answers it, quoted verbatim from the chunk
the model was shown; a question whose quote is not in that chunk is
dropped and counted, so every kept question can be checked by anyone
holding the text. Measurement then needs no chunk ids: a returned
passage that holds the quote is a hit, so one set measures the same
corpus stored with a different chunker, store or embedder, as long as
the text is the same.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Optional, Sequence, TYPE_CHECKING

from ..core.errors import InvalidInput
from ..providers.llm import ChatError, ChatModel

if TYPE_CHECKING:
    from ..ingestion.pdf import PdfParser
    from ..memory.engine import MemoryEngine

VERSION = "questions-v1"
#: Chunks one set is written from; a larger corpus is sampled, and the
#: set says how many chunks there were.
MAX_CHUNKS = 200
#: A chunk longer than this is not shown to the model; the set counts it.
MAX_CHUNK_BYTES = 8_000
#: Questions asked of one chunk, at most.
MAX_PER_CHUNK = 5
MAX_QUESTION_CHARS = 400
#: A quote shorter than this tells no passage from another ("the" is in
#: every chunk), so it is not a truth a question can rest on.
MIN_QUOTE_WORDS = 4
#: Files a corpus root is read up to, and the largest file read.
MAX_FILES = 2_000
MAX_FILE_BYTES = 400_000
TEXT_SUFFIXES = (".md", ".txt", ".rst", ".markdown")

SYSTEM = ("You write questions that a passage answers. Reply with JSON only: a list of objects, "
          "each {\"question\": string, \"quote\": string}. The quote is one sentence copied exactly, "
          "character for character, from the passage, and it answers the question on its own. "
          "Ask about what the passage states, never about the passage or the text itself. "
          "No preamble, no markdown fence.")


@dataclass(frozen=True)
class Question:
    question: str
    #: The sentence that answers it, as it stands in the corpus.
    quote: str
    #: Where the chunk came from, when its episode has a source.
    source: Optional[str]
    #: The episode the model was shown; a reader's convenience, not the truth.
    episode_id: int


@dataclass(frozen=True)
class QuestionSet:
    corpus: str
    model: str
    questions: tuple[Question, ...]
    chunks_total: int
    chunks_asked: int
    per_chunk: int
    seed: int
    dropped_unparsed: int
    dropped_unquoted: int
    skipped_long: int
    calls_failed: int
    #: Questions the model wrote empty or over MAX_QUESTION_CHARS, whatever their quote.
    dropped_unasked: int = 0
    version: str = VERSION

    def as_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "QuestionSet":
        if payload.get("version") != VERSION:
            raise InvalidInput(f"question set version {payload.get('version')!r} is not {VERSION}")
        raw = payload.get("questions")
        if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
            raise InvalidInput("a question set's questions must be a list of objects")
        questions = []
        for item in raw:
            question, quote, source, episode_id = item.get("question"), item.get("quote"), item.get("source"), item.get("episode_id")
            if (not isinstance(question, str) or not isinstance(quote, str) or not (source is None or isinstance(source, str))
                    or type(episode_id) is not int or set(item) != {"question", "quote", "source", "episode_id"}):
                raise InvalidInput("a question is a question, a quote, a source or null, and an episode id, and nothing else")
            questions.append(Question(question, quote, source, episode_id))
        counts = {key: payload.get(key) for key in ("chunks_total", "chunks_asked", "per_chunk", "seed", "dropped_unparsed",
                                                     "dropped_unquoted", "skipped_long", "calls_failed")}
        if any(type(value) is not int for value in counts.values()) or not isinstance(payload.get("corpus"), str) \
                or not isinstance(payload.get("model"), str) or type(payload.get("dropped_unasked", 0)) is not int:
            raise InvalidInput("a question set's counts are integers and its corpus and model are text")
        return cls(corpus=payload["corpus"], model=payload["model"], questions=tuple(questions),  # type: ignore[arg-type]
                   dropped_unasked=payload.get("dropped_unasked", 0), **counts)  # type: ignore[arg-type]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_payload(), ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "QuestionSet":
        try:
            payload = json.loads(Path(path).read_text())
        except (OSError, ValueError) as error:
            raise InvalidInput(f"cannot read a question set from {path}: {error}") from error
        if not isinstance(payload, dict):
            raise InvalidInput(f"{path} does not hold a question set")
        return cls.from_payload(payload)

    def text(self) -> str:
        return (f"{len(self.questions)} question(s) from {self.chunks_asked} of {self.chunks_total} chunk(s) "
                f"of {self.corpus}, {self.per_chunk} asked per chunk, written by {self.model}; dropped "
                f"{self.dropped_unquoted} whose quote was not in the chunk or under {MIN_QUOTE_WORDS} words, "
                f"{self.dropped_unasked} with no question or one over {MAX_QUESTION_CHARS} characters, and "
                f"{self.dropped_unparsed} the model did not write as asked; {self.skipped_long} chunk(s) too long to show, "
                f"{self.calls_failed} call(s) failed")


def _normal(text: str) -> str:
    return " ".join(text.split())


def parse_pairs(reply: str) -> Optional[list[tuple[str, str]]]:
    """The (question, quote) pairs in a reply, or None when it is not a
    list of them. A fence or a sentence around the JSON is tolerated --
    the first bracket that opens a JSON list is the one read, so a
    bracket in the prose around it is not -- and anything else is the
    model not writing what was asked."""
    decoder = json.JSONDecoder()
    parsed: object = None
    start = reply.find("[")
    while start >= 0:
        try:
            parsed, _ = decoder.raw_decode(reply, start)
            break
        except ValueError:
            start = reply.find("[", start + 1)
    if not isinstance(parsed, list):
        return None
    pairs: list[tuple[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            return None
        question, quote = item.get("question"), item.get("quote")
        if not isinstance(question, str) or not isinstance(quote, str):
            return None
        pairs.append((question.strip(), quote.strip()))
    return pairs


@dataclass(frozen=True)
class Anchored:
    """The pairs of one reply that a question can rest on, and what was dropped."""

    #: (question, quote) with the quote's whitespace settled as it stands in the chunk.
    pairs: tuple[tuple[str, str], ...]
    #: No question, or one over MAX_QUESTION_CHARS.
    unasked: int
    #: A quote under MIN_QUOTE_WORDS or not in the chunk.
    unquoted: int
    #: Pairs past the number asked for, not looked at.
    extra: int


def anchored(pairs: Sequence[tuple[str, str]], text: str, per_chunk: int) -> Anchored:
    """The first ``per_chunk`` pairs whose question was asked and whose quote
    is in ``text`` verbatim, whitespace aside; every other pair counted."""
    normal_text = _normal(text)
    kept: list[tuple[str, str]] = []
    unasked = unquoted = 0
    for question, quote in pairs[:per_chunk]:
        if not question or len(question) > MAX_QUESTION_CHARS:
            unasked += 1
            continue
        if len(quote.split()) < MIN_QUOTE_WORDS or _normal(quote) not in normal_text:
            unquoted += 1
            continue
        kept.append((question, _normal(quote)))
    return Anchored(tuple(kept), unasked, unquoted, max(0, len(pairs) - per_chunk))


async def chunks_in(engine: "MemoryEngine", space: str) -> list[tuple[int, Optional[str], int, str]]:
    """Every chunk of the space as (episode id, source, chunk id, text),
    oldest episode first."""
    rows: list[tuple[int, Optional[str], int, str]] = []
    counts = await engine.documents.counts(space)
    episodes = sorted(await engine.documents.recent_episodes(space, max(counts.episodes, 1)),
                      key=lambda episode: (episode.created_at, episode.episode_id))
    for episode in episodes:
        # A document read through the file lane names its original by hash;
        # the path it was stored under is on the episode as its filename.
        source = episode.source
        if source is not None and source.startswith("attachment:"):
            source = episode.metadata.get("document_filename") or source
        for chunk in await engine.documents.chunks_of(space, episode.episode_id):
            rows.append((episode.episode_id, source, chunk.chunk_id, chunk.text))
    return rows


async def write_questions(engine: "MemoryEngine", space: str, model: ChatModel, *, corpus: str = "",
                          model_name: str = "", per_chunk: int = 2, max_chunks: int = MAX_CHUNKS,
                          seed: int = 42) -> QuestionSet:
    """Ask the model for questions about a sample of the space's chunks,
    and keep the ones whose answering sentence is in the chunk shown."""
    if not 1 <= per_chunk <= MAX_PER_CHUNK:
        raise InvalidInput(f"per_chunk must be between 1 and {MAX_PER_CHUNK}")
    if max_chunks < 1:
        raise InvalidInput("max_chunks must be at least 1")
    rows = await chunks_in(engine, space)
    fitting = [row for row in rows if len(row[3].encode("utf-8")) <= MAX_CHUNK_BYTES]
    skipped_long = len(rows) - len(fitting)
    chosen = fitting if len(fitting) <= max_chunks else random.Random(seed).sample(fitting, max_chunks)
    chosen.sort(key=lambda row: (row[0], row[2]))
    kept: list[Question] = []
    seen: set[str] = set()
    unparsed = unquoted = unasked = failed = 0
    for episode_id, source, _, text in chosen:
        user = (f"Passage:\n{text}\n\nWrite {per_chunk} question(s) this passage answers, each with the exact "
                f"sentence from the passage that answers it.")
        try:
            reply = await model.complete(SYSTEM, user)
        except ChatError:
            failed += 1
            continue
        pairs = parse_pairs(reply)
        if pairs is None:
            unparsed += 1
            continue
        found = anchored(pairs, text, per_chunk)
        unasked += found.unasked
        unquoted += found.unquoted
        for question, quote in found.pairs:
            key = _normal(question).lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(Question(question, quote, source, episode_id))
    return QuestionSet(corpus=corpus, model=model_name or type(model).__name__, questions=tuple(kept),
                       chunks_total=len(rows), chunks_asked=len(chosen), per_chunk=per_chunk, seed=seed,
                       dropped_unparsed=unparsed, dropped_unquoted=unquoted, skipped_long=skipped_long,
                       calls_failed=failed, dropped_unasked=unasked)


@dataclass(frozen=True)
class QuestionReport:
    corpus: str
    questions: int
    ks: tuple[int, ...]
    #: Share of questions whose quote is in a returned passage within the top k.
    quote_at: dict[int, float] = field(default_factory=dict)
    #: Share whose source (file) is among the top k, where the question has one.
    source_at: dict[int, float] = field(default_factory=dict)
    with_source: int = 0
    mrr: float = 0.0
    #: Questions whose quote no returned passage held, at the largest k.
    unfound: int = 0
    version: str = VERSION

    def as_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["quote_at"] = {str(k): v for k, v in self.quote_at.items()}
        payload["source_at"] = {str(k): v for k, v in self.source_at.items()}
        return payload

    def text(self) -> str:
        if not self.questions:
            return f"no questions asked of {self.corpus}"
        parts = ", ".join(f"quote in top {k} {self.quote_at[k]:.0%}" for k in self.ks)
        sources = (", ".join(f"source in top {k} {self.source_at[k]:.0%}" for k in self.ks)
                   + f" (of {self.with_source} with a source)") if self.with_source else "no sources"
        return (f"{self.questions} question(s) of {self.corpus}: {parts}; MRR {self.mrr:.3f}; "
                f"{self.unfound} unfound at {max(self.ks)}; {sources}")


async def measure(engine: "MemoryEngine", space: str, questions: QuestionSet, *,
                  ks: Sequence[int] = (1, 5, 10), limit: Optional[int] = None) -> QuestionReport:
    """Ask every question of the space and score the passages that came
    back by whether they hold the quote."""
    ks = tuple(sorted(set(ks)))
    if not ks or ks[0] < 1:
        raise InvalidInput("ks must be positive")
    depth = limit or ks[-1]
    if depth < ks[-1]:
        raise InvalidInput("limit must reach the largest k")
    ranks: list[Optional[int]] = []
    source_ranks: list[Optional[int]] = []
    for question in questions.questions:
        items = (await engine.recall(space, question.question, limit=depth)).items
        rank = next((position for position, item in enumerate(items, 1) if question.quote in _normal(item.text)), None)
        ranks.append(rank)
        if question.source is not None:
            source_ranks.append(next((position for position, item in enumerate(items, 1)
                                      if item.source == question.source), None))
    asked = len(ranks)
    if not asked:
        return QuestionReport(questions.corpus, 0, ks)
    return QuestionReport(
        corpus=questions.corpus, questions=asked, ks=ks,
        quote_at={k: sum(1 for r in ranks if r is not None and r <= k) / asked for k in ks},
        source_at={k: sum(1 for r in source_ranks if r is not None and r <= k) / len(source_ranks) for k in ks}
        if source_ranks else {},
        with_source=len(source_ranks),
        mrr=sum(1 / r for r in ranks if r is not None) / asked,
        unfound=sum(1 for r in ranks if r is None))


async def store_corpus(engine: "MemoryEngine", space: str, root: str | Path,
                       suffixes: Sequence[str] = TEXT_SUFFIXES, pdf_parser: Optional["PdfParser"] = None) -> dict[str, int]:
    """Every text file under a root, and every PDF where the optional
    parser is installed, stored with its path from the root as its
    source; in a settled order, so two stores of one root hold the same
    text. Hidden directories under the root are skipped; the root's own
    ancestors are not looked at, so a root under one is read. Up to
    MAX_FILES files, the rest counted. ``pdf_parser`` chooses how PDFs are
    read, so one question set can measure two readers of the same
    documents."""
    base = Path(root)
    wanted = tuple(suffixes) + (".pdf",)
    files: list[Path] = []
    total = 0
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in wanted:
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(base).parts):
            continue
        total += 1
        if len(files) < MAX_FILES:
            files.append(path)
    counts = {"files": 0, "files_total": total, "files_cut": 0, "pdf_unread": 0}
    for path in files:
        if path.stat().st_size > MAX_FILE_BYTES:
            counts["files_cut"] += 1
            continue
        source = str(path.relative_to(base))
        if path.suffix.lower() == ".pdf":
            try:
                from ..ingestion.files import ingest_document
                from ..ingestion.formats.registry import BuiltinDocumentParser

                await ingest_document(engine, space, path.read_bytes(), filename=source,
                                      parser=BuiltinDocumentParser(pdf_parser=pdf_parser))
            except InvalidInput:
                counts["pdf_unread"] += 1
                continue
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            await engine.remember(space, text, kind="file", source=source)
        counts["files"] += 1
    return counts
