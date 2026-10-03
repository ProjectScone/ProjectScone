"""Questions each chunk answers, written by a model, kept only with their quote, and searched as a lane.

A passage is written in its author's words and asked about in the
reader's. "Is there a winter mooring ban?" shares no word with "The
harbour closes to sailing boats every November", so neither the text lane
nor a hashed vector finds the answer. RAGFlow writes questions for every
chunk at ingestion and searches them at many times the weight of the
content; LlamaIndex's questions-answered extractor puts them into the
text a node is embedded from. Both take the model's questions on trust.

Here a question is kept only with the sentence that answers it, copied
from the chunk: the anchoring the bench's question sets use
(``bench.questions.anchored``). A question whose sentence is not in the
chunk, is too short to tell one passage from another, or was never asked
is dropped and counted. The kept questions go into an index of their own
(``core.ports.QuestionIndex``), apart from the words the chunk is under,
so the recall's question lane finds the chunk by them and nothing else.
Nothing else changes: the stored text, the vectors, the context index and
the passage a recall returns are as ingestion left them, and forgetting
an episode drops its chunks' questions with the chunks.

The pass is opt-in twice over. It runs only by hand
(``engine.build_chunk_questions``, ``scone chunk-questions``, ``POST
/v1/chunk-questions``) with a model, and only on an engine with the
question lane on (``SCONE_QUESTION_LANE``), which is also what makes a
recall search the questions. A store without the index gets no model
calls and a report that says why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Optional, Sequence

from ..bench.questions import MAX_PER_CHUNK, MAX_QUESTION_CHARS, MIN_QUOTE_WORDS, anchored, loose_pairs, parse_pairs
from ..core import forget_after
from ..core.errors import InvalidInput
from ..core.ports import question_index
from ..core.timeutil import parse_rfc3339
from ..core.validation import check_space
from ..providers.llm import ChatError, ChatModel

if TYPE_CHECKING:
    from ..core.models import Chunk, Episode
    from ..memory.engine import MemoryEngine

VERSION = "chunk-questions-v1"
DEFAULT_PER_CHUNK = 3
#: Chunks one pass asks about; the rest are counted, and the report names
#: the chunk a later pass resumes after.
MAX_CHUNKS = 2_000
#: A chunk longer than this is not shown to the model; the report counts it.
MAX_CHUNK_BYTES = 8_000

#: Worded apart from the bench's prompt on purpose: a measurement whose
#: questions come from the same prompt as the index's would find its own
#: questions, not the passages.
SYSTEM = ("You read one passage from someone's stored notes, documents or conversations, and list the questions "
          "a person could later ask that this passage answers. Phrase each question the way someone looking for "
          "the information would ask it, in their own everyday words rather than only the passage's. "
          "Reply with JSON only: a list of objects, each {\"question\": string, \"quote\": string}, where the quote "
          "is the one sentence of the passage that answers the question, copied exactly. "
          "No preamble, no markdown fence.")


@dataclass(frozen=True)
class ChunkQuestions:
    """The questions kept for one chunk, each with the sentence that answers it."""

    chunk_id: int
    episode_id: int
    questions: tuple[str, ...]
    quotes: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return {"chunk_id": self.chunk_id, "episode_id": self.episode_id, "questions": list(self.questions),
                "quotes": list(self.quotes)}


@dataclass(frozen=True)
class QuestionLaneReport:
    space: str
    model: str
    per_chunk: int
    #: Chunks the pass could ask about: in the episodes named, after ``after_chunk``.
    chunks_total: int = 0
    chunks_asked: int = 0
    #: Chunks past ``max_chunks``, not looked at; ``resume_after`` names where they start.
    chunks_cut: int = 0
    resume_after: Optional[int] = None
    #: Chunks over MAX_CHUNK_BYTES, not shown to the model.
    skipped_long: int = 0
    #: Calls made, including those that failed: one per chunk asked.
    model_calls: int = 0
    calls_failed: int = 0
    dropped_unparsed: int = 0
    #: Replies not valid as one list, read object by object (``bench.questions.loose_pairs``).
    read_loosely: int = 0
    #: Objects in those replies that could not be read as a pair, passed over.
    dropped_unread: int = 0
    dropped_unquoted: int = 0
    dropped_unasked: int = 0
    #: Pairs past ``per_chunk`` in one reply.
    dropped_extra: int = 0
    #: A question the model wrote twice for one chunk.
    dropped_repeated: int = 0
    #: Chunks whose question index now holds this pass's questions.
    chunks_indexed: int = 0
    #: Chunks whose reply was read and kept no question: whatever an earlier
    #: pass wrote for them is removed, so each holds this pass's questions only.
    chunks_kept_none: int = 0
    #: Chunks forgotten while the model was answering: nothing is written for them.
    chunks_gone: int = 0
    #: Episodes past their ``forget_after``, whose chunks were not shown to the model.
    episodes_past_forget_after: int = 0
    kept: tuple[ChunkQuestions, ...] = ()
    #: Whether the document store keeps the question index the lane is searched through.
    kept_lane: bool = True
    seconds: float = 0.0
    reasons: tuple[str, ...] = field(default_factory=tuple)
    version: str = VERSION

    @property
    def questions(self) -> int:
        return sum(len(kept.questions) for kept in self.kept)

    def record(self) -> dict[str, object]:
        return {"space": self.space, "model": self.model, "per_chunk": self.per_chunk, "chunks_total": self.chunks_total,
                "chunks_asked": self.chunks_asked, "chunks_cut": self.chunks_cut, "resume_after": self.resume_after,
                "skipped_long": self.skipped_long, "model_calls": self.model_calls, "calls_failed": self.calls_failed,
                "dropped_unparsed": self.dropped_unparsed, "read_loosely": self.read_loosely,
                "dropped_unread": self.dropped_unread, "dropped_unquoted": self.dropped_unquoted,
                "dropped_unasked": self.dropped_unasked, "dropped_extra": self.dropped_extra,
                "dropped_repeated": self.dropped_repeated, "chunks_indexed": self.chunks_indexed,
                "chunks_kept_none": self.chunks_kept_none, "chunks_gone": self.chunks_gone,
                "episodes_past_forget_after": self.episodes_past_forget_after, "questions": self.questions, "kept": [kept.record() for kept in self.kept], "kept_lane": self.kept_lane,
                "seconds": round(self.seconds, 3), "reasons": list(self.reasons), "version": self.version}

    def text(self) -> str:
        said = (f"{self.questions} question(s) kept for {self.chunks_indexed} of {self.chunks_asked} chunk(s) asked "
                f"({self.chunks_total} in scope), {self.per_chunk} asked per chunk, written by {self.model} in "
                f"{self.model_calls} call(s) and {self.seconds:.1f}s; dropped {self.dropped_unquoted} whose quote was not "
                f"in the chunk or under {MIN_QUOTE_WORDS} words, {self.dropped_unasked} with no question or one over "
                f"{MAX_QUESTION_CHARS} characters, {self.dropped_extra} over the per-chunk limit, {self.dropped_repeated} "
                f"repeated, and {self.dropped_unparsed} repl(ies) not written as asked; {self.read_loosely} read object by "
                f"object, with {self.dropped_unread} object(s) in them that could not be read; {self.skipped_long} chunk(s) too "
                f"long to show, {self.calls_failed} call(s) failed; {self.chunks_kept_none} chunk(s) kept no question and "
                f"hold none now; {self.chunks_gone} chunk(s) forgotten while the model answered")
        if self.chunks_cut:
            said += f"; {self.chunks_cut} chunk(s) past max_chunks, resume after chunk {self.resume_after}"
        if self.episodes_past_forget_after:
            said += f"; {self.episodes_past_forget_after} episode(s) past their forget_after not read"
        return "; ".join((said, *self.reasons))


async def _episodes(engine: "MemoryEngine", space: str,
                    episode_ids: Optional[Sequence[int]]) -> tuple[list["Episode"], int]:
    """The episodes a pass reads, and how many were left out because their
    ``forget_after`` had come. Named episodes go through ``engine.episode``,
    which refuses an overdue one as Gone."""
    if episode_ids is not None:
        return [await engine.episode(space, episode_id) for episode_id in dict.fromkeys(episode_ids)], 0
    counts = await engine.documents.counts(space)
    moment = parse_rfc3339(engine.clock())
    read, overdue = [], 0
    for episode in await engine.documents.recent_episodes(space, counts.episodes):
        if forget_after.is_due(episode.metadata, moment):
            overdue += 1
            continue
        read.append(episode)
    return read, overdue


async def build_chunk_questions(engine: "MemoryEngine", space: str, model: ChatModel, *,
                                per_chunk: int = DEFAULT_PER_CHUNK, max_chunks: int = MAX_CHUNKS,
                                after_chunk: Optional[int] = None, episode_ids: Optional[Sequence[int]] = None,
                                model_name: str = "") -> QuestionLaneReport:
    """Ask the model for questions each chunk answers, keep those anchored in
    the chunk, and index them beside the chunk's context; chunks in id order."""
    check_space(space)
    if not getattr(engine, "question_lane", False):
        raise InvalidInput("the question lane is off; start with SCONE_QUESTION_LANE=1 (question_lane=True) to write it")
    if isinstance(per_chunk, bool) or not isinstance(per_chunk, int) or not 1 <= per_chunk <= MAX_PER_CHUNK:
        raise InvalidInput(f"per_chunk must be between 1 and {MAX_PER_CHUNK}")
    if isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or not 1 <= max_chunks <= MAX_CHUNKS:
        raise InvalidInput(f"max_chunks must be between 1 and {MAX_CHUNKS}")
    name = model_name or type(model).__name__
    keeper = question_index(engine.documents)
    if keeper is None:
        return QuestionLaneReport(space, name, per_chunk, kept_lane=False, reasons=(
            f"the {engine.documents.name} document store keeps no question index, so there is nowhere to put "
            "questions and no model call was made",))
    started = time.perf_counter()
    rows: list[tuple["Chunk", "Episode"]] = []
    episodes, overdue = await _episodes(engine, space, episode_ids)
    for episode in episodes:
        rows.extend((chunk, episode) for chunk in await engine.documents.chunks_of(space, episode.episode_id)
                    if after_chunk is None or chunk.chunk_id > after_chunk)
    rows.sort(key=lambda row: row[0].chunk_id)
    counts = dict.fromkeys(("asked", "skipped_long", "failed", "unparsed", "loosely", "unread", "unquoted", "unasked",
                            "extra", "repeated", "indexed", "kept_none", "gone"), 0)
    kept: list[ChunkQuestions] = []
    resume_after: Optional[int] = None
    examined = 0
    for chunk, episode in rows:
        if counts["asked"] >= max_chunks:
            resume_after = rows[examined - 1][0].chunk_id
            break
        examined += 1
        if len(chunk.text.encode("utf-8")) > MAX_CHUNK_BYTES:
            counts["skipped_long"] += 1
            continue
        counts["asked"] += 1
        user = (f"Passage:\n{chunk.text}\n\nWrite {per_chunk} question(s) this passage answers, each with the "
                f"sentence of the passage that answers it, copied exactly.")
        try:
            reply = await model.complete(SYSTEM, user)
        except ChatError:
            counts["failed"] += 1
            continue
        pairs = parse_pairs(reply)
        if pairs is None:
            loose = loose_pairs(reply)
            if loose is None:
                counts["unparsed"] += 1
                continue
            pairs = list(loose.pairs)
            counts["loosely"] += 1
            counts["unread"] += loose.unread
        found = anchored(pairs, chunk.text, per_chunk)
        counts["unasked"] += found.unasked
        counts["unquoted"] += found.unquoted
        counts["extra"] += found.extra
        questions: list[str] = []
        quotes: list[str] = []
        seen: set[str] = set()
        for question, quote in found.pairs:
            key = " ".join(question.split()).casefold()
            if key in seen:
                counts["repeated"] += 1
                continue
            seen.add(key)
            questions.append(question)
            quotes.append(quote)
        # A reply read replaces the chunk's questions, none included; the store
        # writes nothing for a chunk forgotten while the model answered.
        if not await keeper.index_questions(space, chunk.chunk_id, questions):
            counts["gone"] += 1
            continue
        if not questions:
            counts["kept_none"] += 1
            continue
        counts["indexed"] += 1
        kept.append(ChunkQuestions(chunk.chunk_id, episode.episode_id, tuple(questions), tuple(quotes)))
    return QuestionLaneReport(
        space, name, per_chunk, chunks_total=len(rows), chunks_asked=counts["asked"],
        chunks_cut=len(rows) - examined, resume_after=resume_after, skipped_long=counts["skipped_long"],
        model_calls=counts["asked"], calls_failed=counts["failed"], dropped_unparsed=counts["unparsed"],
        read_loosely=counts["loosely"], dropped_unread=counts["unread"], dropped_unquoted=counts["unquoted"],
        dropped_unasked=counts["unasked"], dropped_extra=counts["extra"], dropped_repeated=counts["repeated"],
        chunks_indexed=counts["indexed"], chunks_kept_none=counts["kept_none"], chunks_gone=counts["gone"], episodes_past_forget_after=overdue, kept=tuple(kept),
        seconds=time.perf_counter() - started)
