"""The memory model, shared by every store.

Two kinds of memory live side by side. Episodes are what happened: a
note, a chat turn, a file, kept verbatim and split into chunks that
point back into the original bytes. Facts are what is true: subject,
predicate, object, with the interval over which it held. An episode
never changes; a fact closes when a later one supersedes it, and the
reason is kept.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

#: The same vocabulary as the Rust product's schema CHECK, so an episode
#: means the same thing on both sides (shared spec, section 1).
EpisodeKind = Literal["note", "file", "conversation", "observation", "connector"]
#: active: holds now. closed: held until valid_until. proposed: a model's
#: extraction awaiting a person; outside the ledger until approved.
#: declined: a proposal a person rejected; never held.
FactStatus = Literal["active", "closed", "proposed", "declined"]
#: stated: a person or a trusted program asserted it. extracted: a model
#: read it out of an episode. inferred: derived from other facts.
#: A surface must never show an extracted or inferred fact without saying so.
FactOrigin = Literal["stated", "extracted", "inferred"]

MAX_CONTENT_BYTES = 2_000_000


class Attachment(BaseModel):
    """Bytes an episode carries, named by the SHA-256 of those bytes. The
    same screenshot remembered twice is one attachment with two
    references; an id can be re-fetched forever because nothing stored is
    rewritten."""

    model_config = ConfigDict(frozen=True)

    attachment_id: str
    media_type: str
    bytes: int
    filename: Optional[str] = None


class Episode(BaseModel):
    model_config = ConfigDict(frozen=True)

    episode_id: int
    space: str
    kind: EpisodeKind = "note"
    content: str
    content_hash: str
    source: Optional[str] = None
    tags: tuple[str, ...] = ()
    #: Scope dimensions the caller filters on: ``user_id``, ``agent_id``,
    #: ``session_id`` by convention, any short string key in practice.
    #: The space is the hard boundary; metadata partitions inside it.
    metadata: dict[str, str] = Field(default_factory=dict)
    #: When it happened. Facts distilled from it inherit this.
    created_at: str
    #: When the engine first saw it; the ordering key for "recent".
    ingested_at: str
    #: Filled by the engine from the blob store, not by the document
    #: store, so a backend needs no column for it.
    attachments: tuple[Attachment, ...] = ()


class Chunk(BaseModel):
    """A span of an episode. ``start`` and ``end`` are UTF-8 byte offsets,
    half-open, so ``content.encode()[start:end].decode()`` is exactly
    ``text`` and the span means the same thing in the Rust product.
    (Invariant I1: nothing stored is rewritten.)
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: int
    episode_id: int
    space: str
    ordinal: int
    start: int
    end: int
    text: str
    created_at: str


class Fact(BaseModel):
    fact_id: int
    space: str
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    valid_from: str
    valid_until: Optional[str] = None
    status: FactStatus = "active"
    closed_reason: Optional[str] = None
    source_episode_id: Optional[int] = None
    origin: FactOrigin = "stated"
    #: The exact substring of the source episode this claim rests on, when
    #: it came from one. Checked against the episode at assertion; a claim
    #: with a source but no quote is ungrounded and surfaces say so.
    quote: Optional[str] = None
    #: The fact that truncated or bounded this one, as an id, so the
    #: relation is data and not a sentence to be parsed.
    superseded_by: Optional[int] = None
    #: Set when a person suppressed this fact from recall. The interval
    #: and status are untouched: exclusion is a policy, not a rewrite of
    #: history. Cleared by ``include``.
    excluded_reason: Optional[str] = None

    @property
    def excluded(self) -> bool:
        return self.excluded_reason is not None

    @property
    def grounded(self) -> Optional[bool]:
        """True when a source quote is stored, False when there is a source
        but no quote, None when the claim was stated with no source."""
        if self.source_episode_id is None:
            return None
        return self.quote is not None

    @property
    def in_ledger(self) -> bool:
        """Proposed and declined facts never held; they are not part of
        the partition of time and never answer a question."""
        return self.status in ("active", "closed")

    def holds_at(self, when: str) -> bool:
        from .timeutil import parse_rfc3339

        if not self.in_ledger:
            return False
        t = parse_rfc3339(when)
        if parse_rfc3339(self.valid_from) > t:
            return False
        return self.valid_until is None or parse_rfc3339(self.valid_until) > t


LinkKind = Literal["extends", "derived_from", "contradicts", "supports"]
LINK_KINDS: tuple[str, ...] = ("extends", "derived_from", "contradicts", "supports")
#: The kinds that make one fact depend on another; a cycle among them is refused.
DEPENDENCY_KINDS: tuple[str, ...] = ("extends", "derived_from")


class FactLink(BaseModel):
    """A typed relation between two facts of one space, and the evidence it
    rests on. ``from_fact`` is the one making the claim about ``to_fact``:
    an extension extends, a derivation is derived from, and so on."""

    link_id: int
    space: str
    from_fact: int
    to_fact: int
    kind: LinkKind
    created_at: str
    source_episode_id: Optional[int] = None
    quote: Optional[str] = None


class RecallItem(BaseModel):
    chunk_id: int
    episode_id: int
    text: str
    #: Within-query rank score; the top item is 1.0 whatever its relevance.
    score: float
    #: Cosine from the vector lane when that lane saw the chunk, else None.
    similarity: Optional[float] = None
    #: 1-based rank in each lane that returned this chunk ("vector",
    #: "text"), so a caller can see why an item is here.
    lanes: dict[str, int] = Field(default_factory=dict)
    created_at: str
    source: Optional[str] = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)


class RecallResult(BaseModel):
    #: Id of the evidence event recorded for this recall, when an event
    #: log is attached; feedback refers to it.
    event_id: Optional[int] = None
    items: list[RecallItem] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    #: With ``history``: the closed facts that preceded the matched ones for
    #: the same subject and predicate, oldest first. Empty otherwise.
    history: list[Fact] = Field(default_factory=list)
    #: Best cosine the vector lane saw for this query, before fusion; None
    #: when the lane was degraded or the space had nothing to compare.
    top_similarity: Optional[float] = None
    #: Experiment 9. True when the engine has a similarity floor and
    #: top_similarity fell below it (or nothing was found): the reader is
    #: told the evidence is weak. False when it cleared the floor. None
    #: when no floor is configured or the vector lane could not judge.
    low_confidence: Optional[bool] = None
    #: Lanes that failed and were left out, named so a caller can tell a
    #: thin answer from a broken one.
    degraded: list[str] = Field(default_factory=list)
    returned_bytes: int = 0
    space_bytes: int = 0

    @property
    def context_reduction(self) -> float:
        if self.space_bytes == 0:
            return 0.0
        return max(0.0, 1.0 - self.returned_bytes / self.space_bytes)


class Added(BaseModel):
    episode_id: int
    deduplicated: bool = False
    chunks: int = 0


class Status(BaseModel):
    space: str
    episodes: int = 0
    chunks: int = 0
    bytes: int = 0
    #: Proposed facts awaiting a person.
    pending_review: int = 0
    revision: int = 0
    embedder: str = ""
    document_store: str = ""
    vector_index: str = ""


class DecisionOutcome(BaseModel):
    """What one decision in a batch did, keyed by the id the caller sent.
    Never by the id that came back: approving a duplicate returns the fact
    already held, which is a different row."""

    fact_id: int
    outcome: str
    #: Set when a proposal was folded into a fact already in the ledger.
    held_fact_id: Optional[int] = None
    error: Optional[str] = None


class BatchDecision(BaseModel):
    results: list[DecisionOutcome]
    #: How many rows the ledger actually changed.
    applied: int
    revision: int
