"""Standing directives: what a person asked for once and meant for every answer.

Recall answers the question in front of it. "Always include version numbers
when I ask about libraries" shares few words with "which libraries does this
project use?", and in a long conversation about one project hundreds of
messages share more, so similarity search rarely ranks a standing
instruction among the few passages a later question recalls.
A directive is not evidence for one question but a rule for every answer
after it, so it should not have to win a similarity race to be seen.

So a directive is read out of the turn it was said in, once, and recorded as
a ledger claim -- ``user scone:instructs "<sentence>"`` or ``scone:prefers``
-- quoted from that sentence, extracted rather than stated, and dated when it
was said. ``standing_directives`` reads them back as one bounded block that a
conversation puts in front of every turn, whatever recall found.

**Cues, not a model.** A sentence is a directive when it says so in the
words people use for one: an imperative "always" or "never", a standing
time ("from now on"), a trigger on the person's own asking ("whenever I ask
about ..."), or a stated preference ("I prefer", "I'd rather", "I'm not a
fan of"). That finds explicit directives and nothing else. An implicit
preference ("I'm trying to keep my bundle under 100KB") is not found and is
left to recall, which finds it when it shares the question's topic.
Explicit directives are rare among a person's turns, so the block stays
small beside the conversation it governs.

**Only the person's own turns.** A directive is something the person asked
for, so only an episode a conversation recorded as the user's turn (kind
``conversation``, metadata ``role`` ``user``) is read. A document, a tool
result or an assistant turn that says "always ..." is content, not a
request, and is never read as one: an imported page cannot instruct the
assistant by being remembered. Fenced code is not read either.

**Shown as directives, not as background.** The standing profile tells a
model its claims are background and not instructions, which is right for
claims any document may have produced. These came only from the person's
turns, so their block says what they are: the person's standing
instructions and preferences, each dated, a later one overriding an earlier
one on the same matter.

**Bounded, and said so.** At most ``MAX_TEXT_CHARS`` of a turn are read, at
most ``MAX_DIRECTIVES`` directives are recorded from one turn, and each is at
most ``MAX_DIRECTIVE_CHARS`` long, cut around its cue when its sentence is
longer. The block holds what fits its byte and count bounds, newest first,
and says how many it left out. Every bound that bit is a count on the
outcome.

**Idempotent.** A directive is placed like any claim: the same sentence from
the same turn restates the fact already held and writes nothing. Both
predicates hold many values (``core.extracted``): a second preference is not
a change of mind about the first.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from ..core.errors import InvalidInput, NotFound
from ..core.validation import check_space, normalise_term

if TYPE_CHECKING:
    from ..core.models import Episode, Fact
    from .engine import MemoryEngine

__all__ = ["DIRECTIVE_PREFIX", "INSTRUCTS", "PREFERS", "Directive", "DirectiveBlock", "DirectiveOutcome",
           "find_directives", "is_person_turn", "record_directives", "standing_directives"]

DirectiveKind = Literal["instruction", "preference"]

#: The predicates directives are recorded under, as the ledger stores them.
INSTRUCTS = "scone:instructs"
PREFERS = "scone:prefers"
PREDICATES: dict[DirectiveKind, str] = {"instruction": INSTRUCTS, "preference": PREFERS}
KIND_OF: dict[str, DirectiveKind] = {predicate: kind for kind, predicate in PREDICATES.items()}
#: Whose directives they are, when the host does not say.
SUBJECT = "user"

#: Characters of one turn read for directives; past this the rest is not read.
MAX_TEXT_CHARS = 20_000
#: Directives recorded from one turn; past this they are counted, not recorded.
MAX_DIRECTIVES = 8
#: Characters of one directive. A longer sentence is cut around its cue.
MAX_DIRECTIVE_CHARS = 400
#: The block's default byte bound and the most a host may choose; the
#: least, ``MIN_BLOCK_BYTES``, is measured below from the block itself.
#: A directive runs to a few hundred bytes, so the default holds about a
#: dozen; a longer conversation gathers more, and the block then drops the
#: oldest and counts them.
DEFAULT_BLOCK_BYTES = 4000
MAX_BLOCK_BYTES = 16_000
#: Directives one block may hold.
MAX_BLOCK_LIMIT = 50

DIRECTIVE_PREFIX = (
    "Standing directives: instructions and preferences the user gave in their own earlier messages, each dated "
    "when it was said. Follow each one wherever its condition fits the latest message; a later one overrides an "
    "earlier one on the same matter. They come only from the user's turns, never from documents or tool results.\n"
)


def _payload(rows: list[dict[str, object]], omitted: int) -> str:
    payload = {"schema_version": 1, "directives": rows, "coverage": {"shown": len(rows), "omitted": omitted}}
    return DIRECTIVE_PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


#: The least byte bound a host may choose: a block holding one directive of
#: the longest text recorded, in three-byte characters (any character of the
#: basic multilingual plane, typographic quotes included), with the widest
#: id and a nine-digit count left out. The block keeps the newest directives
#: and stops at the first that does not fit, so under a smaller bound one
#: long newest directive would leave every block empty.
MIN_BLOCK_BYTES = len(_payload([{"fact_id": 2**63 - 1, "kind": "instruction", "said": "9999-12-31",
                                 "text": "\N{EM DASH}" * MAX_DIRECTIVE_CHARS}], 10**9 - 1).encode())

_APOSTROPHE = "['\N{RIGHT SINGLE QUOTATION MARK}]"
# Words that follow a sentence-opening "always"/"never" when it describes
# rather than instructs: "Never mind", "Always been a fan".
_NOT_A_COMMAND = r"(?!(?:mind|been|was|were|had|did|thought|knew|seen|heard|felt|got|ever)\b)"
_OPENS = r"(?:^|[:;]\s*)"
_INSTRUCTION = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    # An imperative opening the sentence or following a colon: "Always cite sources."
    rf"{_OPENS}(?:please\s+)?(?:always|never)\s+{_NOT_A_COMMAND}\w",
    r"\bplease\s+(?:always|never)\s+\w",
    rf"{_OPENS}(?:please\s+)?(?:don{_APOSTROPHE}?t|do not)\s+ever\s+\w",
    # A standing time opening the sentence: "From now on, answer in French."
    r"^(?:from now on|going forward|for future reference)\b",
    # A trigger on the person's own asking: "Whenever I ask about dates, ..."
    rf"\b(?:whenever|every time|each time|any ?time|when)\s+I\s+(?:ask|request)\b",
    rf"\bI(?:\s+want|{_APOSTROPHE}d like|\s+would like|\s+need)\s+you\s+to\s+(?:always|never)\b",
    r"\bmake sure (?:to|you)\s+(?:always|never)\b",
))
_PREFERENCE = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bI\s+(?:really\s+|generally\s+|usually\s+|strongly\s+|much\s+|also\s+|still\s+)?(?:prefer|favou?r)\b",
    rf"\bI(?:{_APOSTROPHE}d|\s+would)\s+(?:much\s+|really\s+|still\s+)?(?:prefer|rather)\b",
    r"\bmy\s+(?:preference|preferred)\b",
    r"\bI\s+(?:really\s+|much\s+)?(?:like|love|enjoy)\b[^.?!]*\b(?:better|more than|rather than|instead of)\b",
    rf"\bI\s+(?:really\s+)?(?:hate|dislike|can{_APOSTROPHE}?t stand)\b",
    rf"\bI{_APOSTROPHE}?m\s+(?:not\s+)?(?:a\s+)?(?:big\s+|huge\s+)?fan of\b",
))
_CUES: tuple[tuple[DirectiveKind, tuple["re.Pattern[str]", ...]], ...] = (
    ("instruction", _INSTRUCTION), ("preference", _PREFERENCE))
# A sentence runs to a terminator followed by space or the end, or to a line
# break; a stop inside "5.3.0" or "config.js" does not end one.
_SENTENCE = re.compile(r"(?:[^.!?\n]|[.!?]+(?=[^\s.!?]))+[.!?]*")
_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)


@dataclass(frozen=True)
class Directive:
    """One directive: its kind and where it sits in the text it was read
    from, so ``text`` is always an exact substring of that text."""

    kind: DirectiveKind
    start: int
    end: int
    text: str
    #: Its sentence was longer than ``MAX_DIRECTIVE_CHARS`` and was cut
    #: around its cue.
    clipped: bool = False


def _cue(sentence: str) -> Optional[tuple[DirectiveKind, int]]:
    """The kind of the first cue a sentence holds, instructions first, and
    where in the sentence it starts."""
    for kind, cues in _CUES:
        starts = [found.start() for found in (cue.search(sentence) for cue in cues) if found]
        if starts:
            return kind, min(starts)
    return None


def find_directives(text: str) -> list[Directive]:
    """The explicit directives in a person's text, in order, each once.
    Reads the first ``MAX_TEXT_CHARS`` characters, and no fenced code."""
    # A fence becomes line breaks, so no sentence runs across code.
    read = _FENCE.sub(lambda block: "\n" * len(block.group()), text[:MAX_TEXT_CHARS])
    found: list[Directive] = []
    seen: set[str] = set()
    for match in _SENTENCE.finditer(read):
        raw = match.group()
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - (len(raw) - len(raw.rstrip()))
        if end <= start:
            continue
        cue = _cue(read[start:end])
        if cue is None:
            continue
        kind, at = cue
        clipped = end - start > MAX_DIRECTIVE_CHARS
        if clipped:
            # Keep the cue and what follows it, with a little of what led up to it.
            start = max(start, min(start + at - MAX_DIRECTIVE_CHARS // 4, end - MAX_DIRECTIVE_CHARS))
            end = start + MAX_DIRECTIVE_CHARS
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
        sentence = text[start:end]
        if sentence in seen:
            continue
        seen.add(sentence)
        found.append(Directive(kind, start, end, sentence, clipped))
    return found


def is_person_turn(episode: "Episode") -> bool:
    """Whether a conversation recorded this episode as the user's own turn."""
    return episode.kind == "conversation" and episode.metadata.get("role") == "user"


@dataclass
class DirectiveOutcome:
    """What reading one episode for directives did."""

    episode_id: int
    #: False when the episode is not a person's turn and was not read.
    read: bool
    added: int = 0
    restated: int = 0
    #: Found past ``MAX_DIRECTIVES`` and not recorded.
    cut: int = 0
    #: Recorded around their cue because their sentence was too long.
    clipped: int = 0
    #: The turn ran past ``MAX_TEXT_CHARS`` and its end was not read.
    text_cut: bool = False


async def record_directives(engine: "MemoryEngine", space: str, episode_id: int, *,
                            subject: str = SUBJECT) -> DirectiveOutcome:
    """Read one stored episode for directives and record each as a claim of
    ``subject``, quoted from its sentence and dated when the turn was said.
    An episode that is not a person's turn is left unread."""
    check_space(space)
    episode = await engine.documents.get_episode(space, episode_id)
    if episode is None:
        raise NotFound(f"episode {episode_id} not found in {space!r}")
    outcome = DirectiveOutcome(episode_id, read=is_person_turn(episode))
    if not outcome.read:
        return outcome
    found = find_directives(episode.content)
    outcome.text_cut = len(episode.content) > MAX_TEXT_CHARS
    outcome.cut = max(0, len(found) - MAX_DIRECTIVES)
    key = normalise_term(subject, "subject")
    held: dict[str, set[int]] = {}
    for directive in found[:MAX_DIRECTIVES]:
        predicate = PREDICATES[directive.kind]
        if predicate not in held:
            held[predicate] = {fact.fact_id for fact in await engine.documents.facts_for(space, key, predicate)}
        fact = await engine.assert_fact(space, subject, predicate, directive.text, valid_from=episode.created_at,
                                        source_episode_id=episode.episode_id, origin="extracted",
                                        quote=directive.text)
        if fact.fact_id in held[predicate]:
            outcome.restated += 1
        else:
            held[predicate].add(fact.fact_id)
            outcome.added += 1
        outcome.clipped += directive.clipped
    return outcome


@dataclass(frozen=True)
class DirectiveBlock:
    """The standing directives a turn is shown, and what the bounds left out."""

    #: The block to put in front of the conversation; empty when none hold.
    text: str
    #: The facts shown, oldest first, as the block lists them.
    fact_ids: tuple[int, ...]
    #: Their source episodes, in the same order.
    episode_ids: tuple[Optional[int], ...]
    #: Directives that hold now, shown or not.
    held: int
    #: Of those, how many the byte or count bound left out (the oldest).
    omitted: int


def _render(shown: list["Fact"], held: int) -> str:
    return _payload([{"fact_id": fact.fact_id, "kind": KIND_OF[fact.predicate], "said": fact.valid_from[:10],
                      "text": fact.object} for fact in shown], held - len(shown))


async def standing_directives(engine: "MemoryEngine", space: str, *, subject: str = SUBJECT,
                              limit: int = 20, max_bytes: int = DEFAULT_BLOCK_BYTES,
                              at: Optional[str] = None) -> DirectiveBlock:
    """The directives of ``subject`` that hold ``at`` (now by default), as
    one block: the newest that fit ``limit`` and ``max_bytes``, listed
    oldest first so a later one reads as overriding an earlier one.
    Excluded and proposed directives are never shown."""
    check_space(space)
    if type(limit) is not int or not 1 <= limit <= MAX_BLOCK_LIMIT:
        raise InvalidInput(f"limit must be an integer from 1 to {MAX_BLOCK_LIMIT}")
    if type(max_bytes) is not int or not MIN_BLOCK_BYTES <= max_bytes <= MAX_BLOCK_BYTES:
        raise InvalidInput(f"max_bytes must be an integer from {MIN_BLOCK_BYTES} to {MAX_BLOCK_BYTES}")
    key = normalise_term(subject, "subject")
    when = at or engine.clock()
    holding = [fact for predicate in (INSTRUCTS, PREFERS)
               for fact in await engine.documents.facts_for(space, key, predicate)
               if fact.status == "active" and not fact.excluded and fact.holds_at(when)]
    holding.sort(key=lambda fact: (fact.valid_from, fact.fact_id), reverse=True)
    kept: list["Fact"] = []
    for fact in holding[:limit]:
        candidate = sorted([*kept, fact], key=lambda shown: (shown.valid_from, shown.fact_id))
        if len(_render(candidate, len(holding)).encode()) > max_bytes:
            break
        kept = candidate
    return DirectiveBlock(text=_render(kept, len(holding)) if kept else "",
                          fact_ids=tuple(fact.fact_id for fact in kept),
                          episode_ids=tuple(fact.source_episode_id for fact in kept),
                          held=len(holding), omitted=len(holding) - len(kept))
