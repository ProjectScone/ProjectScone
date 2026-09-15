"""A follow-up turn searched with what the conversation already named, and said so.

"Where does Alice Chen work?" and then "since when?": memory is searched
for the latest user message, and the second one names nothing. A search
for "since when?" finds whatever says "since", about anyone.

The chat engines that handle this ask a model to condense the history
and the follow-up into one standalone question, and search that instead
of the question (LlamaIndex whenever there is history; RAGFlow when a
conversation has more than one question and the option is on). What was
searched is not kept beside what was found, and the rewrite is taken as
it comes back.

Here there are two ways, both off unless asked for:

``carry`` needs no model. A latest user turn *leans on the conversation*
when it holds "since when", "what about" or "how about", or ends in
"too" or "as well". A turn that names nothing also leans when it refers
back (it, they, that, there ...), is three words or fewer, or has fewer
than two telling words of its own. A turn that names something is taken
to be about what it names: "there" in "Is there a meeting with Bob Smith
on Friday?" and "its" in "When did Kestrel Bank open its Leeds branch?"
point into the question, not back. Then the named or rare terms of the
most recent earlier user turn that has any are carried forward verbatim
-- never the assistant's words, which the person did not say -- and the
second query is those terms followed by the question. A term the
question already names, as whole words ("Ann" is not named by "Annex"),
is not carried again. A standalone question, a first turn, a turn whose
earlier turns name nothing new, and a carried query over the query bound
are searched as asked, and the record says which.

``rewrite`` asks a model, as :mod:`.query_transforms` does, to restate
the follow-up as a standalone question from a bounded history. The
restatement is taken only when it can be read, is within the query
bound, and shares a word with the question or the history it was shown.
A model that fails, is slow, or returns what cannot be trusted falls
back to ``carry``, or to the question as asked, and ``fallback`` names
the cause.

Either way the question as asked is still searched, and the two ranked
lists are fused by rank: interleaved, the question's first, each passage
once. The question's best passage keeps the lead, and the follow-up's
best is second. Reciprocal rank fusion was measured first and dropped:
the second query holds the question's own words, so the question's
matches rank in both lists, collect credit twice, and push down the
passage only the carried terms found -- on the development pairs of
``benchmarks/followup-pairs-v1.json`` it moved no second turn at weights
1, 2 or 4, where interleaving moved five of thirteen. Facts are
interleaved the same way and cut to as many as the longer of the two
recalls gave, so a carried query never doubles the facts put in front of
a model (where facts come before passages, they would push out the
passage the question found); ``facts_dropped`` counts the cut.

Named or rare terms are read from the text as written: a run of
capitalised words that are not common words (asking words, pronouns,
stopwords, openers such as "Tell", and verbs, adverbs and time words that
open a sentence, such as "Explain", "Actually" and "Yesterday" -- so a
sentence may open with a name but not with "Where"), a quoted phrase,
and a word shaped like an
identifier (a digit, a symbol inside it as in node.js or a@b.org, or a
capital after its first letter as in iPhone). "Rare" is a shape, not a
count over the corpus: this module reads no store. Shape cannot tell
"Globex hired Carol" from "Explain what Carol does", so an opening word
missing from the list is still read as a name.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Optional, Sequence, TypeVar

from ..core.errors import InvalidInput
from ..core.models import RecallResult
from ..core.validation import MAX_QUERY
from ..providers.llm import ChatModel
from .lexical import STOPWORDS, tokenize
from .query_transforms import _object as reply_object

#: The settings a conversation can take: off (the default), carry, rewrite.
MODES: tuple[str, ...] = ("off", "carry", "rewrite")
#: Terms carried from one earlier turn, at most; ``cut`` says when more were found.
MAX_CARRIED_TERMS = 6
#: Earlier user turns read, newest first, for one that names something;
#: ``turns_unread`` says how many were left when none of them did.
MAX_LOOKBACK_TURNS = 4
#: A turn of this many words or fewer leans on the conversation.
SHORT_WORDS = 3
#: Letters a word needs to tell one subject from another.
TELLING_LETTERS = 4
#: Bytes of earlier turns shown to a rewriting model, newest kept first;
#: ``history_omitted`` counts the turns left out.
MAX_HISTORY_BYTES = 6_000
#: Seconds a rewriting model is given before the fallback is taken.
REWRITE_TIMEOUT_S = 5.0
#: Characters a quotation may hold and still be read as one term; a longer
#: one is a passage, not a name, and its words are read one by one.
QUOTED_TERM_CHARS = 200

_ASKING = frozenset("""what who whom whose which where why how when do does did is are was were am be been
    have has had can could will would shall should may might must""".split())
_REFERRING = frozenset("""it its itself they them their theirs themselves he him his himself she her hers herself
    that this those these there then""".split())
#: Phrases that lean on the conversation even in a question that names things.
_REFERRING_PHRASE = re.compile(r"\b(?:since when|what about|how about)\b|\b(?:too|as well)(?=\W*$)")
_OPENERS = frozenset("""and also so now ok okay yes no please tell show give list find hi hey hello thanks thank
    not any all some about after before since again more other else just really i i'm i've i'd i'll let let's""".split())
_COMMON = STOPWORDS | _ASKING | _REFERRING | _OPENERS
#: Words that open a sentence capitalised and are not names: verbs of asking,
#: discourse adverbs, and days relative to now.
_OPENING_WORDS = frozenset("""explain remind summarize summarise describe compare check recall remember define clarify
    confirm search look actually basically honestly anyway maybe perhaps yesterday today tomorrow tonight""".split())
_NOT_NAMES = _COMMON | _OPENING_WORDS

_OPENING = "\"'([{\u2018\u201c\u00ab"
_CLOSING = ".,;:!?\"')]}\u2019\u201d\u00bb"
_POSSESSIVE = re.compile(r"['\u2019]s$")
_QUOTED = re.compile(f"\"([^\"\\n]{{1,{QUOTED_TERM_CHARS}}})\"|\u201c([^\u201d\\n]{{1,{QUOTED_TERM_CHARS}}})\u201d")
_SYMBOL_INSIDE = re.compile(r"\w[.@#+/_]+\w")

_REWRITE_SYSTEM = (
    "You restate the last question of a conversation as one standalone question for a search over someone's "
    "own notes and conversations. Use the conversation only to fill in what the question refers to; keep "
    "names, dates and numbers exactly as written. If the question already stands on its own, return it "
    "unchanged. Reply with JSON only, of the form {\"query\": \"...\"}; no answer, no explanation."
)


@dataclass(frozen=True)
class Followup:
    """What a follow-up turn was searched with beside the question, and why."""

    mode: str
    #: How the second query was made: "carry", "rewrite", or "none".
    method: str
    #: Whether a second query is searched beside the question.
    applied: bool
    reason: str
    #: The second query, exactly as searched; None when there is none.
    query: Optional[str] = None
    carried: tuple[str, ...] = ()
    #: Index in the caller's messages of the user turn the terms came from.
    from_message: Optional[int] = None
    #: What made the question read as leaning on the conversation.
    cues: tuple[str, ...] = ()
    #: New terms that turn named, before ``MAX_CARRIED_TERMS``.
    terms_found: int = 0
    #: Earlier user turns not read because the lookback bound was reached.
    turns_unread: int = 0
    #: Facts fusion left out to hold no more than the longer recall gave.
    facts_dropped: int = 0
    model_calls: int = 0
    #: Why a model's rewrite was not used, when a fallback was taken.
    fallback: Optional[str] = None
    history_omitted: int = 0

    @property
    def cut(self) -> bool:
        return self.terms_found > len(self.carried)

    def record(self) -> dict[str, object]:
        return {"mode": self.mode, "method": self.method, "applied": self.applied, "reason": self.reason,
                "query": self.query, "carried": list(self.carried), "from_message": self.from_message,
                "cues": list(self.cues), "terms_found": self.terms_found, "cut": self.cut,
                "turns_unread": self.turns_unread, "facts_dropped": self.facts_dropped,
                "model_calls": self.model_calls, "fallback": self.fallback,
                "history_omitted": self.history_omitted}


@dataclass(frozen=True)
class _Word:
    start: int
    end: int
    text: str
    #: Punctuation or a possessive followed it, or an opening mark led it.
    closed: bool
    opened: bool


def _words(text: str) -> list[_Word]:
    words = []
    for found in re.finditer(r"\S+", text):
        raw = found.group()
        lead = len(raw) - len(raw.lstrip(_OPENING))
        body = raw[lead:]
        core = body.rstrip(_CLOSING)
        possessive = _POSSESSIVE.search(core)
        if possessive:
            core = core[:possessive.start()].rstrip(_CLOSING)
        if core:
            start = found.start() + lead
            words.append(_Word(start, start + len(core), core, len(core) < len(body), lead > 0))
        elif words:
            # A mark standing alone ("...", "-") ends the name before it.
            words[-1] = replace(words[-1], closed=True)
    return words


def _rare(word: str) -> bool:
    return (any(character.isdigit() for character in word) or _SYMBOL_INSIDE.search(word) is not None
            or any(character.isupper() for character in word[1:]))


def named_terms(text: str) -> list[str]:
    """The named or rare terms of ``text``, verbatim, in order, each once."""
    found: list[tuple[int, str]] = []
    quoted = [(match.start(), match.end()) for match in _QUOTED.finditer(text)]
    for match in _QUOTED.finditer(text):
        inner = match.group(1) if match.group(1) is not None else match.group(2)
        if inner.strip():
            found.append((match.start(), inner.strip()))
    run: list[_Word] = []

    def flush() -> None:
        if run:
            found.append((run[0].start, text[run[0].start:run[-1].end]))
            run.clear()

    for word in _words(text):
        if any(start <= word.start < end for start, end in quoted):
            flush()
            continue
        if word.text[0].isupper() and word.text.casefold() not in _NOT_NAMES:
            if word.opened:
                flush()
            run.append(word)
            if word.closed:
                flush()
            continue
        flush()  # a word that is not part of a name ends one
        if _rare(word.text):
            found.append((word.start, word.text))
    flush()  # a name may end the text
    terms: list[str] = []
    seen: set[str] = set()
    for _, term in sorted(found, key=lambda pair: pair[0]):
        if term.casefold() not in seen:
            seen.add(term.casefold())
            terms.append(term)
    return terms


def _cues(question: str, named: Sequence[str]) -> tuple[str, ...]:
    """What makes ``question`` lean on the conversation; empty when it stands alone.

    ``named`` is what the question names: a question that names something
    leans only through a referring phrase."""
    folded = " ".join(question.casefold().split())
    cues = [match.group() for match in _REFERRING_PHRASE.finditer(folded)]
    if named:
        return tuple(cues)
    words = _words(question)
    for word in words:
        folded_word = word.text.casefold()
        if folded_word in _REFERRING and folded_word not in cues:
            cues.append(folded_word)
    if len(words) <= SHORT_WORDS:
        cues.append("short")
    telling = {token for token in tokenize(question) if len(token) >= TELLING_LETTERS and token not in _COMMON}
    if len(telling) < 2:
        cues.append("names nothing")
    return tuple(cues)


def _names(question: str, term: str) -> bool:
    """Whether ``question`` holds ``term`` as whole words, whatever the case and spacing."""
    pattern = r"\s+".join(re.escape(part) for part in term.casefold().split())
    return re.search(rf"(?<!\w){pattern}(?!\w)", question.casefold()) is not None


def _user_turns(messages: Sequence[Mapping[str, object]]) -> list[int]:
    return [index for index, message in enumerate(messages)
            if message.get("role") == "user" and isinstance(message.get("content"), str)
            and str(message["content"]).strip()]


def _latest(messages: Sequence[Mapping[str, object]]) -> tuple[int, str, list[int]]:
    turns = _user_turns(messages)
    if not turns:
        raise InvalidInput("a follow-up needs a user question")
    return turns[-1], str(messages[turns[-1]]["content"]).strip(), turns[:-1]


def carry(messages: Sequence[Mapping[str, object]], *, question: Optional[str] = None,
          max_terms: int = MAX_CARRIED_TERMS, lookback: int = MAX_LOOKBACK_TURNS) -> Followup:
    """The earlier turn's terms carried into a second query, or the reason none is searched.

    ``question`` is the text searched for the latest turn when it differs
    from the message (a long message is searched by excerpts of it); the
    message itself decides whether the turn leans on the conversation."""
    _, asked, earlier = _latest(messages)
    base = asked if question is None else question
    if not earlier:
        return Followup("carry", "none", False, "first turn: there is no earlier user turn to carry from")
    named = named_terms(asked)
    cues = _cues(asked, named)
    if not cues and named:
        return Followup("carry", "none", False, f"standalone: the question names {', '.join(named)}; "
                        "a referring word in it is read as pointing there")
    if not cues:
        return Followup("carry", "none", False, "standalone: the question refers back to nothing and has telling words of its own")
    examined = earlier[::-1][:lookback]
    source, terms = None, []
    for index in examined:
        terms = named_terms(str(messages[index]["content"]))
        if terms:
            source = index
            break
    if source is None:
        unread = len(earlier) - len(examined)
        reason = f"nothing to carry: no named or rare term in the last {len(examined)} earlier user turn(s)"
        if unread:
            reason += f"; {unread} earlier turn(s) not read past the lookback bound of {lookback}"
        return Followup("carry", "none", False, reason, cues=cues, turns_unread=unread)
    new = [term for term in terms if not _names(asked, term)]
    if not new:
        return Followup("carry", "none", False, f"nothing to carry: the question already names what message {source} named",
                        from_message=source, cues=cues)
    kept = tuple(new[:max_terms])
    searched = " ".join(kept) + " " + base
    if len(searched) > MAX_QUERY:
        return Followup("carry", "none", False,
                        f"the carried query would pass the {MAX_QUERY}-character bound; the question was searched as asked",
                        from_message=source, cues=cues)
    return Followup("carry", "carry", True, f"carried {len(kept)} term(s) from message {source}", query=searched,
                    carried=kept, from_message=source, cues=cues, terms_found=len(new))


async def rewrite_followup(model: ChatModel, messages: Sequence[Mapping[str, object]], *,
                           question: Optional[str] = None, timeout_s: float = REWRITE_TIMEOUT_S,
                           max_history_bytes: int = MAX_HISTORY_BYTES) -> Followup:
    """The follow-up restated by ``model`` from the conversation, or the fallback with its reason."""
    latest, asked, before = _latest(messages)
    base = asked if question is None else question
    if not before:
        return Followup("rewrite", "none", False, "first turn: there is no earlier user turn to rewrite from")
    shown: list[str] = []
    spent = omitted = 0
    for message in reversed(messages[:latest]):
        content = message.get("content")
        if message.get("role") not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        line = f"{str(message['role']).upper()}: {content.strip()}"
        if omitted or spent + len(line.encode("utf-8")) + 1 > max_history_bytes:
            omitted += 1
            continue
        shown.append(line)
        spent += len(line.encode("utf-8")) + 1

    def fallback(reason: str, calls: int) -> Followup:
        return replace(carry(messages, question=question), mode="rewrite", fallback=reason, model_calls=calls,
                       history_omitted=omitted)

    if not shown:
        return fallback(f"the latest earlier turn is over the {max_history_bytes}-byte history bound; no model was asked", 0)
    history = "\n".join(reversed(shown))
    try:
        reply = await asyncio.wait_for(model.complete(_REWRITE_SYSTEM, f"Conversation:\n{history}\nQuestion: {base}"),
                                       timeout_s)
    except TimeoutError:
        return fallback(f"timeout after {timeout_s}s", 1)
    except Exception as error:  # a model that fails leaves the conversation to the rule
        return fallback(f"model failed: {type(error).__name__}", 1)
    body = reply_object(reply)
    raw = body.get("query") if body is not None else None
    if not isinstance(raw, str) or not raw.strip():
        return fallback("the reply could not be read as a question", 1)
    query = " ".join(raw.split())
    if len(query) > MAX_QUERY:
        return fallback(f"the rewrite is over the {MAX_QUERY}-character bound", 1)
    if query.casefold() == " ".join(base.split()).casefold():
        return Followup("rewrite", "rewrite", False, "standalone: the model returned the question as asked",
                        model_calls=1, history_omitted=omitted)
    if not set(tokenize(query)) & (set(tokenize(base)) | set(tokenize(history))):
        return fallback("the rewrite shares no word with the question or the conversation", 1)
    return Followup("rewrite", "rewrite", True, "restated from the conversation by the model", query=query,
                    model_calls=1, history_omitted=omitted)


async def plan_followup(messages: Sequence[Mapping[str, object]], mode: str, *, model: Optional[ChatModel] = None,
                        question: Optional[str] = None, timeout_s: float = REWRITE_TIMEOUT_S) -> Followup:
    """The follow-up for ``mode``: "carry", or "rewrite" with a model."""
    if mode == "carry":
        return carry(messages, question=question)
    if mode != "rewrite":
        raise InvalidInput(f"a follow-up mode is carry or rewrite, not {mode!r}")
    if model is None:
        raise InvalidInput("a rewritten follow-up needs a model")
    return await rewrite_followup(model, messages, question=question, timeout_s=timeout_s)


_T = TypeVar("_T")


def _interleaved(first: Sequence[_T], second: Sequence[_T], key: Callable[[_T], int]) -> list[_T]:
    taken: list[_T] = []
    seen: set[int] = set()
    for rank in range(max(len(first), len(second))):
        for ranked in (first, second):
            if rank < len(ranked) and key(ranked[rank]) not in seen:
                seen.add(key(ranked[rank]))
                taken.append(ranked[rank])
    return taken


def fused(original: RecallResult, second: RecallResult, *, limit: int) -> tuple[RecallResult, int]:
    """The question's recall and the follow-up's, fused by rank, and the facts left out.

    Passages and facts are interleaved, the question's first, each once:
    at most ``limit`` passages, and no more facts than the longer of the
    two recalls gave. Each passage keeps the fields of the recall that
    ranked it first. The evidence event stays the question's. Evidence is
    weak only when both recalls said so, and unknown when neither said it
    was strong."""
    confidences = {original.low_confidence, second.low_confidence}
    low = False if False in confidences else True if confidences == {True} else None
    similarities = [value for value in (original.top_similarity, second.top_similarity) if value is not None]
    facts = _interleaved(original.facts, second.facts, lambda fact: fact.fact_id)
    kept_facts = facts[:max(len(original.facts), len(second.facts))]
    return original.model_copy(update={
        "items": _interleaved(original.items, second.items, lambda item: item.chunk_id)[:limit],
        "facts": kept_facts,
        "low_confidence": low,
        "top_similarity": max(similarities) if similarities else None,
        "degraded": sorted(set(original.degraded) | set(second.degraded)),
    }), len(facts) - len(kept_facts)
