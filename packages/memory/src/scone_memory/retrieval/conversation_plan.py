"""Choose evidence retrieval independently of the model that writes the answer."""

from dataclasses import dataclass
import re
from typing import Literal, Sequence

from ..core.models import RecallItem
from ..core.validation import MAX_QUERY
from .lexical import tokenize
from .query_formulation import FormulatedQuery, formulate_query

_GREETING = re.compile(r"^\s*(?:hello|hi|hey)(?: there)?\s*[!,.]\s*", re.IGNORECASE)
_TRANSCRIPT_ROLE = re.compile(r"(?:^|\n)(You|Assistant)↗\s*Source\s+\d+")
# These describe an overview operation rather than a subject in the store.
_OVERVIEW_TERMS = frozenset("""been being us about lately recently recent current currently
    working worked work doing done built building made make progress status update updates
    new latest catch up catchup recap summarize summarise summary overview everything anything
    something things thing main important know known remember learned learning learned discussed
    discussing conversations conversation projects project decisions decided plans planning so far
    previous past today yesterday accomplished accomplish leave left off
    hello hi hey please can could would should tell give what's we've we're how going happening""".split())


@dataclass(frozen=True)
class ConversationPlan:
    mode: Literal["search", "overview"]
    query: str
    #: Set when the message was too long to search as written; ``query`` is
    #: then verbatim excerpts of it, and this records which spans.
    formulation: FormulatedQuery | None = None


def plan_conversation_retrieval(query: str) -> ConversationPlan:
    cleaned = _GREETING.sub("", query).strip() or query
    formulation = formulate_query(query) if len(cleaned) > MAX_QUERY else None
    search = cleaned if formulation is None else formulation.text
    if any(character.isalpha() and not character.isascii() for character in cleaned):
        return ConversationPlan("search", search, formulation)
    terms = set(tokenize(cleaned))
    anchors = terms - _OVERVIEW_TERMS
    return ConversationPlan("search" if anchors else "overview", search, formulation)


def overview_evidence(items: Sequence[RecallItem], limit: int) -> list[RecallItem]:
    """Diversify a bounded recent window; scores are selection heuristics, not truth.

    Long-form updates outrank short acknowledgments/questions. Near duplicates
    and repeated sources lose priority so one chat does not fill the overview.
    Every selected passage stays verbatim and retains its original identifiers.
    """
    candidates: list[tuple[RecallItem, set[str], float]] = []
    seen: set[int] = set()
    for position, item in enumerate(items):
        if item.episode_id in seen:
            continue
        seen.add(item.episode_id)
        stripped = item.text.strip()
        if stripped.startswith(("<task-notification>", "<system-reminder>")):
            continue
        words = tokenize(stripped)
        if len(words) < 5:
            continue
        richness = min(len(words), 100) / 100
        recency = 1 - position / max(len(items), 1)
        score = 0.65 * richness + 0.35 * recency
        if stripped.endswith("?") and len(words) < 40:
            score -= 0.4
        if item.metadata.get("integration") in {"scone-text", "scone-voice"} and item.metadata.get("role") == "assistant":
            score -= 1.0
        if len(set(_TRANSCRIPT_ROLE.findall(stripped))) > 1:
            score -= 1.0
        candidates.append((item, set(words), score))
    selected: list[RecallItem] = []
    selected_words: list[set[str]] = []
    sources: dict[str, int] = {}
    while candidates and len(selected) < limit:
        def utility(candidate: tuple[RecallItem, set[str], float]) -> float:
            item, words, score = candidate
            overlap = max((len(words & previous) / max(len(words | previous), 1)
                           for previous in selected_words), default=0)
            source = item.source or str(item.episode_id)
            return score - 0.7 * overlap - 0.04 * sources.get(source, 0)
        best = max(range(len(candidates)), key=lambda index: utility(candidates[index]))
        if utility(candidates[best]) <= 0:
            break
        item, selected_terms, _ = candidates.pop(best)
        selected.append(item)
        selected_words.append(selected_terms)
        source = item.source or str(item.episode_id)
        sources[source] = sources.get(source, 0) + 1
    return selected
