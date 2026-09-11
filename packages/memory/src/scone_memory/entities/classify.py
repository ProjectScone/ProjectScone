"""Which fact objects name things, and which are values.

Subjects always name things. An object may name a thing ("Acme Robotics"),
or be a value: a date, an amount, an identifier, a phrase, a pronoun. Making
every object a graph node fills the graph with 'tired', '3 MB' and 'Monday';
joining every object by folded case joins 'MB' (megabytes) to 'mb'
(millibits). So objects are values unless a rule below shows they name a
thing, the first rule that fits decides, and every decision names its rule.

The rules, in order:

1. a recorded decision for the predicate;
2. a key that an identity decision names;
3. quoted text, or prose (long, or holding a sentence break);
4. a third-person or indefinite pronoun;
5. a date, quantity, identifier or yes/no shape, which names a thing only
   when a subject carries the same key and the text holds no cased letters
   (a unit, version or path never joins by folding, in either direction);
6. any key that a subject carries;
7. a predicate whose object is a value (role, colour, price, status, ...);
8. a leading determiner: a title-cased rest is a name ("the Web Summit"),
   anything else a description ("the Acme lab");
9. a name shape (a capital letter, or a short name in a script without case);
10. a predicate whose object is a thing (works_at, lives_in, reports_to, ...);
11. a lowercase phrase that two or more subjects share, as a concept;
12. anything else, as a value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal, Mapping

from ..core.validation import entity_key

CLASSIFIER_VERSION = "objects/1"

ObjectClass = Literal["entity", "literal"]
LiteralKind = Literal["date", "quantity", "identifier", "text", "value", "pronoun"]
ClassBasis = Literal[
    "decision", "identity_decision", "quoted_text", "prose", "pronoun", "date_shape", "quantity_shape",
    "identifier_shape", "value_shape", "subject_anchor", "literal_predicate", "determiner_name", "description",
    "name_shape", "uncased_name", "entity_predicate", "shared_object", "common_value",
]


@dataclass(frozen=True, slots=True)
class ObjectClassification:
    object_class: ObjectClass
    literal_kind: LiteralKind | None
    basis: ClassBasis
    #: For a decision, the fact id that recorded it; "case_sensitive" when a
    #: subject carried the key but the value's case could change its meaning.
    detail: str | None = None


@dataclass(frozen=True)
class ClassificationContext:
    #: Keys that some subject carries.
    anchors: frozenset[str] = frozenset()
    #: Keys that an identity decision names (merged aliases, their targets, typed keys).
    identity_keys: frozenset[str] = frozenset()
    #: Keys that are objects of two or more distinct subjects.
    shared_objects: frozenset[str] = frozenset()
    #: Predicate key -> (decided class, deciding fact id).
    overrides: Mapping[str, tuple[ObjectClass, int]] = field(default_factory=dict)


_SPEAKERS = frozenset("i me my mine myself we us our ours ourselves you your yours yourself yourselves".split())
_PRONOUNS = frozenset("""he him his himself she her hers herself it its itself they them their theirs
    themselves this that these those someone somebody something anyone anybody anything everyone
    everybody everything nobody nothing""".split()) | {"no one"}
_DETERMINERS = frozenset("the a an my our his her their its this that your".split())
_CONNECTORS = frozenset("of and for the de la le du des von van der den y on in at to &".split())
_ABBREVIATIONS = frozenset("dr mr mrs ms prof st sr jr inc ltd co corp vs etc no fig approx dept univ".split())
_QUOTES = {'"': '"', "'": "'", "“": "”", "‘": "’", "«": "»", "「": "」", "『": "』"}

_VALUE_NOUNS = frozenset("""age email phone url website title role position status size price cost version
    count amount height weight colour color rating score salary duration quantity percentage level grade
    number total rank balance budget temperature speed length width depth""".split())
_ENTITY_PREDICATES = frozenset("""works_at worked_at works_for worked_for employed_by studied_at studies_at
    attends attended lives_in lived_in lives_at based_in located_in located_at headquartered_in born_in
    moved_to visited knows met married_to reports_to manages managed_by member_of part_of belongs_to
    founded founded_by owns owned_by created_by built_by uses used_by depends_on works_on contributes_to
    leads led_by partner_of friend_of sibling_of parent_of child_of colleague_of acquired acquired_by
    invested_in subsidiary_of competitor_of mentor_of customer_of supplier_of""".split())

_MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december"
           "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
_DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_DATE = re.compile("|".join((
    r"\d{4}-\d{2}(?:-\d{2}(?:[t ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:z|[+-]\d{2}:?\d{2})?)?)?",
    r"\d{1,2}[/.]\d{1,2}[/.]\d{2,4}",
    r"[12]\d{3}s?",
    rf"(?:\d{{1,2}}(?:st|nd|rd|th)?\s+)?(?:{_MONTHS})\.?(?:\s+\d{{1,2}}(?:st|nd|rd|th)?)?(?:,?\s+\d{{4}})?",
    rf"(?:last|next|this|every)?\s*(?:{_DAYS})s?",
    r"(?:q[1-4]|h[12])(?:\s+\d{4})?",
    r"(?:spring|summer|autumn|fall|winter)(?:\s+\d{4})?",
    r"today|tomorrow|yesterday|tonight|now",
    r"(?:last|next|this)\s+(?:week|weekend|month|quarter|year)",
    r"\d+\s+(?:minutes?|hours?|days?|weeks?|months?|years?)\s+(?:ago|later|from now)",
    r"\d{1,2}(?::\d{2}){1,2}(?:\s*[ap]m)?|\d{1,2}\s*[ap]m",
)))
_UNITS = frozenset("""kb mb gb tb pb kib mib gib tib ms us ns sec secs mins hr hrs kg mg lb lbs oz km cm mm
    mi ft ml px pt em rem fps hz khz mhz ghz kw kwh mah usd eur gbp jpy""".split())
_QUANTITY = re.compile(r"[+-]?[$€£¥]?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|[a-zµ°]{1,6}|[$€£¥]))?")
_IDENTIFIER = re.compile("|".join((
    r"[^@\s]+@[^@\s]+\.[^@\s]+",
    r"(?:[a-z][a-z0-9+.-]*://|www\.)\S+",
    r"(?:~?/|\.{1,2}/|[a-z]:\\)\S*|[\w.-]+(?:/[\w.-]+)+/?",
    r"v?\d+(?:\.\d+){1,3}(?:[-+][0-9a-z.-]+)?",
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    r"(?=[0-9a-f]*\d)[0-9a-f]{7,}",
    r"#\d+",
    r"[a-z]{2,10}-\d{2,}",
    r"(?=[a-z0-9_./-]*\d)[a-z0-9]+(?:[-_/.][a-z0-9]+){2,}",
    r"(?=(?:[a-z]*\d){4})(?=\d*[a-z])[a-z0-9]{5,}",
)))
_YES_NO = frozenset("true false yes no none null nil n/a".split())
_SENTENCE_BREAK = re.compile(r"(\S+)[.!?…。！？]+\s+(?=\S)")


def is_case_safe(text: str) -> bool:
    """True when folding case cannot change what the text means: it holds
    no cased letters at all. Text already in lower case is not enough,
    because subject keys are folded too; a subject 'mb' may have been
    written 'MB', and matching it proves nothing about case."""
    return all(character.lower() == character.upper() for character in text)


def reference_flag(key: str) -> Literal["speaker_reference", "unresolved_reference"] | None:
    if key in _SPEAKERS:
        return "speaker_reference"
    return "unresolved_reference" if key in _PRONOUNS else None


def literal_shape(text: str) -> tuple[LiteralKind, ClassBasis] | None:
    folded = " ".join(text.casefold().split())
    if _DATE.fullmatch(folded):
        return "date", "date_shape"
    if folded in _UNITS or _QUANTITY.fullmatch(folded):
        return "quantity", "quantity_shape"
    if _IDENTIFIER.fullmatch(folded):
        return "identifier", "identifier_shape"
    if folded in _YES_NO:
        return "value", "value_shape"
    return None


def _quoted(text: str) -> bool:
    return len(text) >= 2 and _QUOTES.get(text[0]) == text[-1]


def _prose(text: str) -> bool:
    words = text.split()
    if len(words) > 12 or len(text) > 120:
        return True
    # A break inside the text, after at least two words and not after an
    # abbreviation, means two sentences: "It rained. We stayed in".
    for found in _SENTENCE_BREAK.finditer(text):
        before = text[:found.start(1)].split()
        if before and found.group(1).rstrip(".").casefold() not in _ABBREVIATIONS:
            return True
    return False


def _uncased_name(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    return (bool(letters) and len(text) <= 12 and len(text.split()) <= 3
            and all(character.lower() == character.upper() for character in letters))


def name_shaped(text: str) -> bool:
    tokens = text.split()
    return (0 < len(tokens) <= 6 and len(text) <= 80 and any(character.isalpha() for character in text)
            and any(any(character.isupper() for character in token) for token in tokens))


def _title_cased(text: str) -> bool:
    tokens = text.split()
    content = [token for token in tokens if token.casefold() not in _CONNECTORS]
    return bool(content) and len(tokens) <= 6 and all(token[0].isupper() or token[0].isdigit() for token in content)


def _value_predicate(predicate_key: str) -> bool:
    words = predicate_key.replace(" ", "_").split("_")
    return predicate_key == "is_a" or words[-1] in _VALUE_NOUNS


def join_block_reason(text: str) -> ClassBasis | None:
    """Why an object must never join a subject by key, or None when it may.

    Prose, quotations and pronouns never name one thing. A value whose case
    can change its meaning never joins through folding.
    """
    stripped = text.strip()
    if _quoted(stripped):
        return "quoted_text"
    if _prose(stripped):
        return "prose"
    if entity_key(stripped) in _PRONOUNS:
        return "pronoun"
    shape = literal_shape(stripped)
    if shape is not None and not is_case_safe(stripped):
        return shape[1]
    return None


def classify_object(text: str, predicate_key: str, context: ClassificationContext) -> ObjectClassification:
    stripped = text.strip()
    key = entity_key(stripped)
    decided = context.overrides.get(predicate_key)
    if decided is not None:
        return ObjectClassification(decided[0], None if decided[0] == "entity" else "value", "decision", str(decided[1]))
    if key in context.identity_keys:
        return ObjectClassification("entity", None, "identity_decision")
    if _quoted(stripped):
        return ObjectClassification("literal", "text", "quoted_text")
    if _prose(stripped):
        return ObjectClassification("literal", "text", "prose")
    if key in _PRONOUNS:
        return ObjectClassification("literal", "pronoun", "pronoun")
    shape = literal_shape(stripped)
    if shape is not None:
        if key in context.anchors and is_case_safe(stripped):
            return ObjectClassification("entity", None, "subject_anchor")
        return ObjectClassification("literal", shape[0], shape[1],
                                    "case_sensitive" if key in context.anchors else None)
    if key in context.anchors:
        return ObjectClassification("entity", None, "subject_anchor")
    if _value_predicate(predicate_key):
        return ObjectClassification("literal", "value", "literal_predicate")
    words = stripped.split()
    if len(words) > 1 and words[0].casefold() in _DETERMINERS:
        if _title_cased(" ".join(words[1:])):
            return ObjectClassification("entity", None, "determiner_name")
        return ObjectClassification("literal", "text", "description")
    if name_shaped(stripped):
        return ObjectClassification("entity", None, "name_shape")
    if _uncased_name(stripped):
        return ObjectClassification("entity", None, "uncased_name")
    if predicate_key.replace(" ", "_") in _ENTITY_PREDICATES and len(words) <= 6:
        return ObjectClassification("entity", None, "entity_predicate")
    if (key == stripped and len(words) <= 4 and not any(character.isdigit() for character in stripped)
            and key in context.shared_objects):
        return ObjectClassification("entity", None, "shared_object")
    return ObjectClassification("literal", "value", "common_value")
