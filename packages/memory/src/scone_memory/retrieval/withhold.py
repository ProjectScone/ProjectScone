"""Withholding what a caller must not receive, on the way out.

``capture/redact.py`` scrubs secrets on the way **in**, and only for the
agent feed. Memory arrives by many other doors — ``remember``, ``sync``,
document import — and none of them scrub, so a space accumulates whatever
was put into it. For a framework whose business is remembering what
people said, retrieval being unable to withhold anything is a real gap,
and the less glamorous half of it is the half that matters.

The leading RAG framework has a PII postprocessor for this. Ours differs
in two ways that are deliberate:

- **The caller chooses the kinds.** Withholding something nobody asked to
  withhold damages an answer to protect nothing, so each kind is named.
- **A number is checked, not merely matched.** A run of sixteen digits is
  an order number far more often than a card, so the card check runs
  Luhn. A false positive here costs a reader the answer.

And the rule that makes this honest rather than dangerous:

**A report of what was withheld is never a statement that the rest is
clean.** Pattern matching is a net. "0 withheld" means the patterns
matched nothing, not that there is nothing to find, and a caller who
reads it the second way is worse off than one who was told nothing. Every
report says so in as many words.

Nothing is deleted from the store: this is what one answer hands back.
The memory still holds what it held, which is why the report says
"withheld" and not "removed".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Mapping, Sequence

from ..capture.redact import SECRET_PATTERNS
from ..core.errors import InvalidInput
from ..core.models import RecallItem

#: An address. Deliberately narrow: the local part is not allowed spaces
#: or quotes, so prose around an address is not swallowed with it.
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
#: A telephone number in the shapes people actually write, needing either
#: a country prefix or separators -- a bare run of digits is not a phone
#: number, it is a number.
_PHONE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])"
                    r"\d{2,4}[\s.-]?\d{2,4}(?![\w.])")
#: An IPv4 address, each octet in range so 999.1.1.1 is not one.
_IP = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
                 r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b")
#: A candidate card number. Matched loosely and then **checked**, because
#: the check is what tells a card from an order reference.
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")

#: What a caller may ask to withhold.
KINDS: tuple[str, ...] = ("email", "phone", "ip", "card", "secret")
#: Bytes of one item scanned. A passage longer than this is scanned to
#: here and the report says the rest was not looked at, because a scan
#: that stopped early must not read as a scan that found nothing.
MAX_SCANNED = 200_000


def _luhn(digits: str) -> bool:
    """Whether these digits could be a card number.

    Not proof that they are, but enough to keep an order reference out of
    the answer, which is the false positive that costs a reader most.
    """
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0 and len(digits) >= 13


@dataclass(frozen=True)
class Withheld:
    """What one answer withheld, and what that does and does not mean."""

    items: tuple[RecallItem, ...] = ()
    #: Matches replaced across every item.
    withheld: int = 0
    by_kind: Mapping[str, int] = field(default_factory=dict)
    #: Kinds actually applied, so a caller can see their choice echoed
    #: rather than assume it took.
    kinds_applied: tuple[str, ...] = ()
    #: Items longer than the scan bound, whose tails were not examined.
    unscanned: int = 0
    why: str = ""

    def record(self) -> dict[str, object]:
        return {"withheld": self.withheld, "by_kind": dict(self.by_kind),
                "kinds_applied": list(self.kinds_applied), "unscanned": self.unscanned,
                "why": self.why,
                "items": [item.model_dump() for item in self.items]}


def withhold(items: Sequence[RecallItem], *, kinds: Sequence[str] = KINDS,
             scan: int = MAX_SCANNED) -> Withheld:
    """Replace what the named kinds match, and say what was replaced."""
    chosen = tuple(dict.fromkeys(kinds))
    unknown = [kind for kind in chosen if kind not in KINDS]
    if unknown:
        raise InvalidInput(f"there is no pattern for {', '.join(unknown)}; the kinds are "
                           f"{', '.join(KINDS)}")
    if not chosen:
        raise InvalidInput("withholding nothing is not a policy: name at least one kind")
    if not 1 <= scan <= MAX_SCANNED:
        raise InvalidInput(f"the scan bound must be from 1 to {MAX_SCANNED}, not {scan}")

    counted: dict[str, int] = {}
    kept: list[RecallItem] = []
    unscanned = 0
    for item in items:
        text = item.text
        if len(text) > scan:
            unscanned += 1
        head, tail = text[:scan], text[scan:]
        for kind in chosen:
            head, found = _apply(kind, head)
            if found:
                counted[kind] = counted.get(kind, 0) + found
        kept.append(item if head + tail == text else item.model_copy(update={"text": head + tail}))

    total = sum(counted.values())
    why = (f"{total} match(es) of {', '.join(chosen)} withheld from this answer"
           if total else f"the patterns for {', '.join(chosen)} matched nothing here")
    why += ("; this is a net of patterns, not a guarantee -- nothing withheld is "
            "**not a finding** that there is nothing of these kinds in the text, and the "
            "memory still holds whatever it held")
    if unscanned:
        why += (f"; {unscanned} passage(s) are longer than the scan bound and their tails were "
                f"not examined at all")
    return Withheld(items=tuple(kept), withheld=total, by_kind=counted, kinds_applied=chosen,
                    unscanned=unscanned, why=why)


def _apply(kind: str, text: str) -> tuple[str, int]:
    """One kind's replacements, and how many were made."""
    if kind == "secret":
        found = 0
        for pattern in SECRET_PATTERNS:
            text, count = pattern.subn("[withheld: secret]", text)
            found += count
        return text, found
    if kind == "card":
        # Matched loosely, then checked: the check is what separates a
        # card from an order reference, and a false positive here costs
        # the reader the answer.
        found = 0

        def checked(match: "re.Match[str]") -> str:
            nonlocal found
            digits = re.sub(r"[ -]", "", match.group())
            if not _luhn(digits):
                return match.group()
            found += 1
            return "[withheld: card]"

        return _CARD.sub(checked, text), found
    pattern = {"email": _EMAIL, "phone": _PHONE, "ip": _IP}[kind]
    text, count = pattern.subn(f"[withheld: {kind}]", text)
    return text, count
