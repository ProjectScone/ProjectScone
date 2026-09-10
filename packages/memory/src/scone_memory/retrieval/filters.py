"""Choosing which memories a search may see, by what was recorded.

A filter narrows the candidates before anything is ranked, so one
customer's notes cannot be crowded out of a search by another's, and a
question about published work never has to look at drafts.

The shape is deliberately small. A condition names one metadata key and
one test of it; `all` and `any` combine conditions; `not` inverts one.
That is the whole grammar, and it is enough for the filtering people
actually write, while staying something a store can turn into a WHERE
clause rather than a pass over everything it just fetched.

Two decisions worth stating, because both are places where a filter can
silently stop filtering:

An absent key satisfies nothing, negation included. A memory with no
priority is not a memory of priority zero, and "not a draft" must not
quietly match every memory that has no status at all.

Values are stored as text, so "10" sorts before "9". A numeric test
parses both sides first, and a value that is not a number fails the
test rather than being compared as letters.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Union

from ..core.errors import InvalidInput
from ..core.validation import METADATA_KEY

#: Conditions in one filter. Beyond this it is a program in a loop.
MAX_CONDITIONS = 200
#: Groups inside groups. Deeper than anyone reads a query back.
MAX_DEPTH = 8

#: The tests a condition can make, and what each does to one value.
TESTS = ("is", "has", "in", "above", "below", "at_least", "at_most", "present")
#: The tests SQLite can make exactly. The rest narrow generously and are
#: settled by a second pass in Python.
EXACT = frozenset({"is", "has", "in", "present"})
#: Anything outside this cannot be part of a number, so a value holding
#: one is not one, whatever CAST would make of it.
NOT_A_NUMBER = "*[^0-9.eE+-]*"
#: SQL for each numeric test, in the same order they are named.
COMPARISONS = {"above": ">", "below": "<", "at_least": ">=", "at_most": "<="}


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise InvalidInput(f"{what} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise InvalidInput(f"{what} must be a number") from None


def _stored_number(value: str) -> Optional[float]:
    """The stored text as a number, or None when it is not one."""
    try:
        return float(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class Condition:
    """One test of one metadata key."""

    field: str
    test: str
    value: object
    negate: bool = False

    def matches(self, metadata: Mapping[str, str]) -> bool:
        found = metadata.get(self.field)
        if found is None:
            # Absent satisfies nothing, and inverting nothing is still
            # nothing: a key that was never recorded cannot answer.
            return False
        return self._holds(found) != self.negate

    def _holds(self, found: str) -> bool:
        if self.test == "present":
            return True
        if self.test == "is":
            return found == self.value
        if self.test == "has":
            assert isinstance(self.value, str)
            return self.value in found
        if self.test == "in":
            assert isinstance(self.value, Collection)
            return found in self.value
        number = _stored_number(found)
        if number is None:
            return False
        assert isinstance(self.value, (int, float))
        if self.test == "above":
            return number > self.value
        if self.test == "below":
            return number < self.value
        if self.test == "at_least":
            return number >= self.value
        return number <= self.value

    def fields(self) -> set[str]:
        return {self.field}

    def count(self) -> int:
        return 1

    def to_sql(self, column: str) -> tuple[str, list[object]]:
        """A clause keeping at least every row `matches` would keep.

        Being generous costs a row that the second pass then drops. Being
        strict loses a memory that should have been found, and nothing
        later can put it back, so every choice here leans the first way."""
        here, params = self._test(column)
        if not self.negate:
            return here, params
        present, key = self._exists(column, "", [])
        if self.test in EXACT:
            # The test is exact, so its opposite is too. The key still has
            # to be there: absent answers nothing, either way round.
            return f"(NOT {here} AND {present})", [*params, *key]
        # A generous test negated would be strict. Ask only for the key.
        return present, key

    def _exists(self, column: str, check: str, params: list[object]) -> tuple[str, list[object]]:
        return (f"EXISTS (SELECT 1 FROM json_each({column}) AS f"
                f" WHERE f.key = ?{check})", [self.field, *params])

    def _test(self, column: str) -> tuple[str, list[object]]:
        if self.test == "present":
            return self._exists(column, "", [])
        if self.test == "is":
            return self._exists(column, " AND f.value = ?", [self.value])
        if self.test == "has":
            return self._exists(column, " AND instr(f.value, ?) > 0", [self.value])
        if self.test == "in":
            assert isinstance(self.value, Collection)
            slots = ", ".join("?" for _ in self.value)
            return self._exists(column, f" AND f.value IN ({slots})", list(self.value))
        return self._exists(
            column,
            f" AND f.value NOT GLOB ? AND CAST(f.value AS REAL) {COMPARISONS[self.test]} ?",
            [NOT_A_NUMBER, self.value],
        )


@dataclass(frozen=True)
class Group:
    """Conditions that must all hold, or of which any may."""

    every: bool
    parts: tuple["Filter", ...]

    def matches(self, metadata: Mapping[str, str]) -> bool:
        results = (part.matches(metadata) for part in self.parts)
        return all(results) if self.every else any(results)

    def fields(self) -> set[str]:
        return set().union(*(part.fields() for part in self.parts))

    def count(self) -> int:
        return sum(part.count() for part in self.parts)

    def to_sql(self, column: str) -> tuple[str, list[object]]:
        # Generosity survives both: a superset joined by AND or by OR is
        # still a superset, so the whole clause keeps the invariant.
        joiner = " AND " if self.every else " OR "
        clauses, params = [], []
        for part in self.parts:
            clause, values = part.to_sql(column)
            clauses.append(clause)
            params.extend(values)
        return f"({joiner.join(clauses)})", params


Filter = Union[Condition, Group]


#: A filter carried as text, in a query string or on a command line.
#: Longer than this it stopped being a question somebody asked.
MAX_TEXT = 8000


def read_conditions(text: Optional[str]) -> Optional[dict]:
    """A filter written as JSON text, or None when there is none.

    Unreadable is refused rather than dropped, wherever it came from. A
    filter that is quietly ignored answers from everything, and the
    caller cannot tell that from a genuinely wide result: they asked to
    see one team and got the whole space back, looking like an answer."""
    if text is None or not text.strip():
        return None
    if len(text) > MAX_TEXT:
        raise InvalidInput(f"conditions is too long: at most {MAX_TEXT} characters")
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise InvalidInput(f"conditions must be a JSON object: {exc}") from None
    if not isinstance(parsed, dict):
        raise InvalidInput("conditions must be a JSON object, a mapping naming a field, or all, or any")
    return parsed


def parse_filter(spec: object) -> Filter:
    """Read a filter, or say exactly why it cannot mean anything.

    Every refusal here is a question that would otherwise be answered
    with the wrong memories rather than with an error."""
    parsed = _read(spec, depth=1)
    total = parsed.count()
    if total > MAX_CONDITIONS:
        raise InvalidInput(f"a filter may hold at most {MAX_CONDITIONS} conditions, got {total}")
    return parsed


def _read(spec: object, depth: int) -> Filter:
    if not isinstance(spec, Mapping):
        raise InvalidInput("a filter must be a mapping naming a field, or all, or any")
    groups = [name for name in ("all", "any") if name in spec]
    if len(groups) == 2:
        raise InvalidInput("a group is all or any, not both")
    if groups:
        return _group(spec, groups[0], depth)
    return _condition(spec)


def _group(spec: Mapping, name: str, depth: int) -> Group:
    if depth > MAX_DEPTH:
        raise InvalidInput(f"a filter may be nested {MAX_DEPTH} groups deep, no further")
    if set(spec) - {name}:
        raise InvalidInput(f"a {name} group takes only its own list of conditions")
    parts = spec[name]
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        raise InvalidInput(f"{name} must be a list of conditions")
    if not parts:
        raise InvalidInput(f"{name} needs at least one condition to mean anything")
    return Group(every=name == "all", parts=tuple(_read(part, depth + 1) for part in parts))


def _condition(spec: Mapping) -> Condition:
    field = spec.get("field")
    if not isinstance(field, str) or not METADATA_KEY.match(field):
        raise InvalidInput("a condition needs a field naming a metadata key")
    named = [test for test in TESTS if test in spec]
    if len(named) != 1 or set(spec) - {"field", "not", *named}:
        raise InvalidInput(f"a condition makes exactly one test of its field, one of: {', '.join(TESTS)}")
    test, value = named[0], spec[named[0]]
    negate = spec.get("not", False)
    if not isinstance(negate, bool):
        raise InvalidInput("not must be true or false")
    return Condition(field=field, test=test, value=_value(test, field, value), negate=negate)


def _value(test: str, field: str, value: object) -> object:
    if test == "present":
        if value is not True:
            raise InvalidInput(f"present is a question, so {field}'s present must be true")
        return True
    if test in ("is", "has"):
        if not isinstance(value, str):
            raise InvalidInput(f"{field}'s {test} compares against text")
        return value
    if test == "in":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise InvalidInput(f"{field}'s in takes a list of values")
        if not value:
            raise InvalidInput(f"{field}'s in needs at least one value to choose between")
        if any(not isinstance(item, str) for item in value):
            raise InvalidInput(f"{field}'s in compares against text")
        return tuple(value)
    return _number(value, f"{field}'s {test}")
