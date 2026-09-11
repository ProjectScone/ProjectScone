"""Input rules are reusable without depending on memory orchestration."""

import pytest

from scone_memory.core.errors import InvalidInput


def test_shared_rules_normalize_and_reject_invalid_metadata():
    from scone_memory.core.validation import normalise_metadata, normalise_tags, normalise_time

    assert normalise_tags([" Team ", "team", "Ops"]) == ("team", "ops")
    assert normalise_metadata({"team": "Blue"}) == {"team": "Blue"}
    assert normalise_time("2026-09-08T12:00:00-05:00") == "2026-09-08T17:00:00.000Z"
    with pytest.raises(InvalidInput, match="metadata key"):
        normalise_metadata({"private-key": "value"})


import pytest


@pytest.mark.parametrize("text", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"])
def test_an_instant_past_the_ends_of_utc_is_invalid_not_an_overflow(text):
    """A valid offset can carry a first or last day past the representable
    range once it is moved to UTC; callers expect ValueError for bad input."""
    from scone_memory.core.timeutil import parse_rfc3339

    with pytest.raises(ValueError, match="outside"):
        parse_rfc3339(text)
