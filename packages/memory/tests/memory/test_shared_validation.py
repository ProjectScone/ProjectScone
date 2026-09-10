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
