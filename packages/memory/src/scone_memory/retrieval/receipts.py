"""What a retrieval stage's receipt may say about the answer.

Every stage replaces the answer's items with its own output -- widening,
withholding, merging and code context all do -- so a receipt's copy of
the passages is always redundant with ``items`` and always older than it.
Handing one back returns what a later stage removed: text a withholding
policy took out, or a passage whose source a later stage found deleted.

This lives here rather than beside either caller because both the CLI and
the HTTP route stage the same pipeline, and the first version of this rule
was written in one of them only. A rule about what may leave the system
that exists on one of two exits is not a rule.
"""

from __future__ import annotations


def staged(record: dict) -> dict:
    """``record`` without its own copy of the passages.

    The copy is dropped unconditionally, not only under a withholding
    policy. Conditioning it on withholding treats one symptom of the
    class -- it was written that way first, and the deleted-source case
    walked straight through the gap. The counts are the useful part of a
    receipt and they stay.
    """
    kept = {key: value for key, value in record.items() if key != "items"}
    if "items" in record:
        kept["items_not_repeated"] = (
            "the passages this answer returns are in `items`; this receipt described them at "
            "an earlier stage and its copy is not returned")
    return kept
