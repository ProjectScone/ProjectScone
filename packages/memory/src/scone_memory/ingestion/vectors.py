"""Checking what an embedder actually returned, for every caller.

Ingestion has validated embedding responses since it had them. Semantic
chunking asked an embedder a question too, and did not -- so an embedder
answering with nothing produced one chunk covering the whole text and
`subject_changed=False`, which is precisely what a genuinely coherent
passage produces. Broken provider output arrived disguised as a finding
about the text.

It lives here rather than in `batch`, and is not copied, because the
version that drifts from its original is the one that was copied.
"""

from __future__ import annotations

import math

#: Value types whose every instance is a number and never a bool, so a vector
#: made only of them needs only its finiteness checked, in one C-level pass.
#: Anything else -- a subclass, a numpy scalar, a bool -- is judged value by value.
_PLAIN_NUMBERS = frozenset((float, int))


def validated_vectors(response: object, count: int, dimension: int) -> list[list[float]]:
    """Vectors from one batch, checked. ``dimension`` is the width already
    settled for this operation, or 0 when the embedder declares none and
    nothing has settled it yet."""
    if not isinstance(response, list) or len(response) != count:
        raise ValueError('embedding response must contain one vector per input text')
    # Not every embedder declares a width -- a remote model behind an
    # endpoint that does not advertise one reports 0, and the abstention
    # floor reads the width off the vectors themselves for exactly that
    # case. Checking against 0 would refuse every vector such an embedder
    # ever returned. With no declared width the width of the first vector
    # settles it, and the caller carries that forward across every batch of
    # one operation: settling it per batch would let 64 vectors be eight
    # wide and the next sixteen, and an index built from those is silently
    # incoherent.
    expected = dimension if dimension else None
    vectors: list[list[float]] = []
    for vector in response:
        if not isinstance(vector, list) or not vector:
            raise ValueError('embedding vector does not match the configured dimension')
        if expected is None:
            expected = len(vector)
        if len(vector) != expected:
            raise ValueError('embedding vector does not match the configured dimension')
        try:
            if set(map(type, vector)) <= _PLAIN_NUMBERS:
                valid = all(map(math.isfinite, vector))
            else:
                valid = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) for value in vector)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError('embedding vector must contain finite numeric values')
        # Providers may reuse their response buffers on the next call.
        vectors.append(list(vector))
    return vectors

