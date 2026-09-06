"""Shared input checks for vector adapters, before they read or write data."""

import math
from typing import Optional, Sequence


def validate_vector(vector: Sequence[float], dim: Optional[int]) -> None:
    if dim is None:
        raise ValueError("vector index must be initialized with ensure(dim)")
    if len(vector) != dim:
        raise ValueError(f"vector has {len(vector)} dimensions; index requires {dim}")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("vector values must be finite")
