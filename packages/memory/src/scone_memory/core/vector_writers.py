"""What a vector index records about who wrote its vectors.

The invariant every recording index keeps: a recorded writer with basis
``written`` or ``declared`` wrote (or was vouched for) every stored vector.
Any write that would break that claim changes the record in the same
transaction as the write, so a reader never sees a writer's name over
vectors someone else produced.

Records are ``(name, basis)``:

- ``(writer, "written")``: this writer produced every vector.
- ``(writer, "declared")``: an operator vouched that every vector is this writer's.
- ``(writer, "unrecorded")``: this writer added vectors beside older vectors
  that no one recorded; only a declaration or a rebuild settles them.
- ``("mixed", "invalidated")``: vectors from more than one writer.
- ``("rebuilding:<writer>:<nonce>", "rebuilding")``: a rebuild began and has
  not finished; nothing is trusted until it does.
- None: nothing recorded yet.
"""

from __future__ import annotations

Writer = tuple[str, str]

MIXED: Writer = ("mixed", "invalidated")


def rebuilding_token(writer: str, nonce: str) -> Writer:
    return (f"rebuilding:{writer}:{nonce}", "rebuilding")


def after_write(current: Writer | None, writer: str, holds_vectors: bool) -> Writer:
    """The record an index must hold once ``writer`` has added vectors."""
    if current is None:
        return (writer, "unrecorded") if holds_vectors else (writer, "written")
    name, basis = current
    if name == writer and basis in ("written", "declared", "unrecorded"):
        return current
    if basis == "rebuilding" and name.startswith(f"rebuilding:{writer}:"):
        return current
    return MIXED


class VectorsNotComparable(ValueError):
    """The index's record does not vouch that its vectors are this writer's."""


def vouches(current: Writer | None, writer: str, holds_vectors: bool) -> bool:
    """True when every stored vector can be compared with ``writer``'s."""
    if current is None:
        return not holds_vectors
    return current[0] == writer and current[1] in ("written", "declared")
