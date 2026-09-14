"""Predicates the framework reads out of files, and what they are like.

A file defines many things and imports many modules; a project depends
on many packages. A ledger predicate holds one value at a time unless it
is configured to hold many, and that is right for what people state:
"lives in Lisbon" retires "lives in Austin". It is wrong for what a
reader extracts from a file, where a second import is not a change of
mind about the first. Read under the one-value rule, a module with three
imports held one and closed two as superseded, and every graph built on
the ledger kept the last claim of each kind and called the rest history.

So these predicates hold many values by their nature. The declaration
sits here, in the core, where the engine can read it without reaching
into the readers, and no configuration takes a predicate out of it. The
readers' own constants are checked against this set by test, so a reader
cannot gain a predicate the ledger does not know the shape of.
"""

from __future__ import annotations

#: What a source file says about itself, and what a manifest declares.
MANY_VALUED: frozenset[str] = frozenset({
    "defines", "imports", "calls", "inherits", "mixes_in", "uses_type", "notes", "flags", "cites",
    "depends_on", "develops_with", "references",
})
