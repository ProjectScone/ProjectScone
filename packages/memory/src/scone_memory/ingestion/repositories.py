"""More than one repository in a space: each named, its files kept apart, and an import of what another publishes followed to its file.

A map reads one directory, and every file it reads is named by its path
below that directory. Two repositories mapped into one space would then
share names -- both have a `src/main.py` -- and the graph would fold them
into one thing. Naming a repository (`scone map ROOT --repo lib`) puts
its name in front of every path (`lib/src/main.py`), so the two stay
two, and the explorer shows each repository as a top directory.

What one repository publishes is written in its manifest: a
`pyproject.toml` names the package `scone_memory`, a `package.json`
names `@acme/ui`, a `Cargo.toml` names the crate `acme-core`, a
`go.mod` names the module `example.com/acme`. An import in another
repository that names one of those (`import scone_memory.x`, `from
"@acme/ui/button"`, `use acme_core::store`, `"example.com/acme/pkg"`)
is a link between the two repositories, and the map follows it to the
file when the file is in the space: the manifest says which directory
publishes the package, the language says how a module maps to a path
below it, and the space's own record of files says whether that file
was read. Nothing is guessed: a package nobody here publishes, or a
module whose file was never mapped, stays a name.
"""
from __future__ import annotations

import posixpath
import re
from typing import TYPE_CHECKING, Iterable

from .manifests import is_manifest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

#: What a repository may be called: a path segment without a slash.
TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def repository_prefix(tag: str | None) -> str:
    """The prefix a repository's paths carry, or none; a bad name is refused."""
    if tag is None:
        return ""
    if not isinstance(tag, str) or not TAG.match(tag) or tag in (".", ".."):
        raise ValueError("a repository name is one path segment: letters, digits, dots, dashes and underscores, "
                         "at most 64 of them")
    if is_manifest(tag) or is_manifest(f"{tag}/a.txt"):
        # `requirements/` is where manifests are looked for; a repository
        # so named would make a manifest of every text file in it.
        raise ValueError(f"a repository cannot be named {tag!r}: files under a directory of that name are read as "
                         "package manifests")
    return tag + "/"


def published_by(claims: Iterable[tuple[str, str, str]], prefix: str = "") -> dict[str, str]:
    """What the manifests among ``claims`` (subject, predicate, object)
    publish: package name -> the directory that publishes it, from every
    `defines` claim a manifest makes about a package."""
    from ..entities.kinds import is_file_name

    found: dict[str, str] = {}
    for subject, predicate, obj in claims:
        # A manifest defines its package (`libpkg`, `@acme/ui`, `example.com/svc`)
        # and nothing else; a declaration (`path:name`) or a file is not one.
        if predicate != "defines" or not is_manifest(subject.split("/")[-1]) or ":" in obj or is_file_name(obj):
            continue
        found.setdefault(obj, posixpath.dirname(subject) if "/" in subject else prefix.rstrip("/"))
    return found


async def space_publishes(engine: "MemoryEngine", space: str, prefix: str = "") -> tuple[dict[str, str], set[str]]:
    """What the space already holds of other repositories: the packages
    their manifests publish, by directory, and the files they mapped --
    everything not under ``prefix``, this map's own repository. Without a
    prefix there are no other repositories to speak of: an unnamed map
    reads one tree, and what the space holds of it may be stale."""
    from ..entities.kinds import is_file_name
    from ..entities.read import load_projection

    if not prefix:
        return {}, set()
    projection, _ = await load_projection(engine, space, mode="current")
    keys = {entity.entity_id: entity.key for entity in projection.entities}
    claims = [(keys.get(relation.subject_id, ""), relation.predicate, keys.get(relation.object_id, ""))
              for relation in projection.relations]
    published = {name: where for name, where in published_by(claims).items() if not (where + "/").startswith(prefix)}
    known = {key for key in keys.values() if is_file_name(key) and "/" in key and not key.startswith(prefix)}
    return published, known
