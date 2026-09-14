"""What a document says about the files beside it, read into the graph.

A codebase's documents point at its code and at each other: a README
links the design note, the design note names the module it describes,
an ADR is cited from both. Left unread, the graph has every call in the
code and nothing that says which document explains a file -- so "what
rests on this module" answers with callers and never with the page that
would go stale. The leading code-graph tool turns Markdown links and
wikilinks into edges between documents; this reads a document for
three things and records them as claims like every other file's:

- **A link to a file the walk read** (``[text](./other.md)``,
  ``[[Other]]``, a reStructuredText ``:doc:`` or ``<target>`_``, a path
  in backticks such as ```src/pkg/engine.py```) becomes ``references``
  from the document to that file. One relationship per pair, however
  many times the page links it.
- **A decision record or standard** (``ADR-12``, ``RFC 7231``) becomes
  ``cites``, the same node the code's comments cite, so the graph reaches
  the document and the code that follow one decision from either side.
- Nothing else. A link that leads outside the tree (a URL) is counted,
  not claimed: there is nothing there to bind to. A link to a file the
  walk did not read is unresolved and said so, like a call the code graph
  cannot place, never guessed at.

Links inside fenced code blocks are examples, not references, and are
skipped. Every claim quotes the line it came from with its byte span, so
it can be checked against the document the way every quote is.
"""

from __future__ import annotations

from dataclasses import dataclass
import posixpath
import re
from typing import Optional, Protocol

from .code_graph import CITES, CodeClaim, MAX_CLAIMS, MAX_LINES, REFERENCES, cited

#: The documents this reads. Plain text is included because a tree's
#: notes are often `.txt`, and a link is a link whatever the suffix.
DOC_SUFFIXES: tuple[str, ...] = (".md", ".markdown", ".rst", ".txt")
#: A backticked path is taken as one only with a suffix the walk reads,
#: so `` `x.y` `` (an attribute) and `` `a/b` `` (a route) are not.
PATH_SUFFIXES = ("py", "pyi", "ts", "tsx", "js", "jsx", "mjs", "cjs", "go", "rs", "java", "kt", "scala", "cs",
                 "c", "h", "cc", "cpp", "hpp", "rb", "php", "swift", "md", "markdown", "rst", "txt", "toml",
                 "json", "yaml", "yml")

_INLINE = re.compile(r"(?<!\\)(!?)\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_DEFINITION = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?\s*(?:\"[^\"]*\")?\s*$")
_WIKI = re.compile(r"\[\[([^\]|#]+?)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
_BACKTICKED = re.compile(r"`([A-Za-z0-9_][\w./-]*\.(?:" + "|".join(PATH_SUFFIXES) + r"))`")
_RST_DOC = re.compile(r":doc:`(?:[^`<]*<)?([^`>]+)>?`")
_RST_LINK = re.compile(r"`[^`<]*<([^>`]+)>`_")
_RST_INCLUDE = re.compile(r"^\s*\.\.\s+(?:include|literalinclude)::\s+(\S+)")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


class LinkResolve(Protocol):
    """How a document's link is followed to a file the walk read."""

    def link(self, path: str, target: str) -> Optional[str]: ...


def is_document(path: str) -> bool:
    name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return any(name.endswith(suffix) for suffix in DOC_SUFFIXES)


@dataclass(frozen=True)
class DocLinks:
    """What a document's links came to: the claims for those that bound,
    the targets that did not, and how many led outside the tree."""

    claims: tuple[CodeClaim, ...]
    unresolved: tuple[str, ...]
    outside: int


def _targets(line: str) -> list[tuple[str, bool]]:
    """Every link target on a line, and whether it is a wikilink (which
    names a page by title rather than by path)."""
    found: list[tuple[str, bool]] = []
    definition = _DEFINITION.match(line)
    if definition:
        found.append((definition.group(1), False))
    for image, target in _INLINE.findall(line):
        if not image:
            found.append((target, False))
    found.extend((name.strip(), True) for name in _WIKI.findall(line))
    found.extend((target, False) for target in _BACKTICKED.findall(line))
    found.extend((target.strip(), False) for target in _RST_DOC.findall(line))
    found.extend((target.strip(), False) for target in _RST_LINK.findall(line))
    include = _RST_INCLUDE.match(line)
    if include:
        found.append((include.group(1), False))
    return found


def _bare(target: str) -> Optional[str]:
    """The path a target names, without fragment or query; None for a
    fragment alone."""
    target = target.split("#", 1)[0].split("?", 1)[0].strip()
    return target or None


def doc_links(content: str, path: str, *, resolve: Optional[LinkResolve] = None) -> DocLinks:
    """What a document references and cites, with what it could not bind."""
    if not content or content.count("\n") > MAX_LINES:
        return DocLinks((), (), 0)
    lines = content.split("\n")
    claims: list[CodeClaim] = []
    bound: set[tuple[str, str]] = set()
    unresolved: list[str] = []
    outside = 0
    inside_fence: Optional[str] = None
    offset = 0
    for number, line in enumerate(lines, start=1):
        begins = offset
        offset += len(line.encode("utf-8")) + 1
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if inside_fence is None:
                inside_fence = marker[0]
            elif marker[0] == inside_fence:
                inside_fence = None
            continue
        if inside_fence is not None:
            continue
        quote = line.strip()
        ends = begins + len(line.encode("utf-8"))

        def say(predicate: str, obj: str) -> None:
            if len(claims) < MAX_CLAIMS and (predicate, obj) not in bound:
                bound.add((predicate, obj))
                claims.append(CodeClaim(path, predicate, obj, quote, number, begins, ends))

        for target, by_title in _targets(line):
            if _SCHEME.match(target) and not by_title:
                outside += 1
                continue
            named = target if by_title else _bare(target)
            if named is None:
                continue
            where = None
            if resolve is not None:
                where = resolve.link(path, f"[[{named}]]" if by_title else named)
            if where is None or where == path:
                if where is None:
                    unresolved.append(named)
                continue
            say(REFERENCES, where)
        for document in cited(line):
            say(CITES, document)
    return DocLinks(tuple(claims), tuple(dict.fromkeys(unresolved)), outside)


def doc_claims(content: str, path: str, *, resolve: Optional[LinkResolve] = None) -> tuple[CodeClaim, ...]:
    """The claims alone, for the one place that records every file's."""
    return doc_links(content, path, resolve=resolve).claims


def link_target(paths: set[str], path: str, target: str) -> Optional[str]:
    """The file a document's link names, among the files a walk read.

    Relative to the document first, then from the root, then the one
    file whose path ends in the target (a page that names
    ``retrieval/recall.py`` means the one file spelt that way, wherever
    the package root is), and none when two would answer. ``[[Title]]``
    names a document by its stem, case folded. Without a suffix a link
    tries the document suffixes and an index page.
    """
    if target.startswith("[[") and target.endswith("]]"):
        title = target[2:-2].strip().lower()
        pages = [seen for seen in paths if is_document(seen)
                 and posixpath.splitext(posixpath.basename(seen))[0].lower() == title]
        return pages[0] if len(pages) == 1 else None
    here = posixpath.dirname(path)
    relative = posixpath.normpath(posixpath.join(here, target)) if not target.startswith("/") else None
    rooted = posixpath.normpath(target.lstrip("/"))
    for candidate in dict.fromkeys(c for c in (relative, rooted) if c is not None and not c.startswith("..")):
        if candidate in paths:
            return candidate
        for suffix in DOC_SUFFIXES:
            if f"{candidate}{suffix}" in paths:
                return f"{candidate}{suffix}"
        for index in ("index.md", "README.md", "index.rst"):
            if f"{candidate}/{index}" in paths:
                return f"{candidate}/{index}"
    tail = posixpath.normpath(target.strip("/"))
    if tail.startswith(".."):
        return None
    ending = [seen for seen in paths if seen == tail or seen.endswith("/" + tail)]
    return ending[0] if len(ending) == 1 else None
