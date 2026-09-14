"""A codebase in the graph as the tree of directories, files and declarations it is.

Code enters the graph as facts its readers record: a file ``defines`` a
declaration (``pkg/store.py:Shelf``), a declaration defines a method
(``pkg/store.py:Shelf.put``), a file ``imports`` another, a declaration
``calls`` one. The tree is those paths and definitions laid out as a
file explorer lays them out, each node saying how many files and
declarations sit beneath it and what it calls, imports and inherits,
and what does each of those to it.

It holds only what the facts hold, and says what it left out:

- an entity is placed only when a code relation holds it and its label is
  a path or a declaration qualified by one, so a prose name shaped like a
  file (``Node.js``) is not taken for one;
- a file or declaration that only a call or an import names was not read
  where it is defined, and is marked so rather than drawn like one that was;
- children past ``max_children`` under one node are counted, never dropped
  in silence, and so are the relations past ``MAX_LINKS`` in one list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from ..core.errors import InvalidInput
from .classify import CODE_PREDICATES
from .project import EntityProjection

NodeKind = Literal["directory", "file", "declaration"]
#: Children listed under one node before the rest are only counted.
MAX_CHILDREN = 200
#: Relations listed in one direction of one predicate before the rest are only counted.
MAX_LINKS = 50
#: The predicates a node's relations are read from, and the name of each in reverse.
_REVERSED = {"defines": "defined_by", "imports": "imported_by", "calls": "called_by", "inherits": "inherited_by",
             "mixes_in": "mixed_into", "depends_on": "depended_on_by", "develops_with": "developed_with_by"}
# A file: a last path segment carrying an extension. A path may hold spaces.
_FILE = re.compile(r"[^:\n]*?[^/:\n]\.[A-Za-z0-9_+-]{1,16}")


@dataclass(frozen=True)
class Link:
    entity_id: str
    label: str


@dataclass
class TreeNode:
    name: str
    kind: NodeKind
    #: The full path or qualified declaration; a directory's path.
    path: str
    entity_id: Optional[str] = None
    #: False for a file or declaration the graph knows only because
    #: something calls or imports it: it was not read where it is defined.
    defined: bool = True
    files: int = 0
    declarations: int = 0
    children: list["TreeNode"] = field(default_factory=list)
    #: Children past the cap, counted and not listed.
    more: int = 0
    #: Predicate (or its reverse) to the entities it links, in label order, capped.
    links: dict[str, tuple[Link, ...]] = field(default_factory=dict)
    #: Predicate (or its reverse) to how many it links, the uncapped count.
    link_counts: dict[str, int] = field(default_factory=dict)

    @property
    def calls(self) -> tuple[Link, ...]:
        return self.links.get("calls", ())

    @property
    def called_by(self) -> tuple[Link, ...]:
        return self.links.get("called_by", ())

    @property
    def imports(self) -> tuple[Link, ...]:
        return self.links.get("imports", ())

    @property
    def imported_by(self) -> tuple[Link, ...]:
        return self.links.get("imported_by", ())


@dataclass(frozen=True)
class CodeTree:
    root: Optional[TreeNode]
    #: Entities in the projection that are not placed in the tree.
    not_code: int = 0
    #: Files and declarations known only as something called or imported.
    undefined: int = 0
    #: Children counted and not listed, over the whole tree.
    cut: int = 0
    max_children: int = MAX_CHILDREN
    why: str = ""


def _split(label: str) -> tuple[str, Optional[str]]:
    """A label as (file, declaration), or (file, None) for a file, or ("", None) for neither."""
    head, colon, tail = label.rpartition(":")
    if colon and tail and _FILE.fullmatch(head):
        return head, tail
    return (label, None) if _FILE.fullmatch(label) else ("", None)


def code_tree(projection: EntityProjection, *, max_children: int = MAX_CHILDREN) -> CodeTree:
    """The projection's code as a tree of directories, files and declarations."""
    if isinstance(max_children, bool) or not isinstance(max_children, int) or max_children < 1:
        raise InvalidInput(f"max_children is a whole number from 1, not {max_children!r}")
    labels = {entity.entity_id: entity.label for entity in projection.entities}
    code = [relation for relation in projection.relations if relation.predicate in CODE_PREDICATES]
    held = {end for relation in code for end in (relation.subject_id, relation.object_id)}
    # A file was read when it says something; a declaration, when something defines it.
    read = {relation.subject_id for relation in code}
    defined = {relation.object_id for relation in code if relation.predicate == "defines"}
    placed: dict[str, TreeNode] = {}
    for entity_id in sorted(held, key=lambda one: labels[one]):
        file, declaration = _split(labels[entity_id])
        if file:
            placed[labels[entity_id]] = TreeNode(
                name=declaration if declaration is not None else file.rpartition("/")[2],
                kind="declaration" if declaration is not None else "file", path=labels[entity_id],
                entity_id=entity_id, defined=entity_id in (defined if declaration is not None else read))
    not_code = len(projection.entities) - len(placed)
    if not placed:
        why = (f"no source files in this graph: {not_code} entities, none a path a code relation holds; "
               "code is read into the graph with `scone map`")
        return CodeTree(root=None, not_code=not_code, max_children=max_children, why=why)
    by_id = {node.entity_id: node for node in placed.values()}
    for relation in code:
        for node, other, name in ((by_id.get(relation.subject_id), relation.object_id, relation.predicate),
                                  (by_id.get(relation.object_id), relation.subject_id,
                                   _REVERSED[relation.predicate])):
            if node is not None:
                node.link_counts[name] = node.link_counts.get(name, 0) + 1
                node.links[name] = (*node.links.get(name, ()), Link(other, labels[other]))
    for node in placed.values():
        node.links = {name: tuple(sorted(links, key=lambda link: (link.label.casefold(), link.label)))[:MAX_LINKS]
                      for name, links in node.links.items()}

    root = TreeNode(name="", kind="directory", path="")
    directories: dict[str, TreeNode] = {"": root}

    def directory(path: str) -> TreeNode:
        if path not in directories:
            parent, _, name = path.rpartition("/")
            directories[path] = TreeNode(name=name, kind="directory", path=path)
            directory(parent).children.append(directories[path])
        return directories[path]

    files: dict[str, TreeNode] = {}
    for label, node in placed.items():
        file, declaration = _split(label)
        if declaration is None:
            files[file] = node
    for label in placed:
        file, _ = _split(label)
        if file not in files:
            # A declaration of a file the graph holds no entity for: the file is still where it sits.
            files[file] = TreeNode(name=file.rpartition("/")[2], kind="file", path=file, defined=False)
    for file, node in files.items():
        directory(file.rpartition("/")[0]).children.append(node)
    for label, node in placed.items():
        file, declaration = _split(label)
        if declaration is None:
            continue
        holder, dot, _ = declaration.rpartition(".")
        parent = placed.get(f"{file}:{holder}") if dot else None
        (parent if parent is not None else files[file]).children.append(node)

    cut = 0

    def settle(node: TreeNode) -> None:
        nonlocal cut
        for child in node.children:
            settle(child)
        node.files = sum(child.files for child in node.children) + (node.kind == "file")
        node.declarations = (sum(child.declarations for child in node.children)
                             + sum(child.kind == "declaration" for child in node.children))
        node.children.sort(key=lambda child: (child.kind == "declaration", child.kind == "file",
                                              child.name.casefold(), child.name))
        if len(node.children) > max_children:
            node.more = len(node.children) - max_children
            cut += node.more
            node.children = node.children[:max_children]

    settle(root)
    # A chain of directories each holding only the next is one node, and so is a root holding one entry.
    while root.kind == "directory" and len(root.children) == 1 and not root.more:
        only = root.children[0]
        if root.name and only.kind != "directory":
            break
        root = TreeNode(**{**only.__dict__, "name": f"{root.name}/{only.name}" if root.name else only.name})

    def compact(node: TreeNode) -> None:
        for index, child in enumerate(node.children):
            while (child.kind == "directory" and len(child.children) == 1 and not child.more
                   and child.children[0].kind == "directory"):
                only = child.children[0]
                child = TreeNode(**{**only.__dict__, "name": f"{child.name}/{only.name}"})
            node.children[index] = child
            compact(child)

    compact(root)
    undefined = sum(not node.defined for node in placed.values()) + sum(
        not node.defined and node.entity_id is None for node in files.values())
    parts = [f"{root.files} files and {root.declarations} declarations"]
    if not_code:
        parts.append(f"{not_code} entities are not code a path names and are not in the tree")
    if undefined:
        parts.append(f"{undefined} files or declarations are known only from a call or an import, "
                     "not read where they are defined")
    if cut:
        parts.append(f"{cut} children past the cap of {max_children} under one node are counted, not listed")
    return CodeTree(root=root, not_code=not_code, undefined=undefined, cut=cut, max_children=max_children,
                    why="; ".join(parts))
