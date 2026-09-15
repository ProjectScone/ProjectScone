"""A codebase in the graph as the tree of directories, files and declarations it is.

Code enters the graph as facts its readers record: a file ``defines`` a
declaration (``pkg/store.py:Shelf``), a declaration defines a method
(``pkg/store.py:Shelf.put``), a file ``imports`` another, a declaration
``calls`` one, a document ``references`` a file it links. The tree is those paths and definitions laid out as a
file explorer lays them out, each node saying how many files and
declarations sit beneath it and what it calls, imports and inherits,
and what does each of those to it.

It holds only what the facts hold, and says what it left out:

- an entity is placed only when a code relation holds it and its label is
  a path or a declaration qualified by one, so a prose name shaped like a
  file (``Node.js``) is not taken for one;
- a file or declaration that only a call, an import or a link in a document names was not read
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
             "mixes_in": "mixed_into", "depends_on": "depended_on_by", "develops_with": "developed_with_by",
             "references": "referenced_by", "imports_when_called": "imported_when_called_by",
             "imports_for_types": "imported_for_types_by", "uses_type": "type_used_by", "runs_with": "run_with_by",
             "requires_env": "required_by", "connects_to": "connected_to_by"}
# A file: a last path segment carrying an extension. A path may hold spaces.
_FILE = re.compile(r"[^:\n]*?[^/:\n]\.[A-Za-z0-9_+-]{1,16}")


@dataclass(frozen=True)
class Link:
    entity_id: str
    label: str
    #: The facts behind the relation.
    fact_ids: tuple[int, ...] = ()


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


def code_tree(projection: EntityProjection, *, max_children: Optional[int] = None) -> CodeTree:
    """The projection's code as a tree of directories, files and declarations.
    ``max_children`` defaults to ``MAX_CHILDREN`` as it stands when called."""
    if max_children is None:
        max_children = MAX_CHILDREN
    if isinstance(max_children, bool) or not isinstance(max_children, int) or max_children < 1:
        raise InvalidInput(f"max_children is a whole number from 1, not {max_children!r}")
    labels = {entity.entity_id: entity.label for entity in projection.entities}
    code = [relation for relation in projection.relations if relation.predicate in CODE_PREDICATES]
    held = {end for relation in code for end in (relation.subject_id, relation.object_id)}
    # A file was read when it says something; a declaration, when something defines it.
    read = {relation.subject_id for relation in code}
    defined = {relation.object_id for relation in code if relation.predicate == "defines"}
    placed: dict[str, TreeNode] = {}
    # A name with no directory that nothing was read from and that holds no
    # declaration is a package or module (`lodash.merge`, `socket.io`), not a file here.
    holding = {_split(labels[one])[0] for one in held if _split(labels[one])[1] is not None}
    for entity_id in sorted(held, key=lambda one: labels[one]):
        file, declaration = _split(labels[entity_id])
        if file and declaration is None and "/" not in file and entity_id not in read and file not in holding:
            continue
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
                node.links[name] = (*node.links.get(name, ()), Link(other, labels[other], relation.fact_ids))
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
        parts.append(f"{not_code} entities are not source files or declarations and are not in the tree")
    if undefined:
        parts.append(f"{undefined} files or declarations are known only from a call, an import or a link in a "
                     "document, not read where they are defined")
    if cut:
        parts.append(f"{cut} children past the cap of {max_children} under one node are counted, not listed")
    return CodeTree(root=root, not_code=not_code, undefined=undefined, cut=cut, max_children=max_children,
                    why="; ".join(parts))



_PAGE_STYLE = """
:root{--ground:#f7f8f6;--ink:#1d2320;--muted:#5d6a63;--rule:#d5dbd6;--accent:#1f6f5c;--mark:#fbe7a8;--panel:#ffffff}
@media (prefers-color-scheme: dark){:root{--ground:#141816;--ink:#e3e8e5;--muted:#9aa8a0;--rule:#2d3531;
--accent:#6cc4a8;--mark:#5a4a12;--panel:#1b201d}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:16px 20px;border-bottom:1px solid var(--rule)}
h1{margin:0 0 4px;font-size:18px}
header p{margin:0;color:var(--muted);max-width:70ch}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px}
.controls input{padding:6px 8px;min-width:16rem;max-width:100%;border:1px solid var(--rule);border-radius:4px;
background:var(--panel);color:var(--ink)}
.controls button{padding:6px 10px;border:1px solid var(--rule);border-radius:4px;background:var(--panel);color:var(--ink)}
#filter-count{color:var(--muted)}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(16rem,26rem);gap:0}
@media (max-width:760px){main{grid-template-columns:1fr}}
.tree{padding:12px 20px;overflow-x:auto}
ul{list-style:none;margin:0;padding-left:18px}
.tree>ul{padding-left:0}
summary{cursor:pointer;padding:1px 0}
.leaf{display:block;padding:1px 0 1px 14px}
.kind{display:inline-block;min-width:5.5em;color:var(--muted);font-size:12px;font-variant:small-caps}
button.name{border:0;background:none;padding:0 2px;color:var(--ink);font:inherit;cursor:pointer}
button.name:focus-visible,summary:focus-visible,.controls :focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.k-directory button.name{font-weight:600}
.k-declaration button.name{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}
.count{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums;margin-left:6px}
.unread{color:var(--muted);font-size:12px;font-style:italic;margin-left:6px}
.more{color:var(--muted);font-size:12px;padding-left:14px}
li.chosen>details>summary button.name,li.chosen>.leaf button.name{background:var(--mark)}
li.hidden{display:none}
aside{border-left:1px solid var(--rule);padding:12px 16px;background:var(--panel);min-height:100%}
@media (max-width:760px){aside{border-left:0;border-top:1px solid var(--rule)}}
aside h2{font-size:15px;margin:0 0 4px;overflow-wrap:anywhere}
aside h3{font-size:13px;margin:12px 0 4px;color:var(--muted)}
aside ul{padding-left:0}
aside li{overflow-wrap:anywhere;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
aside button.name{text-align:left;color:var(--accent)}
.muted{color:var(--muted)}
"""

_PAGE_CODE = """
(function(){
  var data = JSON.parse(document.getElementById('tree-data').textContent);
  var byId = {};
  data.nodes.forEach(function(node){ byId[node.id] = node; });
  var inspect = document.getElementById('inspect');
  var filter = document.getElementById('filter');
  var said = document.getElementById('filter-count');
  function item(id){ return document.querySelector('li[data-id="' + id + '"]'); }
  function openTo(id){
    var node = byId[id];
    while (node && node.parent){
      var holder = item(node.parent);
      var fold = holder && holder.querySelector(':scope > details');
      if (fold){ fold.open = true; }
      node = byId[node.parent];
    }
  }
  function text(tag, words, cls){ var el = document.createElement(tag); el.textContent = words; if (cls){ el.className = cls; } return el; }
  var NAMES = {calls:'Calls', called_by:'Called by', imports:'Imports', imported_by:'Imported by',
    defines:'Defines', defined_by:'Defined by', inherits:'Inherits', inherited_by:'Inherited by',
    mixes_in:'Mixes in', mixed_into:'Mixed into', depends_on:'Depends on', depended_on_by:'Depended on by',
    develops_with:'Develops with', developed_with_by:'Developed with by', references:'References',
    referenced_by:'Referenced by', imports_when_called:'Imports when called',
    imported_when_called_by:'Imported when called by', imports_for_types:'Imports for types',
    imported_for_types_by:'Imported for types by', uses_type:'Uses type', type_used_by:'Type used by',
    runs_with:'Runs with', run_with_by:'Run with by', requires_env:'Requires env', required_by:'Required by',
    connects_to:'Connects to', connected_to_by:'Connected to by'};
  function choose(id){
    var node = byId[id];
    if (!node){ return; }
    document.querySelectorAll('li.chosen').forEach(function(el){ el.classList.remove('chosen'); });
    var li = item(id);
    if (li){ li.classList.add('chosen'); }
    inspect.replaceChildren();
    inspect.appendChild(text('h2', node.path || node.name));
    var counts = node.kind === 'declaration' ? '' : node.files + ' files, ';
    inspect.appendChild(text('p', node.kind + ' · ' + counts + node.declarations + ' declarations beneath' +
      (node.defined ? '' : ' · known only from a call, an import or a link in a document, not read where it is defined'), 'muted'));
    Object.keys(node.counts).sort().forEach(function(name){
      inspect.appendChild(text('h3', (NAMES[name] || name) + ' (' + node.counts[name] + ')'));
      var list = document.createElement('ul');
      node.links[name].forEach(function(pair){
        var entry = document.createElement('li');
        entry.title = 'facts ' + pair[2].join(', ');
        if (pair[1]){
          var go = text('button', pair[0], 'name');
          go.type = 'button';
          go.addEventListener('click', function(){
            var at = item(pair[1]);
            if (at && at.classList.contains('hidden')){ filter.value = ''; filter.dispatchEvent(new Event('input')); }
            openTo(pair[1]); choose(pair[1]);
            if (at){ at.scrollIntoView({block:'nearest'}); }
          });
          entry.appendChild(go);
        } else { entry.textContent = pair[0]; }
        list.appendChild(entry);
      });
      inspect.appendChild(list);
      var hidden = node.counts[name] - node.links[name].length;
      if (hidden > 0){ inspect.appendChild(text('p', hidden + ' more not listed', 'muted')); }
    });
    if (!Object.keys(node.counts).length){ inspect.appendChild(text('p', 'No calls, imports or definitions recorded for this node.', 'muted')); }
  }
  document.querySelector('.tree').addEventListener('click', function(event){
    var button = event.target.closest('button.name');
    if (!button){ return; }
    event.preventDefault();
    choose(button.closest('li').dataset.id);
  });
  function every(open){ document.querySelectorAll('.tree details').forEach(function(fold){ fold.open = open; }); }
  document.getElementById('expand-all').addEventListener('click', function(){ every(true); });
  document.getElementById('collapse-all').addEventListener('click', function(){ every(false); });
  filter.addEventListener('input', function(){
    var wanted = filter.value.trim().toLocaleLowerCase();
    var keep = {};
    var found = 0;
    data.nodes.forEach(function(node){
      if (!wanted || (node.path || node.name).toLocaleLowerCase().indexOf(wanted) !== -1){
        if (wanted){ found += 1; }
        var at = node;
        while (at){ keep[at.id] = true; at = at.parent ? byId[at.parent] : null; }
      }
    });
    document.querySelectorAll('.tree li[data-id]').forEach(function(li){ li.classList.toggle('hidden', !keep[li.dataset.id]); });
    if (wanted){ data.nodes.forEach(function(node){ if (keep[node.id]){ openTo(node.id); } }); }
    said.textContent = wanted ? found + ' matching' : '';
  });
})();
"""


def tree_page(tree: CodeTree, space: str, *, notes: tuple[str, ...] = (), projection: str = "") -> str:
    """``tree`` as one self-contained page: nested disclosures, a filter,
    expand and collapse all, and a panel of the chosen node's relations.
    ``notes`` name the projection and the read it lays out, before the tree's own account."""
    import base64
    import hashlib
    import html as markup
    import json

    nodes: list[dict[str, object]] = []
    ids: dict[int, str] = {}

    def number(node: TreeNode, parent: Optional[str]) -> None:
        ids[id(node)] = f"n{len(ids)}"
        nodes.append({"id": ids[id(node)], "name": node.name, "kind": node.kind, "path": node.path,
                      "parent": parent, "defined": node.defined, "files": node.files,
                      "declarations": node.declarations, "more": node.more})
        for child in node.children:
            number(child, ids[id(node)])

    if tree.root is not None:
        number(tree.root, None)
    by_entity = {node.entity_id: ids[id(node)] for node in _walk(tree.root) if node.entity_id is not None}
    for record, node in zip(nodes, _walk(tree.root)):
        record["links"] = {name: [[link.label, by_entity.get(link.entity_id), list(link.fact_ids)] for link in links]
                           for name, links in node.links.items()}
        record["counts"] = dict(node.link_counts)

    def listed(node: TreeNode, depth: int) -> str:
        count = (f"{node.declarations} declarations" if node.kind != "directory"
                 else f"{node.files} files · {node.declarations} declarations")
        head = (f'<span class="kind">{node.kind}</span><button type="button" class="name">'
                f'{markup.escape(node.name or "(root)")}</button>'
                + (f'<span class="count">{count}</span>' if node.kind != "declaration" or node.children else "")
                + ('' if node.defined else '<span class="unread">not read</span>'))
        more = f'<li class="more">{node.more} more not listed</li>' if node.more else ""
        if node.children or node.more:
            inner = "".join(listed(child, depth + 1) for child in node.children) + more
            return (f'<li class="k-{node.kind}" data-id="{ids[id(node)]}"><details{" open" if depth < 2 else ""}>'
                    f"<summary>{head}</summary><ul>{inner}</ul></details></li>")
        return f'<li class="k-{node.kind}" data-id="{ids[id(node)]}"><span class="leaf">{head}</span></li>'

    body = (f'<ul>{listed(tree.root, 0)}</ul>' if tree.root is not None
            else '<p class="muted">Nothing to lay out.</p>')
    said = "; ".join((*notes, tree.why))
    block = (json.dumps({"about": said, "projection": projection, "nodes": nodes}, ensure_ascii=True, sort_keys=True)
             .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))

    def pinned(text: str) -> str:
        return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode() + "'"

    policy = f"default-src 'none'; img-src data:; script-src {pinned(_PAGE_CODE)}; style-src {pinned(_PAGE_STYLE)}"
    title = f"Code tree {space}"
    return "\n".join([
        "<!DOCTYPE html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<meta http-equiv="Content-Security-Policy" content="{policy}">', '<link rel="icon" href="data:,">',
        f"<title>{markup.escape(title)}</title>", f"<style>{_PAGE_STYLE}</style>", "</head>", "<body>",
        f"<header><h1>{markup.escape(title)}</h1><p>{markup.escape(said)}</p>",
        '<div class="controls"><input id="filter" type="search" placeholder="Filter by path or name" '
        'aria-label="Filter by path or name">',
        '<button type="button" id="expand-all">Expand all</button>',
        '<button type="button" id="collapse-all">Collapse all</button>',
        '<span id="filter-count" aria-live="polite"></span></div></header>',
        f'<main><nav class="tree" aria-label="Code tree">{body}</nav>',
        '<aside id="inspect" aria-live="polite"><p class="muted">Choose a directory, file or declaration to see '
        "what it calls, imports and defines, and what does each of those to it.</p></aside></main>",
        f'<script id="tree-data" type="application/json">{block}</script>',
        f'<script id="tree-code">{_PAGE_CODE}</script>', "</body>", "</html>", ""])


def _walk(node: Optional[TreeNode]):
    """``node`` and every node beneath it, depth first, in listed order."""
    if node is None:
        return
    yield node
    for child in node.children:
        yield from _walk(child)
