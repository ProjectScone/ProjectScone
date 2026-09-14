"""The whole graph as one page a person can move through.

The `html` export draws the two hundred most connected entities as a
fixed picture with a panel beside it. A codebase's graph is thousands
of entities, and a fixed picture of the busiest two hundred is not the
graph: it is a poster of it. The leading code-graph tool's `graph.html`
lays the whole graph out in the browser, colours it by community, and
lets the reader search, click, hover and filter. This is that, written
here: every entity up to `MAX_NODES`, every relation up to `MAX_EDGES`,
laid out by a force simulation the page runs itself (no library, no
network), coloured by the communities `analysis.py` found, with a
legend that turns each community on and off, a search box, the
neighbourhood lit on hover, a panel that lists a chosen entity's
relations with the facts behind each, and a filter by predicate. What
the graph names but never reads (`typing`, a package, a cited record)
is drawn small and dim and can be hidden, so the picture is of the
codebase and not of the standard library.

Everything the page holds comes from the projection; nothing is
fetched. Names reach the page through one JSON block and are written
as text, never parsed as markup, and the page's content security
policy allows only its own style and code, pinned by hash. Bounded and
said: past `MAX_NODES` or `MAX_EDGES` the busiest are drawn and the
header says how many were left out.
"""

from __future__ import annotations

import base64
from collections import Counter
import hashlib
import html as markup
import json
from typing import Mapping

from .analysis import cached_analysis
from .export import Export, _about_lines
from .project import EntityProjection

#: Entities drawn before the page says it left some out. Past this a
#: browser's canvas is still fine; a person's eye is not.
MAX_NODES = 5_000
MAX_EDGES = 20_000
#: Relations whose object may be something the graph only names.
_NAMED_ONLY = frozenset(("imports", "depends_on", "develops_with", "cites", "uses_type", "references"))


def _external(projection: EntityProjection) -> frozenset[str]:
    """Named by the graph and never read by it: the object of imports,
    dependencies or citations that is the subject of nothing."""
    subjects = {relation.subject_id for relation in projection.relations}
    return frozenset(relation.object_id for relation in projection.relations
                     if relation.predicate in _NAMED_ONLY and relation.object_id not in subjects)


def explorer_page(projection: EntityProjection, about: Mapping[str, object]) -> Export:
    analysis = cached_analysis(projection)
    external = _external(projection)
    degree: Counter[str] = Counter()
    for relation in projection.relations:
        degree[relation.subject_id] += 1
        if relation.object_id != relation.subject_id:
            degree[relation.object_id] += 1
    # The busiest first, the graph's own before what it only names.
    ranked = sorted(projection.entities,
                    key=lambda e: (e.entity_id in external, -degree[e.entity_id], e.label.casefold(), e.entity_id))
    shown = ranked[:MAX_NODES]
    kept = {entity.entity_id for entity in shown}
    edges = sorted((r for r in projection.relations if r.subject_id in kept and r.object_id in kept),
                   key=lambda r: (-len(r.fact_ids), r.subject_id, r.predicate, r.object_id))[:MAX_EDGES]
    membership = {member: community.community_id for community in analysis.communities for member in community.members}
    communities = [{"id": community.community_id, "label": community.label, "size": len(community.members)}
                   for community in sorted(analysis.communities, key=lambda c: (-len(c.members), c.community_id))]
    left_out = (len(projection.entities) - len(shown), len(projection.relations) - len(edges))
    notes = [line for line in _about_lines(about) if line]
    if left_out[0] or left_out[1]:
        notes.append(f"{left_out[0]} entities and {left_out[1]} relations left out of the page: the busiest are drawn")
    named_shown, named_out = len(external & kept), len(external - kept)
    if named_shown:
        notes.append(f"{named_shown} entities named but never read are drawn small and can be hidden")
    if named_out:
        notes.append(f"{named_out} entities named but never read were left out first")
    said = "; ".join(notes)
    data = {
        "space": projection.space, "revision": projection.revision, "digest": projection.digest[:12],
        "about": said,
        "nodes": [{"id": e.entity_id, "label": e.label, "kind": e.kind, "community": membership.get(e.entity_id),
                   "degree": degree[e.entity_id], "external": e.entity_id in external} for e in shown],
        "edges": [{"id": r.relation_id, "source": r.subject_id, "target": r.object_id, "predicate": r.predicate,
                   "facts": list(r.fact_ids)} for r in edges],
        "communities": communities,
        "predicates": sorted({r.predicate for r in edges}),
        "left_out": {"entities": left_out[0], "relations": left_out[1]},
    }
    block = (json.dumps(data, ensure_ascii=True, sort_keys=True)
             .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))

    def pinned(text: str) -> str:
        return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode() + "'"

    policy = f"default-src 'none'; img-src data:; script-src {pinned(CODE)}; style-src {pinned(STYLE)}"
    title = f"Graph explorer {projection.space}"
    page = "\n".join([
        "<!DOCTYPE html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<meta http-equiv="Content-Security-Policy" content="{policy}">',
        '<link rel="icon" href="data:,">',
        f"<title>{markup.escape(title)}</title>", f"<style>{STYLE}</style>", "</head>", "<body>",
        f"<header><h1>{markup.escape(title)}</h1><p id=\"about\">{markup.escape(said)}</p>"
        '<input id="search" type="search" placeholder="Find an entity" aria-label="Find an entity">'
        '<button id="fit" type="button">Fit</button><button id="pause" type="button">Pause</button></header>',
        '<main><div id="stage"><canvas id="graph"></canvas><div id="hover" hidden></div></div>',
        '<aside><section id="legend"><h2>Communities</h2><label><input id="externals" type="checkbox"> '
        "Show what is named but never read</label><ul id=\"communities\"></ul>"
        '<h2>Relations</h2><ul id="predicates"></ul></section>'
        '<section id="details" aria-live="polite"><p class="muted">Choose an entity to see its relations and the '
        "facts behind them. Hover to light its neighbourhood; drag to move; wheel to zoom.</p></section></aside></main>",
        f'<script id="graph-data" type="application/json">{block}</script>',
        f'<script id="graph-code">{CODE}</script>', "</body>", "</html>", ""])
    return Export(page.encode("utf-8", "backslashreplace"), "text/html", "explorer.html")


STYLE = """
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  color: #1d1d1f; background: #f6f6f4; }
body { display: grid; grid-template-rows: auto 1fr; }
header { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; padding: 10px 16px;
  border-bottom: 1px solid #dcdcd8; background: #ffffff; }
header h1 { margin: 0; font-size: 16px; }
header p { margin: 0; color: #5b5b5b; font-size: 12px; flex: 1 1 320px; }
input, button { font: inherit; padding: 6px 10px; border: 1px solid #c6c6c2; border-radius: 6px; background: #fff; }
button { cursor: pointer; }
main { display: grid; grid-template-columns: 1fr minmax(280px, 360px); min-height: 0; }
#stage { position: relative; min-width: 0; min-height: 0; overflow: hidden; background: #ffffff; }
#graph { width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }
#hover { position: absolute; pointer-events: none; background: #1d1d1f; color: #fff; padding: 4px 8px;
  border-radius: 4px; font-size: 12px; max-width: 320px; overflow-wrap: anywhere; }
aside { border-left: 1px solid #dcdcd8; overflow: auto; background: #fbfbfa; display: grid;
  grid-template-rows: auto 1fr; min-height: 0; }
aside section { padding: 12px 16px; }
#legend { border-bottom: 1px solid #dcdcd8; max-height: 45vh; overflow: auto; }
aside h2 { margin: 8px 0 4px; font-size: 13px; color: #5b5b5b; text-transform: uppercase; letter-spacing: .04em; }
aside ul { padding-left: 0; list-style: none; margin: 0; }
aside li { margin: 3px 0; overflow-wrap: anywhere; display: flex; align-items: center; gap: 6px; font-size: 13px; }
.swatch { width: 12px; height: 12px; border-radius: 50%; flex: 0 0 12px; }
.count { color: #5b5b5b; font-size: 12px; margin-left: auto; }
#details h3 { margin: 0 0 4px; font-size: 16px; overflow-wrap: anywhere; }
#details ul { padding-left: 16px; list-style: disc; }
#details li { display: list-item; }
#details button.go { color: #0b57d0; background: none; border: 0; padding: 0; text-decoration: underline; }
.muted { color: #5b5b5b; }
.badge { font-size: 11px; padding: 1px 6px; border-radius: 10px; background: #e8e8e4; color: #5b5b5b; }
#results { margin: 0; padding: 0; }
#results li button { color: #0b57d0; background: none; border: 0; padding: 2px 0; text-decoration: underline; }
@media (max-width: 720px) { main { grid-template-columns: 1fr; grid-template-rows: 1fr auto; }
  aside { border-left: 0; border-top: 1px solid #dcdcd8; max-height: 45vh; } }
"""

CODE = r"""
(function () {
  "use strict";
  var data = JSON.parse(document.getElementById("graph-data").textContent);
  var canvas = document.getElementById("graph"), ctx = canvas.getContext("2d");
  var stage = document.getElementById("stage"), hoverBox = document.getElementById("hover");
  var details = document.getElementById("details"), search = document.getElementById("search");
  var PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                 "#9c755f", "#bab0ac", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"];
  var nodes = data.nodes, edges = data.edges, byId = {}, links = {}, colour = {}, index = 0;
  data.communities.forEach(function (c) { colour[c.id] = PALETTE[index++ % PALETTE.length]; });
  // Deterministic start: the same graph opens the same way every time.
  function seeded(text) { var h = 2166136261; for (var i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0) / 4294967296; }
  var maxDegree = 1;
  nodes.forEach(function (n) {
    byId[n.id] = n; links[n.id] = [];
    var a = seeded(n.id) * Math.PI * 2, r = 200 + seeded(n.id + "r") * 400;
    n.x = Math.cos(a) * r; n.y = Math.sin(a) * r; n.vx = 0; n.vy = 0;
    if (n.degree > maxDegree) { maxDegree = n.degree; }
  });
  edges.forEach(function (e) { links[e.source].push(e); if (e.target !== e.source) { links[e.target].push(e); } });
  nodes.forEach(function (n) { n.r = n.external ? 2.5 : 3 + 9 * Math.sqrt(n.degree / maxDegree); });
  var hidden = {}, hiddenPredicates = {}, showExternals = false, chosen = null, hovered = null;
  var view = { x: 0, y: 0, k: 1 }, running = true, iterations = 0, MAX_ITERATIONS = 400;

  function visible(n) { return !hidden[n.community] && (showExternals || !n.external); }
  function edgeVisible(e) { return !hiddenPredicates[e.predicate] && visible(byId[e.source]) && visible(byId[e.target]); }

  // The force simulation: repulsion through a grid so a large graph
  // stays cheap, springs along the edges, a pull toward the centre and
  // toward each community's own centre so communities settle apart.
  function step() {
    var alive = nodes.filter(visible), cell = 60, grid = {}, i, n;
    for (i = 0; i < alive.length; i++) {
      n = alive[i]; var key = Math.floor(n.x / cell) + ":" + Math.floor(n.y / cell);
      (grid[key] = grid[key] || []).push(n);
    }
    var centres = {}, counts = {};
    for (i = 0; i < alive.length; i++) {
      n = alive[i]; if (!n.community) { continue; }
      centres[n.community] = centres[n.community] || { x: 0, y: 0 };
      centres[n.community].x += n.x; centres[n.community].y += n.y; counts[n.community] = (counts[n.community] || 0) + 1;
    }
    for (i = 0; i < alive.length; i++) {
      n = alive[i]; var gx = Math.floor(n.x / cell), gy = Math.floor(n.y / cell), fx = 0, fy = 0;
      for (var dx = -1; dx <= 1; dx++) { for (var dy = -1; dy <= 1; dy++) {
        var bucket = grid[(gx + dx) + ":" + (gy + dy)]; if (!bucket) { continue; }
        for (var j = 0; j < bucket.length; j++) {
          var m = bucket[j]; if (m === n) { continue; }
          var ddx = n.x - m.x, ddy = n.y - m.y, d2 = ddx * ddx + ddy * ddy + 0.01;
          if (d2 > cell * cell * 2) { continue; }
          var f = 900 / d2; fx += ddx * f; fy += ddy * f;
        }
      } }
      fx -= n.x * 0.002; fy -= n.y * 0.002;
      if (n.community && counts[n.community] > 1) {
        var c = centres[n.community]; fx += (c.x / counts[n.community] - n.x) * 0.01; fy += (c.y / counts[n.community] - n.y) * 0.01;
      }
      n.vx = (n.vx + fx) * 0.6; n.vy = (n.vy + fy) * 0.6;
    }
    for (i = 0; i < edges.length; i++) {
      var e = edges[i]; if (!edgeVisible(e)) { continue; }
      var a = byId[e.source], b = byId[e.target], ex = b.x - a.x, ey = b.y - a.y;
      var dist = Math.sqrt(ex * ex + ey * ey) + 0.01, want = 40 + a.r + b.r, pull = (dist - want) * 0.02;
      a.vx += ex / dist * pull; a.vy += ey / dist * pull; b.vx -= ex / dist * pull; b.vy -= ey / dist * pull;
    }
    for (i = 0; i < alive.length; i++) { n = alive[i]; n.x += Math.max(-30, Math.min(30, n.vx)); n.y += Math.max(-30, Math.min(30, n.vy)); }
    iterations++;
    if (iterations >= MAX_ITERATIONS) { running = false; document.getElementById("pause").textContent = "Resume"; }
  }

  function resize() {
    var rect = stage.getBoundingClientRect(), scale = window.devicePixelRatio || 1;
    canvas.width = rect.width * scale; canvas.height = rect.height * scale;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
  }
  function toScreen(n) { return { x: (n.x + view.x) * view.k + canvas.clientWidth / 2, y: (n.y + view.y) * view.k + canvas.clientHeight / 2 }; }
  function neighbourhood(id) { var set = {}; set[id] = true; links[id].forEach(function (e) { set[e.source] = true; set[e.target] = true; }); return set; }

  function draw() {
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    var focus = hovered || chosen, lit = focus ? neighbourhood(focus) : null, i;
    ctx.lineWidth = 1;
    for (i = 0; i < edges.length; i++) {
      var e = edges[i]; if (!edgeVisible(e)) { continue; }
      var a = toScreen(byId[e.source]), b = toScreen(byId[e.target]);
      var on = !lit || (lit[e.source] && lit[e.target] && (e.source === focus || e.target === focus));
      ctx.strokeStyle = on ? "rgba(90,90,90,0.35)" : "rgba(90,90,90,0.06)";
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    for (i = 0; i < nodes.length; i++) {
      var n = nodes[i]; if (!visible(n)) { continue; }
      var p = toScreen(n), on = !lit || lit[n.id];
      ctx.globalAlpha = on ? (n.external ? 0.55 : 1) : 0.12;
      ctx.fillStyle = n.external ? "#9a9a96" : (colour[n.community] || "#777");
      ctx.beginPath(); ctx.arc(p.x, p.y, n.r * Math.max(0.6, Math.min(1.6, view.k)), 0, Math.PI * 2); ctx.fill();
      if (n.id === chosen) { ctx.strokeStyle = "#1d1d1f"; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1; }
      if (view.k > 0.9 && (n.r > 6 || on && lit) && !n.external) {
        ctx.fillStyle = "#1d1d1f"; ctx.font = "11px system-ui, sans-serif"; ctx.fillText(n.label.slice(0, 40), p.x + n.r + 3, p.y + 4);
      }
    }
    ctx.globalAlpha = 1;
  }
  function frame() { if (running) { step(); } draw(); window.requestAnimationFrame(frame); }

  function fit() {
    var alive = nodes.filter(visible); if (!alive.length) { return; }
    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    alive.forEach(function (n) { minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x); minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y); });
    var w = Math.max(1, maxX - minX + 80), h = Math.max(1, maxY - minY + 80);
    view.k = Math.min(canvas.clientWidth / w, canvas.clientHeight / h, 3);
    view.x = -(minX + maxX) / 2; view.y = -(minY + maxY) / 2;
  }
  function at(px, py) {
    var best = null, bestD = 1e9;
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i]; if (!visible(n)) { continue; }
      var p = toScreen(n), d = (p.x - px) * (p.x - px) + (p.y - py) * (p.y - py), reach = Math.max(6, n.r * view.k) + 3;
      if (d < reach * reach && d < bestD) { best = n; bestD = d; }
    }
    return best;
  }
  function element(tag, text, cls) { var made = document.createElement(tag); made.textContent = text; if (cls) { made.className = cls; } return made; }
  function show(id) {
    chosen = id; var n = byId[id];
    details.replaceChildren();
    details.appendChild(element("h3", n.label));
    var meta = element("p", (n.kind || "kind unknown") + " · " + n.degree + " relation(s)" + (n.community ? " · " + communityLabel(n.community) : ""), "muted");
    if (n.external) { meta.appendChild(document.createTextNode(" ")); meta.appendChild(element("span", "named, never read", "badge")); }
    details.appendChild(meta);
    var groups = {};
    links[id].forEach(function (e) {
      var out = e.source === id, key = (out ? "→ " : "← ") + e.predicate;
      (groups[key] = groups[key] || []).push(e);
    });
    Object.keys(groups).sort().forEach(function (key) {
      details.appendChild(element("h2", key + " (" + groups[key].length + ")"));
      var list = document.createElement("ul");
      groups[key].sort(function (a, b) { return byId[a.source === id ? a.target : a.source].label.localeCompare(byId[b.source === id ? b.target : b.source].label); })
        .slice(0, 200).forEach(function (e) {
          var other = byId[e.source === id ? e.target : e.source], item = document.createElement("li");
          var go = element("button", other.label, "go"); go.type = "button";
          go.addEventListener("click", function () { show(other.id); centre(other); });
          item.appendChild(go);
          item.appendChild(element("span", " " + (e.facts.length === 1 ? "fact " : "facts ") + e.facts.join(", "), "muted"));
          list.appendChild(item);
        });
      details.appendChild(list);
    });
  }
  function communityLabel(id) { for (var i = 0; i < data.communities.length; i++) { if (data.communities[i].id === id) { return data.communities[i].label; } } return id; }
  function centre(n) { view.x = -n.x; view.y = -n.y; }

  // Legend: one row per community with a switch, and one per predicate.
  var communityList = document.getElementById("communities");
  data.communities.forEach(function (c) {
    var item = document.createElement("li"), box = document.createElement("input"), swatch = element("span", "", "swatch");
    box.type = "checkbox"; box.checked = true; swatch.style.background = colour[c.id];
    box.addEventListener("change", function () { hidden[c.id] = !box.checked; });
    item.appendChild(box); item.appendChild(swatch); item.appendChild(element("span", c.label));
    item.appendChild(element("span", String(c.size), "count")); communityList.appendChild(item);
  });
  var predicateList = document.getElementById("predicates");
  data.predicates.forEach(function (p) {
    var item = document.createElement("li"), box = document.createElement("input");
    box.type = "checkbox"; box.checked = true;
    box.addEventListener("change", function () { hiddenPredicates[p] = !box.checked; });
    item.appendChild(box); item.appendChild(element("span", p)); predicateList.appendChild(item);
  });
  document.getElementById("externals").addEventListener("change", function (event) { showExternals = event.target.checked; });
  document.getElementById("fit").addEventListener("click", fit);
  document.getElementById("pause").addEventListener("click", function () {
    running = !running; if (running) { iterations = 0; } this.textContent = running ? "Pause" : "Resume";
  });

  // Search: up to twenty matches by name, each a button that chooses.
  var results = document.createElement("ul"); results.id = "results"; search.insertAdjacentElement("afterend", results);
  search.addEventListener("input", function () {
    var q = search.value.trim().toLowerCase(); results.replaceChildren(); if (!q) { return; }
    nodes.filter(function (n) { return n.label.toLowerCase().indexOf(q) !== -1; }).slice(0, 20).forEach(function (n) {
      var item = document.createElement("li"), go = element("button", n.label); go.type = "button";
      go.addEventListener("click", function () { if (n.external) { showExternals = true; document.getElementById("externals").checked = true; } show(n.id); centre(n); results.replaceChildren(); });
      item.appendChild(go); results.appendChild(item);
    });
  });

  // Mouse: hover lights a neighbourhood, click chooses, drag pans, wheel zooms.
  var dragging = null, moved = false;
  canvas.addEventListener("pointerdown", function (event) { dragging = { x: event.clientX, y: event.clientY, vx: view.x, vy: view.y }; moved = false; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", function (event) {
    var rect = canvas.getBoundingClientRect(), px = event.clientX - rect.left, py = event.clientY - rect.top;
    if (dragging) {
      if (Math.abs(event.clientX - dragging.x) + Math.abs(event.clientY - dragging.y) > 3) { moved = true; }
      view.x = dragging.vx + (event.clientX - dragging.x) / view.k; view.y = dragging.vy + (event.clientY - dragging.y) / view.k; return;
    }
    var n = at(px, py); hovered = n ? n.id : null;
    if (n) { hoverBox.hidden = false; hoverBox.textContent = n.label + (n.external ? " (named, never read)" : ""); hoverBox.style.left = (px + 12) + "px"; hoverBox.style.top = (py + 12) + "px"; }
    else { hoverBox.hidden = true; }
  });
  canvas.addEventListener("pointerup", function (event) {
    if (dragging && !moved) { var rect = canvas.getBoundingClientRect(), n = at(event.clientX - rect.left, event.clientY - rect.top); if (n) { show(n.id); } }
    dragging = null;
  });
  canvas.addEventListener("wheel", function (event) {
    event.preventDefault();
    var factor = event.deltaY < 0 ? 1.1 : 1 / 1.1, rect = canvas.getBoundingClientRect();
    var px = event.clientX - rect.left - canvas.clientWidth / 2, py = event.clientY - rect.top - canvas.clientHeight / 2;
    view.x -= px / view.k * (1 - 1 / factor); view.y -= py / view.k * (1 - 1 / factor); view.k *= factor;
  }, { passive: false });
  window.addEventListener("resize", resize);
  resize();
  for (var warm = 0; warm < 120; warm++) { step(); }
  fit();
  frame();
})();
"""
