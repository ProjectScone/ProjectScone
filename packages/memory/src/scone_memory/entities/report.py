"""A whole-space knowledge report: what the analysis found, readable and cited.

The report states which projection and analysis it came from, keeps every
computed score apart from the recorded facts it rests on, and cites fact
ids for every connection and question. ``render_markdown`` turns it into a
document a person can read or keep.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .analysis import GraphAnalysis
from .markdown import literal
from .project import EntityProjection

REPORT_SCHEMA_VERSION = 1


def _name(entities: Mapping[str, object], entity_id: str) -> dict[str, object]:
    entity = entities[entity_id]
    return {"id": entity_id, "label": getattr(entity, "label"), "key": getattr(entity, "key")}


def _hubs(analysis: GraphAnalysis, percentile: float | None) -> set[str]:
    """Entities whose number of neighbours is above the given percentile:
    utility hubs a ranking by centrality would otherwise always lead with."""
    if percentile is None or not analysis.importance:
        return set()
    degrees = sorted(item.degree for item in analysis.importance)
    cut = degrees[max(0, math.ceil(len(degrees) * percentile / 100) - 1)]  # nearest-rank percentile
    return {item.entity_id for item in analysis.importance if item.degree > cut}


def build_report(projection: EntityProjection, analysis: GraphAnalysis, *, meta: Mapping[str, object],
                 filters: Mapping[str, object], coverage: Mapping[str, object], central: int = 15,
                 exclude_hubs: float | None = None) -> dict[str, object]:
    """``exclude_hubs`` leaves entities above that degree percentile out of
    the central ranking and lists them apart; they stay in their communities."""
    entities = {entity.entity_id: entity for entity in projection.entities}
    communities = {community.community_id: community for community in analysis.communities}
    hubs = _hubs(analysis, exclude_hubs)
    ranked = [item for item in analysis.importance if item.entity_id not in hubs][:central]
    bridging = sorted((item for item in analysis.importance if item.degree >= 2 and item.participation > 0.3),
                      key=lambda item: (-item.participation, -item.betweenness, item.entity_id))[:10]
    return {
        "schema_version": REPORT_SCHEMA_VERSION, "space": projection.space, "projection": dict(meta),
        "filters": dict(filters),
        "analysis": {"version": analysis.version, "modularity": analysis.modularity,
                     "levels": analysis.coverage.levels, "betweenness": analysis.coverage.betweenness,
                     "resolution": analysis.coverage.resolution, "exclude_hubs": exclude_hubs,
                     "basis": "computed", "coverage": analysis.coverage.record()},
        "summary": {"entities": len(projection.entities), "relations": len(projection.relations),
                    "attributes": len(projection.attributes), "communities": len(analysis.communities),
                    "isolated_entities": analysis.coverage.isolated_entities},
        "communities": [{
            "id": community.community_id, "label": community.label, "size": len(community.members),
            "central": [_name(entities, member) for member in community.top_entities],
            "internal_links": community.internal_links, "boundary_links": community.boundary_links,
            "cohesion": community.cohesion, "kinds": [list(item) for item in community.kinds],
            "predicates": [list(item) for item in community.predicates]}
            for community in analysis.communities],
        "central_entities": [{
            **_name(entities, item.entity_id), "kind": entities[item.entity_id].kind,
            "community_id": item.community_id, "community": communities[item.community_id].label,
            "degree": item.degree, "weight": item.weight, "pagerank": item.pagerank,
            "betweenness": item.betweenness, "participation": item.participation} for item in ranked],
        "hubs_excluded": [{**_name(entities, item.entity_id), "degree": item.degree, "pagerank": item.pagerank}
                          for item in analysis.importance if item.entity_id in hubs],
        "bridging_entities": [{
            **_name(entities, item.entity_id), "community_id": item.community_id,
            "participation": item.participation, "betweenness": item.betweenness} for item in bridging],
        "surprising_connections": [{
            "relation_id": surprise.relation_id, "subject": _name(entities, surprise.subject_id),
            "predicate": surprise.predicate, "object": _name(entities, surprise.object_id),
            "fact_ids": list(surprise.fact_ids), "communities": list(surprise.communities),
            "links_between_communities": surprise.links_between_communities, "reason": surprise.reason}
            for surprise in analysis.surprising_connections],
        "suggestions": [{"kind": suggestion.kind, "text": suggestion.text, "entity_ids": list(suggestion.entity_ids),
                         "relation_ids": list(suggestion.relation_ids), "fact_ids": list(suggestion.fact_ids)}
                        for suggestion in analysis.suggestions],
        "coverage": {**dict(coverage), "entities_analysed": analysis.coverage.entities_analysed,
                     "truncated": bool(_reasons(coverage)) or analysis.coverage.truncated,
                     "reasons": [*_reasons(coverage), *analysis.coverage.reasons]},
    }


def _reasons(coverage: Mapping[str, object]) -> list[str]:
    found = coverage.get("reasons", [])
    return [str(reason) for reason in found] if isinstance(found, list) else []


def _facts(fact_ids: list[int]) -> str:
    return "fact " + str(fact_ids[0]) if len(fact_ids) == 1 else "facts " + ", ".join(str(i) for i in fact_ids)


def _sampled_sources(analysis: Mapping[str, Any]) -> str | None:
    method = str(analysis.get("betweenness", "exact"))
    return method.split(":", 1)[1] if method.startswith("sampled:") else None


def render_markdown(report: Mapping[str, Any]) -> str:
    """The report as Markdown. Every name, predicate and question comes from
    the ledger, so each is escaped to read as the text it is."""
    projection, filters = report["projection"], report["filters"]
    summary, analysis = report["summary"], report["analysis"]
    sampled = _sampled_sources(analysis)
    lines = [f"# Knowledge report: {literal(report['space'])}", "",
             f"_Projection `{str(projection['digest'])[:12]}` at revision {projection['revision']}, "
             f"{literal(filters['status'])} facts as of {literal(filters['as_of'])}. Communities, scores and "
             f"questions are computed from recorded facts; they are analysis, not facts._", "", "## Summary", "",
             f"- {summary['entities']} entities, {summary['relations']} relations, {summary['attributes']} attributes",
             f"- {summary['communities']} communities (modularity {analysis['modularity']}, "
             f"resolution {analysis.get('resolution', 1.0):g}), "
             f"{summary['isolated_entities']} entities known only by their values"]
    if analysis.get("exclude_hubs") is not None:
        lines.append(f"- Central entities leave out entities whose links are above the "
                     f"{analysis['exclude_hubs']:g}th percentile; they are listed under Hubs left out of the ranking")
    lines += ["", "## Communities", ""]
    for number, community in enumerate(report["communities"] or [], 1):
        cohesion = "n/a" if community["cohesion"] is None else community["cohesion"]
        lines.append(f"### {number}. {literal(community['label'])}")
        lines.append(f"{community['size']} entities, cohesion {cohesion}, {community['internal_links']} links inside, "
                     f"{community['boundary_links']} across.")
        extras = [", ".join(f"{literal(name)} {count}" for name, count in community[key]) for key in ("kinds", "predicates")]
        if any(extras):
            lines.append(f"Kinds: {extras[0] or 'none'}. Predicates: {extras[1] or 'none'}.")
        lines.append("")
    if not report["communities"]:
        lines += ["No entity is linked to another yet.", ""]
    betweenness = "Betweenness (estimated)" if sampled else "Betweenness"
    lines += ["## Central entities", "", f"| Entity | Kind | Links | PageRank | {betweenness} | Community |",
              "| --- | --- | --- | --- | --- | --- |"]
    for item in report["central_entities"] or []:
        lines.append(f"| {literal(item['label'])} | {literal(item['kind'] or '')} | {item['degree']} | "
                     f"{item['pagerank']} | {item['betweenness']} | {literal(item['community'])} |")
    if analysis.get("exclude_hubs") is not None:
        lines += ["", "## Hubs left out of the ranking", ""]
        lines += [f"- {literal(hub['label'])}: {hub['degree']} links, PageRank {hub['pagerank']}"
                  for hub in report.get("hubs_excluded") or []] or ["None: no entity is above that percentile."]
    lines += ["", "## Surprising connections", ""]
    for surprise in report["surprising_connections"] or []:
        lines.append(f"- **{literal(surprise['subject']['label'])}** {literal(surprise['predicate'])} "
                     f"**{literal(surprise['object']['label'])}**: {literal(surprise['reason'])} "
                     f"({_facts(surprise['fact_ids'])})")
    if not report["surprising_connections"]:
        lines.append("None: no relation links two communities.")
    lines += ["", "## Questions worth asking", ""]
    for suggestion in report["suggestions"] or []:
        cited = f" ({_facts(suggestion['fact_ids'])})" if suggestion["fact_ids"] else ""
        lines.append(f"- {literal(suggestion['text'])}{cited}")
    if not report["suggestions"]:
        lines.append("None yet.")
    coverage = report["coverage"]
    lines += ["", "## Coverage", "", f"- {coverage.get('facts_counted', 0)} facts counted of "
              f"{coverage.get('facts_read', 0)} read; {coverage['entities_analysed']} entities analysed."]
    if sampled:
        lines.append(f"- Betweenness is estimated from {sampled} sampled sources, not computed over every entity.")
    if coverage["reasons"]:
        lines.append(f"- Limited by: {literal(', '.join(coverage['reasons']))}. This is a sample, not the whole space.")
    return "\n".join(lines) + "\n"
