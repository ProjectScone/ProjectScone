"""Construct a cited text-association network without reading or writing stores."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from difflib import SequenceMatcher
import hashlib
import json

from ..entities.analysis import partition_associations
from .matching import candidate_names, first_match, prose_spans
from .models import (NetworkAnalysis, NetworkBridge, NetworkCandidate, NetworkCommunity,
                     NetworkCoverage, NetworkDocument, NetworkEdge, NetworkLimits,
                     NetworkMention, NetworkNode, NetworkPassage, NetworkVariant,
                     TextDocument, TextNetwork, TextTerm)


class TextNetworkCancelled(InterruptedError):
    """The caller canceled an in-memory analysis; no partial result is returned."""


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def _cancel(check: Callable[[], bool] | None) -> None:
    if check is not None and check():
        raise TextNetworkCancelled('text_network_cancelled')


def _terms(values: Sequence[TextTerm], maximum: int) -> list[TextTerm]:
    if len(values) > maximum:
        raise ValueError('max_terms')
    terms = sorted((TextTerm.model_validate(value.model_dump()) for value in values), key=lambda term: (term.label.casefold(), term.label))
    owners: dict[str, int] = {}
    for index, term in enumerate(terms):
        for name in (term.label, *term.aliases):
            folded = name.casefold()
            if folded in owners and owners[folded] != index:
                raise ValueError('ambiguous_terms')
            owners[folded] = index
    return [term.model_copy(update={'aliases': sorted(set(term.aliases), key=lambda value: (value.casefold(), value))}) for term in terms]


def _topology(nodes: list[NetworkNode], edges: list[NetworkEdge]) -> tuple[list[NetworkNode], NetworkAnalysis]:
    labels = {node.id: node.label for node in nodes}
    groups = partition_associations(list(labels), [(edge.source, edge.target, edge.weight) for edge in edges])
    communities: list[NetworkCommunity] = []
    membership: dict[str, str] = {}
    for group in groups:
        identity = 'group:' + _hash(group)[:32]
        membership.update((node, identity) for node in group)
        names = sorted((labels[node] for node in group), key=lambda name: (name.casefold(), name))
        communities.append(NetworkCommunity(id=identity, node_ids=list(group), label=' · '.join(names[:3])))
    adjacent: dict[str, set[str]] = {node.id: set() for node in nodes}
    for edge in edges:
        adjacent[edge.source].add(edge.target)
        adjacent[edge.target].add(edge.source)
    bridges: list[NetworkBridge] = []
    for edge in edges:
        # Remove only this edge in a bounded reachability walk. There are at
        # most 60 nodes/600 edges, so this cannot scale with an external graph.
        visited = {edge.source}
        queue = [edge.source]
        while queue:
            node = queue.pop()
            for other in adjacent[node]:
                if {node, other} == {edge.source, edge.target} or other in visited:
                    continue
                visited.add(other)
                queue.append(other)
        cut = edge.target not in visited
        cross = membership[edge.source] != membership[edge.target]
        if cut or cross:
            bridges.append(NetworkBridge(source=edge.source, target=edge.target, cut_edge=cut, cross_community=cross))
    return ([node.model_copy(update={'community_id': membership[node.id]}) for node in nodes],
            NetworkAnalysis(communities=communities, bridges=bridges,
                            isolated_node_ids=sorted(node for node, neighbors in adjacent.items() if not neighbors)))


def _variants(nodes: list[NetworkNode], maximum: int) -> list[NetworkVariant]:
    found: list[NetworkVariant] = []
    for index, left in enumerate(nodes):
        for right in nodes[index+1:]:
            a, b = left.label.casefold(), right.label.casefold()
            if min(len(a), len(b)) < 4 or a == b or abs(len(a)-len(b)) > 3:
                continue
            if SequenceMatcher(None, a, b, autojunk=False).ratio() >= .82:
                found.append(NetworkVariant(left=left.id, right=right.id))
                if len(found) >= maximum:
                    return found
    return found


def build_text_network(documents: Sequence[TextDocument], terms: Sequence[TextTerm], *,
                       limits: NetworkLimits | None = None,
                       should_cancel: Callable[[], bool] | None = None) -> TextNetwork:
    """Analyze explicit documents, preserving original Unicode offsets.

    No Fact objects, inference, model calls, I/O, logs, caches or retained state.
    Input is rejected before text analysis when its UTF-8 budget is exceeded.
    Cancellation is checked between bounded passages and before graph work.
    """
    bound = NetworkLimits.model_validate(limits.model_dump()) if limits else NetworkLimits()
    _cancel(should_cancel)
    if len(documents) > bound.max_documents:
        raise ValueError('max_documents')
    if len({document.key for document in documents}) != len(documents):
        raise ValueError('duplicate_documents')
    selected = sorted(documents, key=lambda document: document.key)
    summaries: list[NetworkDocument] = []
    total_bytes = 0
    for document in selected:
        _cancel(should_cancel)
        document = TextDocument.model_validate(document.model_dump())
        # The character count rejects obviously oversized strings before UTF-8
        # allocation; each admitted encode remains within four times this cap.
        if len(document.content) > bound.max_input_bytes:
            raise ValueError('input_bytes')
        raw = document.content.encode('utf-8')
        total_bytes += len(raw)
        if total_bytes > bound.max_input_bytes:
            raise ValueError('input_bytes')
        summaries.append(NetworkDocument(**document.model_dump(exclude={'content'}), content_sha256=hashlib.sha256(raw).hexdigest()))
    tracked = _terms(terms, bound.max_terms)
    digest = _hash({'version': 1, 'documents': [row.model_dump() for row in summaries], 'terms': [row.model_dump() for row in tracked]})
    identities = ['term:' + _hash([digest, term.label.casefold()])[:32] for term in tracked]
    counts = dict.fromkeys(identities, 0)
    document_keys: dict[str, set[str]] = {identity: set() for identity in identities}
    pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    passages: dict[str, NetworkPassage] = {}
    suggestions: dict[str, tuple[str, set[str], int]] = {}
    known = {name.casefold() for term in tracked for name in (term.label, *term.aliases)}
    reasons: set[str] = set()
    passage_count = mention_count = 0
    for document in selected:
        for start, end in prose_spans(document.content):
            _cancel(should_cancel)
            if passage_count >= bound.max_passages:
                reasons.add('max_passages')
                break
            passage_count += 1
            text = document.content[start:end]
            folded = text.casefold()
            seen_candidates: set[str] = set()
            for label in sorted(candidate_names(text)):
                key = label.casefold()
                if key in known or key in seen_candidates:
                    continue
                seen_candidates.add(key)
                if key in suggestions:
                    kept, keys, count = suggestions[key]
                    keys.add(document.key)
                    suggestions[key] = (min(kept, label), keys, count+1)
                elif len(suggestions) < 4000:
                    suggestions[key] = (label, {document.key}, 1)
                else:
                    reasons.add('candidate_scan_limit')
            mentions: list[NetworkMention] = []
            for identity, term in zip(identities, tracked):
                _cancel(should_cancel)
                ranges = [match for name in (term.label, *term.aliases) if (match := first_match(text, folded, name)) is not None]
                if not ranges:
                    continue
                first, last = min(ranges, key=lambda span: (span[0], -span[1]))
                mentions.append(NetworkMention(node_id=identity, start=start+first, end=start+last))
                counts[identity] += 1
                document_keys[identity].add(document.key)
            mention_count += len(mentions)
            if not mentions:
                continue
            passage_id = 'passage:' + _hash([digest, document.key, start, end])[:32]
            needed = any(counts[mention.node_id] <= 3 for mention in mentions)
            found = sorted(mention.node_id for mention in mentions)
            for index, left in enumerate(found):
                for right in found[index+1:]:
                    pair = (left, right)
                    pair_counts[pair] += 1
                    if len(pairs[pair]) < 3:
                        pairs[pair].append(passage_id)
                        needed = True
                    else:
                        reasons.add('max_edge_citations')
            if needed:
                excerpt_start = max(start, min(mention.start for mention in mentions)-100)
                passages[passage_id] = NetworkPassage(id=passage_id, document_key=document.key, start=start, end=end,
                    excerpt_start=excerpt_start, excerpt=document.content[excerpt_start:min(end, excerpt_start+600)], mentions=mentions)
            else:
                reasons.add('max_node_citations')
        if 'max_passages' in reasons:
            break
    _cancel(should_cancel)
    ranked = sorted(pairs, key=lambda pair: (-pair_counts[pair], pair))
    if len(ranked) > bound.max_edges:
        reasons.add('max_edges')
    kept_passages: dict[str, NetworkPassage] = {}
    edges: list[NetworkEdge] = []
    for left, right in ranked[:bound.max_edges]:
        evidence: list[str] = []
        for identity in pairs[(left, right)]:
            if identity not in kept_passages and len(kept_passages) >= bound.max_cited_passages:
                reasons.add('max_cited_passages')
                continue
            kept_passages[identity] = passages[identity]
            evidence.append(identity)
        if evidence:
            edges.append(NetworkEdge(id='edge:' + _hash([digest, left, right])[:32], source=left, target=right,
                weight=pair_counts[(left, right)], passage_ids=evidence, passages_truncated=len(evidence) != pair_counts[(left, right)]))
    # Lone terms also retain a passage when citation capacity permits.
    for identity, passage in passages.items():
        if identity in kept_passages:
            continue
        if len(kept_passages) >= bound.max_cited_passages:
            reasons.add('max_cited_passages')
            break
        kept_passages[identity] = passage
    nodes = [NetworkNode(id=identity, label=term.label, kind=term.kind, aliases=term.aliases,
        passage_count=counts[identity], document_keys=sorted(document_keys[identity]), community_id='')
        for identity, term in zip(identities, tracked)]
    nodes, analysis = _topology(nodes, edges)
    candidates = [NetworkCandidate(label=label, document_keys=sorted(keys), passage_count=count)
        for label, keys, count in sorted(suggestions.values(), key=lambda row: (-row[2], row[0].casefold(), row[0]))[:bound.max_candidates]]
    variants = _variants(nodes, bound.max_variants+1)
    if len(variants) > bound.max_variants:
        reasons.add('max_variants')
        variants = variants[:bound.max_variants]
    if len(suggestions) > bound.max_candidates:
        reasons.add('max_candidates')
    result = TextNetwork(digest=digest, documents=summaries, nodes=nodes, edges=edges,
        passages=sorted(kept_passages.values(), key=lambda passage: (passage.document_key, passage.start)),
        candidates=candidates, variants=variants, analysis=analysis,
        coverage=NetworkCoverage(partial=bool(reasons), reasons=sorted(reasons), documents=len(selected),
            passages=passage_count, mentions=mention_count, edges=len(pairs)))
    _cancel(should_cancel)
    if len(result.model_dump_json().encode()) > bound.max_response_bytes:
        raise ValueError('response_bytes')
    return result
