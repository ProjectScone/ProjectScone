"""Bounded raw-text inspection; no stores, model calls or retained state."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib

from .build import _cancel, _hash, _terms
from .matching import first_match, prose_spans
from .models import NetworkDocument, StrictValue, TextDocument, TextTerm
from .research_models import (ContextBatch, ContextRequest, ContextWindow, CoverageTerm,
                              ResearchCoverage, ResearchLimits, SectionCoverage,
                              SectionStatistics, SectionTermCount, TextSection)

Cancel = Callable[[], bool] | None


@dataclass(frozen=True)
class _Scan:
    spans: dict[str, list[tuple[int, int]]]
    cutoffs: dict[str, int | None]
    coverage: ResearchCoverage


def _documents(values: Sequence[TextDocument], limits: ResearchLimits,
               should_cancel: Cancel) -> tuple[list[TextDocument], list[NetworkDocument]]:
    _cancel(should_cancel)
    if len(values) > limits.max_documents:
        raise ValueError('max_documents')
    documents: list[TextDocument] = []
    summaries: list[NetworkDocument] = []
    total = 0
    for value in values:
        _cancel(should_cancel)
        document = TextDocument.model_validate(value.model_dump())
        if len(document.content) > limits.max_input_bytes:
            raise ValueError('input_bytes')
        raw = document.content.encode('utf-8')
        total += len(raw)
        if total > limits.max_input_bytes:
            raise ValueError('input_bytes')
        documents.append(document)
        summaries.append(NetworkDocument(**document.model_dump(exclude={'content'}),
                                         content_sha256=hashlib.sha256(raw).hexdigest()))
    if len({value.key for value in documents}) != len(documents):
        raise ValueError('duplicate_documents')
    return sorted(documents, key=lambda value: value.key), sorted(summaries, key=lambda value: value.key)


def _scan(documents: list[TextDocument], limits: ResearchLimits, should_cancel: Cancel) -> _Scan:
    spans: dict[str, list[tuple[int, int]]] = {document.key: [] for document in documents}
    cutoffs: dict[str, int | None] = {document.key: None for document in documents}
    count = 0
    partial = False
    for document in documents:
        _cancel(should_cancel)
        if partial:
            cutoffs[document.key] = 0
            continue
        for start, end in prose_spans(document.content):
            _cancel(should_cancel)
            if count >= limits.max_passages:
                cutoffs[document.key] = start
                partial = True
                break
            spans[document.key].append((start, end))
            count += 1
    return _Scan(spans, cutoffs, ResearchCoverage(partial=partial,
        reasons=['max_passages'] if partial else [], scanned_passages=count))


def _response(value: StrictValue, limits: ResearchLimits, should_cancel: Cancel) -> None:
    _cancel(should_cancel)
    if len(value.model_dump_json().encode()) > limits.max_response_bytes:
        raise ValueError('response_bytes')


def _sections(values: Sequence[TextSection], documents: list[TextDocument],
              limits: ResearchLimits) -> list[TextSection]:
    if len(values) > limits.max_sections:
        raise ValueError('max_sections')
    sections = sorted((TextSection.model_validate(value.model_dump()) for value in values),
                      key=lambda value: (value.document_key, value.start, value.key))
    if len({value.key for value in sections}) != len(sections):
        raise ValueError('duplicate_sections')
    sizes = {document.key: len(document.content) for document in documents}
    ends: dict[str, int] = {}
    for section in sections:
        if section.document_key not in sizes or section.end > sizes[section.document_key]:
            raise ValueError('section_bounds')
        if section.start < ends.get(section.document_key, 0):
            raise ValueError('overlapping_sections')
        ends[section.document_key] = section.end
    return sections


def _count_section(section: TextSection, document: TextDocument, scan: _Scan,
                   terms: list[CoverageTerm], should_cancel: Cancel) -> SectionStatistics:
    counts: dict[str, SectionTermCount] = {}
    passages = 0
    for start, end in scan.spans[document.key]:
        _cancel(should_cancel)
        if end <= section.start or start >= section.end:
            continue
        if start < section.start or end > section.end:
            raise ValueError('section_splits_passage')
        passages += 1
        text = document.content[start:end]
        folded = text.casefold()
        for term in terms:
            _cancel(should_cancel)
            matches = [match for name in (term.label, *term.aliases)
                       if (match := first_match(text, folded, name)) is not None]
            if not matches:
                continue
            first, last = min(matches, key=lambda value: (value[0], -value[1]))
            prior = counts.get(term.id)
            counts[term.id] = SectionTermCount(node_id=term.id,
                passage_count=(prior.passage_count if prior else 0)+1,
                first_passage_start=prior.first_passage_start if prior else start,
                first_passage_end=prior.first_passage_end if prior else end,
                first_start=prior.first_start if prior else start+first,
                first_end=prior.first_end if prior else start+last,
                last_start=start+first, last_end=start+last)
    cutoff = scan.cutoffs[document.key]
    return SectionStatistics(**section.model_dump(), passage_count=passages,
        complete=cutoff is None or section.end <= cutoff, counts=list(counts.values()))


def build_section_coverage(documents: Sequence[TextDocument], terms: Sequence[TextTerm],
                           sections: Sequence[TextSection], *, limits: ResearchLimits | None = None,
                           should_cancel: Cancel = None) -> SectionCoverage:
    """Count matched prose passages in supplied nonoverlapping source sections.

    Sections may cover a subset of a document, but may not split a scanned
    prose passage. Counts are computed before any network citation sampling.
    A zero is evidence of absence only within a section marked complete.
    """
    bound = ResearchLimits.model_validate(limits.model_dump()) if limits else ResearchLimits()
    selected, summaries = _documents(documents, bound, should_cancel)
    tracked = _terms(terms, bound.max_terms)
    chosen = _sections(sections, selected, bound)
    network_digest = _hash({'version': 1, 'documents': [row.model_dump() for row in summaries],
                            'terms': [row.model_dump() for row in tracked]})
    identified = [CoverageTerm(**term.model_dump(),
        id='term:' + _hash([network_digest, term.label.casefold()])[:32]) for term in tracked]
    scan = _scan(selected, bound, should_cancel)
    by_key = {document.key: document for document in selected}
    statistics = [_count_section(section, by_key[section.document_key], scan, identified, should_cancel)
                  for section in chosen]
    digest = _hash({'version': 1, 'network_digest': network_digest,
                   'sections': [row.model_dump() for row in statistics], 'coverage': scan.coverage.model_dump()})
    result = SectionCoverage(digest=digest, network_digest=network_digest, documents=summaries,
                             terms=identified, sections=statistics, coverage=scan.coverage)
    _response(result, bound, should_cancel)
    return result


def _window(request: ContextRequest, document: TextDocument, spans: list[tuple[int, int]],
            cutoff: int | None, limit: int) -> ContextWindow:
    try:
        index = spans.index((request.start, request.end))
    except ValueError:
        if cutoff is not None and request.start >= cutoff:
            raise ValueError('context_scan_limit') from None
        raise ValueError('context_passage') from None
    left = spans[max(0, index-request.before)][0]
    right = spans[min(len(spans)-1, index+request.after)][1]
    unknown_after = cutoff is not None and index+request.after >= len(spans)
    if request.end-request.start >= limit:
        start, end = request.start, request.start+limit
    else:
        spare = limit-(request.end-request.start)
        before = min(request.start-left, spare//2)
        after = min(right-request.end, spare-before)
        before = min(request.start-left, spare-after)
        start, end = request.start-before, request.end+after
    return ContextWindow(key=request.key, document_key=request.document_key,
        content_sha256=request.content_sha256, target_start=request.start, target_end=request.end,
        start=start, end=end, text=document.content[start:end], truncated_before=start > left,
        truncated_after=end < right or unknown_after, target_truncated=end < request.end)


def surrounding_context(documents: Sequence[TextDocument], requests: Sequence[ContextRequest], *,
                        limits: ResearchLimits | None = None, should_cancel: Cancel = None) -> ContextBatch:
    """Return bounded source windows around fingerprinted whole prose passages.

    The caller authorizes and supplies documents; this function performs no I/O.
    A request must match a complete original prose span, never a code block or
    an arbitrary substring. Source gaps between neighbors remain unmodified.
    """
    bound = ResearchLimits.model_validate(limits.model_dump()) if limits else ResearchLimits()
    selected, summaries = _documents(documents, bound, should_cancel)
    if len(requests) > bound.max_context_requests:
        raise ValueError('max_context_requests')
    chosen = sorted((ContextRequest.model_validate(value.model_dump()) for value in requests), key=lambda value: value.key)
    if len({value.key for value in chosen}) != len(chosen):
        raise ValueError('duplicate_context_requests')
    by_key = {document.key: document for document in selected}
    fingerprints = {summary.key: summary.content_sha256 for summary in summaries}
    for request in chosen:
        _cancel(should_cancel)
        if request.document_key not in by_key or request.end > len(by_key[request.document_key].content):
            raise ValueError('context_bounds')
        if fingerprints[request.document_key] != request.content_sha256:
            raise ValueError('context_source_changed')
    scan = _scan(selected, bound, should_cancel)
    windows: list[ContextWindow] = []
    for request in chosen:
        _cancel(should_cancel)
        windows.append(_window(request, by_key[request.document_key], scan.spans[request.document_key],
                               scan.cutoffs[request.document_key], bound.max_context_characters))
    digest = _hash({'version': 1, 'windows': [window.model_dump() for window in windows],
                   'coverage': scan.coverage.model_dump()})
    result = ContextBatch(digest=digest, windows=windows, coverage=scan.coverage)
    _response(result, bound, should_cancel)
    return result
