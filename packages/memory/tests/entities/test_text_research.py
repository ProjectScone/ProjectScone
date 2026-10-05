"""Raw-text research remains bounded and independent of sampled citations."""
import hashlib

import pytest


def api():
    import scone_memory.text_network as module
    return module


def document(content, key='text:one'):
    return api().TextDocument(key=key, kind='document', id=key, title='Example', content=content)


def term(label, aliases=()):
    return api().TextTerm(label=label, kind='subject', aliases=list(aliases))


def section(content, start=0, end=None, key='one', document_key='text:one'):
    return api().TextSection(key=key, document_key=document_key, title='Section', start=start,
                             end=len(content) if end is None else end)


def request(content, start, end, key='one', **kwargs):
    return api().ContextRequest(key=key, document_key='text:one',
        content_sha256=hashlib.sha256(content.encode()).hexdigest(), start=start, end=end, **kwargs)


def test_section_counts_precede_citation_sampling_and_match_network_identity():
    text = '\n\n'.join(['Mira walks.'] * 220)
    docs, terms = [document(text)], [term('Mira')]
    network = api().build_text_network(docs, terms)
    result = api().build_section_coverage(docs, terms, [section(text)])
    assert len(network.passages) == 3
    assert result.network_digest == network.digest
    assert result.terms[0].id == network.nodes[0].id
    assert result.sections[0].passage_count == 220
    assert result.sections[0].counts[0].passage_count == 220
    assert result.sections[0].counts[0].first_start == 0
    assert result.sections[0].counts[0].last_start == text.rindex('Mira')
    assert result.sections[0].counts[0].first_passage_start == 0
    assert result.sections[0].counts[0].first_passage_end == len('Mira walks.')
    assert result.sections[0].complete and not result.coverage.partial
    assert result.persisted is False


def test_sections_keep_unicode_offsets_aliases_and_exclude_fenced_code():
    text = '# First\n\n🙂 Straße and Mira.\n\n> ```\n> MIRA\n> ```\n\n# Next\n\nDoctor Mira returns.'
    split = text.index('# Next')
    result = api().build_section_coverage([document(text)], [term('Mira', ['Doctor Mira']), term('strasse'), term('s')],
        [section(text, 0, split, 'first'), section(text, split, key='next')])
    names = {row.id: row.label for row in result.terms}
    counts = [{names[row.node_id]: row.passage_count for row in part.counts} for part in result.sections]
    assert counts == [{'Mira': 1, 'strasse': 1}, {'Mira': 1}]
    assert [part.passage_count for part in result.sections] == [2, 2]
    first = next(row for row in result.sections[0].counts if names[row.node_id] == 'strasse')
    assert text[first.first_start:first.first_end] == 'Straße'


def test_section_limit_marks_unscanned_sections_without_claiming_absence():
    text = 'Mira.\n\nRowan.\n\nMira.'
    result = api().build_section_coverage([document(text)], [term('Mira'), term('Rowan')],
        [section(text, 0, 7, 'first'), section(text, 7, key='later')],
        limits=api().ResearchLimits(max_passages=1))
    assert result.coverage.partial and result.coverage.reasons == ['max_passages']
    assert result.sections[0].complete
    assert not result.sections[1].complete and result.sections[1].counts == []


@pytest.mark.parametrize('spans', [[(0, 8), (7, 12)], [(0, 100)], [(1, 4)]])
def test_invalid_or_passage_splitting_sections_are_rejected(spans):
    text = 'Mira.\n\nRowan.'
    with pytest.raises(ValueError):
        api().build_section_coverage([document(text)], [term('Mira')],
            [section(text, start, end, str(index)) for index, (start, end) in enumerate(spans)])


def test_section_identity_includes_section_selection_but_not_input_order():
    text = 'Mira.\n\nRowan.'
    sections = [section(text, 0, 7, 'first'), section(text, 7, key='last')]
    terms = [term('Mira'), term('Rowan')]
    result = api().build_section_coverage([document(text)], terms, sections)
    assert result == api().build_section_coverage([document(text)], terms[::-1], sections[::-1])
    assert result.digest != api().build_section_coverage([document(text)], terms, [sections[0]]).digest


def test_context_verifies_source_and_exact_paragraph_boundaries():
    text = 'Before.\n\n🙂 Mira meets Rowan.\n\nAfter.'
    start, end = text.index('🙂'), text.index('\n\nAfter')
    result = api().surrounding_context([document(text)], [request(text, start, end)])
    window = result.windows[0]
    assert window.text == text and window.start == 0 and window.end == len(text)
    assert window.target_start == start and window.target_end == end
    assert not window.truncated_before and not window.truncated_after and not window.target_truncated
    for invalid in [request(text, start+1, end), request(text, start, end-1),
                    request(text, start, end).model_copy(update={'content_sha256': 'a'*64})]:
        with pytest.raises(ValueError):
            api().surrounding_context([document(text)], [invalid])


def test_context_excludes_code_targets_and_uses_unicode_character_cap():
    text = 'Before.\n\n```\nMira\n```\n\n' + '🙂' * 1000 + '\n\nAfter.'
    code = text.index('Mira')
    with pytest.raises(ValueError, match='context_passage'):
        api().surrounding_context([document(text)], [request(text, code, code+4)])
    start, end = text.index('🙂'), text.index('\n\nAfter')
    result = api().surrounding_context([document(text)], [request(text, start, end)],
                                      limits=api().ResearchLimits(max_context_characters=600))
    window = result.windows[0]
    assert window.text == '🙂'*600
    assert text[window.start:window.end] == window.text
    assert window.target_truncated and window.truncated_after and window.truncated_before


def test_context_scan_limit_refuses_unverified_late_passage():
    text = 'Before.\n\nMira.'
    with pytest.raises(ValueError, match='context_scan_limit'):
        api().surrounding_context([document(text)], [request(text, 9, len(text))],
                                  limits=api().ResearchLimits(max_passages=1))


def test_context_request_controls_neighbors_and_retains_fingerprint():
    text = 'First.\n\nMira.\n\nLast.'
    result = api().surrounding_context([document(text)], [request(text, 8, 13, before=0, after=0)])
    assert result.windows[0].text == 'Mira.'
    assert result.windows[0].content_sha256 == hashlib.sha256(text.encode()).hexdigest()


def test_research_rejects_duplicate_unknown_or_excessive_inputs():
    text = 'Mira.'
    cases = [([document(text)]*2, [section(text)]),
             ([document(text)], [section(text)]*2),
             ([document(text)], [section(text, document_key='missing')]),
             ([document(text)], [section(text, key=str(index)) for index in range(129)])]
    for docs, sections in cases:
        with pytest.raises(ValueError):
            api().build_section_coverage(docs, [], sections)
    with pytest.raises(ValueError):
        api().surrounding_context([document(text)], [request(text, 0, 5)]*2)
    with pytest.raises(ValueError):
        api().surrounding_context([document(text)], [request(text, 0, 5, key=str(index)) for index in range(11)])


@pytest.mark.parametrize('operation', ['coverage', 'context'])
def test_research_checks_input_bytes_output_bytes_and_cancellation(operation):
    text = 'Mira Rowan ' * 300
    docs = [document(text)]
    def run(limits=None, should_cancel=None):
        if operation == 'coverage':
            return api().build_section_coverage(docs, [term('Mira'), term('Rowan')], [section(text)],
                                               limits=limits, should_cancel=should_cancel)
        return api().surrounding_context(docs, [request(text, 0, len(text))],
                                         limits=limits, should_cancel=should_cancel)
    with pytest.raises(ValueError, match='input_bytes'):
        run(api().ResearchLimits(max_input_bytes=100))
    with pytest.raises(ValueError, match='response_bytes'):
        run(api().ResearchLimits(max_response_bytes=1024))
    with pytest.raises(api().TextNetworkCancelled):
        run(should_cancel=lambda: True)


def test_cancellation_during_matching_stops_without_partial_result():
    text = '\n\n'.join(['Mira Rowan']*100)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        return calls > 25
    with pytest.raises(api().TextNetworkCancelled):
        api().build_section_coverage([document(text)], [term('Mira'), term('Rowan')], [section(text)], should_cancel=cancel)
