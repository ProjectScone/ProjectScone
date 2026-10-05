"""The temporary network never upgrades a literal mention to a recorded fact."""
import json

import pytest


def api():
    import scone_memory.text_network as module
    return module


def document(text, key='draft:one'):
    return api().TextDocument(key=key, kind='draft', id=key, title='Private title', revision=1, content=text)


def term(label, aliases=(), kind='character'):
    return api().TextTerm(label=label, kind=kind, aliases=list(aliases))


def test_empty_network_stays_ephemeral_and_has_name_candidates_without_terms():
    graph = api().build_text_network([document('Mira visits Alder Bay.\n\nMira returns to Alder Bay.')], [])
    assert graph.persisted is False and graph.basis == 'literal_shared_passage'
    assert graph.nodes == [] and graph.edges == []
    assert {'Mira', 'Alder Bay'} <= {row.label for row in graph.candidates}
    assert all(row.passage_count == 2 for row in graph.candidates)
    assert graph.coverage.documents == 1 and graph.coverage.passages == 2


def test_only_shared_prose_passages_connect_and_each_edge_has_exact_provenance():
    text = 'Mira alone.\n\nRowan alone.\n\nMira meets Rowan.\n\n```\nMira sees Harbor\n```\n\n| Name | Place |\n|---|---|\n| Mira | Harbor |\n| Rowan | Forest |'
    graph = api().build_text_network([document(text)], [term(name) for name in ('Mira', 'Rowan', 'Harbor', 'Forest', 'Missing')])
    names = {node.id: node.label for node in graph.nodes}
    pairs = {frozenset((names[edge.source], names[edge.target])) for edge in graph.edges}
    assert pairs == {frozenset(('Mira', 'Rowan')), frozenset(('Mira', 'Harbor')), frozenset(('Rowan', 'Forest'))}
    cited = {passage.id: passage for passage in graph.passages}
    for edge in graph.edges:
        assert edge.kind == 'shared_passage' and edge.passage_ids
        for identity in edge.passage_ids:
            passage = cited[identity]
            assert text[passage.excerpt_start:passage.excerpt_start + len(passage.excerpt)] == passage.excerpt
            assert {edge.source, edge.target} <= {mention.node_id for mention in passage.mentions}
            for mention in passage.mentions:
                assert text[mention.start:mention.end].casefold() == names[mention.node_id].casefold()
    assert {names[identity] for identity in graph.analysis.isolated_node_ids} == {'Missing'}


def test_unicode_casefold_expansion_offsets_and_boundaries_are_original():
    text = '🙂 Straße meets Élodie. STRASSE returns. Strassex is elsewhere.'
    graph = api().build_text_network([document(text)], [term('strasse'), term('élodie'), term('s')])
    labels = {node.id: node.label for node in graph.nodes}
    matches = {labels[mention.node_id]: text[mention.start:mention.end] for mention in graph.passages[0].mentions}
    assert matches == {'strasse': 'Straße', 'élodie': 'Élodie'}
    assert next(node for node in graph.nodes if node.label == 's').passage_count == 0


def test_aliases_explicit_and_ambiguous_aliases_are_refused():
    graph = api().build_text_network([document('Doctor Mira meets Rowan.')], [term('Mira', ['Doctor Mira']), term('Rowan')])
    assert len(graph.edges) == 1
    with pytest.raises(ValueError, match='ambiguous'):
        api().build_text_network([document('Mira')], [term('Mira'), term('Someone', ['MIRA'])])


def test_input_order_does_not_change_ids_and_content_changes_do():
    documents = [document('Mira Rowan', 'draft:a'), document('Mira Harbor', 'source:b')]
    terms = [term('Mira'), term('Rowan'), term('Harbor')]
    first = api().build_text_network(documents, terms)
    assert first == api().build_text_network(list(reversed(documents)), list(reversed(terms)))
    changed = api().build_text_network([document('Mira Rowan!', 'draft:a'), documents[1]], terms)
    assert first.digest != changed.digest
    assert {row.id for row in first.nodes}.isdisjoint(row.id for row in changed.nodes)


def test_spelling_suggestions_never_merge_distinct_names_or_create_edges():
    graph = api().build_text_network([document('Marianne waits.\n\nMariane leaves.')], [term('Marianne'), term('Mariane')])
    assert len(graph.nodes) == 2 and graph.edges == []
    assert len(graph.variants) == 1 and graph.variants[0].reason == 'similar_spelling'
    assert {graph.variants[0].left, graph.variants[0].right} == {row.id for row in graph.nodes}


def test_generic_kinds_and_multi_document_provenance_do_not_create_cross_document_edges():
    graph = api().build_text_network([document('Mira here', 'x:a'), document('Rowan here', 'x:b')], [term('Mira', kind='animal'), term('Rowan', kind='element')])
    assert graph.edges == []
    assert {node.kind for node in graph.nodes} == {'animal', 'element'}
    assert [node.document_keys for node in sorted(graph.nodes, key=lambda node: node.label)] == [['x:a'], ['x:b']]


def test_output_caps_never_leave_an_uncited_edge_and_mark_partial_coverage():
    graph = api().build_text_network([document('Mira Rowan\n\nRowan Harbor\n\nHarbor Forest')],
        [term(name) for name in ('Mira', 'Rowan', 'Harbor', 'Forest')],
        limits=api().NetworkLimits(max_edges=2, max_cited_passages=1))
    assert len(graph.passages) == 1 and len(graph.edges) <= 2
    assert graph.coverage.partial and {'max_edges', 'max_cited_passages'} <= set(graph.coverage.reasons)
    assert graph.coverage.edges == 3
    assert all(edge.passage_ids for edge in graph.edges)


def test_paragraph_limit_reports_incomplete_input_not_absence():
    graph = api().build_text_network([document('Mira\n\nRowan Mira')], [term('Mira'), term('Rowan')], limits=api().NetworkLimits(max_passages=1))
    assert graph.edges == [] and graph.coverage.partial
    assert 'max_passages' in graph.coverage.reasons


@pytest.mark.parametrize('terms', [[term] * 61, []])
def test_hard_input_limits(terms):
    with pytest.raises(ValueError):
        api().build_text_network([document('x' * (512 * 1024 + 1))], [] if not terms else [term(str(n)) for n in range(61)])


def test_byte_limit_counts_utf8_not_characters_and_response_limit_is_enforced():
    with pytest.raises(ValueError, match='input_bytes'):
        api().build_text_network([document('é' * 100)], [], limits=api().NetworkLimits(max_input_bytes=199))
    with pytest.raises(ValueError, match='response_bytes'):
        api().build_text_network([document('Mira Rowan')], [term('Mira'), term('Rowan')], limits=api().NetworkLimits(max_response_bytes=1024))


def test_cancellation_checked_before_content_processing():
    with pytest.raises(api().TextNetworkCancelled):
        api().build_text_network([document('Mira Rowan')], [], should_cancel=lambda: True)


def test_long_excerpts_bound_size_but_all_offsets_refer_to_unmodified_source():
    text = '🙂 ' * 1000 + 'Mira meets Rowan.'
    graph = api().build_text_network([document(text)], [term('Mira'), term('Rowan')])
    passage = graph.passages[0]
    assert len(passage.excerpt) <= 600 and passage.end == len(text)
    assert text[passage.excerpt_start:passage.excerpt_start + len(passage.excerpt)] == passage.excerpt
    assert len(graph.model_dump_json().encode()) < 512 * 1024
    assert not {'fact_ids', 'episode_id', 'confidence'} & set(json.loads(graph.model_dump_json()))


def test_combining_mark_is_not_a_word_boundary_and_candidates_count_passages_once():
    graph = api().build_text_network([document('Cafe\u0301. Mira. MIRA.')], [term('Cafe')])
    assert graph.nodes[0].passage_count == 0
    assert next(row for row in graph.candidates if row.label.casefold() == 'mira').passage_count == 1


def test_explicit_topology_partition_has_no_dependency_on_ledger_records():
    from scone_memory.entities.analysis import partition_associations
    groups = partition_associations(['a', 'b', 'c', 'd', 'e', 'f', 'alone'], [
        ('a', 'b', 10), ('a', 'c', 10), ('b', 'c', 10), ('d', 'e', 10), ('d', 'f', 10), ('e', 'f', 10), ('c', 'd', 1)])
    assert {frozenset(group) for group in groups} == {frozenset('abc'), frozenset('def'), frozenset(['alone'])}
    with pytest.raises(ValueError):
        partition_associations(['a', 'b'], [('a', 'b', 1), ('b', 'a', 1)])
    with pytest.raises(ValueError):
        partition_associations(['a'], [('a', 'missing', 1)])


def test_dense_valid_input_returns_bounded_citations_instead_of_rejecting_response():
    names = [f'Term{number:02}' for number in range(60)]
    graph = api().build_text_network([document('\n\n'.join([' '.join(names)]*1000))], [term(name) for name in names])
    assert len(graph.edges) == 600 and len(graph.model_dump_json().encode()) <= 512*1024
    assert all(edge.weight == 1000 and edge.passages_truncated and len(edge.passage_ids) <= 3 for edge in graph.edges)
    assert len(graph.passages) <= 3 and graph.coverage.partial
    assert 'max_edge_citations' in graph.coverage.reasons


@pytest.mark.parametrize('fenced', [
    '> ```python\n> Mira Port\n> ```',
    '- ```python\n  Mira Port\n  ```',
    '> - ~~~~text\n>   Mira Port\n>   ~~~~',
    '1. ```python\n   Mira Port\n   ```',
    '    ```python\n    Mira Port\n    ```',
])
def test_container_fences_do_not_create_associations_and_keep_following_offsets(fenced):
    text = fenced + '\n\nMira meets Rowan.'
    graph = api().build_text_network([document(text)], [term(name) for name in ('Mira', 'Port', 'Rowan')])
    labels = {node.id: node.label for node in graph.nodes}
    assert [{labels[edge.source], labels[edge.target]} for edge in graph.edges] == [{'Mira', 'Rowan'}]
    assert graph.passages[0].start == len(fenced)+2
    assert graph.passages[0].excerpt == 'Mira meets Rowan.'


def test_lone_term_citation_sampling_reports_omitted_evidence():
    graph = api().build_text_network([document('\n\n'.join(['Mira walks alone.']*20))], [term('Mira')])
    assert graph.nodes[0].passage_count == 20 and len(graph.passages) == 3
    assert graph.coverage.partial and 'max_node_citations' in graph.coverage.reasons


def test_spelling_coverage_reports_actual_overflow_only():
    terms = [term(name) for name in ('Marianne', 'Mariane', 'Marianna')]
    graph = api().build_text_network([], terms, limits=api().NetworkLimits(max_variants=1))
    assert len(graph.variants) == 1
    assert graph.coverage.partial and 'max_variants' in graph.coverage.reasons
    complete = api().build_text_network([], terms[:2], limits=api().NetworkLimits(max_variants=1))
    assert len(complete.variants) == 1 and 'max_variants' not in complete.coverage.reasons


@pytest.mark.parametrize('prefix', [
    '> ```\n> hidden Mira Port\n\n',
    '- ```\n  hidden Mira Port\n\n',
    '    ```\n    hidden Mira Port\n\n',
    '> - ```\n>   hidden Mira Port\n\n',
    '- > ```\n  > hidden Mira Port\n\n',
])
def test_unclosed_container_fence_ends_at_container_boundary(prefix):
    text = prefix + 'Mira visits Rowan.'
    graph = api().build_text_network([document(text)], [term(name) for name in ('Mira', 'Port', 'Rowan')])
    labels = {node.id: node.label for node in graph.nodes}
    assert [{labels[edge.source], labels[edge.target]} for edge in graph.edges] == [{'Mira', 'Rowan'}]
    assert graph.passages[0].start == len(prefix)


def test_top_level_unclosed_fence_still_excludes_everything_to_end():
    graph = api().build_text_network([document('```\nMira Port\n\nMira Port')], [term('Mira'), term('Port')])
    assert graph.passages == [] and graph.edges == []
