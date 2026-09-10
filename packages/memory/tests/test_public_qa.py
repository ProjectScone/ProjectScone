import json
from pathlib import Path

import pytest

from scone_memory.testing.public_qa import build_bundle, answer_score, benchmark_messages


def datasets(tmp_path):
    hotpot=[]
    squad=[]
    for i in range(4):
        hotpot.append({'_id':str(i),'question':f'Question {i}?','answer':'GOLD_ONLY_CANARY',
            'context':[[f'First{i}',['Visible source.']],[f'Distractor{i}',['Other source.']]],
            'supporting_facts':[[f'First{i}',0]]})
        squad.append({'title':f'Squad{i}','paragraphs':[{'context':'A source has cobalt.',
            'qas':[{'id':str(i),'question':f'Which color {i}?',
                    'answers':[{'text':'cobalt','answer_start':13}]}]}]})
    h=tmp_path/'hotpot.json';s=tmp_path/'squad.json'
    h.write_text(json.dumps(hotpot));s.write_text(json.dumps({'data':squad}))
    return h,s


def test_selection_is_deterministic_and_gold_stays_out_of_queries_and_documents(tmp_path):
    h,s=datasets(tmp_path)
    first=build_bundle(h,s,count=1)
    second=build_bundle(h,s,count=1)
    assert first==second
    assert len(first.queries)==2 and len(first.reserved)==2
    assert not {q.id for q in first.queries}&{q.id for q in first.reserved}
    assert len(first.documents)==6, 'all selected/reserved distractors must remain in the pool'
    visible=json.dumps([d.model_dump() for d in first.documents]+[benchmark_messages(q) for q in first.queries])
    assert 'GOLD_ONLY_CANARY' not in visible
    assert any('GOLD_ONLY_CANARY' in g.answers for g in first.gold)
    assert all(set(q.model_dump())=={'id','dataset','question'} for q in first.queries)


def test_sampling_does_not_change_when_answer_labels_change(tmp_path):
    h,s=datasets(tmp_path)
    before=build_bundle(h,s,count=1)
    rows=json.loads(h.read_text())
    for row in rows:row['answer']='DIFFERENT_GOLD'
    h.write_text(json.dumps(rows))
    after=build_bundle(h,s,count=1)
    assert before.queries==after.queries and before.documents==after.documents


def test_invalid_support_index_and_duplicate_ids_fail_instead_of_dropping_rows(tmp_path):
    h,s=datasets(tmp_path)
    rows=json.loads(h.read_text())
    for row in rows:row['supporting_facts'][0][1]=99
    h.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='support'):build_bundle(h,s,count=1)
    rows[1]['_id']=rows[0]['_id'];h.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='duplicate'):build_bundle(h,s,count=1)


def test_standard_answer_metrics_and_failure_denominators():
    assert answer_score('The Eiffel Tower',['Eiffel Tower'],'squad',True)=={'em':1.,'f1':1.}
    assert answer_score('red blue',['red green'],'squad',True)=={'em':0.,'f1':.5}
    assert answer_score('yes indeed',['yes'],'hotpotqa',True)=={'em':0.,'f1':0.}
    assert answer_score('cobalt',['wrong','cobalt'],'squad',True)=={'em':1.,'f1':1.}
    assert answer_score('cobalt',['cobalt'],'squad',False)=={'em':0.,'f1':0.}
    assert answer_score('INSUFFICIENT_EVIDENCE',['cobalt'],'squad',True)=={'em':0.,'f1':0.}


def test_blank_sentence_slots_preserve_original_support_indices(tmp_path):
    h,s=datasets(tmp_path)
    rows=json.loads(h.read_text())
    for row in rows:
        row['context'][0][1]=['', 'Visible source.', ' ']
        row['supporting_facts'][0][1]=1
    h.write_text(json.dumps(rows))
    bundle=build_bundle(h,s,count=1)
    assert any(d.text=='\nVisible source.\n ' for d in bundle.documents)
    assert all(sentence=='Visible source.' for g in bundle.gold for _,sentence in g.support_sentences)
