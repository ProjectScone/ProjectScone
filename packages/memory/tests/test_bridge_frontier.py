"""Bounded alternative bridges must survive a deeper dead end."""
import pytest

from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever
from scone_memory.retrieval.recall_scope import RecallScope
from scone_memory.retrieval.structured_evidence import EvidenceRequirement, StructuredEvidenceAssessor
from test_reachable_fact import fact


PLAN = EvidenceRequirement(kind='reachable_fact',subject='ticket',predicate='located in',
                          via=('assigned to','managed by'),max_hops=3)
BRANCHES = (('ticket','assigned to','first'),('first','managed by','dead-end'),
            ('ticket','managed by','second'),('second','managed by','annex'),('annex','located in','Dakar'))


async def test_frontier_keeps_shorter_alternative_to_deeper_dead_end():
    records = tuple(fact(i+1,*triple) for i,triple in enumerate(BRANCHES[:3]))
    report = await StructuredEvidenceAssessor('question',(PLAN,),followup_strategy='bridge').assess_with_coverage(
        'question',records)
    assert report.decision.status == 'insufficient'
    assert report.decision.followup_queries == ('dead-end located in','second assigned to managed by located in')
    assert set(report.coverage[0].bridge_ids) == {'fact:1','fact:2','fact:3'}
    assert report.coverage[0].alternative_queries == ('second assigned to managed by located in',)
    assert report.decision.selected_groups == (('fact:1','fact:2'),)


async def test_frontier_queries_are_bounded_and_prioritize_other_requirements():
    records=tuple(fact(i+1,'ticket','assigned to',f'office-{i}') for i in range(8))
    missing=EvidenceRequirement(kind='fact',subject='juniper',predicate='uses')
    report=await StructuredEvidenceAssessor('question',(PLAN,missing),followup_strategy='bridge').assess_with_coverage(
        'question',records)
    assert len(report.decision.followup_queries)==3
    assert report.decision.followup_queries[1]=='juniper uses'
    assert len(report.coverage[0].alternative_queries)==2


async def test_query_collision_keeps_both_distinct_bridges_without_duplicate_searches():
    plan=EvidenceRequirement(kind='reachable_fact',subject='root',predicate='p',via=('via',),max_hops=3)
    records=(fact(1,'root','via','a'),fact(2,'a','via','X via'),fact(3,'root','via','X'))
    report=await StructuredEvidenceAssessor('question',(plan,),followup_strategy='bridge').assess_with_coverage(
        'question',records)
    assert report.decision.followup_queries==('X via p',)
    assert set(report.coverage[0].bridge_ids)=={'fact:1','fact:2','fact:3'}
    assert report.coverage[0].alternative_queries==()


async def test_real_recall_resolves_alternative_branch_within_same_budget(engine, monkeypatch):
    async def seed(triple):
        left,predicate,right=triple
        quote=f'{left} {predicate} {right}.'
        episode=await engine.remember('alpha',quote,kind='file',source=f'manuals/{left}',metadata={'team':'blue'})
        return await engine.assert_fact('alpha',left,predicate,right,source_episode_id=episode.episode_id,quote=quote)

    for i in range(80):
        await seed((f'archive-{i}',['depends on','uses','located in','managed by'][i%4],f'depot-{i}'))
    records=[await seed(triple) for triple in BRANCHES]
    # Different root predicates keep both branches active under ledger semantics.
    for record in records:
        assert (await engine.documents.get_fact('alpha',record.fact_id)).status=='active'
    question='Where is the office responsible for ticket located?'
    observed=[]
    native_recall=engine.recall

    async def observe(*args, **kwargs):
        result=await native_recall(*args, **kwargs)
        observed.append((args[1],[(f.subject,f.predicate,f.object) for f in result.facts]))
        return result

    monkeypatch.setattr(engine,'recall',observe)
    assessor=StructuredEvidenceAssessor(question,(PLAN,),followup_strategy='bridge')
    result=await AdaptiveRetriever(engine,assessor,limits=AdaptiveLimits(
        max_rounds=4,max_queries=6,candidate_limit=16)).retrieve('alpha',question,
            scope=RecallScope.validated(where={'team':'blue'},source_prefix='manuals/'))
    assert result.status=='sufficient', (observed,result.model_dump())
    assert {record.fact_id for record in records[2:]} <= {record.fact_id for record in result.recall.facts}
    assert result.queries_used<=6 and len(result.rounds)<=4
