import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parseClaimDetail, relationTarget, relationLabel} from '../src/memory/claim-relations.ts';
const fact={fact_id:7,subject:'Mira',predicate:'works_at',object:'Cedar',status:'proposed',origin:'inferred',valid_from:'2026-09-06T00:00:00Z',valid_until:null,confidence:0.8};
const link={link_id:4,from_fact:7,to_fact:2,kind:'derived_from',created_at:'2026-09-06T00:00:00Z',source_episode_id:8,quote:'A recorded connection'};
const detail={fact,links:[link],sources:[]};
test('claim detail validates identity and preserves directions, statuses and source ids',()=>{
  const value=parseClaimDetail(detail,7);
  assert.equal(value.fact.status,'proposed');
  assert.equal(value.fact.origin,'inferred');
  assert.equal(relationTarget(value.links[0],7),2);
  assert.equal(relationTarget(value.links[0],2),7);
  assert.equal(relationLabel(value.links[0],7),'Derived from');
  assert.equal(relationLabel(value.links[0],2),'Used to derive');
  assert.deepEqual(value.sources,[]);
  assert.equal(value.links[0].source_episode_id,8);
  assert.equal(value.links[0].quote,'A recorded connection');
});
test('unsupported, unrelated or mismatched detail never becomes a navigable claim',()=>{
  for(const value of [null,{}, {...detail,fact:{...fact,fact_id:8}}, {...detail,fact:{...fact,status:'approved'}},
    {...detail,links:[{...link,from_fact:3}]}, {...detail,links:[{...link,to_fact:7}]},
    {...detail,links:[{...link,kind:'causes'}]}, {...detail,links:[{...link,to_fact:'../../status'}]},
    {...detail,links:[link,link]}, {...detail,sources:['8']}, {...detail,sources:[-1]},
    {...detail,fact:{...fact,valid_from:'not a date'}}, {...detail,fact:{...fact,object:null}},
    {...detail,links:[{...link,source_episode_id:'8'}]}])assert.throws(()=>parseClaimDetail(value,7),/claim|relationship/i);
});
test('all stored relationship kinds have explicit forward and inverse descriptions',()=>{
  for(const [kind,forward,inverse] of [['extends','Extends','Extended by'],['supports','Supports','Supported by'],['contradicts','Contradicts','Contradicted by'],['derived_from','Derived from','Used to derive']]){
    const value=parseClaimDetail({...detail,links:[{...link,kind}]},7);
    assert.equal(relationLabel(value.links[0],7),forward);
    assert.equal(relationLabel(value.links[0],2),inverse);
  }
});
