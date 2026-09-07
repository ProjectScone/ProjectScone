import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parseIntegrityReport} from '../src/memory/integrity.ts';

const clean={space:'alpha',episodes:4,chunks:8,facts:3,links:2,tombstones:1,
  chunks_without_episode:[],vectors_without_chunk:[],facts_citing_forgotten:[],
  facts_citing_unknown:[],links_with_missing_ends:[],attachments_unlinked:[],not_inspected:[],healthy:true};
test('integrity preserves checked counts and explicit uninspected stores',()=>{
  assert.equal(parseIntegrityReport(clean,'alpha').tombstones,1);
  const partial=parseIntegrityReport({...clean,vectors_without_chunk:null,not_inspected:['vectors']},'alpha');
  assert.equal(partial.vectors_without_chunk,null);
  assert.deepEqual(partial.not_inspected,['vectors']);
});
test('integrity distinguishes forgotten support from an unknown source',()=>{
  const result=parseIntegrityReport({...clean,healthy:false,facts_citing_forgotten:[12],facts_citing_unknown:[19]},'alpha');
  assert.deepEqual(result.facts_citing_forgotten,[12]);
  assert.deepEqual(result.facts_citing_unknown,[19]);
});
test('integrity cannot accept cross-space, missing, contradictory or malformed reports',()=>{
  for(const bad of [null,{}, {...clean,space:'beta'},{...clean,chunks:-1},{...clean,facts:1.5},
    {...clean,healthy:'true'},{...clean,facts_citing_unknown:[0]},
    {...clean,attachments_unlinked:['https://untrusted.example/image']},
    {...clean,facts_citing_forgotten:[12]},{...clean,vectors_without_chunk:null},
    {...clean,not_inspected:['vectors']},{...clean,facts_citing_unknown:[12,12]}]){
    assert.throws(()=>parseIntegrityReport(bad,'alpha'),/integrity report/i);
  }
});
