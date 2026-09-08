import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parseProfile} from '../src/memory/profile.ts';

const claim={fact_id:4,subject:'Juniper',predicate:'prefers',object:'local storage',confidence:.8,source_episode_id:9};
const value={static_facts:[claim],dynamic:['A **retained** source'],recent:[{episode_id:9,excerpt:'A **retained** source',created_at:'2026-09-07T10:00:00Z'}]};
test('profile preserves actual claim and source identities without fabricating provenance',()=>{
  assert.deepEqual(parseProfile(value),{claims:[claim],recent:value.recent});
  const minimal={...claim};delete minimal.source_episode_id;
  assert.equal(parseProfile({...value,static_facts:[minimal]}).claims[0].source_episode_id,undefined);
  assert.deepEqual(parseProfile({static_facts:[],dynamic:[],recent:[]}),{claims:[],recent:[]});
});
test('profile refuses missing evidence, inconsistent excerpts and duplicate identities',()=>{
  for(const bad of [null,{}, {...value,recent:undefined},{...value,recent:[]},
    {...value,dynamic:['unrelated text']},{...value,recent:[{...value.recent[0],episode_id:-1}]},
    {...value,recent:[{...value.recent[0],created_at:'bad'}]},
    {...value,static_facts:[claim,claim]},
    {...value,dynamic:[...value.dynamic,...value.dynamic],recent:[...value.recent,...value.recent]},
    {...value,static_facts:[{...claim,confidence:NaN}]},
    {...value,static_facts:[{...claim,source_episode_id:'9'}]},
    {...value,static_facts:[{...claim,subject:3}]},
    {...value,static_facts:Array.from({length:51},(_,i)=>({...claim,fact_id:i+1}))},
  ])assert.throws(()=>parseProfile(bad),/profile/i);
});
test('profile counts Unicode characters and preserves server order for historical backfills',()=>{
  const recent=[{episode_id:12,excerpt:'🪴'.repeat(200),created_at:'2001-01-01T00:00:00Z'},value.recent[0]];
  assert.deepEqual(parseProfile({...value,recent,dynamic:recent.map(r=>r.excerpt)}).recent,recent);
  const over=[{...recent[0],excerpt:'🪴'.repeat(201)}];
  assert.throws(()=>parseProfile({...value,recent:over,dynamic:over.map(r=>r.excerpt)}),/profile/i);
});
test('profile preserves multiple native source references and refuses malformed provenance',()=>{
  const native={...claim,sources:[9,12]};delete native.source_episode_id;
  assert.deepEqual(parseProfile({...value,static_facts:[native]}).claims[0].sources,[9,12]);
  for(const sources of [[9,9],[0],['9'],{},null])assert.throws(()=>parseProfile({...value,static_facts:[{...native,sources}]}),/profile/i);
});
