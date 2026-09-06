import {test} from 'node:test';
import assert from 'node:assert/strict';
import {scopeFromDraft,readScope,sameScope} from '../src/conversations/recall-scope.ts';

const empty={kind:'',source_prefix:'',since:'',until:'',metadata:[]};
test('scope form builds literal filters without widening partially entered metadata',()=>{
  assert.deepEqual(scopeFromDraft(empty),{});
  assert.deepEqual(scopeFromDraft({...empty,kind:'file',source_prefix:'docs/',since:'2026-09-01',metadata:[{key:'collection',value:'manuals'}]}),
    {kind:'file',source_prefix:'docs/',since:'2026-09-01T00:00:00.000Z',where:{collection:'manuals'}});
  for(const metadata of [[{key:'collection',value:''}],[{key:'',value:'manuals'}],[{key:'collection',value:'a'},{key:'collection',value:'b'}]])assert.throws(()=>scopeFromDraft({...empty,metadata}));
  assert.throws(()=>scopeFromDraft({...empty,since:'2026-02-30'}));
  assert.throws(()=>scopeFromDraft({...empty,since:'2026-09-01T24:00:00Z'}));
  assert.throws(()=>scopeFromDraft({...empty,since:'2026-09-02',until:'2026-09-01'}));
});
test('scope acknowledgement compares normalized constraints and preserves empty prefix',()=>{
  assert.equal(sameScope({where:{team:'a',collection:'b'},since:'2026-09-01T01:00:00+01:00'},{where:{collection:'b',team:'a'},since:'2026-09-01T00:00:00.000Z'}),true);
  assert.equal(sameScope({source_prefix:''},{}),false);
  assert.equal(sameScope({kind:'file'},undefined),false);
  assert.equal(sameScope({},{}),true);
  for(const value of [{kind:'future'},{where:{'bad-key':'x'}},{source_prefix:3},{until:'not-a-date'},{where:{team:''}},{extra:'ignored?'}])assert.throws(()=>readScope(value));
});
