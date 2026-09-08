import {test} from 'node:test';
import assert from 'node:assert/strict';
import {compileFilter,type FilterGroup,type FilterRule} from '../src/memory/metadata-filter.ts';
const rule=(change:Partial<FilterRule>={}):FilterRule=>({kind:'rule',field:'status',operator:'is',value:'published',negated:false,...change});
test('metadata filters preserve nested groups, negation and exact text',()=>{
  const draft:FilterGroup={kind:'group',mode:'all',children:[rule({value:' a,b:c '}),{kind:'group',mode:'any',children:[rule({field:'priority',operator:'at_least',value:'10'}),rule({field:'category',operator:'in',value:'guide\nreference',negated:true})]}]};
  assert.deepEqual(JSON.parse(compileFilter(draft)),{all:[{field:'status',is:' a,b:c '},{any:[{field:'priority',at_least:10},{field:'category',in:['guide','reference'],not:true}]}]});
});
test('presence has no stale value and numeric conditions reject missing or nonfinite operands',()=>{
  assert.deepEqual(JSON.parse(compileFilter({kind:'group',mode:'all',children:[rule({operator:'present',value:'ignored'})]})),{all:[{field:'status',present:true}]});
  for(const value of ['', ' ', 'soon','Infinity','NaN','1e309'])assert.throws(()=>compileFilter({kind:'group',mode:'all',children:[rule({operator:'above',value})]}));
});
test('invalid metadata drafts never compile to a wider query',()=>{
  for(const node of [rule({field:'space-name'}),rule({field:''}),rule({operator:'in',value:''}),rule({operator:'in',value:'one\n\ntwo'})])assert.throws(()=>compileFilter({kind:'group',mode:'all',children:[node]}));
  assert.throws(()=>compileFilter({kind:'group',mode:'any',children:[]}));
});
test('filter limits match native group depth, condition count and text bounds',()=>{
  let nested:FilterGroup={kind:'group',mode:'all',children:[rule()]};
  for(let i=1;i<8;i++)nested={kind:'group',mode:'all',children:[nested]};
  assert.doesNotThrow(()=>compileFilter(nested));
  assert.throws(()=>compileFilter({kind:'group',mode:'all',children:[nested]}));
  assert.throws(()=>compileFilter({kind:'group',mode:'all',children:Array.from({length:201},()=>rule())}));
  assert.throws(()=>compileFilter({kind:'group',mode:'all',children:[rule({value:'a'.repeat(8000)})]}));
});
