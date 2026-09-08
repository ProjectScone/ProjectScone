import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseCapabilities } from '../src/capabilities.ts';

const fixtures = JSON.parse(readFileSync(new URL('../../tests/fixtures/http-capabilities.json', import.meta.url), 'utf8'));
test('image inference requires its own explicit capability',()=>{
  const features={...fixtures.python.features};delete features['images.understand'];
  assert.equal(parseCapabilities({...fixtures.python,features}).features['images.understand'],false);
  assert.equal(parseCapabilities({...fixtures.python,features:{...features,'images.understand':true}}).features['images.understand'],true);
  for(const value of ['true',1,null])assert.throws(()=>parseCapabilities({...fixtures.python,features:{...features,'images.understand':value}}));
});
test('model management requires explicit host administration capability',()=>{
  const features={...fixtures.python.features};delete features['models.manage'];
  assert.equal(parseCapabilities({...fixtures.python,features}).features['models.manage'],false);
  assert.equal(parseCapabilities({...fixtures.python,features:{...features,'models.manage':true}}).features['models.manage'],true);
  assert.equal(parseCapabilities({...fixtures.python,features:{...features,'models.manage':false}}).features['models.manage'],false);
  for(const value of ['true',1,null])assert.throws(()=>parseCapabilities({...fixtures.python,features:{...features,'models.manage':value}}));
});
test('source reads need an independent explicit boolean capability',()=>{
  const features={...fixtures.python.features};delete features['episodes.read'];
  assert.equal(parseCapabilities({...fixtures.python,features}).features['episodes.read'],false);
  assert.equal(parseCapabilities({...fixtures.python,features:{...features,'episodes.read':true,'episodes.list':false}}).features['episodes.read'],true);
  for(const value of ['true',1,null])assert.throws(()=>parseCapabilities({...fixtures.python,features:{...features,'episodes.read':value}}));
});
test('metadata conditions require explicit native recall support',()=>{
  const features={...fixtures.python.features};delete features['recall.conditions'];
  assert.equal(parseCapabilities({...fixtures.python,features}).features['recall.conditions'],false);
  assert.equal(parseCapabilities({...fixtures.python,features:{...features,'recall.conditions':true}}).features['recall.conditions'],true);
  for(const value of ['true',1,null])assert.throws(()=>parseCapabilities({...fixtures.python,features:{...features,'recall.conditions':value}}));
});
test('job history reads require their own explicit boolean capability',()=>{
  const features={...fixtures.python.features};delete features['jobs.read'];
  assert.equal(parseCapabilities({...fixtures.python,features}).features['jobs.read'],false);
  assert.equal(parseCapabilities({...fixtures.python,features:{...features,'jobs.read':true}}).features['jobs.read'],true);
  for(const value of ['true',1,null])assert.throws(()=>parseCapabilities({...fixtures.python,features:{...features,'jobs.read':value}}));
});
test('processing actions require independent explicit boolean capabilities',()=>{
  for(const key of ['processing.distill','processing.derive'] as const){
    assert.equal(parseCapabilities(fixtures.python).features[key],false);
    assert.equal(parseCapabilities({...fixtures.python,features:{...fixtures.python.features,[key]:true}}).features[key],true);
    for(const value of ['true',1,null])assert.throws(()=>parseCapabilities({...fixtures.python,features:{...fixtures.python.features,[key]:value}}),/capabilit/i);
  }
});
test('profiles require an explicit boolean capability before evidence is read',()=>{
  const features={...fixtures.python.features};delete features['profile.read'];
  assert.equal(parseCapabilities({...fixtures.python,features}).features['profile.read'],false);
  assert.equal(parseCapabilities({...fixtures.python,features:{...features,'profile.read':true}}).features['profile.read'],true);
  for(const value of ['true',1,null])assert.throws(()=>parseCapabilities({...fixtures.python,features:{...features,'profile.read':value}}),/capabilit/i);
});
test('integrity checks require an explicit boolean capability',()=>{
  assert.equal(parseCapabilities({...fixtures.python,features:{...fixtures.python.features,'integrity.read':true}}).features['integrity.read'],true);
  assert.equal(parseCapabilities(fixtures.rust).features['integrity.read'],false);
  assert.throws(()=>parseCapabilities({...fixtures.python,features:{...fixtures.python.features,'integrity.read':'true'}}),/capabilit/i);
});
test('claim relationship inspection requires an explicit valid capability', () => {
  assert.equal(parseCapabilities(fixtures.python).features['facts.links'],true);
  assert.equal(parseCapabilities(fixtures.rust).features['facts.links'],false);
  for(const value of ['true',1,null])assert.throws(()=>parseCapabilities({...fixtures.python,features:{...fixtures.python.features,'facts.links':value}}),/capabilit/i);
});
test('source browsing requires an explicit valid optional inventory capability', () => {
  for (const native of [fixtures.rust, fixtures.python]) assert.equal(parseCapabilities(native).features['episodes.list'], true);
  const features = {...fixtures.python.features}; delete features['episodes.list'];
  assert.equal(parseCapabilities({...fixtures.python,features}).features['episodes.list'], false);
  assert.throws(()=>parseCapabilities({...fixtures.python,features:{...features,'episodes.list':'true'}}),/capabilit/i);
});
test('capability parser preserves explicit false and both native contracts', () => {
  assert.equal(parseCapabilities(fixtures.rust).features['facts.review'], true);
  assert.equal(parseCapabilities(fixtures.rust).features['facts.exclude'], false);
  assert.equal(parseCapabilities({...fixtures.rust, features:{...fixtures.rust.features,'facts.review':false}}).features['facts.review'], false);
  assert.equal(parseCapabilities(fixtures.python).features['facts.review'], true);
});
test('incomplete, mistyped and unknown-version capability contracts cannot authorize workflows', () => {
  for (const bad of [null, {}, {...fixtures.python, schema_version:2},
    {...fixtures.python, features:{}},
    {...fixtures.python, features:{...fixtures.python.features, 'facts.review':'false'}},
    {...fixtures.python, features:{...fixtures.python.features, 'facts.close':null}}]) {
    assert.throws(() => parseCapabilities(bad), /capabilit/i);
  }
});
test('additive boolean features remain forward compatible', () => {
  assert.equal(parseCapabilities({...fixtures.python, features:{...fixtures.python.features, 'attachments.read':true}}).features['facts.read'], true);
});
test('source upload requires explicit combined episode attachment support', () => {
  assert.equal(parseCapabilities(fixtures.rust).features['episodes.attachments'], false);
  assert.equal(parseCapabilities({...fixtures.python,features:{...fixtures.python.features,'episodes.attachments':true}}).features['episodes.attachments'], true);
  assert.throws(()=>parseCapabilities({...fixtures.python,features:{...fixtures.python.features,'episodes.attachments':'true'}}),/capabilit/i);
});
