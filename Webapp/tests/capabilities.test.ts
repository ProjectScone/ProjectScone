import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseCapabilities } from '../src/capabilities.ts';

const fixtures = JSON.parse(readFileSync(new URL('../../tests/fixtures/http-capabilities.json', import.meta.url), 'utf8'));
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
