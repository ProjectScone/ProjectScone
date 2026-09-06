import test from 'node:test';
import assert from 'node:assert/strict';
import {claimGroups,filterClaimGroups} from '../src/memory/claim-groups.ts';
import type {Fact} from '../src/memory/types';

const fact=(id:number,extra:Partial<Fact>={}):Fact=>({fact_id:id,subject:'Ada',predicate:'prefers',object:'local storage',confidence:1,status:'active',valid_from:'2020-01-01T00:00:00Z',valid_until:null,...extra});
const now=Date.parse('2026-09-06T12:00:00Z');

test('groups retain history and choose currently eligible records, not future or excluded versions',()=>{
  const groups=claimGroups([fact(1,{status:'closed',valid_until:'2021-01-01T00:00:00Z'}),fact(2,{valid_from:'2021-01-01T00:00:00Z'}),fact(3,{valid_from:'2030-01-01T00:00:00Z'}),fact(4,{excluded_reason:'incorrect'}),fact(5,{status:'proposed'}),fact(6,{status:'declined'})],now);
  assert.equal(groups.length,1);
  assert.deepEqual(groups[0].versions.map(f=>f.fact_id),[1,4,2,3]);
  assert.equal(groups[0].current?.fact_id,2);
  assert.equal(filterClaimGroups(groups,'','all')[0].displayed.fact_id,2);
});

test('search exposes a matching historical version without calling it current',()=>{
  const groups=claimGroups([fact(1,{object:'cloud storage',status:'closed',valid_until:'2021-01-01T00:00:00Z'}),fact(2,{valid_from:'2021-01-01T00:00:00Z'})],now);
  const result=filterClaimGroups(groups,'cloud','all');
  assert.equal(result.length,1);
  assert.equal(result[0].displayed.fact_id,1);
  assert.equal(result[0].current?.fact_id,2);
  assert.equal(result[0].versions.length,2);
  assert.equal(filterClaimGroups(groups,'cloud','current').length,0);
});

test('a closed interval still holds before its future successor starts',()=>{
  const groups=claimGroups([fact(1,{status:'closed',valid_until:'2030-01-01T00:00:00Z'}),fact(2,{object:'remote storage',valid_from:'2030-01-01T00:00:00Z'})],now);
  assert.equal(groups[0].current?.fact_id,1);
  assert.equal(filterClaimGroups(groups,'','current')[0].displayed.fact_id,1);
  assert.equal(filterClaimGroups(groups,'remote','all')[0].current?.fact_id,1);
  assert.equal(claimGroups(groups[0].versions,Date.parse('2030-01-01T00:00:00Z'))[0].current?.fact_id,2);
});

test('evidence and exclusion filters expose the matching version and preserve originals',()=>{
  const rows=[fact(1,{status:'closed',quote:null,valid_until:'2021-01-01T00:00:00Z'}),fact(2,{quote:'local storage',grounded:true,valid_from:'2021-01-01T00:00:00Z'}),fact(3,{excluded_reason:'wrong source',quote:'local storage'})];
  const before=JSON.stringify(rows),groups=claimGroups(rows,now);
  assert.equal(filterClaimGroups(groups,'','no-quote')[0].displayed.fact_id,1);
  assert.equal(filterClaimGroups(groups,'','excluded')[0].displayed.fact_id,3);
  assert.equal(JSON.stringify(rows),before);
});

test('subject and predicate grouping cannot collide and ordering is stable',()=>{
  const rows=[fact(1,{subject:'A:b',predicate:'c'}),fact(2,{subject:'A',predicate:'b:c'})];
  assert.equal(claimGroups(rows,now).length,2);
  assert.deepEqual(claimGroups(rows,now).map(g=>g.key),claimGroups([...rows].reverse(),now).map(g=>g.key));
});
