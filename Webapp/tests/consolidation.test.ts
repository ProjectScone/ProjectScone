import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parsePassReceipt} from '../src/memory/consolidation.ts';

const receipt={space:'alpha',scope:'derive',groups:2,sent:1,proposed:1,restated:0,rejected:0,rejected_reasons:{},latency_ms:12};
test('processing receipts preserve counters and require the requested scope and space',()=>{
  const parsed=parsePassReceipt(receipt,'alpha','derive');
  assert.equal(parsed.counts[2][1],1);
  assert.equal(parsed.error,null);
  assert.throws(()=>parsePassReceipt(receipt,'beta','derive'));
  assert.throws(()=>parsePassReceipt(receipt,'alpha','distill'));
});
test('processing receipts reject incomplete and malformed responses',()=>{
  for(const bad of [null,[],{}, {...receipt,sent:undefined},{...receipt,proposed:-1},
    {...receipt,groups:0.5},{...receipt,rejected:NaN},{...receipt,latency_ms:Infinity},
    {...receipt,rejected_reasons:[]},{...receipt,rejected_reasons:{invalid:'1'}},
  ])assert.throws(()=>parsePassReceipt(bad,'alpha','derive'));
});
