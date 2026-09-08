import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parseProcessingStatus} from '../src/memory/processing.ts';

test('processing preserves the last extraction failure without inventing failed episode counts',()=>{
  const status=parseProcessingStatus({space:'alpha',last_distill:{error:'DistillError: 2 episode(s) failed'}});
  assert.equal(status.last_distill?.error,'DistillError: 2 episode(s) failed');
  assert.equal(status.failed_distill,undefined);
  assert.equal(parseProcessingStatus({space:'alpha',last_distill:null}).last_distill,null);
  assert.equal(parseProcessingStatus({space:'alpha'}).last_distill,undefined);
  assert.equal(parseProcessingStatus({space:'alpha',last_distill:{error:null}}).last_distill?.error,null);
  for(const last_distill of [[],true,{error:4}])assert.throws(()=>parseProcessingStatus({space:'alpha',last_distill}),/Invalid memory status/);
});

test('processing status preserves reported zero and leaves omitted counters unknown',()=>{
  const minimal=parseProcessingStatus({space:'alpha',episodes:0,pending_derivation:null});
  assert.equal(minimal.episodes,0);assert.equal(minimal.pending_derivation,undefined);
  assert.equal(minimal.pending_review,undefined);
  const python=parseProcessingStatus({space:'alpha',episodes:5,chunks:12,revision:7,pending_distill:2,pending_derivation:1,pending_review:3,semantic_lane:'stopped',derivation:'on'});
  assert.equal(python.pending_distill,2);assert.equal(python.pending_derivation,1);assert.equal(python.derivation,'on');
  const rust=parseProcessingStatus({space:'alpha',episodes:5,chunks:12,revision:7,pending_distill:20,semantic_lane:'paused'});
  assert.equal(rust.pending_distill,20);assert.equal(rust.derivation,undefined);
});
test('processing status rejects malformed counters and metadata instead of rendering false status',()=>{
  for(const bad of [null,[],{}, {space:''},{space:'alpha',episodes:-1},{space:'alpha',chunks:'12'},
    {space:'alpha',pending_distill:1.2},{space:'alpha',pending_review:NaN},
    {space:'alpha',pending_derivation:Infinity},{space:'alpha',revision:Number.MAX_SAFE_INTEGER+1},
    {space:'alpha',derivation:true},{space:'alpha',semantic_lane:[]},{space:'alpha',embedder:{}},
  ])assert.throws(()=>parseProcessingStatus(bad),/Invalid memory status/);
});
test('unknown processing modes remain visible without becoming enabled defaults',()=>{
  const result=parseProcessingStatus({space:'alpha',semantic_lane:'maintenance',derivation:'suspended'});
  assert.equal(result.semantic_lane,'maintenance');assert.equal(result.derivation,'suspended');
});

test('processing keeps failed work separate from pending and preserves explicit model absence',()=>{
  const configured=parseProcessingStatus({space:'alpha',pending_distill:0,failed_distill:2,model:'local-extractor'});
  assert.equal(configured.failed_distill,2);assert.equal(configured.pending_distill,0);
  assert.equal(configured.model,'local-extractor');
  assert.equal(parseProcessingStatus({space:'alpha',model:null}).model,null);
  assert.equal(parseProcessingStatus({space:'alpha'}).model,undefined);
  for(const bad of [{space:'alpha',failed_distill:-1},{space:'alpha',failed_distill:'2'},{space:'alpha',model:42},{space:'alpha',model:''}])assert.throws(()=>parseProcessingStatus(bad),/Invalid memory status/);
});
