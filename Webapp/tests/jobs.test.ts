import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parseJobPage} from '../src/memory/jobs.ts';

const at='2026-09-07T12:00:00.000Z';
const item={index:0,episode_id:7,outcome:'accepted',state:'searchable',searchable_at:at,consolidated_at:null,error:null,attempts:0};
const job={job_id:'batch-a',space:'alpha',created_at:at,request_id:null,cancelled_at:null,items:[item],searchable:1,consolidated:0,state:'searchable'};
test('job history preserves searchable and consolidated as separate receipts',()=>{
  const page=parseJobPage({jobs:[job],next:null},'alpha');
  assert.equal(page.jobs[0].searchable,1);assert.equal(page.jobs[0].consolidated,0);
  assert.equal(page.jobs[0].items[0].episode_id,7);assert.equal(page.next,null);
  const failed={...job,state:'failed',items:[{...item,state:'failed',attempts:2,error:'Extractor unavailable'}]};
  assert.equal(parseJobPage({jobs:[failed],next:null},'alpha').jobs[0].items[0].attempts,2);
  const cancelled={...failed,state:'cancelled',cancelled_at:at};
  assert.equal(parseJobPage({jobs:[cancelled],next:null},'alpha').jobs[0].searchable,1);
});
test('job history rejects cross-space and inconsistent receipt totals or states',()=>{
  for(const changed of [{space:'beta'},{searchable:0},{consolidated:1},{state:'consolidated'},{cancelled_at:at},{searchable:-1},{job_id:'../secret'}]){
    assert.throws(()=>parseJobPage({jobs:[{...job,...changed}],next:null},'alpha'));
  }
});
test('job items retain exact identities and reject malformed lifecycle fields',()=>{
  for(const changed of [{episode_id:0},{episode_id:Number.MAX_SAFE_INTEGER+1},{index:1},{attempts:-1},{state:'done'},{outcome:'made-up'},{searchable_at:null},{searchable_at:'yesterday'},{consolidated_at:at},{error:3}]){
    assert.throws(()=>parseJobPage({jobs:[{...job,items:[{...item,...changed}]}],next:null},'alpha'));
  }
  assert.throws(()=>parseJobPage({jobs:[{...job,created_at:'2026-02-30T12:00:00Z'}],next:null},'alpha'));
});
test('job cursors advance using server identity without treating UUIDs as chronological',()=>{
  const page=parseJobPage({jobs:[{...job,job_id:'z'},{...job,job_id:'a'}],next:'a'},'alpha',['previous']);
  assert.equal(page.next,'a');
  for(const value of [{jobs:[job],next:'elsewhere'},{jobs:[job],next:'batch-a'},{jobs:[job,job],next:null},{jobs:[],next:'batch-a'}]){
    assert.throws(()=>parseJobPage(value,'alpha',['batch-a']));
  }
  assert.throws(()=>parseJobPage({jobs:Array.from({length:21},(_,i)=>({...job,job_id:String(i)})),next:null},'alpha'));
});
test('job enum arrays cannot masquerade as strings or hide failed items',()=>{
  for(const changed of [{outcome:['accepted']},{state:['failed']},{state:['searchable']}]){
    assert.throws(()=>parseJobPage({jobs:[{...job,items:[{...item,...changed}]}],next:null},'alpha'));
  }
});
