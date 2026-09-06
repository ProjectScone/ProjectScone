import {test} from 'node:test';
import assert from 'node:assert/strict';
import {capabilities, session, transcript, turnReceipt} from '../src/conversations/contracts.ts';

test('live text requires an explicit supported SSE active-window contract',()=>{
  const wire={schema_version:1,text_configured:true,reply_transport:'poll',reply_replay:'durable_receipts'};
  const text_stream={transport:'sse',replay:'active_window',max_bytes:65536,max_chunks:256};
  assert.equal(capabilities({...wire,streaming:true,text_stream}).streaming,true);
  for(const extra of [{},{streaming:'true',text_stream},{streaming:false,text_stream},{streaming:true},
    {streaming:true,text_stream:{...text_stream,transport:'websocket'}},
    {streaming:true,text_stream:{...text_stream,replay:'durable'}},
    {streaming:true,text_stream:{...text_stream,max_bytes:0}},
    {streaming:true,text_stream:{...text_stream,max_chunks:'256'}}]){
    assert.equal(capabilities({...wire,...extra}).streaming,false);
  }
});

test('recall scope requires explicit support and missing scope remains unknown',()=>{
  const wire={schema_version:1,text_configured:true,reply_transport:'poll',reply_replay:'durable_receipts'};
  for(const value of [undefined,false,'true',1])assert.equal(capabilities({...wire,recall_scope:value}).recall_scope,false);
  assert.equal(capabilities({...wire,recall_scope:true}).recall_scope,true);
  const saved={session_id:'scoped',space:'alpha',state:'running',revision:2,created_at:'2026-09-06T12:00:00Z'};
  assert.equal(session(saved).recall_scope,undefined);
  assert.deepEqual(session({...saved,recall_scope:{}}).recall_scope,{});
  assert.deepEqual(session({...saved,recall_scope:{source_prefix:''}}).recall_scope,{source_prefix:''});
  for(const scope of [null,[],{space:'beta'},{kind:'invented'},{where:{collection:2}},{since:'tomorrow'}])assert.throws(()=>session({...saved,recall_scope:scope}));
});

test('conversation deletion requires an explicit true capability',()=>{
  const wire={schema_version:1,text_configured:false,reply_transport:'poll',reply_replay:'durable_receipts'};
  for(const value of [undefined,false,'true',1,null])assert.equal(capabilities({...wire,session_deletion:value}).session_deletion,false);
  assert.equal(capabilities({...wire,session_deletion:true}).session_deletion,true);
});

test('reply cancellation is enabled only by an explicit capability',()=>{
  const wire={schema_version:1,text_configured:true,reply_transport:'poll',reply_replay:'durable_receipts'};
  for(const value of [undefined,false,'true',1])assert.equal(capabilities({...wire,turn_cancellation:value}).turn_cancellation,false);
  assert.equal(capabilities({...wire,turn_cancellation:true}).turn_cancellation,true);
});

test('transcript pagination requires explicit support and valid advancing cursors',()=>{
  const wire={schema_version:1,text_configured:true,reply_transport:'poll',reply_replay:'durable_receipts'};
  for(const value of [undefined,false,'true',1])assert.equal(capabilities({...wire,transcript_pagination:value}).transcript_pagination,false);
  assert.equal(capabilities({...wire,transcript_pagination:true}).transcript_pagination,true);
  const page={episodes:[{episode_id:2,content:'Message',metadata:{}}],has_more:true,next_before:'valid_cursor-123'};
  assert.equal(transcript(page).next_before,'valid_cursor-123');
  assert.equal(transcript({episodes:[],has_more:false,next_before:null}).next_before,null);
  // Legacy partial transcripts stay readable, but cannot invent a paging cursor.
  assert.equal(transcript({episodes:[],has_more:true}).next_before,null);
  for(const invalid of ['',123,'../secret','a'.repeat(1025)])assert.throws(()=>transcript({...page,next_before:invalid}));
  assert.throws(()=>transcript({...page,has_more:false}));
  assert.throws(()=>transcript({...page,episodes:[]}));
});

test('a cancelled turn is a settled receipt, not missing work to resend',()=>{
  const receipt=turnReceipt({request_id:'cancelled-turn',status:'cancelled',result_state:'unavailable',result:null});
  assert.equal(receipt.status,'cancelled');assert.equal(receipt.result,undefined);
});

test('conversation capability checks do not enable a string boolean or unknown protocol',()=>{
  const wire={schema_version:1,text_configured:true,reply_transport:'poll',reply_replay:'process_lifetime'};
  assert.equal(capabilities(wire).text_configured,true);
  assert.throws(()=>capabilities({...wire,text_configured:'false'}));
  assert.throws(()=>capabilities({...wire,schema_version:2}));
  assert.throws(()=>capabilities({...wire,reply_transport:'stream'}));
  assert.equal(capabilities({...wire,reply_replay:'durable_receipts'}).text_configured,true);
  assert.throws(()=>capabilities({...wire,reply_replay:'anything'}));
  assert.throws(()=>capabilities({...wire,reply_replay:['durable_receipts']}));
});
test('session controls require a valid ID, known state and positive lifecycle revision',()=>{
  const wire={session_id:'abc-123',space:'alpha',state:'running',revision:2,created_at:'2026-09-06T12:00:00Z'};
  assert.equal(session(wire).state,'running');
  assert.equal(session({...wire,latest_request_id:'newest-turn'}).latest_request_id,'newest-turn');
  assert.equal(session(wire).latest_request_id,null);
  assert.throws(()=>session({...wire,latest_request_id:'../bad'}));
  for(const invalid of [{session_id:'../secret'},{state:'live'},{revision:0},{revision:'2'},{active_request_id:'../bad'}])assert.throws(()=>session({...wire,...invalid}));
});
test('transcripts keep public content as text and reject unsafe source identifiers',()=>{
  const wire={episodes:[{episode_id:7,content:'<script>not markup</script>',metadata:{role:'assistant'}}],has_more:false};
  assert.equal(transcript(wire).episodes[0].content,'<script>not markup</script>');
  assert.throws(()=>transcript({...wire,episodes:[{...wire.episodes[0],episode_id:'../../bad'}]}));
  assert.throws(()=>transcript({...wire,has_more:'false'}));
});
test('successful turn receipt requires aggregate text and validates source references',()=>{
  const wire={request_id:'turn-1',status:'completed',result:{text:'Reply',assistant_episode_id:7,memory_context:{status:'prepared',references:[{episode_id:2,chunk_id:3}]}}};
  assert.equal(turnReceipt(wire).result?.text,'Reply');
  assert.throws(()=>turnReceipt({...wire,result:{text:42}}));
  assert.throws(()=>turnReceipt({...wire,result:{text:'ok',assistant_episode_id:-1}}));
  assert.throws(()=>turnReceipt({...wire,result:{text:'ok',memory_context:{status:'prepared',references:[{episode_id:'../bad'}]}}}));
});

test('completed receipts can explicitly report forgotten or unavailable text without becoming failed',()=>{
  for(const result_state of ['forgotten','unavailable','unreadable']){
    const value=turnReceipt({request_id:'turn-1',status:'completed',result_state,result:null});
    assert.equal(value.status,'completed');
    assert.equal(value.result_state,result_state);
    assert.equal(value.result,undefined);
  }
  const available=turnReceipt({request_id:'turn-2',status:'completed',result_state:'available',result:{text:'Saved reply',assistant_episode_id:7}});
  assert.equal(available.result?.text,'Saved reply');
  const interrupted=turnReceipt({request_id:'turn-3',status:'interrupted',result_state:'unavailable',result:null});
  assert.equal(interrupted.status,'interrupted');
  assert.equal(interrupted.result,undefined);
  assert.equal(turnReceipt({request_id:'turn-4',status:'pending',result_state:null,result:null}).status,'pending');
});

test('receipt availability rejects contradictions and cannot silently drop unrecognized states',()=>{
  const completed={request_id:'turn-1',status:'completed'};
  for(const invalid of [
    {result_state:'unknown',result:null},
    {result_state:['unavailable'],result:null},
    {result_state:'available',result:{text:'No source ID'}},
    {result_state:'available',result:null},
    {result_state:'forgotten',result:{text:'stale text'}},
    {result_state:'unavailable',result:{text:'stale text'}},
    {status:'pending',result_state:'forgotten',result:null},
    {status:'failed',result_state:'available',result:{text:'contradictory'}},
    {status:'pending',result_state:'unreadable',result:null},
    {result_state:null,result:null},
    {result:null},
  ])assert.throws(()=>turnReceipt({...completed,...invalid}));
});
