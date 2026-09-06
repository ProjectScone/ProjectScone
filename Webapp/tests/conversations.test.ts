import {test} from 'node:test';
import assert from 'node:assert/strict';
import {capabilities, session, transcript, turnReceipt} from '../src/conversations/contracts.ts';

test('conversation capability checks do not enable a string boolean or unknown protocol',()=>{
  const wire={schema_version:1,text_configured:true,reply_transport:'poll',reply_replay:'process_lifetime'};
  assert.equal(capabilities(wire).text_configured,true);
  assert.throws(()=>capabilities({...wire,text_configured:'false'}));
  assert.throws(()=>capabilities({...wire,schema_version:2}));
  assert.throws(()=>capabilities({...wire,reply_transport:'stream'}));
});
test('session controls require a valid ID, known state and positive lifecycle revision',()=>{
  const wire={session_id:'abc-123',space:'alpha',state:'running',revision:2,created_at:'2026-09-06T12:00:00Z'};
  assert.equal(session(wire).state,'running');
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
