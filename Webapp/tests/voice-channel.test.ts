import {test} from 'node:test';
import assert from 'node:assert/strict';
import {attachVoiceSocket} from '../src/conversations/voice/channel.ts';

class Peer extends EventTarget {
  readyState=0;binaryType='blob';bufferedAmount=0;sent:unknown[]=[];closed=0;
  send(data:unknown){this.sent.push(data);}
  close(){this.closed++;this.readyState=3;this.dispatchEvent(new Event('close'));}
  open(){this.readyState=1;this.dispatchEvent(new Event('open'));}
  receive(data:unknown){this.dispatchEvent(new MessageEvent('message',{data}));}
}
function setup(){
  const peer=new Peer(),abort=new AbortController(),states:string[]=[],audio:string[]=[],cleared:string[]=[];
  const channel=attachVoiceSocket(peer,'private-key','mine',{sampleRate:48000,channels:1},{
    state:state=>states.push(state),audio:value=>audio.push(value.turnId),clear:id=>cleared.push(id),
  },abort.signal);
  return {peer,abort,states,audio,cleared,channel};
}
const ready=JSON.stringify({type:'ready',session_id:'mine'});
function pcm(){const b=new ArrayBuffer(9),v=new DataView(b);v.setUint8(0,1);v.setUint8(1,116);v.setUint32(2,16000,true);v.setUint8(6,1);v.setInt16(7,1024,true);return b;}
test('voice sends authentication once and no microphone bytes before matching readiness',()=>{
  const {peer,channel,states}=setup();
  channel.send(new Float32Array([.5]));assert.equal(peer.sent.length,0);
  peer.open();assert.deepEqual(peer.sent,[JSON.stringify({type:'hello',key:'private-key',sample_rate:48000,channels:1})]);
  channel.send(new Float32Array([.5]));assert.equal(peer.sent.length,1);
  peer.receive(ready);channel.send(new Float32Array([.5]));
  assert.equal(new DataView(peer.sent[1] as ArrayBuffer).getInt16(0,true),16384);
  assert.deepEqual(states,['connecting','ready']);channel.stop();
  assert.equal(peer.sent[2],'{"type":"end"}');assert.equal(peer.closed,0);
  channel.stop();assert.equal(peer.sent.length,3);
  peer.close();assert.equal(peer.closed,1);
});
test('voice processes exact audio and interruption IDs only after readiness',()=>{
  const {peer,channel,audio,cleared}=setup();peer.open();peer.receive(ready);
  peer.receive(pcm());peer.receive('{"type":"clear","turn_id":"t"}');
  assert.deepEqual(audio,['t']);assert.deepEqual(cleared,['t']);channel.stop();
  peer.receive(pcm());assert.deepEqual(audio,['t'],'late socket events cannot resurrect stopped playback');peer.close();
});
test('Stop closes local media immediately but lets the server complete its end handshake',()=>{
  const {peer,channel,states,audio,cleared}=setup();peer.open();peer.receive(ready);
  channel.stop();
  assert.equal(states.at(-1),'closed','local capture must stop before server cleanup');
  assert.equal(peer.closed,0,'an end message must reach an open peer before transport closure');
  channel.send(new Float32Array([1]));peer.receive(pcm());peer.receive('{"type":"clear","turn_id":"t"}');
  assert.equal(peer.sent.length,2);assert.deepEqual(audio,[]);assert.deepEqual(cleared,[]);
  peer.close();assert.deepEqual(states,['connecting','ready','closed']);
});
test('a stalled end handshake cannot retain a socket indefinitely',async()=>{
  const {peer,channel,states}=setup();peer.open();peer.receive(ready);channel.stop();
  assert.equal(peer.closed,0);
  await new Promise(resolve=>setTimeout(resolve,2100));
  assert.equal(peer.closed,1);assert.deepEqual(states,['connecting','ready','closed']);
});
test('end-send failure falls back to immediate transport cleanup',()=>{
  const {peer,channel,states}=setup();peer.open();peer.receive(ready);
  peer.send=()=>{throw Error('closed peer');};
  channel.stop();assert.equal(peer.closed,1);assert.equal(states.at(-1),'closed');
});
test('invalid handshake, early PCM and peer errors close without disclosing raw diagnostics',()=>{
  for(const message of ['{"type":"ready","session_id":"other"}',pcm(),'{"type":"error","reason":"secret"}']){
    const {peer,states,channel}=setup();peer.open();peer.receive(message);
    assert.deepEqual(states,['connecting','failed']);assert.equal(peer.closed,1);
    channel.send(new Float32Array([1]));assert.equal(peer.sent.length,1);
  }
});
test('backpressure is a visible failure rather than an unbounded microphone queue',()=>{
  const {peer,states,channel}=setup();peer.open();peer.receive(ready);peer.bufferedAmount=128000;
  channel.send(new Float32Array([1]));assert.equal(states.at(-1),'failed');assert.equal(peer.closed,1);assert.equal(peer.sent.length,1);
});
test('abort before opening sends no key or audio, and disconnection never reconnects',()=>{
  const {peer,abort,channel,states}=setup();abort.abort();peer.open();peer.receive(ready);
  channel.send(new Float32Array([1]));assert.equal(peer.sent.length,0);assert.equal(peer.closed,1);assert.deepEqual(states,['connecting','closed']);
  const active=setup();active.peer.open();active.peer.receive(ready);active.peer.close();
  assert.deepEqual(active.states,['connecting','ready','closed']);assert.equal(active.peer.sent.length,1);
});
test('a stalled voice handshake releases the socket',async()=>{
  const {peer,states}=setup();peer.open();
  await new Promise(resolve=>setTimeout(resolve,5100));
  assert.equal(states.at(-1),'failed');assert.equal(peer.closed,1);
});
