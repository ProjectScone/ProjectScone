import {test} from 'node:test';
import assert from 'node:assert/strict';
import {decodeAudio,parseVoiceControl,encodeMicrophone,voiceSocketUrl} from '../src/conversations/voice/wire.ts';

function frame(id='turn-1',rate=16000,channels=1,samples=[-32768,0,32767]){
  const bytes=new ArrayBuffer(1+id.length+5+samples.length*2),view=new DataView(bytes);
  view.setUint8(0,id.length);for(let i=0;i<id.length;i++)view.setUint8(i+1,id.charCodeAt(i));
  view.setUint32(1+id.length,rate,true);view.setUint8(5+id.length,channels);
  samples.forEach((s,i)=>view.setInt16(6+id.length+i*2,s,true));return bytes;
}
test('voice frames retain turn identity, explicit PCM rate/channels and signed samples',()=>{
  const audio=decodeAudio(frame());assert.equal(audio.turnId,'turn-1');assert.equal(audio.sampleRate,16000);assert.equal(audio.channels,1);
  assert.deepEqual([...audio.samples],[-1,0,32767/32768]);
  assert.equal(decodeAudio(frame('stereo',48000,2,[1,2,3,4])).channels,2);
});
test('malformed, empty, oversized and misaligned audio never reaches playback',()=>{
  for(const data of [new ArrayBuffer(0),frame('',16000),frame('x',1),frame('x',16000,3),frame('x',16000,2),frame('x',16000,1,[]),frame().slice(0,-1),new ArrayBuffer(1_000_001),frame('\u00ff')])assert.throws(()=>decodeAudio(data));
});
test('voice control is bounded and rejects readiness for another session or unknown instructions',()=>{
  assert.deepEqual(parseVoiceControl('{"type":"ready","session_id":"mine"}','mine'),{type:'ready',sessionId:'mine'});
  assert.deepEqual(parseVoiceControl('{"type":"clear","turn_id":"t1"}','mine'),{type:'clear',turnId:'t1'});
  assert.deepEqual(parseVoiceControl('{"type":"error","reason":"provider secret"}','mine'),{type:'error'});
  for(const value of ['{}','[]','bad','{"type":"ready","session_id":"other"}','{"type":"clear","turn_id":""}','{"type":"redirect","url":"https://evil.invalid"}',' '.repeat(8193)])assert.throws(()=>parseVoiceControl(value,'mine'));
});
test('microphone encoding is explicit signed little-endian mono PCM with finite bounded samples',()=>{
  const data=encodeMicrophone(new Float32Array([-2,-1,-.5,0,.5,1,2]));const view=new DataView(data);
  assert.deepEqual(Array.from({length:7},(_,i)=>view.getInt16(i*2,true)),[-32768,-32768,-16384,0,16384,32767,32767]);
  for(const values of [new Float32Array(),new Float32Array([NaN]),new Float32Array([Infinity]),new Float32Array(32001)])assert.throws(()=>encodeMicrophone(values));
});
test('socket addresses never carry keys or leave the page origin; remote plaintext is refused',()=>{
  assert.equal(voiceSocketUrl('s1','https://scone.example'),'wss://scone.example/v1/conversations/s1/audio');
  assert.equal(voiceSocketUrl('s1','http://127.0.0.1:7437'),'ws://127.0.0.1:7437/v1/conversations/s1/audio');
  for(const id of ['..','.','a/b','https://evil.invalid','x?key=secret'])assert.throws(()=>voiceSocketUrl(id,'https://scone.example'));
  for(const origin of ['http://scone.example','https://name:secret@scone.example','file:///tmp/index.html'])assert.throws(()=>voiceSocketUrl('s1',origin));
});
