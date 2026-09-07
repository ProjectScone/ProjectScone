import {test} from 'node:test';
import assert from 'node:assert/strict';
import {VoicePlayback} from '../src/conversations/voice/playback.ts';
class Output {
  buffer:Buffer|null=null;onended:(()=>void)|null=null;startAt=-1;stopped=false;disconnected=false;
  connect(){}start(time:number){this.startAt=time;}stop(){this.stopped=true;}disconnect(){this.disconnected=true;}
}
class Buffer {
  channels:Float32Array[];duration:number;
  constructor(channels:number,length:number,rate:number){this.channels=Array.from({length:channels},()=>new Float32Array(length));this.duration=length/rate;}
  copyToChannel(data:Float32Array,channel:number){this.channels[channel].set(data);}
}
class Device {
  currentTime=10;state='running';destination={};outputs:Output[]=[];formats:number[][]=[];
  createBuffer(channels:number,length:number,rate:number){this.formats.push([channels,length,rate]);return new Buffer(channels,length,rate);}
  createBufferSource(){const node=new Output();this.outputs.push(node);return node;}
}
function setup(){const device=new Device();return {device,queue:new VoicePlayback(device as unknown as AudioContext)};}
const audio=(turnId='t1',seconds=.02)=>({turnId,sampleRate:16000,channels:1,samples:new Float32Array(Math.round(16000*seconds))});
test('playback retains sample rate and deinterleaves stereo rather than assuming mono',()=>{
  const {device,queue}=setup();queue.enqueue({turnId:'t',sampleRate:48000,channels:2,samples:new Float32Array([.25,.5,-.25,-.5])});
  assert.deepEqual(device.formats,[[2,2,48000]]);
  assert.deepEqual(device.outputs[0].buffer?.channels.map(c=>[...c]),[[.25,-.25],[.5,-.5]]);
  queue.stop();
});
test('PCM chunks play in arrival order and interruption frees only the named turn',()=>{
  const {device,queue}=setup();queue.enqueue(audio('a',.1));queue.enqueue(audio('a',.1));queue.enqueue(audio('b',.1));
  assert.ok(device.outputs[1].startAt>=device.outputs[0].startAt+.1);
  assert.ok(device.outputs[2].startAt>=device.outputs[1].startAt+.1);
  queue.clear('a');assert.deepEqual(device.outputs.map(n=>n.stopped),[true,true,false]);
  queue.stop();assert.equal(device.outputs.every(n=>n.stopped&&n.disconnected),true);
  assert.throws(()=>queue.enqueue(audio()),'stopped output cannot resume');
});
test('played nodes are released and a cleared queue does not leave a silent scheduling gap',()=>{
  const {device,queue}=setup();queue.enqueue(audio('old',2));queue.clear('old');queue.enqueue(audio('new'));
  assert.ok(device.outputs[1].startAt<10.1,'interruption immediately releases scheduled time');
  device.outputs[1].onended?.();assert.equal(device.outputs[1].disconnected,true);
  queue.stop();
});
test('playback limits queued duration and tiny frame floods before allocating another buffer',()=>{
  const {device,queue}=setup();assert.throws(()=>queue.enqueue(audio('long',9)));assert.equal(device.outputs.length,0);
  for(let i=0;i<128;i++)queue.enqueue(audio(String(i),1/16000));
  assert.throws(()=>queue.enqueue(audio('overflow',1/16000)));assert.equal(device.outputs.length,128);queue.stop();
});
test('suspended output is not falsely reported as playing',()=>{
  const {device,queue}=setup();device.state='suspended';assert.throws(()=>queue.enqueue(audio()));assert.equal(device.outputs.length,0);queue.stop();
});
