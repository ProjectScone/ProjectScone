import {test} from 'node:test';
import assert from 'node:assert/strict';
import {runInNewContext} from 'node:vm';
import {installCaptureProcessor} from '../src/conversations/voice/capture-worklet.ts';
function capture(rate=48000){
  const messages:any[]=[];let Processor:any;
  class Worklet {port={onmessage:null as any,postMessage:(value:any)=>messages.push(value)}}
  runInNewContext('('+installCaptureProcessor.toString()+')()',{
    sampleRate:rate,Float32Array,AudioWorkletProcessor:Worklet,
    registerProcessor:(name:string,processor:any)=>{assert.equal(name,'scone-microphone');Processor=processor;},
  });
  return {processor:new Processor(),messages};
}
test('capture batches exact 20ms mono samples across rendering blocks',()=>{
  for(const rate of [16000,44100,48000,192000]){
    const {processor,messages}=capture(rate),length=Math.round(rate*.02);
    const samples=Float32Array.from({length},(_,i)=>i/length);
    processor.process([[samples.subarray(0,128)]]);assert.equal(messages.length,0);
    processor.process([[samples.subarray(128)]]);
    assert.equal(messages.length,1);assert.deepEqual([...messages[0].samples],[...samples]);
    processor.port.onmessage({data:'ack'});assert.equal(processor.outstanding,0);
  }
});
test('capture stops on main-thread backlog rather than accumulating audio indefinitely',()=>{
  const {processor,messages}=capture(16000),samples=new Float32Array(320);
  for(let i=0;i<4;i++)assert.equal(processor.process([[samples]]),true);
  assert.equal(processor.process([[samples]]),false);
  assert.equal(messages.length,5);assert.equal(messages[4].type,'overrun');
  assert.equal(processor.process([[samples]]),false);assert.equal(messages.length,5);
});
