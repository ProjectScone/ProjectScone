import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createApiClient} from '../src/api.ts';
test('API voice keeps its key out of the socket URL and authenticates only after opening',async()=>{
  const previous=globalThis.WebSocket,peers:Peer[]=[];
  class Peer extends EventTarget {
    url:string;readyState=0;binaryType='blob';bufferedAmount=0;sent:string[]=[];
    constructor(url:string){super();this.url=url;peers.push(this);}
    send(value:string){this.sent.push(value);}close(){this.readyState=3;}
  }
  globalThis.WebSocket=Peer as unknown as typeof WebSocket;
  try{
    const abort=new AbortController(),states:string[]=[];
    const api=createApiClient('only-in-hello',()=>{},'https://memory.example');
    const connection=api.voiceConnection('sid',{sampleRate:16000,channels:1},{state:value=>states.push(value),audio:()=>{},clear:()=>{}},abort.signal);
    assert.equal(peers[0].url,'wss://memory.example/v1/conversations/sid/audio');assert.equal(peers[0].sent.length,0);
    peers[0].readyState=1;peers[0].dispatchEvent(new Event('open'));
    assert.equal(JSON.parse(peers[0].sent[0]).key,'only-in-hello');
    connection.stop();assert.equal(peers[0].readyState,3);
    abort.abort();assert.throws(()=>api.voiceConnection('sid',{sampleRate:16000,channels:1},{state:()=>{},audio:()=>{},clear:()=>{}},abort.signal));
    assert.equal(peers.length,1,'an already cancelled connection must not open a socket');
  }finally{globalThis.WebSocket=previous;}
});
