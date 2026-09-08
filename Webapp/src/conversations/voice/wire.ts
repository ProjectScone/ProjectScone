/** Scone's native WebSocket protocol; never reinterpret a provider's encoded audio. */
export interface VoiceAudio {turnId:string;sampleRate:number;channels:number;samples:Float32Array}
export type VoiceControl={type:'ready';sessionId:string}|{type:'clear';turnId:string}|{type:'error'};
const asciiId=(value:unknown):value is string=>typeof value==='string'&&/^[\x20-\x7e]{1,255}$/.test(value);
export function decodeAudio(data:ArrayBuffer):VoiceAudio {
  if(!(data instanceof ArrayBuffer)||data.byteLength<9||data.byteLength>1_000_000)throw Error('Invalid audio frame size');
  const view=new DataView(data),length=view.getUint8(0),offset=6+length;
  if(!length||offset>=data.byteLength)throw Error('Incomplete audio header');
  const turnId=String.fromCharCode(...new Uint8Array(data,1,length));
  const sampleRate=view.getUint32(1+length,true),channels=view.getUint8(5+length);
  if(!asciiId(turnId)||sampleRate<8000||sampleRate>192000||![1,2].includes(channels)
    ||(data.byteLength-offset)%(2*channels))throw Error('Unsupported audio format');
  const samples=new Float32Array((data.byteLength-offset)/2);
  for(let i=0;i<samples.length;i++)samples[i]=view.getInt16(offset+i*2,true)/32768;
  return {turnId,sampleRate,channels,samples};
}
export function parseVoiceControl(text:string,sessionId:string):VoiceControl {
  if(typeof text!=='string'||text.length>8192)throw Error('Invalid voice control size');
  const value:unknown=JSON.parse(text);
  if(!value||typeof value!=='object'||Array.isArray(value))throw Error('Invalid voice control');
  const control=value as Record<string,unknown>;
  if(control.type==='ready'&&control.session_id===sessionId)return {type:'ready',sessionId};
  if(control.type==='clear'&&asciiId(control.turn_id))return {type:'clear',turnId:control.turn_id};
  // Do not show or log arbitrary provider diagnostics, which may contain credentials.
  if(control.type==='error')return {type:'error'};
  throw Error('Unsupported voice control');
}
export function encodeMicrophone(samples:Float32Array):ArrayBuffer {
  if(!(samples instanceof Float32Array)||!samples.length||samples.length>32000)throw Error('Invalid microphone frame size');
  const data=new ArrayBuffer(samples.length*2),view=new DataView(data);
  for(let i=0;i<samples.length;i++){
    if(!Number.isFinite(samples[i]))throw Error('Invalid microphone sample');
    const sample=Math.max(-1,Math.min(1,samples[i]));
    view.setInt16(i*2,Math.round(sample*(sample<0?32768:32767)),true);
  }
  return data;
}
export function voiceSocketUrl(id:string,origin:string):string {
  if(!/^[A-Za-z0-9._:-]{1,128}$/.test(id)||id==='.'||id==='..')throw Error('Invalid voice session');
  const url=new URL(origin);
  const loopback=['localhost','127.0.0.1','[::1]'].includes(url.hostname);
  if(url.username||url.password||url.search||url.hash||url.pathname!=='/'
    ||!(url.protocol==='https:'||(url.protocol==='http:'&&loopback)))throw Error('Voice requires a secure origin');
  url.protocol=url.protocol==='https:'?'wss:':'ws:';
  url.pathname='/v1/conversations/'+encodeURIComponent(id)+'/audio';
  return url.href;
}
