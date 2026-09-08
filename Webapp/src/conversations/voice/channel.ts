import {decodeAudio,encodeMicrophone,parseVoiceControl,type VoiceAudio} from './wire.ts';
export interface VoiceSocket extends EventTarget {readyState:number;binaryType:string;bufferedAmount:number;send(data:string|ArrayBuffer):void;close():void}
export interface VoiceEvents {state:(state:'connecting'|'ready'|'closed'|'failed')=>void;audio:(audio:VoiceAudio)=>void;clear:(turnId:string)=>void}

/** Stop media immediately; retain only the socket for a bounded peer close. */
function endSocket(socket:VoiceSocket){
  const release=()=>{
    clearTimeout(deadline);
    socket.removeEventListener('close',release);socket.removeEventListener('error',close);
  };
  const close=()=>{release();if(socket.readyState<2)socket.close();};
  const deadline=setTimeout(close,2000);
  socket.addEventListener('close',release,{once:true});socket.addEventListener('error',close,{once:true});
  try{socket.send('{"type":"end"}');}catch{close();}
}

export function attachVoiceSocket(socket:VoiceSocket,key:string,sessionId:string,format:{sampleRate:number;channels:number},events:VoiceEvents,signal:AbortSignal){
  if(!Number.isInteger(format.sampleRate)||format.sampleRate<8000||format.sampleRate>192000||format.channels!==1){
    socket.close();throw Error('Unsupported microphone format');
  }
  let ready=false,closed=false,helloSent=false;
  socket.binaryType='arraybuffer';
  const timeout=setTimeout(()=>finish('failed'),5000);
  const finish=(state:'closed'|'failed',notifyEnd=false)=>{
    if(closed)return;
    closed=true;key='';clearTimeout(timeout);
    socket.removeEventListener('open',onOpen);socket.removeEventListener('message',onMessage);
    socket.removeEventListener('error',onError);socket.removeEventListener('close',onClose);
    signal.removeEventListener('abort',onAbort);
    try{
      if(notifyEnd&&ready&&socket.readyState===1)endSocket(socket);
      else if(socket.readyState<2)socket.close();
    }finally{events.state(state);}
  };
  const onOpen=()=>{
    if(closed||helloSent)return;
    helloSent=true;
    try{socket.send(JSON.stringify({type:'hello',key,sample_rate:format.sampleRate,channels:format.channels}));}
    catch{finish('failed');}
    finally{key='';}
  };
  const onMessage=(event:Event)=>{
    if(closed)return;
    try{
      const data=(event as MessageEvent).data;
      if(typeof data==='string'){
        const control=parseVoiceControl(data,sessionId);
        if(control.type==='error'){finish('failed');return;}
        if(control.type==='ready'){
          if(ready||!helloSent)throw Error('Unexpected readiness');
          ready=true;clearTimeout(timeout);events.state('ready');
        }else{
          if(!ready)throw Error('Early interruption');
          events.clear(control.turnId);
        }
      }else{
        if(!ready)throw Error('Early audio');
        events.audio(decodeAudio(data));
      }
    }catch{finish('failed');}
  };
  const onError=()=>finish('failed'),onClose=()=>finish('closed'),onAbort=()=>finish('closed',true);
  socket.addEventListener('open',onOpen);socket.addEventListener('message',onMessage);
  socket.addEventListener('error',onError);socket.addEventListener('close',onClose);
  signal.addEventListener('abort',onAbort,{once:true});
  events.state('connecting');
  if(signal.aborted)onAbort();else if(socket.readyState===1)onOpen();else if(socket.readyState>1)finish('failed');
  return {
    send(samples:Float32Array){
      // Permission may arrive before authentication; no microphone data is queued.
      if(closed||!ready)return;
      try{
        const data=encodeMicrophone(samples);
        if(socket.readyState!==1||socket.bufferedAmount+data.byteLength>128000)throw Error('Microphone queue exceeded');
        socket.send(data);
      }catch{finish('failed');}
    },
    stop:()=>finish('closed',true),
  };
}
