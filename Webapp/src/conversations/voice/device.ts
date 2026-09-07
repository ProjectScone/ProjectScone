import type {VoiceEvents} from './channel.ts';
import {VoicePlayback} from './playback.ts';
import {installCaptureProcessor} from './capture-worklet.ts';
export type VoiceDeviceState='permission'|'connecting'|'listening'|'muted'|'closed'|'failed';
export type VoiceConnector=(format:{sampleRate:number;channels:number},events:VoiceEvents,signal:AbortSignal)=>{send:(samples:Float32Array)=>void;stop:()=>void};
/** Call directly from a user gesture. Never invoke on mount or on reconnect. */
export function startVoiceDevice(connect:VoiceConnector,state:(state:VoiceDeviceState)=>void,signal:AbortSignal){
  let context:AudioContext|undefined,stream:MediaStream|undefined,source:MediaStreamAudioSourceNode|undefined;
  let processor:AudioWorkletNode|undefined,silent:GainNode|undefined,playback:VoicePlayback|undefined;
  let connection:ReturnType<VoiceConnector>|undefined,stopped=false,muted=false,listening=false,moduleUrl:string|undefined;
  const stopTracks=(value:MediaStream)=>{for(const track of value.getTracks()){track.removeEventListener('ended',failed);track.stop();}};
  const finish=(verdict:'closed'|'failed')=>{
    if(stopped)return;stopped=true;
    signal.removeEventListener('abort',stop);window.removeEventListener('pagehide',stop);
    if(stream)stopTracks(stream);
    if(processor){processor.port.onmessage=null;processor.port.close();processor.onprocessorerror=null;processor.disconnect();}
    source?.disconnect();silent?.disconnect();playback?.stop();connection?.stop();
    if(moduleUrl){URL.revokeObjectURL(moduleUrl);moduleUrl=undefined;}
    if(context){context.removeEventListener('statechange',contextChanged);void context.close().catch(()=>{});}
    state(verdict);
  };
  const stop=()=>finish('closed'),failed=()=>finish('failed');
  const contextChanged=()=>{if(listening&&context?.state!=='running')failed();};
  signal.addEventListener('abort',stop,{once:true});window.addEventListener('pagehide',stop);
  state('permission');
  const ready=(async()=>{
    try{
      if(signal.aborted){stop();return;}
      if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia||!window.AudioContext)throw Error('Browser audio unavailable');
      // Resume during the originating click, before a permission dialog yields.
      context=new AudioContext({latencyHint:'interactive'});
      const resume=context.resume();
      const permission=navigator.mediaDevices.getUserMedia({audio:{channelCount:{ideal:1},echoCancellation:true,noiseSuppression:true},video:false}).then(value=>{
        if(stopped){stopTracks(value);return;}
        stream=value;for(const track of stream.getTracks())track.addEventListener('ended',failed,{once:true});
      });
      await Promise.all([resume,permission]);
      if(stopped)return;
      if(!stream||context.state!=='running'||!context.audioWorklet)throw Error('Microphone or audio output unavailable');
      moduleUrl=URL.createObjectURL(new Blob(['('+installCaptureProcessor.toString()+')()'],{type:'text/javascript'}));
      await context.audioWorklet.addModule(moduleUrl);
      if(moduleUrl){URL.revokeObjectURL(moduleUrl);moduleUrl=undefined;}
      if(stopped)return;
      playback=new VoicePlayback(context);
      processor=new AudioWorkletNode(context,'scone-microphone',{numberOfInputs:1,numberOfOutputs:1,outputChannelCount:[1],channelCount:1,channelCountMode:'explicit'});
      processor.onprocessorerror=failed;
      processor.port.onmessage=event=>{
        if(stopped)return;
        if(event.data?.type!=='samples'||!(event.data.samples instanceof Float32Array)){failed();return;}
        processor?.port.postMessage('ack');
        if(!muted)connection?.send(event.data.samples);
      };
      source=context.createMediaStreamSource(stream);silent=context.createGain();silent.gain.value=0;
      source.connect(processor);processor.connect(silent);silent.connect(context.destination);
      context.addEventListener('statechange',contextChanged);
      connection=connect({sampleRate:context.sampleRate,channels:1},{
        state:value=>{
          if(stopped)return;
          if(value==='failed'||value==='closed'){finish(value);return;}
          if(value==='ready'&&context?.state!=='running'){failed();return;}
          listening=value==='ready';state(listening?(muted?'muted':'listening'):'connecting');
        },
        audio:audio=>{if(!stopped)playback?.enqueue(audio);},
        clear:turnId=>{if(!stopped)playback?.clear(turnId);},
      },signal);
      if(stopped)connection.stop(); // A connector may fail synchronously.
    }catch{failed();}
  })();
  return {ready,stop,mute(value:boolean){
    if(stopped||!listening)return;
    muted=value;
    for(const track of stream?.getAudioTracks()??[])track.enabled=!muted;
    state(muted?'muted':'listening');
  }};
}
