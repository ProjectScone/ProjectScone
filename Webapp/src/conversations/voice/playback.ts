import type {VoiceAudio} from './wire.ts';
export class VoicePlayback {
  private context:AudioContext;
  private pending=new Map<AudioBufferSourceNode,{turnId:string;end:number}>();
  private stopped=false;
  constructor(context:AudioContext){this.context=context;}
  enqueue(audio:VoiceAudio):void{
    if(this.stopped||this.context.state!=='running')throw Error('Audio output is not running');
    const frames=audio.samples.length/audio.channels;
    if(![1,2].includes(audio.channels)||!Number.isInteger(frames)||frames<1
      ||!Number.isInteger(audio.sampleRate)||audio.sampleRate<8000||audio.sampleRate>192000)throw Error('Invalid playback format');
    const start=Math.max(this.context.currentTime+.015,...Array.from(this.pending.values(),item=>item.end));
    const end=start+frames/audio.sampleRate;
    // Bound both duration and node overhead. Never silently discard output.
    if(end-this.context.currentTime>8||this.pending.size>=128)throw Error('Audio output queue exceeded');
    const buffer=this.context.createBuffer(audio.channels,frames,audio.sampleRate);
    for(let channel=0;channel<audio.channels;channel++){
      const samples=new Float32Array(frames);
      for(let i=0;i<frames;i++)samples[i]=audio.samples[i*audio.channels+channel];
      buffer.copyToChannel(samples,channel);
    }
    const node=this.context.createBufferSource();node.buffer=buffer;
    try{
      node.connect(this.context.destination);
      node.onended=()=>this.release(node,false);
      this.pending.set(node,{turnId:audio.turnId,end});node.start(start);
    }catch{this.release(node,true);throw Error('Audio output could not start');}
  }
  private release(node:AudioBufferSourceNode,stop:boolean){
    this.pending.delete(node);node.onended=null;
    try{if(stop)node.stop();}catch{/* A device can end a node first. */}
    node.disconnect();node.buffer=null;
  }
  clear(turnId:string):void{
    for(const [node,item] of this.pending)if(item.turnId===turnId)this.release(node,true);
  }
  stop():void{
    this.stopped=true;
    for(const node of this.pending.keys())this.release(node,true);
  }
}
