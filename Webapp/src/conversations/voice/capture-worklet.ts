// This function is serialized into the audio worklet's separate realm. Keep it
// self-contained: imported values/outer closures do not exist in that realm.
declare const sampleRate:number;
declare class AudioWorkletProcessor {port:MessagePort}
declare function registerProcessor(name:string,processor:unknown):void;
export function installCaptureProcessor(){
  class SconeMicrophone extends AudioWorkletProcessor {
    buffer=new Float32Array(Math.round(sampleRate*.02));
    offset=0;
    outstanding=0;
    live=true;
    constructor(){
      super();
      this.port.onmessage=event=>{if(event.data==='ack')this.outstanding=Math.max(0,this.outstanding-1);};
    }
    process(inputs:Float32Array[][]){
      if(!this.live)return false;
      const input=inputs[0]?.[0];if(!input)return true;
      for(const sample of input){
        this.buffer[this.offset++]=sample;
        if(this.offset===this.buffer.length){
          if(this.outstanding>=4){this.port.postMessage({type:'overrun'});this.live=false;return false;}
          this.outstanding++;
          this.port.postMessage({type:'samples',samples:this.buffer},[this.buffer.buffer]);
          this.buffer=new Float32Array(Math.round(sampleRate*.02));this.offset=0;
        }
      }
      return true;
    }
  }
  registerProcessor('scone-microphone',SconeMicrophone);
}
