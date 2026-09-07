// Public-text transport only. A terminal event is a cue to read the saved
// receipt, never a substitute for it. No model command is issued here.
export type TextEvent = {kind:'text';sequence:number;text:string}|{kind:'gap';next:number}
  |{kind:'terminal';status:string}|{kind:'end';reason:string};
const MAX_FRAME=524288; // accommodates JSON escapes around a 64 KiB native chunk
const positive=(value:unknown):value is number=>typeof value==='number'&&Number.isSafeInteger(value)&&value>0;

export async function* readTextStream(body:ReadableStream<Uint8Array>,requestId:string,after:number,onActivity=()=>{}):AsyncGenerator<TextEvent>{
  const reader=body.getReader(),decoder=new TextDecoder('utf-8',{fatal:true});
  let pending='',event='',id='',data:string[]=[],size=0,cursor=after;
  try{
    while(true){
      const {value,done}=await reader.read();
      if(done){decoder.decode();throw Error('Live stream ended before a recorded outcome');}
      onActivity();
      const decoded=decoder.decode(value,{stream:true});
      // Bound each frame, not an entire network read that may contain many events.
      let offset=0;
      while(offset<decoded.length){
        const newline=decoded.indexOf('\n',offset),end=newline<0?decoded.length:newline;
        pending+=decoded.slice(offset,end);size+=end-offset;
        if(size>MAX_FRAME)throw Error('Live stream frame exceeds the preview limit');
        if(newline<0)break;
        offset=newline+1;size++;
        const line=pending.endsWith('\r')?pending.slice(0,-1):pending;pending='';
        if(line===''){
          if(data.length){
            const wire=JSON.parse(data.join('\n'));
            if(!wire||typeof wire!=='object'||Array.isArray(wire))throw Error('Invalid live stream event');
            if(event==='text'){
              if(!positive(wire.sequence)||id!==String(wire.sequence)||wire.provisional!==true||typeof wire.text!=='string'||!wire.text.length)throw Error('Invalid live text chunk');
              if(wire.sequence>cursor){
                if(wire.sequence!==cursor+1)throw Error('Live text sequence is missing a gap');
                cursor=wire.sequence;yield {kind:'text',sequence:cursor,text:wire.text};
              }
            }else if(event==='gap'){
              if(id||wire.after!==cursor||!positive(wire.next_sequence)||wire.next_sequence<=cursor+1)throw Error('Invalid live text gap');
              cursor=wire.next_sequence-1;yield {kind:'gap',next:wire.next_sequence};
            }else if(event==='terminal'||event==='end'){
              if(id||wire.request_id!==requestId||wire.read_receipt!==true)throw Error('Mismatched live stream outcome');
              if(event==='terminal'){
                if(!['completed','failed','cancelled','interrupted'].includes(wire.status))throw Error('Invalid live stream outcome');
                yield {kind:'terminal',status:wire.status};
              }else{
                if(!['service_shutdown','deleted','window_unavailable'].includes(wire.reason))throw Error('Invalid live stream ending');
                yield {kind:'end',reason:wire.reason};
              }
              return;
            }else throw Error('Unsupported live stream event');
          }
          event='';id='';data=[];size=0;
        }else if(!line.startsWith(':')){
          const colon=line.indexOf(':'),field=colon<0?line:line.slice(0,colon);
          let content=colon<0?'':line.slice(colon+1);if(content.startsWith(' '))content=content.slice(1);
          if(field==='event')event=content;
          else if(field==='id')id=content;
          else if(field==='data')data.push(content);
        }
      }
    }
  }finally{try{await reader.cancel();}finally{reader.releaseLock();}}
}
