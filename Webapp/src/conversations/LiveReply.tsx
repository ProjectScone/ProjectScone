import {useEffect,useRef,useState} from 'react';
import {ApiError,type ApiClient} from '../api';
import {readTextStream} from './text-stream';

const MAX_PREVIEW_BYTES=1024*1024;
type Phase='connecting'|'live'|'interrupted'|'waiting'|'limited';
export function LiveReply({api,sid,requestId,onTerminal}:{api:ApiClient;sid:string;requestId:string;onTerminal:(id:string)=>void}){
  const [attempt,setAttempt]=useState(0),[phase,setPhase]=useState<Phase>('connecting');
  const [text,setText]=useState(''),[missing,setMissing]=useState(false),[retryable,setRetryable]=useState(true);
  const cursor=useRef(0),received=useRef(''),bytes=useRef(0),owner=useRef(api);
  useEffect(()=>{
    if(owner.current!==api){owner.current=api;cursor.current=0;received.current='';bytes.current=0;setText('');setMissing(false);}
    const controller=new AbortController();let disposed=false,timer:ReturnType<typeof setTimeout>|undefined;
    const heartbeat=()=>{clearTimeout(timer);timer=setTimeout(()=>controller.abort(),30000);};
    setPhase('connecting');setRetryable(true);heartbeat();
    void(async()=>{
      try{
        const body=await api.conversationStream(sid,requestId,cursor.current,controller.signal);
        if(disposed){await body.cancel();return;}
        for await(const event of readTextStream(body,requestId,cursor.current,heartbeat)){
          if(disposed)return;
          if(event.kind==='text'){
            const nextBytes=bytes.current+new TextEncoder().encode(event.text).length;
            if(nextBytes>MAX_PREVIEW_BYTES){setPhase('limited');return;}
            cursor.current=event.sequence;bytes.current=nextBytes;received.current+=event.text;
            setText(received.current);setPhase('live');
          }else if(event.kind==='gap'){
            // Do not concatenate non-contiguous passages as though nothing was lost.
            cursor.current=event.next-1;received.current='';bytes.current=0;setText('');setMissing(true);
          }else{
            // Completed transport is not yet a saved message. Keep its labelled
            // preview until the matching assistant episode reaches the transcript.
            if(event.kind!=='terminal'||event.status!=='completed'){received.current='';bytes.current=0;setText('');}
            setPhase('waiting');
            if(event.kind==='terminal')onTerminal(requestId);
            return;
          }
        }
      }catch(error){if(!disposed){
        setRetryable(!(error instanceof ApiError&&[401,403,404,409,422,501].includes(error.status)));
        setPhase('interrupted');
      }}finally{clearTimeout(timer);controller.abort();}
    })();
    return()=>{disposed=true;clearTimeout(timer);controller.abort();};
  },[api,sid,requestId,attempt,onTerminal]);
  const status=phase==='connecting'?'Connecting to live reply…':phase==='live'?'Receiving public text':phase==='interrupted'?'Live preview interrupted. Checking the saved reply.':phase==='limited'?'Preview size limit reached. Waiting for the saved reply.':'Live preview ended. Checking the saved reply.';
  return <section className="conversation-live" aria-label="Live reply preview">
    <header><span className="conversation-live-badge"><span aria-hidden="true">◉</span> Live preview</span><span>Not yet a saved reply</span></header>
    <div className="conversation-live-status" role="status">{status}</div>
    {missing&&<p className="conversation-live-gap">Earlier live text is missing. Showing only the latest continuous segment.</p>}
    {text&&owner.current===api&&<div className="conversation-live-text" tabIndex={0} aria-label="Provisional reply text">{text}</div>}
    <footer><span>Public text chunks, not a token count. The saved receipt determines the final outcome.</span>
      {phase==='interrupted'&&retryable&&<button type="button" onClick={()=>setAttempt(n=>n+1)}>Reconnect preview</button>}
    </footer>
  </section>;
}
