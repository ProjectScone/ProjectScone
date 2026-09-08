import {useCallback,useEffect,useRef,useState} from 'react';
import {ApiError,type ApiClient} from '../api';
import {ConversationEvidence} from './ConversationEvidence';
import {DeleteConversation} from './DeleteConversation';
import {RecallScopeSummary} from './RecallScopeControls';
import {LiveReply} from './LiveReply';
import {VoiceControls} from './VoiceControls';
import {useTranscriptHeight} from './useTranscriptHeight';
import {session,transcript,turnReceipt,type ConversationSession as Session,type Transcript,type TurnResult} from './contracts';

export function ConversationSession({api,sid,onSession,textConfigured,voiceSupported,deletionSupported,cancellationSupported,paginationSupported,streamingSupported,onRemoved}:{api:ApiClient;sid:string;onSession:(value:Session)=>void;textConfigured:boolean;voiceSupported:boolean;deletionSupported:boolean;cancellationSupported:boolean;paginationSupported:boolean;streamingSupported:boolean;onRemoved:(sid:string,acknowledged:boolean)=>void}){
  const threadRef=useTranscriptHeight();
  const [current,setCurrent]=useState<Session|null>(null),[saved,setSaved]=useState<Transcript|null>(null);
  const [error,setError]=useState(''),[draft,setDraft]=useState(''),[busy,setBusy]=useState(false),[verified,setVerified]=useState(false);
  const [voiceVerified,setVoiceVerified]=useState(false),[audioStopRequested,setAudioStopRequested]=useState(false);
  const [delivery,setDelivery]=useState(''),[resultReceipt,setResultReceipt]=useState<{api:ApiClient;sid:string;value:TurnResult|undefined}>(),[selected,setSelected]=useState<number|null>(null);
  const result=resultReceipt?.api===api&&resultReceipt.sid===sid?resultReceipt.value:undefined;
  const [attempt,setAttempt]=useState(0);
  const [cancelTarget,setCancelTarget]=useState<string|null>(null),[cancelling,setCancelling]=useState(false);
  const [pages,setPages]=useState<(string|null)[]>([null]);
  const lifetime=useRef(new AbortController()),mutation=useRef(false),request=useRef<string|null>(null),settled=useRef<string|null>(null),stopId=useRef<string|null>(null);
  const commandGeneration=useRef(0),invalidatedReceipt=useRef<string|null>(null);
  const pendingRequest=useRef<string|null>(null);
  const [liveTarget,setLiveTarget]=useState<string|null>(null),suppressedPreview=useRef<string|null>(null);
  const streamTerminal=useCallback((id:string)=>{suppressedPreview.current=id;setAttempt(n=>n+1);},[]);
  const url=`/v1/conversations/${encodeURIComponent(sid)}`;
  const before=pages[pages.length-1];
  const transcriptUrl=url+'/transcript'+(paginationSupported?'?limit=50'+(before?'&before='+encodeURIComponent(before):''):'');
  useEffect(()=>{
    const controller=new AbortController();lifetime.current=controller;
    let timer:ReturnType<typeof setTimeout>;
    function clearUnavailableEvidence(id:string){
      setResultReceipt(undefined);
      // Revoke once, then let people inspect other retained sources normally.
      if(invalidatedReceipt.current!==id){invalidatedReceipt.current=id;setSelected(null);}
    }
    async function refresh(){
      if(controller.signal.aborted)return;
      if(document.hidden||mutation.current){timer=setTimeout(refresh,1500);return;}
      const generation=commandGeneration.current;
      const isCurrent=()=>!controller.signal.aborted&&generation===commandGeneration.current;
      try{
        const options={signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])};
        let next:Session;
        try{next=session(await api.request(url,options));}
        catch(error){if(isCurrent())setVoiceVerified(false);throw error;}
        // Transcript paging is not an audio authorization failure. Lifecycle checks
        // still revoke audio immediately, even if the transcript fetch then stalls.
        if(isCurrent()){setVoiceVerified(true);setCurrent(previous=>previous&&previous.revision>next.revision?previous:next);onSession(next);}
        if(isCurrent()&&next.state!=='running')setLiveTarget(null);
        async function readTranscript(){
          const records=transcript(await api.request(transcriptUrl,options));
          if(paginationSupported&&(records.episodes.length>50||(records.has_more&&(!records.next_before||pages.includes(records.next_before)))))throw Error('Transcript pagination did not advance');
          return records;
        }
        const records=await readTranscript();
        if(!isCurrent())return;
        setCurrent(previous=>previous&&previous.revision>next.revision?previous:next);onSession(next);setSaved(records);setVerified(true);setError('');
        const active=pendingRequest.current||next.active_request_id||next.latest_request_id||request.current;
        if(active){
          try{
            const receipt=turnReceipt(await api.request(url+'/turns/'+encodeURIComponent(active),options));
            if(!isCurrent())return;
            if(receipt.request_id!==active)throw Error('Mismatched reply receipt');
            if(receipt.status==='pending'&&next.state==='running'&&suppressedPreview.current!==active)setLiveTarget(active);
            else if(receipt.status!=='pending'&&(receipt.result_state!=='available'||records.episodes.some(item=>item.episode_id===receipt.result?.assistant_episode_id)))setLiveTarget(null);
            setCancelTarget(receipt.status==='pending'?active:null);
            setDelivery(receipt.status==='cancelled'?'Reply cancelled locally. The provider may still finish processing.':receipt.status==='pending'?'Reply in progress':receipt.status==='completed'?
              receipt.result_state==='forgotten'?'Reply completed; its saved text was forgotten.':
              receipt.result_state==='unreadable'?'Reply completed; its saved text cannot be read right now. Checking again…':
              receipt.result_state==='unavailable'?'Reply completed; its text is unavailable.':'Reply saved':`Reply ${receipt.status}`);
            if(receipt.status!=='pending'){
              const newlySettled=settled.current!==active;
              if(pendingRequest.current===active)pendingRequest.current=null;
              settled.current=active;setBusy(receipt.status==='cancelled'&&Boolean(next.active_request_id));
              if(receipt.result_state==='available'){invalidatedReceipt.current=null;setResultReceipt({api,sid,value:receipt.result});}
              else clearUnavailableEvidence(active);
              // Capture may finish after the transcript request above.
              if(newlySettled){const latest=await readTranscript();
                if(isCurrent()){setSaved(latest);if(latest.episodes.some(item=>item.episode_id===receipt.result?.assistant_episode_id))setLiveTarget(null);}}
            }else setBusy(true);
          }catch{
            if(isCurrent()){setLiveTarget(null);setCancelTarget(null);setDelivery('Reply receipt unavailable. No message was resent.');clearUnavailableEvidence(active);setBusy(Boolean(next.active_request_id)||settled.current!==active);}
          }
        }else if(!next.active_request_id&&!mutation.current){setLiveTarget(null);setBusy(false);}
      }catch{
        if(isCurrent()){setLiveTarget(null);setVerified(false);setError('Connection interrupted. Controls are paused where session state cannot be verified; saved messages may be out of date.');}
      }finally{if(!controller.signal.aborted)timer=setTimeout(refresh,1500);}
    }
    void refresh();return()=>{controller.abort();clearTimeout(timer);};
  },[api,url,attempt,onSession,transcriptUrl,paginationSupported,pages]);
  function changePage(next:(string|null)[]){
    if(!verified||mutation.current)return;
    setSaved(null);setVerified(false);setPages(next);
  }
  async function send(){
    if(!textConfigured||!current||current.mode==='voice'||!verified||busy||mutation.current||current.state!=='running'||!draft.trim())return;
    if(new TextEncoder().encode(draft).length>32000){setError('Keep messages within 32,000 UTF-8 bytes.');return;}
    commandGeneration.current++;
    const controller=lifetime.current,id=crypto.randomUUID();request.current=id;pendingRequest.current=id;mutation.current=true;setBusy(true);setDelivery('Sending message…');setError('');
    try{
      await api.request(url+'/turns',{method:'POST',body:JSON.stringify({request_id:id,text:draft,expected_revision:current.revision}),signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])});
      if(!controller.signal.aborted){setDraft('');setDelivery('Reply in progress');setLiveTarget(id);}
    }catch(error){if(!controller.signal.aborted){
      if(error instanceof ApiError&&[400,401,403,404,409,422,429].includes(error.status)){
        request.current=null;pendingRequest.current=null;setBusy(false);setDelivery('Message was not accepted. Your draft is kept.');
        if(error.status===409)setVerified(false);
      }else setDelivery('Delivery uncertain. Checking the same request; nothing will be resent automatically.');
    }}
    finally{mutation.current=false;}
  }
  async function stop(){
    if(!current||!verified||mutation.current||current.state!=='running')return;
    setAudioStopRequested(true); // Local capture never waits for a remote stop receipt.
    commandGeneration.current++;
    suppressedPreview.current=liveTarget;setLiveTarget(null);
    const controller=lifetime.current;stopId.current??=crypto.randomUUID();mutation.current=true;setBusy(true);
    try{
      const next=session(await api.request(url+'/stop',{method:'POST',body:JSON.stringify({request_id:stopId.current,expected_revision:current.revision}),signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])}));
      if(!controller.signal.aborted){setCurrent(next);onSession(next);}
    }catch{if(!controller.signal.aborted)setError('End request was not confirmed. Checking recorded session state; cleanup may still be running.');}
    finally{mutation.current=false;}
  }
  async function cancelReply(){
    if(!cancellationSupported||!cancelTarget||!verified||mutation.current||current?.state!=='running')return;
    const id=cancelTarget,controller=lifetime.current;
    suppressedPreview.current=id;setLiveTarget(null);
    commandGeneration.current++;mutation.current=true;setCancelling(true);setError('');
    try{
      const receipt=turnReceipt(await api.request(url+'/turns/'+encodeURIComponent(id)+'/cancel',{method:'POST',signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])}));
      if(receipt.request_id!==id||receipt.status!=='cancelled')throw Error('Cancellation was not acknowledged');
      if(!controller.signal.aborted)setDelivery('Reply cancelled locally. Checking whether this session can continue.');
    }catch{if(!controller.signal.aborted)setDelivery('Cancellation was not confirmed. Checking the recorded outcome; nothing will be retried automatically.');}
    finally{
      mutation.current=false;
      if(!controller.signal.aborted){setCancelling(false);setCancelTarget(null);setVerified(false);}
    }
  }
  const terminal=current&&['ended','failed','interrupted'].includes(current.state);
  const voice=current?.mode==='voice';
  return <><section ref={threadRef} className="conversation-thread" aria-label={voice?'Voice conversation':'Text conversation'}>
    <header className="conversation-thread-header"><div><span className="eyebrow">{voice?'Voice':'Text'} session · {sid.slice(0,8)}</span><h2>{current?.state==='ended'?'Conversation ended':current?.state==='interrupted'?'Conversation interrupted':current?.state==='failed'?'Conversation failed':'A conversation that remembers'}</h2></div>
      {terminal&&deletionSupported?<DeleteConversation api={api} sid={sid} enabled={verified} onRemoved={onRemoved}/>:<button onClick={stop} disabled={!verified||current?.state!=='running'}>End conversation</button>}</header>
    {current&&<RecallScopeSummary scope={current.recall_scope}/>}
    {current?.persona!==undefined&&<p className="session-persona" aria-label="Session persona"><span>Persona</span> <strong>{current.persona?(current.persona.current===false?current.persona.id:current.persona.name??current.persona.id):'Server default'}</strong>{current.persona&&<>
      <small>{!current.persona.fingerprint?'Provider configuration version was not recorded.':current.persona.current===false?'Recorded configuration differs from the current catalog or is unavailable.':'Provider choices match the current catalog.'}</small>
      {current.persona.fingerprint&&<code title="Recorded provider-choice fingerprint; does not verify instructions or model weights">{current.persona.fingerprint}</code>}
    </>}</p>}
    {error&&<div className="conversation-notice" role="alert">{error}<button onClick={()=>setAttempt(n=>n+1)}>Check connection</button></div>}
    {!voice&&!textConfigured&&<p className="conversation-notice">History is available. Sending messages requires a text model configured on this server.</p>}
    {voice&&current&&<VoiceControls api={api} session={current} supported={voiceSupported} verified={voiceVerified} stopRequested={audioStopRequested}/>}
    {paginationSupported&&<nav className="conversation-transcript-nav" aria-label="Transcript pages">
      <div><strong>{before?'Earlier messages':'Latest messages'}</strong><span>{saved?`${saved.episodes.length} messages on this page`:'Loading page…'}{before?' · New replies stay in Latest':''}</span></div>
      <div className="conversation-page-actions">
        {before&&<button onClick={()=>changePage([null])} disabled={!verified||mutation.current}>Latest messages</button>}
        <button onClick={()=>changePage(pages.slice(0,-1))} disabled={!verified||mutation.current||pages.length===1}>Newer messages</button>
        <button onClick={()=>saved?.next_before&&changePage([...pages,saved.next_before])} disabled={!verified||mutation.current||!saved?.next_before}>Older messages</button>
      </div>
    </nav>}
    <div className={`conversation-messages${saved?.episodes.length===0&&busy?' is-pending-empty':''}`} role="region" aria-label="Conversation transcript">
      <div className="conversation-saved" role="region" aria-label="Saved messages">
      {saved===null?<p role="status">Loading saved messages…</p>:saved.episodes.length?saved.episodes.map(item=><article className={`conversation-message ${item.metadata.role==='user'?'from-user':'from-agent'}`} key={item.episode_id}>
        <div className="conversation-message-label">{item.metadata.role==='user'?'You':item.metadata.role==='assistant'?'Assistant':'Recorded message'}<button onClick={()=>setSelected(item.episode_id)} aria-label={`Inspect message episode ${item.episode_id}`}>↗ Source {item.episode_id}</button></div><p>{item.content}</p>
      </article>):before?<p>No retained messages on this page. Return to a newer page.</p>:busy?<p className="conversation-caption">Waiting for saved messages…</p>:<div className="conversation-welcome"><div className="conversation-orbit" aria-hidden="true">✳</div><h3>{voice?'Your conversation, in words.':'Start with a question.'}</h3><p>{voice?'Completed public transcripts and replies appear here as they are saved.':'Bring your knowledge into the conversation.'}<br/>{voice?'Audio activity is not a saved transcript.':'Public messages and replies will be saved to this space.'}</p></div>}
      {saved?.has_more&&!paginationSupported&&<p className="conversation-notice">This is a partial transcript. This server does not support browsing older messages.</p>}
      </div>
      {!voice&&streamingSupported&&liveTarget&&<LiveReply key={liveTarget} api={api} sid={sid} requestId={liveTarget} onTerminal={streamTerminal}/>}
    </div>
    {current&&!voice&&<form className="conversation-composer" onSubmit={e=>{e.preventDefault();void send();}}>
      <label htmlFor="conversation-message">Message</label><textarea id="conversation-message" value={draft} onChange={e=>setDraft(e.target.value)} onKeyDown={event=>{
        if(event.key!=='Enter'||event.shiftKey||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229)return;
        event.preventDefault();
        if(!event.repeat)void send();
      }} aria-describedby="conversation-keyboard-hint" placeholder={!textConfigured?'Text runtime not configured. Saved messages are still available.':terminal?'This session is closed. Start a new conversation.':'Ask about something in your memory…'} disabled={!textConfigured||!verified||Boolean(terminal)} rows={3}/>
      <span id="conversation-keyboard-hint" className="conversation-caption">Enter to send · Shift+Enter for a new line</span>
      <div className="conversation-composer-footer"><span role="status">{delivery|| (terminal?'Saved messages remain in your memory.':streamingSupported?'Live public text · saved replies verified separately':'Completed replies · not a token stream')}</span>
        {cancellationSupported&&cancelTarget&&current?.state==='running'?<button type="button" disabled={!verified||cancelling} onClick={cancelReply}>{cancelling?'Cancelling…':'Cancel reply'}</button>:<button className="primary" type="submit" aria-label="Send message" disabled={!textConfigured||!verified||current?.state!=='running'||busy||!draft.trim()}>Send ↑</button>}
      </div>
    </form>}
  </section><ConversationEvidence api={api} result={result} selected={selected} onSelect={setSelected}/></>;
}
