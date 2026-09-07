import {useCallback,useEffect,useRef,useState} from 'react';
import {NavLink,useNavigate,useParams} from 'react-router-dom';
import {ApiError,type ApiClient} from '../api';
import {Modal} from '../components/Modal';
import {capabilities,idPattern,session,sessionPage,type Capabilities,type ConversationSession as Session} from './contracts';
import {ConversationSession} from './ConversationSession';
import {ConversationReadiness,type ServiceState} from './ConversationReadiness';
import {RecallScopeEditor} from './RecallScopeControls';
import {emptyScopeDraft,scopeFromDraft,sameScope,type RecallScope} from './recall-scope';
import './conversations.css';

export function ConversationPage({api,enabled}:{api:ApiClient;enabled:boolean}){
  const {sid}=useParams(),navigate=useNavigate();
  const [cap,setCap]=useState<Capabilities|null>(null),[items,setItems]=useState<Session[]>([]),[after,setAfter]=useState<string|null>(null);
  const [service,setService]=useState<ServiceState>('checking'),[listState,setListState]=useState<'idle'|'loading'|'ready'|'failed'>('idle'),[listAttempt,setListAttempt]=useState(0);
  const [moreError,setMoreError]=useState(false),[loadingMore,setLoadingMore]=useState(false);
  const [error,setError]=useState(''),[attempt,setAttempt]=useState(0),[dialog,setDialog]=useState(false),[consent,setConsent]=useState(false),[starting,setStarting]=useState(false);
  const pendingCreate=useRef<{request_id:string;capture:true;recall_scope?:RecallScope}|null>(null),lifetime=useRef(new AbortController()),locked=useRef(false);
  const [scopeDraft,setScopeDraft]=useState(emptyScopeDraft),[scopeError,setScopeError]=useState('');
  const removed=useRef(new Set<string>()),[notice,setNotice]=useState('');
  const deleted=useCallback((id:string,acknowledged:boolean)=>{
    removed.current.add(id);setItems(previous=>previous.filter(item=>item.session_id!==id));
    setNotice(acknowledged?'Conversation deleted.':'Conversation is no longer available.');
    navigate('/conversations');
  },[navigate]);
  const changed=useCallback((next:Session)=>setItems(previous=>previous.map(item=>item.session_id===next.session_id&&next.revision>=item.revision&&(item.state!==next.state||item.revision!==next.revision)?next:item)),[]);
  useEffect(()=>{
    const controller=new AbortController();lifetime.current=controller;setItems([]);setError('');setAfter(null);setNotice('');removed.current.clear();
    pendingCreate.current=null;setDialog(false);setConsent(false);setScopeDraft(emptyScopeDraft());setScopeError('');
    return()=>controller.abort();
  },[api,enabled]);
  useEffect(()=>{
    const controller=new AbortController();setCap(null);setService('checking');setListState('idle');
    if(enabled){void(async()=>{try{
      const options={signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])};
      const response=await api.request('/v1/conversations/capabilities',options);
      if(controller.signal.aborted)return;
      try{setCap(capabilities(response));setService('ready');}catch{setService('incompatible');}
    }catch(error){if(!controller.signal.aborted)setService(error instanceof ApiError&&[404,501].includes(error.status)?'missing':error instanceof ApiError&&[401,403].includes(error.status)?'unauthorized':'unavailable');}})();}
    return()=>controller.abort();
  },[api,enabled,attempt]);
  useEffect(()=>{
    if(!enabled||!cap)return;
    const controller=new AbortController();setListState('loading');setMoreError(false);
    void(async()=>{try{
      const page=sessionPage(await api.request('/v1/conversations?limit=100',{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])}));
      if(controller.signal.aborted)return;
      // A late list must not erase a just-created session or resurrect a deleted one.
      setItems(previous=>{
        const merged=new Map(previous.map(item=>[item.session_id,item]));
        for(const item of page.items){const old=merged.get(item.session_id);if(!removed.current.has(item.session_id)&&(!old||item.revision>old.revision))merged.set(item.session_id,item);}
        return [...merged.values()];
      });setAfter(page.next_after);setListState('ready');
    }catch{if(!controller.signal.aborted)setListState('failed');}})();
    return()=>controller.abort();
  },[api,enabled,cap,listAttempt]);
  async function start(){
    if(!consent||!cap?.text_configured||locked.current)return;
    if(!pendingCreate.current){
      try{pendingCreate.current={request_id:crypto.randomUUID(),capture:true,...(cap.recall_scope?{recall_scope:scopeFromDraft(scopeDraft)}:{})};}
      catch(error){setScopeError(error instanceof Error?error.message:'Check the memory filters.');return;}
    }
    const body=pendingCreate.current;locked.current=true;setStarting(true);setError('');setScopeError('');const controller=lifetime.current;
    try{
      const next=session(await api.request('/v1/conversations',{method:'POST',body:JSON.stringify(body),signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])}));
      if(body.recall_scope!==undefined&&!sameScope(body.recall_scope,next.recall_scope))throw Error('The server did not confirm the requested recall scope.');
      if(controller.signal.aborted)return;
      setItems(previous=>[next,...previous.filter(item=>item.session_id!==next.session_id)]);setDialog(false);setConsent(false);pendingCreate.current=null;setScopeDraft(emptyScopeDraft());navigate('/conversations/'+encodeURIComponent(next.session_id));
    }catch(error){if(!controller.signal.aborted){
      if(error instanceof ApiError&&[400,422].includes(error.status)){pendingCreate.current=null;setScopeError('The server rejected these settings. Check your memory filters and try again.');}
      else setError('Start was not confirmed. Retry uses the same request ID and memory selection. No message will be sent automatically.');
    }}
    finally{locked.current=false;if(!controller.signal.aborted)setStarting(false);}
  }
  async function more(){
    if(!after||locked.current)return;locked.current=true;setLoadingMore(true);setMoreError(false);const controller=lifetime.current;
    try{const page=sessionPage(await api.request('/v1/conversations?limit=100&after='+encodeURIComponent(after),{signal:controller.signal}));if(!controller.signal.aborted){setItems(previous=>[...previous,...page.items.filter(item=>!removed.current.has(item.session_id)&&!previous.some(old=>old.session_id===item.session_id))]);setAfter(page.next_after);}}
    catch{if(!controller.signal.aborted)setMoreError(true);}finally{locked.current=false;if(!controller.signal.aborted)setLoadingMore(false);}
  }
  return <main id="main" className="conversation-page">
    <header className="conversation-heading"><div><span className="eyebrow">Your knowledge, in conversation</span><h1>Conversations</h1><p>Ask. Remember. Pick up where the context left off.</p></div><span className="conversation-mode">{!enabled?'Connection needed':service==='checking'?'Checking service…':cap?cap.text_configured?'Text configured':'History only':'Setup needed'}</span></header>
    {error&&!dialog&&<div role="alert" className="conversation-notice">{error}</div>}
    {notice&&<p role="status" className="conversation-notice">{notice}</p>}
    <div className="conversation-layout"><aside className="conversation-list" aria-label="Conversations"><button className="primary" disabled={!enabled||!cap?.text_configured} onClick={()=>setDialog(true)}>New conversation</button>
      <div className="conversation-list-label"><span className="eyebrow">Saved sessions</span><span aria-label="Saved session count">{listState==='ready'||items.length?items.length:'—'}</span></div>
      {listState==='loading'&&<p role="status" className="conversation-list-status">Loading saved sessions…</p>}
      {listState==='failed'&&<div className="conversation-list-status"><p role="alert">Saved sessions could not be loaded. The conversation service is connected.</p><button onClick={()=>setListAttempt(n=>n+1)}>Retry saved sessions</button></div>}
      {listState==='ready'&&!items.length&&<p className="conversation-list-status">No saved sessions yet.</p>}
      <nav aria-label="Saved conversations">{items.map(item=><NavLink key={item.session_id} to={'/conversations/'+encodeURIComponent(item.session_id)}><b>{item.session_id.slice(0,8)}</b><span>{item.state} · {new Date(item.created_at).toLocaleDateString()}</span></NavLink>)}</nav>
      {moreError&&<p role="alert" className="conversation-list-status">More sessions could not be loaded. Try again below.</p>}
      {after&&<button disabled={loadingMore} onClick={more}>{loadingMore?'Loading more…':'Load more sessions'}</button>}
      <p className="conversation-list-note">Sessions belong to this memory space. Switching views does not stop an active conversation.</p>
    </aside>
    {enabled&&cap&&sid&&idPattern.test(sid)?<ConversationSession key={sid} api={api} sid={sid} onSession={changed} textConfigured={cap.text_configured} deletionSupported={cap.session_deletion} cancellationSupported={cap.turn_cancellation} paginationSupported={cap.transcript_pagination} streamingSupported={cap.streaming} onRemoved={deleted}/>:<ConversationReadiness enabled={enabled} state={service} cap={cap} invalidAddress={Boolean(sid&&!idPattern.test(sid))} onRetry={()=>setAttempt(n=>n+1)}/>}
    </div>
    {dialog&&<Modal title="Start a conversation" onClose={()=>{if(!starting)setDialog(false);}}><p className="setup-intro">Your configured text model can receive context retrieved from this memory space. Public messages and replies will be saved as sources.</p>
      {cap?.recall_scope?<RecallScopeEditor draft={scopeDraft} disabled={starting||Boolean(pendingCreate.current)} onChange={value=>{setScopeDraft(value);setScopeError('');}}/>:<p className="conversation-caption">Memory selection is managed by this server.</p>}
      {scopeError&&<p role="alert" className="conversation-notice">{scopeError}</p>}
      {pendingCreate.current&&!starting&&<p className="conversation-caption">Memory selection is locked while this start request is unresolved.</p>}
      <label className="conversation-consent"><input type="checkbox" checked={consent} onChange={e=>setConsent(e.target.checked)}/>Save my public messages and replies to this memory space</label>
      {error&&<p role="alert" className="conversation-notice">{error}</p>}<button className="primary" disabled={!consent||starting} onClick={start}>{starting?'Starting…':'Start text conversation'}</button></Modal>}
  </main>;
}
