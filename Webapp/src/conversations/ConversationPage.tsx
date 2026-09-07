import {useCallback,useEffect,useRef,useState} from 'react';
import {NavLink,useNavigate,useParams} from 'react-router-dom';
import {ApiError,type ApiClient} from '../api';
import {Modal} from '../components/Modal';
import {WorkspaceHeading} from '../components/WorkspaceHeading';
import {capabilities,idPattern,session,sessionPage,type Capabilities,type ConversationSession as Session} from './contracts';
import {ConversationSession} from './ConversationSession';
import {ConversationReadiness,type ServiceState} from './ConversationReadiness';
import {RecallScopeEditor} from './RecallScopeControls';
import {PersonaPicker,usePersonaCatalog} from './PersonaPicker';
import {emptyScopeDraft,scopeFromDraft,sameScope,type RecallScope} from './recall-scope';
import './conversations.css';

export function ConversationPage({api,enabled}:{api:ApiClient;enabled:boolean}){
  const {sid}=useParams(),navigate=useNavigate();
  const [cap,setCap]=useState<Capabilities|null>(null),[items,setItems]=useState<Session[]>([]),[after,setAfter]=useState<string|null>(null);
  const [service,setService]=useState<ServiceState>('checking'),[listState,setListState]=useState<'idle'|'loading'|'ready'|'failed'>('idle'),[listAttempt,setListAttempt]=useState(0);
  const [moreError,setMoreError]=useState(false),[loadingMore,setLoadingMore]=useState(false);
  const [error,setError]=useState(''),[attempt,setAttempt]=useState(0),[dialog,setDialog]=useState(false),[consent,setConsent]=useState(false),[starting,setStarting]=useState(false);
  const [mode,setMode]=useState<'text'|'voice'>('text');
  const pendingCreate=useRef<{request_id:string;capture:true;mode?:'voice';recall_scope?:RecallScope;persona?:string;persona_fingerprint?:string}|null>(null),lifetime=useRef(new AbortController()),locked=useRef(false);
  const uncertainCreate=useRef(false);
  const catalog=usePersonaCatalog(api,enabled?cap?.personas??0:0),[selectedPersona,setSelectedPersona]=useState('');
  const personaReady=mode==='voice'?Boolean(cap?.voice)&&catalog.state==='ready'&&catalog.items.some(p=>p.id===selectedPersona&&p.voice_ready&&p.fingerprint):!cap?.personas||(catalog.state==='ready'&&catalog.items.some(p=>p.id===selectedPersona&&p.text_ready));
  const [scopeDraft,setScopeDraft]=useState(emptyScopeDraft),[scopeError,setScopeError]=useState('');
  const removed=useRef(new Set<string>()),[notice,setNotice]=useState('');
  const runtimeLabel=cap?.voice?(cap.text_configured?'Text + voice configured':'Voice configured'):cap?.text_configured?'Text configured':'History only';
  const deleted=useCallback((id:string,acknowledged:boolean)=>{
    removed.current.add(id);setItems(previous=>previous.filter(item=>item.session_id!==id));
    setNotice(acknowledged?'Conversation deleted.':'Conversation is no longer available.');
    navigate('/conversations');
  },[navigate]);
  const changed=useCallback((next:Session)=>setItems(previous=>previous.map(item=>item.session_id===next.session_id&&next.revision>=item.revision&&(item.state!==next.state||item.revision!==next.revision)?next:item)),[]);
  useEffect(()=>{
    const controller=new AbortController();lifetime.current=controller;setItems([]);setError('');setAfter(null);setNotice('');removed.current.clear();
    pendingCreate.current=null;setDialog(false);setConsent(false);setScopeDraft(emptyScopeDraft());setScopeError('');setSelectedPersona('');setMode('text');
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
    if(!consent||!cap||(mode==='voice'?!cap.voice:!cap.text_configured)||locked.current)return;
    if(!pendingCreate.current){
      if(!personaReady)return;
      uncertainCreate.current=false;
      const fingerprint=catalog.items.find(p=>p.id===selectedPersona)?.fingerprint;
      try{pendingCreate.current={request_id:crypto.randomUUID(),capture:true,...(mode==='voice'?{mode:'voice' as const}:{}),...(cap.recall_scope?{recall_scope:scopeFromDraft(scopeDraft)}:{}),...(cap.personas?{persona:selectedPersona,...(fingerprint?{persona_fingerprint:fingerprint}:{})}:{})};}
      catch(error){setScopeError(error instanceof Error?error.message:'Check the memory filters.');return;}
    }
    const body=pendingCreate.current;locked.current=true;setStarting(true);setError('');setScopeError('');const controller=lifetime.current;
    try{
      const next=session(await api.request('/v1/conversations',{method:'POST',body:JSON.stringify(body),signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])}));
      if((next.mode??'text')!==(body.mode??'text'))throw Error('The server did not confirm the requested conversation mode.');
      if(body.recall_scope!==undefined&&!sameScope(body.recall_scope,next.recall_scope))throw Error('The server did not confirm the requested recall scope.');
      if((body.persona??null)!==(next.persona?.id??null))throw Error('The server did not confirm the selected persona.');
      if(body.persona_fingerprint!==undefined&&body.persona_fingerprint!==next.persona?.fingerprint)throw Error('The server did not confirm the selected provider configuration.');
      if(controller.signal.aborted)return;
      setItems(previous=>[next,...previous.filter(item=>item.session_id!==next.session_id)]);setDialog(false);setConsent(false);pendingCreate.current=null;setScopeDraft(emptyScopeDraft());setSelectedPersona('');navigate('/conversations/'+encodeURIComponent(next.session_id));
    }catch(error){if(!controller.signal.aborted){
      if(error instanceof ApiError&&error.status===409&&error.code==='persona_selection_stale'){
        pendingCreate.current=null;setSelectedPersona('');catalog.retry();
        setScopeError('The provider configuration changed before this conversation was created. Review the refreshed choices and select a persona again.');
      }
      else if(!uncertainCreate.current&&error instanceof ApiError&&error.status===403){
        pendingCreate.current=null;
        setScopeError('This key cannot start conversations. Use a key with write access. No session was started by this request; your selection is kept.');
      }
      else if(!uncertainCreate.current&&error instanceof ApiError&&[400,422].includes(error.status)){pendingCreate.current=null;setScopeError('The server rejected these settings. Check your persona and memory filters, then try again.');}
      else {uncertainCreate.current=true;setError('Start was not confirmed. Retry uses the same request ID, persona and memory selection. No message will be sent automatically.');}
    }}
    finally{locked.current=false;if(!controller.signal.aborted)setStarting(false);}
  }
  async function more(){
    if(!after||locked.current)return;locked.current=true;setLoadingMore(true);setMoreError(false);const controller=lifetime.current;
    try{const page=sessionPage(await api.request('/v1/conversations?limit=100&after='+encodeURIComponent(after),{signal:controller.signal}));if(!controller.signal.aborted){setItems(previous=>[...previous,...page.items.filter(item=>!removed.current.has(item.session_id)&&!previous.some(old=>old.session_id===item.session_id))]);setAfter(page.next_after);}}
    catch{if(!controller.signal.aborted)setMoreError(true);}finally{locked.current=false;if(!controller.signal.aborted)setLoadingMore(false);}
  }
  return <main id="main" className="conversation-page">
    <aside className="conversation-list" aria-label="Conversations"><button className="primary" disabled={!enabled||(!cap?.text_configured&&!cap?.voice)} onClick={()=>{if(!pendingCreate.current)setMode(cap?.text_configured?'text':'voice');setDialog(true);}}>New conversation</button>
      <div className="conversation-list-label"><span className="eyebrow">Saved sessions</span><span aria-label="Saved session count">{listState==='ready'||items.length?items.length:'—'}</span></div>
      {listState==='loading'&&<p role="status" className="conversation-list-status">Loading saved sessions…</p>}
      {listState==='failed'&&<div className="conversation-list-status"><p role="alert">Saved sessions could not be loaded. The conversation service is connected.</p><button onClick={()=>setListAttempt(n=>n+1)}>Retry saved sessions</button></div>}
      {listState==='ready'&&!items.length&&<p className="conversation-list-status">No saved sessions yet.</p>}
      <nav aria-label="Saved conversations">{items.map(item=><NavLink key={item.session_id} to={'/conversations/'+encodeURIComponent(item.session_id)}><b>{item.session_id.slice(0,8)}</b><span>{item.state} · {new Date(item.created_at).toLocaleDateString()}</span></NavLink>)}</nav>
      {moreError&&<p role="alert" className="conversation-list-status">More sessions could not be loaded. Try again below.</p>}
      {after&&<button disabled={loadingMore} onClick={more}>{loadingMore?'Loading more…':'Load more sessions'}</button>}
      <p className="conversation-list-note">Sessions belong to this memory space. Leaving a voice session disconnects this tab’s microphone. Saved messages remain available.</p>
    </aside>
    <div className="conversation-workspace">
      <WorkspaceHeading className="conversation-heading" eyebrow="Your knowledge, in conversation" title="Conversations" description="Ask. Remember. Pick up where the context left off." actions={<span className="conversation-mode">{!enabled?'Connection needed':service==='checking'?'Checking service…':cap?runtimeLabel:'Setup needed'}</span>}/>
      {error&&!dialog&&<div role="alert" className="conversation-notice">{error}</div>}
      {notice&&<p role="status" className="conversation-notice">{notice}</p>}
      <div className="conversation-layout">
    {enabled&&cap&&sid&&idPattern.test(sid)?<ConversationSession key={sid} api={api} sid={sid} onSession={changed} textConfigured={cap.text_configured} voiceSupported={cap.voice} deletionSupported={cap.session_deletion} cancellationSupported={cap.turn_cancellation} paginationSupported={cap.transcript_pagination} streamingSupported={cap.streaming} onRemoved={deleted}/>:<ConversationReadiness enabled={enabled} state={service} cap={cap} invalidAddress={Boolean(sid&&!idPattern.test(sid))} onRetry={()=>setAttempt(n=>n+1)}/>}
      </div>
    </div>
    {dialog&&<Modal title="Start a conversation" onClose={()=>{if(!starting)setDialog(false);}}><p className="setup-intro">Your configured model can receive context retrieved from this memory space. Public messages and replies will be saved as sources.</p>
      {cap?.voice&&<fieldset className="conversation-mode-picker" disabled={starting||Boolean(pendingCreate.current)}><legend>How would you like to talk?</legend>{(['text','voice'] as const).map(value=><label key={value} className={mode===value?'is-selected':''}><input type="radio" name="conversation-mode" aria-label={value==='voice'?'Voice':'Text'} checked={mode===value} disabled={value==='text'&&!cap.text_configured} onChange={()=>{setMode(value);setSelectedPersona('');}}/><span><strong>{value==='voice'?'Voice':'Text'}</strong><small>{value==='voice'?'Speak and listen · microphone off until you enable it':'Write at your own pace'}</small></span></label>)}</fieldset>}
      {Boolean(cap?.personas)&&<PersonaPicker catalog={catalog} selected={selectedPersona} disabled={starting||Boolean(pendingCreate.current)} mode={mode} onChange={setSelectedPersona}/>}
      {cap?.recall_scope?<RecallScopeEditor draft={scopeDraft} disabled={starting||Boolean(pendingCreate.current)} onChange={value=>{setScopeDraft(value);setScopeError('');}}/>:<p className="conversation-caption">Memory selection is managed by this server.</p>}
      {scopeError&&<p role="alert" className="conversation-notice">{scopeError}</p>}
      {pendingCreate.current&&!starting&&<p className="conversation-caption">Memory selection is locked while this start request is unresolved.</p>}
      <label className="conversation-consent"><input type="checkbox" checked={consent} onChange={e=>setConsent(e.target.checked)}/>Save my public messages and replies to this memory space</label>
      {error&&<p role="alert" className="conversation-notice">{error}</p>}<button className="primary" disabled={!consent||starting||(!pendingCreate.current&&!personaReady)} onClick={start}>{starting?'Starting…':mode==='voice'?'Start voice conversation':'Start text conversation'}</button></Modal>}
  </main>;
}
