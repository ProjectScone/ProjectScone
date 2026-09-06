import {useCallback,useEffect,useRef,useState} from 'react';
import {NavLink,useNavigate,useParams} from 'react-router-dom';
import type {ApiClient} from '../api';
import {Modal} from '../components/Modal';
import {capabilities,idPattern,session,sessionPage,type Capabilities,type ConversationSession as Session} from './contracts';
import {ConversationSession} from './ConversationSession';
import './conversations.css';

export function ConversationPage({api,enabled}:{api:ApiClient;enabled:boolean}){
  const {sid}=useParams(),navigate=useNavigate();
  const [cap,setCap]=useState<Capabilities|null>(null),[items,setItems]=useState<Session[]>([]),[after,setAfter]=useState<string|null>(null);
  const [error,setError]=useState(''),[attempt,setAttempt]=useState(0),[dialog,setDialog]=useState(false),[consent,setConsent]=useState(false),[starting,setStarting]=useState(false);
  const createId=useRef<string|null>(null),lifetime=useRef(new AbortController()),locked=useRef(false);
  const removed=useRef(new Set<string>()),[notice,setNotice]=useState('');
  const deleted=useCallback((id:string,acknowledged:boolean)=>{
    removed.current.add(id);setItems(previous=>previous.filter(item=>item.session_id!==id));
    setNotice(acknowledged?'Conversation deleted.':'Conversation is no longer available.');
    navigate('/conversations');
  },[navigate]);
  const changed=useCallback((next:Session)=>setItems(previous=>previous.map(item=>item.session_id===next.session_id&&next.revision>=item.revision&&(item.state!==next.state||item.revision!==next.revision)?next:item)),[]);
  useEffect(()=>{
    const controller=new AbortController();lifetime.current=controller;setCap(null);setItems([]);setError('');setAfter(null);setNotice('');removed.current.clear();
    if(enabled){void(async()=>{try{
      const options={signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])};
      const next=capabilities(await api.request('/v1/conversations/capabilities',options));
      if(controller.signal.aborted)return;setCap(next);
      const page=sessionPage(await api.request('/v1/conversations?limit=100',options));if(!controller.signal.aborted){setItems(page.items.filter(item=>!removed.current.has(item.session_id)));setAfter(page.next_after);}
    }catch{if(!controller.signal.aborted)setError('The conversation service could not be verified. Your memory connection alone does not enable conversations.');}})();}
    return()=>controller.abort();
  },[api,enabled,attempt]);
  async function start(){
    if(!consent||!cap?.text_configured||locked.current)return;
    createId.current??=crypto.randomUUID();locked.current=true;setStarting(true);setError('');const controller=lifetime.current;
    try{
      const next=session(await api.request('/v1/conversations',{method:'POST',body:JSON.stringify({request_id:createId.current,capture:true}),signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])}));
      if(controller.signal.aborted)return;
      setItems(previous=>[next,...previous.filter(item=>item.session_id!==next.session_id)]);setDialog(false);setConsent(false);createId.current=null;navigate('/conversations/'+encodeURIComponent(next.session_id));
    }catch{if(!controller.signal.aborted)setError('Start was not confirmed. Retry uses the same request ID and will not create a second session.');}
    finally{locked.current=false;if(!controller.signal.aborted)setStarting(false);}
  }
  async function more(){
    if(!after||locked.current)return;locked.current=true;const controller=lifetime.current;
    try{const page=sessionPage(await api.request('/v1/conversations?limit=100&after='+encodeURIComponent(after),{signal:controller.signal}));if(!controller.signal.aborted){setItems(previous=>[...previous,...page.items.filter(item=>!removed.current.has(item.session_id)&&!previous.some(old=>old.session_id===item.session_id))]);setAfter(page.next_after);}}
    catch{if(!controller.signal.aborted)setError('More sessions could not be loaded.');}finally{locked.current=false;}
  }
  return <main id="main" className="conversation-page">
    <header className="conversation-heading"><div><span className="eyebrow">Your knowledge, in conversation</span><h1>Conversations</h1><p>Ask. Remember. Pick up where the context left off.</p></div><span className="conversation-mode">◉ Text workspace</span></header>
    {error&&<div role="alert" className="conversation-notice">{error}<button onClick={()=>setAttempt(n=>n+1)}>Retry service connection</button></div>}
    {notice&&<p role="status" className="conversation-notice">{notice}</p>}
    <div className="conversation-layout"><aside className="conversation-list" aria-label="Conversations"><button className="primary" disabled={!enabled||!cap?.text_configured||Boolean(error)} onClick={()=>setDialog(true)}>New conversation</button>
      <div className="conversation-list-label"><span className="eyebrow">Saved sessions</span><span>{items.length}</span></div>
      <nav aria-label="Saved conversations">{items.map(item=><NavLink key={item.session_id} to={'/conversations/'+encodeURIComponent(item.session_id)}><b>{item.session_id.slice(0,8)}</b><span>{item.state} · {new Date(item.created_at).toLocaleDateString()}</span></NavLink>)}</nav>
      {after&&<button onClick={more}>Load more sessions</button>}
      <p className="conversation-list-note">Sessions belong to this memory space. Switching views does not stop an active conversation.</p>
    </aside>
    {enabled&&cap&&sid&&idPattern.test(sid)?<ConversationSession key={sid} api={api} sid={sid} onSession={changed} textConfigured={cap.text_configured} deletionSupported={cap.session_deletion} onRemoved={deleted}/>:<section className="conversation-landing">
      <span className="conversation-orbit" aria-hidden="true">✳</span><span className="eyebrow">Context that carries forward</span>
      <h2>{!enabled?'Connect your memory first':cap&&!cap.text_configured?'Text runtime not configured':sid&&!idPattern.test(sid)?'Invalid session address':'Good conversations build on what you know.'}</h2>
      <p>{!enabled?'Use Connect memory to authorize this workspace.':cap&&!cap.text_configured?'Saved conversations remain available. Configure a text model on this server to start or send messages. No provider key belongs in this page.':'Start a text session or open a saved conversation. Inspect the original sources alongside each exchange.'}</p>
      <div className="conversation-landing-facts"><div><b>01</b><strong>Bring context</strong><span>Recall from your authorized memory.</span></div><div><b>02</b><strong>Keep the original</strong><span>Public messages become source records.</span></div><div><b>03</b><strong>See the evidence</strong><span>Inspect what was prepared and saved.</span></div></div>
      <p className="conversation-caption">This workspace supports completed text replies. Voice and video controls will arrive with verified transport support.</p>
    </section>}
    </div>
    {dialog&&<Modal title="Start a conversation" onClose={()=>{if(!starting)setDialog(false);}}><p className="setup-intro">Your configured text model can receive context retrieved from this memory space. Public messages and replies will be saved as sources.</p>
      <label className="conversation-consent"><input type="checkbox" checked={consent} onChange={e=>setConsent(e.target.checked)}/>Save my public messages and replies to this memory space</label>
      {error&&<p role="alert" className="conversation-notice">{error}</p>}<button className="primary" disabled={!consent||starting} onClick={start}>{starting?'Starting…':'Start text conversation'}</button></Modal>}
  </main>;
}
