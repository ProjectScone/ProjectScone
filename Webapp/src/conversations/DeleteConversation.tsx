import {useEffect,useRef,useState} from 'react';
import {ApiError,type ApiClient} from '../api';
import {Modal} from '../components/Modal';
import {session} from './contracts';

/** A terminal session is never deleted implicitly or retried after ambiguity. */
export function DeleteConversation({api,sid,enabled,onRemoved}:{api:ApiClient;sid:string;enabled:boolean;onRemoved:(sid:string,acknowledged:boolean)=>void}){
  const [open,setOpen]=useState(false),[consent,setConsent]=useState(false);
  const [phase,setPhase]=useState<'ready'|'deleting'|'uncertain'|'checking'>('ready');
  const [error,setError]=useState('');
  const lifetime=useRef(new AbortController()),locked=useRef(false);
  useEffect(()=>{const controller=new AbortController();lifetime.current=controller;return()=>controller.abort();},[api,sid]);
  const url='/v1/conversations/'+encodeURIComponent(sid);
  const working=phase==='deleting'||phase==='checking';
  function close(){if(!locked.current){setConsent(false);setOpen(false);}}
  async function remove(){
    if(!enabled||!consent||phase!=='ready'||locked.current)return;
    const controller=lifetime.current;locked.current=true;setPhase('deleting');setError('');
    try{
      const result=await api.request(url,{method:'DELETE',signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])});
      if(result!==undefined)throw Error('Unrecognized delete acknowledgement');
      if(!controller.signal.aborted)onRemoved(sid,true);
    }catch(cause){if(!controller.signal.aborted){
      setConsent(false);
      if(cause instanceof ApiError&&[400,401,403,409,422].includes(cause.status)){
        setPhase('ready');setError('Deletion was refused. Nothing was retried. Check the session state and your access before trying again.');
      }else{
        setPhase('uncertain');setError('Deletion was not confirmed. It may have removed some or all records. Check its status; nothing will be resent automatically.');
      }
    }}finally{locked.current=false;}
  }
  async function check(){
    if(locked.current)return;
    const controller=lifetime.current;locked.current=true;setPhase('checking');
    try{
      const found=session(await api.request(url,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])}));
      if(found.session_id!==sid)throw Error('Different session returned');
      if(!controller.signal.aborted){setPhase('ready');setConsent(false);setError('The conversation still exists. Some records may already have been removed. Inspect it before confirming another deletion.');}
    }catch(cause){if(!controller.signal.aborted){
      if(cause instanceof ApiError&&cause.status===404)onRemoved(sid,false);
      else{setPhase('uncertain');setError('Status could not be checked. No additional delete request was sent.');}
    }}finally{locked.current=false;}
  }
  return <>
    <button className="conversation-delete-trigger" disabled={!enabled} onClick={()=>setOpen(true)}>Delete conversation</button>
    {open&&<Modal title="Delete this conversation?" onClose={close}>
      <div className="conversation-delete-summary"><span className="eyebrow">Permanent removal</span><strong>Session {sid.slice(0,8)}</strong>
        <p>Remove this saved conversation, its event history and reply receipts, and the message sources held only by this conversation.</p>
        <p>Imported knowledge and message sources referenced by another conversation remain. This does not erase copies held by a model provider or in your backups.</p>
      </div>
      <label className="conversation-consent"><input type="checkbox" checked={consent} disabled={working||phase==='uncertain'} onChange={event=>setConsent(event.target.checked)}/>I understand this cannot be undone</label>
      {error&&<p role="alert" className="conversation-notice">{error}</p>}
      {!enabled&&<p role="status" className="conversation-notice">Session state must be verified and closed before deletion.</p>}
      <div className="conversation-delete-actions"><button disabled={working} onClick={close}>Keep conversation</button>
        {phase==='uncertain'||phase==='checking'?<button className="primary" disabled={working} onClick={check}>{working?'Checking…':'Check deletion status'}</button>:<button className="conversation-delete-confirm" disabled={!enabled||!consent||working} onClick={remove}>{working?'Deleting…':'Permanently delete'}</button>}
      </div>
    </Modal>}
  </>;
}
