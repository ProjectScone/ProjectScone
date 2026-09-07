import {useEffect,useRef,useState} from 'react';
import {ApiError,type ApiClient} from '../api';
import {Modal} from '../components/Modal';
import {parsePassReceipt,type PassReceipt,type PassScope} from './consolidation';

type Outcome={kind:'reported';report:PassReceipt}|{kind:'denied'|'uncertain';message:string};
export function ConsolidationPanel({api,space,maintenance,inference,onBusy}:{api:ApiClient;space:string;maintenance:boolean;inference:boolean;onBusy:(busy:boolean)=>void}){
  const [confirm,setConfirm]=useState<PassScope|null>(null),[running,setRunning]=useState(false),[outcome,setOutcome]=useState<Outcome|null>(null);
  const request=useRef<AbortController|null>(null);
  useEffect(()=>{
    setConfirm(null);setOutcome(null);setRunning(false);
    return()=>{request.current?.abort();request.current=null;onBusy(false);};
  },[api,space,onBusy]);
  const locked=running||outcome?.kind==='uncertain';
  async function run(){
    if(!confirm||request.current||locked)return;
    const scope=confirm;
    if(scope==='distill'?!maintenance:!inference)return;
    const controller=new AbortController();request.current=controller;
    setConfirm(null);setOutcome(null);setRunning(true);onBusy(true);
    let uncertain=false;
    try{
      const value=await api.request<unknown>('/v1/consolidate',{method:'POST',body:JSON.stringify({scope}),signal:AbortSignal.any([controller.signal,AbortSignal.timeout(120000)])});
      const report=parsePassReceipt(value,space,scope);
      if(!controller.signal.aborted)setOutcome({kind:'reported',report});
    }catch(error){
      if(!controller.signal.aborted){
        const denied=error instanceof ApiError&&[403,501].includes(error.status);
        uncertain=!denied;
        setOutcome({kind:denied?'denied':'uncertain',message:error instanceof Error?error.message:'The server response could not be verified.'});
      }
    }finally{
      if(request.current===controller){request.current=null;setRunning(false);onBusy(uncertain);}
    }
  }
  return <section className="processing-actions" aria-labelledby="processing-actions-title">
    <header><div><span className="eyebrow">Explicit server operations</span><h2 id="processing-actions-title">Run a processing pass</h2><p>Uses this server’s configured models and policies in space <strong>{space}</strong>. Provider charges may apply. Nothing runs until you confirm.</p></div>
      <div className="processing-action-buttons">{maintenance&&<button className="btn quiet" disabled={locked} onClick={()=>setConfirm('distill')}>Run maintenance</button>}{inference&&<button className="btn quiet" disabled={locked} onClick={()=>setConfirm('derive')}>Run inference</button>}</div></header>
    {confirm&&<Modal title={confirm==='distill'?'Confirm maintenance pass':'Confirm inference pass'} onClose={()=>setConfirm(null)}>
      <p>Run one {confirm==='distill'?'maintenance':'inference'} pass in <strong>{space}</strong>?</p>
      {confirm==='distill'?<p>This runs the configured worker, not only extraction. It may add proposals, automatically accept or close claims under the operator’s policy, apply retention that can forget sources, and derive new claims.</p>:<p>This sends eligible claim groups to the configured model and may store inferred proposals and their premise links. It is not an approval of those conclusions.</p>}
      <p>Provider charges may apply. This is one bounded pass, not a promise to finish the backlog. Leaving the page does not cancel work already accepted by the server.</p>
      <div className="processing-action-buttons"><button className="btn quiet" onClick={()=>setConfirm(null)}>Cancel</button><button className="btn" onClick={()=>void run()}>Run one {confirm==='distill'?'maintenance':'inference'} pass</button></div>
    </Modal>}
    {running&&<p role="status">Waiting for this pass’s report… Navigating away does not cancel server work. No automatic retries will be sent.</p>}
    {outcome&&<section className="processing-result" aria-label="Processing pass result">
      {outcome.kind==='reported'?<>
        <h3>{outcome.report.error?'Pass reported an error':'Pass report received'}</h3>
        <p>{outcome.report.scope==='distill'?'Maintenance':'Inference'} · {outcome.report.space} · {outcome.report.latencyMs.toLocaleString()} ms reported by the server</p>
        {outcome.report.error&&<p role="alert">{outcome.report.error}</p>}
        <p>These are the effects reported by this pass, not an atomic job receipt or proof the backlog is complete. An error can occur after changes have been made.</p>
        <dl>{outcome.report.counts.map(([label,n])=><div key={label}><dt>{label}</dt><dd>{n.toLocaleString()}</dd></div>)}</dl>
        {!!outcome.report.reasons.length&&<details><summary>Rejection reasons</summary><ul>{outcome.report.reasons.map(([reason,n])=><li key={reason}>{reason}: {n}</li>)}</ul></details>}
        <p>Refresh status to read new counts. This report is held only on this page.</p>
      </>:<>
        <h3>{outcome.kind==='denied'?'Pass not started':'Pass outcome unconfirmed'}</h3><p role="alert">{outcome.message}</p>
        <p>{outcome.kind==='denied'?'The server refused this request. Check your key permissions and the configured operation before starting a new pass.':'The server may have applied changes or may still be working. Inspect retained activity and review before deciding whether to start a separate pass. This request will not be retried automatically.'}</p>
        {outcome.kind==='uncertain'&&<button className="btn quiet" onClick={()=>{setOutcome(null);onBusy(false);}}>Acknowledge unconfirmed outcome</button>}
      </>}
    </section>}
  </section>;
}
