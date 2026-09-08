import {useEffect, useState} from 'react';
import {ApiError, type ApiClient} from '../api';
import {Modal} from '../components/Modal';
import {MarkdownText, SourceContent} from '../components/SourceContent';
import {parseRetainedSource} from './source-inventory';
import {parseClaimDetail, relationLabel, relationTarget, type ClaimDetail, type ClaimLink} from './claim-relations';
import './claim-relations.css';

function readError(error:unknown,kind:'Claim'|'Source'):string {
  if(kind==='Source'&&error instanceof ApiError&&error.status===410)return 'This source was forgotten. Its text is no longer retained.';
  if(error instanceof ApiError&&error.status===404)return `${kind} unavailable in this space. It may have been forgotten.`;
  return error instanceof Error?error.message:`${kind} could not be read. Retry to check its state.`;
}

function Evidence({id,api}:{id:number;api:ApiClient}) {
  const [open,setOpen]=useState(false),[attempt,setAttempt]=useState(0);
  const [state,setState]=useState<{text?:string;error?:string}>({});
  useEffect(()=>{
    if(!open)return;
    const controller=new AbortController();setState({});
    api.request<unknown>(`/v1/episodes/${id}`,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])})
      .then(value=>parseRetainedSource(value,id)).then(value=>{if(!controller.signal.aborted)setState({text:value.content});})
      .catch(error=>{if(!controller.signal.aborted)setState({error:readError(error,'Source')});});
    return ()=>controller.abort();
  },[api,id,open,attempt]);
  return <div className="relation-source">
    <button type="button" aria-expanded={open} onClick={()=>setOpen(!open)}>{open?'Hide source':`Read source episode #${id}`}</button>
    {open&&<div className="relation-source-body">{state.error?<><p role="alert">{state.error}</p><button onClick={()=>setAttempt(value=>value+1)}>Retry source</button></>:state.text===undefined?<p role="status">Loading source…</p>:<SourceContent text={state.text}/>}</div>}
  </div>;
}

function ClaimContent({id,api,navigate}:{id:number;api:ApiClient;navigate:(id:number)=>void}) {
  const [state,setState]=useState<{data?:ClaimDetail;error?:string}>({}),[attempt,setAttempt]=useState(0);
  const [premiseLimit,setPremiseLimit]=useState(20),[relationshipLimit,setRelationshipLimit]=useState(20);
  useEffect(()=>{
    const controller=new AbortController();setState({});
    api.request<unknown>(`/v1/facts/${id}`,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])})
      .then(value=>parseClaimDetail(value,id)).then(data=>{if(!controller.signal.aborted)setState({data});})
      .catch(error=>{if(!controller.signal.aborted)setState({error:readError(error,'Claim')});});
    return ()=>controller.abort();
  },[api,id,attempt]);
  const detail=state.data,fact=detail?.fact;
  const premises=detail?.links.filter(link=>link.kind==='derived_from'&&link.from_fact===id)??[];
  const otherLinks=fact?.origin==='inferred'?detail?.links.filter(link=>!premises.includes(link))??[]:detail?.links??[];
  const renderLinks=(links:ClaimLink[],limit:number)=><ol>{links.slice(0,limit).map(link=>{const target=relationTarget(link,id);return <li key={link.link_id}>
    <button className="claim-link-target" onClick={()=>navigate(target)} aria-label={`${relationLabel(link,id)} · claim #${target}`}><span>{relationLabel(link,id)}</span><strong>Claim #{target}</strong><span aria-hidden="true">↗</span></button>
    <p className="claim-link-meta">Link #{link.link_id} · #{link.from_fact} → #{link.to_fact} · {link.kind}</p>
    {link.quote&&<blockquote><MarkdownText text={link.quote} inline/></blockquote>}
    {link.source_episode_id!=null&&<Evidence key={link.source_episode_id} id={link.source_episode_id} api={api}/>}
  </li>;})}</ol>;
  return <section className="claim-detail" aria-label={`Claim #${id} detail`}>
    <header><h3>Claim #{id}</h3><button onClick={()=>setAttempt(value=>value+1)}>Refresh claim</button></header>
    {state.error?<div className="claim-detail-error" role="alert"><p>{state.error}</p><p>No claim details were confirmed. Use Refresh claim to try again.</p></div>:!detail||!fact?<p role="status">Loading claim relationships…</p>:<>
      <div className="claim-detail-statement"><p><MarkdownText text={`${fact.subject} ${fact.predicate.replaceAll('_',' ')} ${fact.object}`} inline/></p>
        <span className={`claim-status claim-status-${fact.status}`}>{fact.status==='proposed'?'Proposed · not accepted':fact.status==='declined'?'Declined · not accepted':fact.status==='closed'?'Closed · historical record':'Active · approval not established'}</span>
        <dl><div><dt>Origin</dt><dd>{fact.origin??'Not recorded'}</dd></div><div><dt>Valid from</dt><dd>{fact.valid_from}</dd></div><div><dt>Valid until</dt><dd>{fact.valid_until??'No end recorded'}</dd></div></dl>
        {fact.excluded_reason!=null&&<p className="claim-excluded">Excluded from claim recall: {fact.excluded_reason}</p>}
        {fact.closed_reason&&<p>Closure reason: {fact.closed_reason}</p>}
      </div>
      {fact.origin==='inferred'&&<section className="claim-detail-links" aria-label="Inference premises">
        <h4>Inference premises ({premises.length})</h4>
        <p>This conclusion was inferred, not quoted. Open each premise to inspect its statement, status and original evidence. Recorded links do not establish that the conclusion follows.</p>
        {!premises.length&&<p className="claim-detail-empty">No premise links were returned. Support for this inference has not been established here.</p>}
        {renderLinks(premises,premiseLimit)}
        {premises.length>premiseLimit&&<button onClick={()=>setPremiseLimit(value=>value+20)}>Show more premises ({premiseLimit} of {premises.length})</button>}
      </section>}
      <section className="claim-detail-evidence"><h4>Direct source evidence</h4>
        {fact.quote&&<blockquote><MarkdownText text={fact.quote} inline/></blockquote>}
        {detail.sources.length?detail.sources.map(source=><Evidence key={source} id={source} api={api}/>):<p>No source episode is attached directly to this claim. Related claims may name their own sources.</p>}
      </section>
      <section className="claim-detail-links"><h4>{fact.origin==='inferred'?'Other stored relationships':'Stored relationships'} <span>{otherLinks.length}</span></h4>
        <p>Labels describe the link from this claim’s perspective. A stored link is not proof that either statement is correct.</p>
        {!otherLinks.length&&<div className="claim-detail-empty">No {fact.origin==='inferred'?'other ':''}typed relationships were returned for this claim.</div>}
        {renderLinks(otherLinks,relationshipLimit)}
        {otherLinks.length>relationshipLimit&&<button onClick={()=>setRelationshipLimit(value=>value+20)}>Show more relationships ({relationshipLimit} of {otherLinks.length})</button>}
      </section>
    </>}
  </section>;
}

function RelationshipDialog({id,api,close}:{id:number;api:ApiClient;close:()=>void}) {
  const [history,setHistory]=useState([id]),current=history[history.length-1];
  function navigate(next:number){setHistory(previous=>{
    const found=previous.indexOf(next);
    return found>=0?previous.slice(0,found+1):[...previous.slice(-49),next];
  });}
  return <Modal title="Claim relationships" onClose={close}>
    <div className="claim-detail-nav"><button disabled={history.length<2} onClick={()=>setHistory(previous=>previous.slice(0,-1))}>Back to previous claim</button><span>Read-only inspection · claim #{current}</span></div>
    <ClaimContent key={current} id={current} api={api} navigate={navigate}/>
  </Modal>;
}

export function ClaimRelationsButton({id,api,disabled=false,purpose='relationships'}:{id:number;api:ApiClient;disabled?:boolean;purpose?:'relationships'|'premises'}) {
  const [open,setOpen]=useState(false);
  return <div className="claim-relations-entry"><button className="btn small quiet" disabled={disabled} aria-label={`Inspect ${purpose} for claim #${id}`} onClick={()=>setOpen(true)}>Inspect {purpose} <span aria-hidden="true">↗</span></button>
    {open&&<RelationshipDialog key={id} id={id} api={api} close={()=>setOpen(false)}/>}
  </div>;
}
