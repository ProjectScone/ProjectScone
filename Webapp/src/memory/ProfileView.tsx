import {useEffect,useState} from 'react';
import {ApiError,type ApiClient} from '../api';
import {WorkspaceState} from '../components/WorkspaceState';
import {MarkdownText,SourceContent} from '../components/SourceContent';
import {parseRetainedSource} from './source-inventory';
import {parseProfile,type ProfileSnapshot,type ProfileClaim} from './profile';
import './profile.css';

function RetainedEvidence({id,api}:{id:number;api:ApiClient}){
  const [attempt,setAttempt]=useState(0),[result,setResult]=useState<{api:ApiClient;text?:string;error?:string}|null>(null);
  useEffect(()=>{
    const controller=new AbortController();setResult(null);
    api.request<unknown>(`/v1/episodes/${id}`,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])})
      .then(v=>parseRetainedSource(v,id)).then(v=>{if(!controller.signal.aborted)setResult({api,text:v.content});})
      .catch(e=>{if(!controller.signal.aborted)setResult({api,error:e instanceof ApiError&&e.status===410?'This source was forgotten. Its text is no longer retained.':e instanceof ApiError&&e.status===404?'This source is unavailable in this space.':e instanceof Error?e.message:'Source could not be read.'});});
    return()=>controller.abort();
  },[api,id,attempt]);
  const current=result?.api===api?result:null;
  return <div className="profile-evidence">{current?.error?<><p role="alert">{current.error}</p><button className="btn quiet small" onClick={()=>{setResult(null);setAttempt(n=>n+1);}}>Retry source</button></>:current?.text===undefined?<p role="status">Loading original source…</p>:<SourceContent text={current.text}/>}</div>;
}
function SourceDisclosure({id,api}:{id:number;api:ApiClient}){
  const [open,setOpen]=useState(false);
  return <div className="profile-source"><button className="btn quiet small" aria-expanded={open} onClick={()=>setOpen(v=>!v)}>{open?'Hide source':`Read source episode #${id}`}</button>
    {open&&<RetainedEvidence key={id} id={id} api={api}/>}</div>;
}
function ClaimSources({claim,api}:{claim:ProfileClaim;api:ApiClient}){
  const sources=[...new Set([...(claim.sources??[]),...(claim.source_episode_id!=null?[claim.source_episode_id]:[])])];
  return sources.length?<>{sources.map(id=><SourceDisclosure key={id} id={id} api={api}/>)}</>:<p className="profile-meta">No source reference was returned with this claim.</p>;
}

export function ProfileView({api,onOpenDocuments}:{api:ApiClient;onOpenDocuments?:()=>void}){
  const [attempt,setAttempt]=useState(0),[result,setResult]=useState<{api:ApiClient;data?:ProfileSnapshot;error?:string;checked?:string}|null>(null);
  useEffect(()=>{
    const controller=new AbortController();setResult(null);
    api.request<unknown>('/v1/profile',{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])})
      .then(parseProfile).then(data=>{if(!controller.signal.aborted)setResult({api,data,checked:new Date().toLocaleTimeString()});})
      .catch(e=>{if(!controller.signal.aborted)setResult({api,error:e instanceof ApiError&&[404,501].includes(e.status)?'This server advertised profiles, but its source-linked profile endpoint is unavailable.':e instanceof Error?e.message:'Profile could not be read.'});});
    return()=>controller.abort();
  },[api,attempt]);
  const current=result?.api===api?result:null,data=current?.data;
  const refresh=()=>{setResult(null);setAttempt(n=>n+1);};
  if(current?.error)return <WorkspaceState role="alert" icon="status" className="workspace-state-notice" title="Profile could not be loaded" description={current.error} actions={<button className="btn quiet" onClick={refresh}>Retry profile</button>}/>;
  if(!data)return <WorkspaceState role="status" busy icon="profile" title="Loading profile context…" description="Reading this space’s selected claims and source-linked excerpts."/>;
  return <section className="profile-workspace" aria-label="Profile context">
    <header className="profile-summary"><div><span className="eyebrow">Context for an agent</span><h2>{data.claims.length} selected claims <span>·</span> {data.recent.length} recent sources</h2><p>A bounded selection from this memory space, not a new source or a verified identity.</p></div><button className="btn quiet" onClick={refresh}>Refresh profile</button></header>
    {!data.claims.length&&!data.recent.length?<WorkspaceState icon="profile" title="No profile context yet" description="Add sources and review their claims to build context in this space." actions={onOpenDocuments&&<button className="btn quiet" onClick={onOpenDocuments}>Open documents</button>}/>:<div className="profile-columns">
      <section aria-label="Selected claim context"><header><h3>Selected claim context</h3><p>Claims returned by this server’s profile selection. Selection does not establish truth or approval.</p></header>
        {!data.claims.length&&<p className="profile-empty">No claims were returned in this snapshot.</p>}
        {data.claims.map(f=><article key={f.fact_id} className="profile-card"><span className="eyebrow">Claim #{f.fact_id}</span><p className="profile-statement"><MarkdownText text={`${f.subject} ${f.predicate.replaceAll('_',' ')} ${f.object}`} inline/></p><p className="profile-meta">Model confidence {f.confidence.toFixed(2)} · uncalibrated</p><ClaimSources claim={f} api={api}/></article>)}
      </section>
      <section aria-label="Recent source context"><header><h3>Recent source context</h3><p>Up to 200 characters from each original, most recent by its recorded source time—not its import time.</p></header>
        {!data.recent.length&&<p className="profile-empty">No recent sources were returned in this snapshot.</p>}
        {data.recent.map(r=><article key={r.episode_id} className="profile-card"><span className="eyebrow">Episode #{r.episode_id}</span><p className="profile-statement"><MarkdownText text={r.excerpt} inline/></p><time className="profile-meta" dateTime={r.created_at}>Source date · {new Date(r.created_at).toLocaleString()}</time><SourceDisclosure id={r.episode_id} api={api}/></article>)}
      </section>
    </div>}
    <p className="profile-footnote">Read at {current.checked}. Refresh after changing memory to inspect a new selection. This is not a complete memory inventory.</p>
  </section>;
}
