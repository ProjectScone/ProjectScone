import {useEffect,useRef,useState} from 'react';
import {ApiError,type ApiClient} from '../api';
import {SourceContent} from '../components/SourceContent';
import {SourceImages} from '../components/SourceImages';
import {parseRetainedSource} from './source-inventory';
import {parseJobPage,type IngestJob,type JobPage,type JobState} from './jobs';
import './jobs.css';

const labels:Record<JobState,string>={searchable:'Searchable',consolidated:'Consolidated',failed:'Needs attention',cancelled:'Cancelled'};
const date=(value:string)=>new Date(value).toLocaleString();

export function JobsPanel({api,space}:{api:ApiClient;space:string}){
  const [cursors,setCursors]=useState<string[]>([]),[attempt,setAttempt]=useState(0),[selected,setSelected]=useState<string|null>(null);
  const [result,setResult]=useState<{api:ApiClient;space:string;cursors:string[];attempt:number;data?:JobPage;error?:string}|null>(null);
  useEffect(()=>{
    const controller=new AbortController();setResult(null);setSelected(null);
    const params=new URLSearchParams({limit:'20'}),before=cursors.at(-1);
    if(before)params.set('before',before);
    api.request<unknown>('/v1/jobs?'+params,{cache:'no-store',signal:AbortSignal.any([controller.signal,AbortSignal.timeout(15000)])})
      .then(value=>parseJobPage(value,space,cursors)).then(data=>{if(!controller.signal.aborted)setResult({api,space,cursors,attempt,data});})
      .catch(()=>{if(!controller.signal.aborted)setResult({api,space,cursors,attempt,error:'Batch history could not be verified. No processing was started. Retry the read or return to the newest batches.'});});
    return()=>controller.abort();
  },[api,space,cursors,attempt]);
  const current=result?.api===api&&result.space===space&&result.cursors===cursors&&result.attempt===attempt?result:null;
  const data=current?.data,job=data?.jobs.find(j=>j.job_id===selected);
  return <section className="jobs-panel" aria-label="Batch history">
    <header className="jobs-heading"><div><span className="eyebrow">Import receipts</span><h2>Batch history</h2><p>Follow what was stored, what became searchable, and what has been read into memory.</p></div>
      <button className="btn quiet small" disabled={!current} onClick={()=>{setCursors([]);setAttempt(n=>n+1);}}>Refresh batches</button></header>
    <p className="jobs-note">Read-only history · Space {space}. Searchable sources are not necessarily consolidated or approved claims. Cancellation does not delete stored sources.</p>
    {!current?<p role="status">Reading batch receipts…</p>:current.error?<div role="alert"><p>{current.error}</p><button className="btn quiet small" onClick={()=>setAttempt(n=>n+1)}>Retry batches</button></div>:data&&!data.jobs.length?<p>No batch receipts on this page. Individually saved sources may exist without a batch receipt.</p>:<>
      <div className="jobs-list">{data?.jobs.map(batch=><button key={batch.job_id} className="job-card" aria-label={`Inspect batch ${batch.job_id}`} aria-pressed={selected===batch.job_id} onClick={()=>setSelected(selected===batch.job_id?null:batch.job_id)}>
        <span className="job-card-title"><strong>{batch.request_id||'Batch '+batch.job_id.slice(0,12)}</strong><span className={'job-state state-'+batch.state}>{labels[batch.state]}</span></span>
        <span className="job-counts"><span>{batch.items.length} records</span><span>{batch.searchable} searchable</span><span>{batch.consolidated} consolidated</span></span>
        <span className="job-date">{date(batch.created_at)} · {batch.job_id}</span>
      </button>)}</div>
      {job&&<JobDetail key={job.job_id} api={api} job={job} close={()=>setSelected(null)}/>}
    </>}
    <nav className="jobs-paging" aria-label="Batch pages"><button className="btn quiet small" disabled={!current||!cursors.length} onClick={()=>setCursors(c=>c.slice(0,-1))}>Newer batches</button><span>Page {cursors.length+1}</span><button className="btn quiet small" disabled={!data?.next} onClick={()=>{if(data?.next)setCursors(c=>[...c,data.next!]);}}>Older batches</button></nav>
    <p className="jobs-note">Refresh checks the server; this is not a live worker monitor. Reading receipts never retries extraction or cancels a job.</p>
  </section>;
}

function JobDetail({api,job,close}:{api:ApiClient;job:IngestJob;close:()=>void}){
  const [page,setPage]=useState(0),[source,setSource]=useState<number|null>(null);
  const detail=useRef<HTMLElement>(null),heading=useRef<HTMLHeadingElement>(null);
  useEffect(()=>{detail.current?.scrollIntoView({block:'start'});heading.current?.focus({preventScroll:true});},[]);
  return <section ref={detail} className="job-detail" aria-label="Batch details"><header><div><span className="eyebrow">Recorded outcomes</span><h3 ref={heading} tabIndex={-1}>{job.request_id||job.job_id}</h3></div><button className="btn quiet small" onClick={close}>Close batch</button></header>
    <p>{job.searchable} searchable · {job.consolidated} consolidated · {job.items.length} records</p>
    {job.cancelled_at&&<p>Cancelled {date(job.cancelled_at)}. Stored records remain available unless separately forgotten.</p>}
    <ol className="job-items" start={page*20+1}>{job.items.slice(page*20,page*20+20).map(item=><li key={item.index}>
      <div className="job-item-heading"><strong>Source #{item.episode_id}</strong><span className={'job-state state-'+item.state}>{labels[item.state]}</span></div>
      <p>{item.outcome} · {item.attempts} failed extraction attempts</p>
      <dl><div><dt>Searchable</dt><dd>{item.searchable_at?date(item.searchable_at):'Not reported'}</dd></div><div><dt>Consolidated</dt><dd>{item.consolidated_at?date(item.consolidated_at):'Not recorded'}</dd></div></dl>
      {item.error&&<details className="job-error"><summary>Last extraction error</summary><pre>{item.error}</pre></details>}
      <button className="btn quiet small" aria-pressed={source===item.episode_id} onClick={()=>setSource(item.episode_id)}>Inspect source {item.episode_id}</button>
    </li>)}</ol>
    {job.items.length>20&&<nav className="jobs-paging" aria-label="Batch records"><button disabled={!page} onClick={()=>{setSource(null);setPage(n=>n-1);}}>Previous records</button><span>{page+1} / {Math.ceil(job.items.length/20)}</span><button disabled={(page+1)*20>=job.items.length} onClick={()=>{setSource(null);setPage(n=>n+1);}}>Next records</button></nav>}
    {source!==null&&<JobSource key={source} api={api} id={source}/>}
  </section>;
}

function JobSource({api,id}:{api:ApiClient;id:number}){
  const [attempt,setAttempt]=useState(0),[result,setResult]=useState<{api:ApiClient;attempt:number;text?:string;error?:string}|null>(null);
  const source=useRef<HTMLElement>(null),heading=useRef<HTMLHeadingElement>(null);
  useEffect(()=>{source.current?.scrollIntoView({block:'start'});heading.current?.focus({preventScroll:true});},[]);
  useEffect(()=>{
    const controller=new AbortController();setResult(null);
    api.request<unknown>(`/v1/episodes/${id}`,{cache:'no-store',signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])})
      .then(value=>parseRetainedSource(value,id)).then(value=>{if(!controller.signal.aborted)setResult({api,attempt,text:value.content});})
      .catch(error=>{if(!controller.signal.aborted)setResult({api,attempt,error:error instanceof ApiError&&error.status===410?'This source was forgotten.':'Source unavailable or could not be verified in this space.'});});
    return()=>controller.abort();
  },[api,id,attempt]);
  const current=result?.api===api&&result.attempt===attempt?result:null;
  return <section ref={source} className="job-source" aria-label="Batch source"><h3 ref={heading} tabIndex={-1}>Original source #{id}</h3>{!current?<p role="status">Reading original source…</p>:current.error?<div role="alert"><p>{current.error}</p><button onClick={()=>setAttempt(n=>n+1)}>Retry source</button></div>:<><SourceContent text={current.text??''}/><SourceImages api={api} episodeId={id}/></>}</section>;
}
