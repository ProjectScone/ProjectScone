import {useEffect,useState} from 'react';
import {Link} from 'react-router-dom';
import type {ApiClient} from '../api';
import {WorkspaceState} from '../components/WorkspaceState';
import {WorkspaceIcon} from '../components/WorkspaceIcon';
import {IntegrityPanel} from './IntegrityPanel';
import {ConsolidationPanel} from './ConsolidationPanel';
import {JobsPanel} from './JobsPanel';
import {parseProcessingStatus,type ProcessingStatus} from './processing';
import './processing.css';

const count=(value:number|undefined)=>value===undefined?'Not reported':value.toLocaleString();
const extractionLabels:Record<string,string>={manual:'Not configured',active:'Enabled',stopped:'Stopped',paused:'Paused'};

export function StatusView({api,integrity,review,documents,maintenance=false,inference=false,jobs=false}:{api:ApiClient;integrity:boolean;review:boolean;documents:boolean;maintenance?:boolean;inference?:boolean;jobs?:boolean}){
  const [attempt,setAttempt]=useState(0);
  const [processingBusy,setProcessingBusy]=useState(false);
  const [result,setResult]=useState<{api:ApiClient;data?:ProcessingStatus;error?:string;checked?:string}|null>(null);
  useEffect(()=>{
    const controller=new AbortController();setResult(null);
    api.request<unknown>('/v1/status',{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])})
      .then(parseProcessingStatus).then(data=>{if(!controller.signal.aborted)setResult({api,data,checked:new Date().toLocaleTimeString()});})
      .catch(error=>{if(!controller.signal.aborted)setResult({api,error:error instanceof Error?error.message:'Status could not be read.'});});
    return()=>controller.abort();
  },[api,attempt]);
  const current=result?.api===api?result:null,s=current?.data;
  const extractionMode=s?.model===null?'Not configured':s?.model!==undefined?'Configured':s?.semantic_lane?extractionLabels[s.semantic_lane]??`Unrecognized mode: ${s.semantic_lane}`:'Not reported';
  const loading=!current;
  const refresh=()=>{setResult(null);setAttempt(n=>n+1);};
  return <div className="processing-workspace">
    <header className="processing-toolbar"><div className="status-access"><span className="status-access-icon"><WorkspaceIcon name="memory"/></span><div><span className="eyebrow">{s?'Memory access verified':'Connection status'}</span><h2>{s?'Your memory, at a glance':'Checking your memory'}</h2><p>{s?`${s.space} · Checked ${current?.checked}`:'Inspecting the status reported by this memory server.'}</p></div></div>
      <button className="btn quiet" disabled={loading||processingBusy} onClick={refresh}>{loading?'Reading status…':'Refresh status'}</button></header>
    {current?.error?<WorkspaceState icon="status" role="alert" title="Status unavailable" description={<><p>{current.error}</p><p>Previous counts have been cleared. Refresh to request a new report.</p></>}/>:!s?<WorkspaceState icon="status" role="status" busy title="Reading memory status…" description="No workers or models are started by this page."/>:<>
      {s.last_distill?.error&&<div className="status-attention" role="alert"><WorkspaceIcon name="status"/><div><h2>The last extraction pass failed</h2><p>Retained sources can still be searched. Some sources may not have produced review proposals.</p><details><summary>View extraction error</summary><p>{s.last_distill.error}</p></details></div></div>}
      <section className="processing-overview" aria-labelledby="processing-heading">
        <header><h2 id="processing-heading">Memory processing</h2><p>From saved sources to reviewed memory.</p></header>
        <div className="processing-grid">
          <article aria-label="Sources"><span className="eyebrow">01 · Retained material</span><h3>Sources</h3><strong className="processing-count">{count(s.episodes)}</strong><span>source episodes</span><p>{count(s.chunks)} stored chunks</p>{documents&&<Link to="#documents">Open documents</Link>}</article>
          <article aria-label="Extraction"><span className="eyebrow">02 · From sources</span><h3>Extraction</h3><strong className="processing-count">{count(s.pending_distill)}</strong><span>episodes awaiting extraction</span><p className="processing-mode">{extractionMode}</p><p className={s.failed_distill? 'processing-failure':''}>{s.failed_distill===undefined?'Failed extraction count not reported':`${count(s.failed_distill)} failed episode${s.failed_distill===1?'':'s'}`}</p></article>
          <article aria-label="Inference"><span className="eyebrow">03 · From claims</span><h3>Inference</h3><strong className="processing-count">{count(s.pending_derivation)}</strong><span>groups awaiting derivation</span><p className="processing-mode">{s.derivation==='on'?'Configured':s.derivation==='off'?'Off':s.derivation?`Unrecognized mode: ${s.derivation}`:'Not reported'}</p></article>
          <article aria-label="Review"><span className="eyebrow">04 · Your decision</span><h3>Review</h3><strong className="processing-count">{count(s.pending_review)}</strong><span>proposals awaiting review</span><p>{s.pending_review===0?'Your review queue is clear.':'Ready for your decision.'}</p>{review&&<Link to="#review">Open review</Link>}</article>
        </div>
        <details className="processing-note"><summary>About these counts</summary><p>Sources and claims have different lifecycles. These counts are not a per-document readiness receipt. Storage counts do not establish search readiness, and configured processing is not proof of a running worker. An empty backlog does not establish that every source was processed successfully. Reading or refreshing status never runs processing or approval. Missing counters remain unknown, not zero.</p></details>
      </section>
      {(maintenance||inference)&&<ConsolidationPanel api={api} space={s.space} maintenance={maintenance} inference={inference} onBusy={setProcessingBusy}/>}
      <div className="status-details">
      <section className="processing-storage" aria-labelledby="storage-heading"><h2 id="storage-heading">Storage & search configuration</h2><dl>{([['Stored bytes',count(s.bytes)],['Revision',count(s.revision)],['Extraction model',s.model===null?'Not configured':s.model??'Not reported'],['Embedder',s.embedder??'Not reported'],['Vector index',s.vector_index??'Not reported'],['Document store',s.document_store??'Not reported']] as const).map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl></section>
      {integrity&&<IntegrityPanel api={api} space={s.space}/>}
      </div>
      {jobs&&<JobsPanel api={api} space={s.space}/>}
    </>}
  </div>;
}
