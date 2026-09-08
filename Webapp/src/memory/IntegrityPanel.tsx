import {useEffect,useRef,useState} from 'react';
import {ApiError,type ApiClient} from '../api';
import {WorkspaceIcon} from '../components/WorkspaceIcon';
import {findingKeys,parseIntegrityReport,type FindingKey,type IntegrityReport} from './integrity';
import './integrity.css';

const findings:Record<FindingKey,{title:string;record:string;help:string}>={
  chunks_without_episode:{title:'Excerpts without sources',record:'Chunk',help:'These chunks refer to a source episode the checker could not find.'},
  vectors_without_chunk:{title:'Search vectors without excerpts',record:'Chunk',help:'These vector entries have no matching chunk in the inspected document store.'},
  facts_citing_forgotten:{title:'Claims with forgotten sources',record:'Claim',help:'The source was deliberately forgotten. The claim remains recorded, but its original evidence is no longer retained. This does not establish that the claim is false.'},
  facts_citing_unknown:{title:'Claims with unknown sources',record:'Claim',help:'Neither the source nor a record of its deliberate forgetting was found.'},
  links_with_missing_ends:{title:'Relationships with missing claims',record:'Link',help:'At least one claim referenced by each relationship was not found.'},
  attachments_unlinked:{title:'Attachments without sources',record:'Attachment',help:'These stored attachments are not linked to a retained source. They may be uploads that have not yet been attached; this check does not delete them.'},
};

function FindingGroup({kind,ids}:{kind:FindingKey;ids:readonly (number|string)[]}){
  const [limit,setLimit]=useState(50);const info=findings[kind];
  return <details className="integrity-finding"><summary><span>{info.title}</span><span>{ids.length}</span></summary>
    <p>{info.help}</p><ul>{ids.slice(0,limit).map(id=><li key={id}>{info.record} #{id}</li>)}</ul>
    {limit<ids.length&&<button className="btn quiet small" onClick={()=>setLimit(n=>n+50)}>Show more IDs</button>}
    <small>Showing {Math.min(limit,ids.length)} of {ids.length} recorded IDs.</small>
  </details>;
}

export function IntegrityPanel({api,space}:{api:ApiClient;space:string}){
  const [report,setReport]=useState<IntegrityReport|null>(null),[error,setError]=useState('');
  const [running,setRunning]=useState(false),[checkedAt,setCheckedAt]=useState('');
  const request=useRef<AbortController|null>(null);
  useEffect(()=>{setReport(null);setError('');setCheckedAt('');setRunning(false);return()=>{request.current?.abort();request.current=null;};},[api,space]);
  const run=async()=>{
    if(!space||request.current)return;
    const controller=new AbortController();request.current=controller;
    setRunning(true);setReport(null);setError('');setCheckedAt('');
    try{
      const value=await api.request<unknown>('/v1/doctor',{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(30000)])});
      const next=parseIntegrityReport(value,space);
      if(!controller.signal.aborted){setReport(next);setCheckedAt(new Date().toLocaleTimeString());}
    }catch(e){if(!controller.signal.aborted)setError(e instanceof ApiError&&(e.status===404||e.status===501)
      ?'This server advertised integrity checks but the operation is unavailable. Check the server version before trying again.'
      :e instanceof Error?e.message:'The integrity check could not finish. No repairs were requested.');
    }finally{if(request.current===controller){request.current=null;setRunning(false);}}
  };
  const partial=report&&(report.not_inspected.length>0||report.tombstones===null);
  return <section className="integrity-panel" aria-labelledby="integrity-heading">
    <header><div><span className="eyebrow">Read-only diagnostics</span><h2 id="integrity-heading"><WorkspaceIcon name="status"/>Memory integrity</h2></div>
      <button className="btn quiet" disabled={running||!space} onClick={()=>void run()}>{running?'Checking references…':'Run integrity check'}</button></header>
    <p className="integrity-intro">Check whether sources, excerpts, claims and search entries still point to retained records. Nothing is repaired or removed.</p>
    {!space&&<p role="status">The memory space must be identified before a check can run.</p>}
    {running&&<p role="status">Inspecting this space’s stores. Large spaces may take longer; this check does not run automatically.</p>}
    {error&&<p role="alert" className="integrity-error">{error}</p>}
    {report&&<div className="integrity-result" key={checkedAt}>
      <div className={'integrity-verdict '+(partial||!report.healthy?'needs-attention':'clear')} role="status">
        <h3>{partial?'Partial check':report.healthy?'No dangling references found':'References need attention'}</h3>
        <p>{partial?'Some stores were not inspected. Findings below cover only the stores the server could check.':report.healthy?'The inspected references were consistent. This does not verify the truth of claims or the relevance of recall.':'Inspect the categories below. Missing support may reflect deliberate forgetting, not data corruption.'}</p>
        {report.not_inspected.length>0&&<p>Not inspected: {report.not_inspected.join(', ')}</p>}
        {report.tombstones===null&&<p>Records of forgotten sources were not counted.</p>}
      </div>
      <dl className="integrity-counts">{([['Sources',report.episodes],['Excerpts',report.chunks],['Claims',report.facts],['Relationships',report.links],['Forgotten-source records',report.tombstones]] as const).map(([label,n])=><div key={label}><dt>{label}</dt><dd>{n===null?'Not reported':n.toLocaleString()}</dd></div>)}</dl>
      {findingKeys.map(kind=>{const ids=report[kind];return ids&&ids.length>0?<FindingGroup key={kind} kind={kind} ids={ids}/>:null;})}
      <p className="integrity-footnote">Space {report.space} · Checked at {checkedAt}. New writes can change these results. This report does not certify an atomic snapshot or perform repairs.</p>
    </div>}
  </section>;
}
