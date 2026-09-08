import {SourceContent} from '../components/SourceContent';
import {WorkspaceState} from '../components/WorkspaceState';
import {useCallback,useEffect,useRef,useState} from 'react';
import {ApiError,type ApiClient} from '../api';
import {SourceImages} from '../components/SourceImages';
import {SourceComposer} from './SourceComposer';
import {parseRetainedSource,parseSourcePage,type SourcePage,type SourceSummary} from './source-inventory';
import './documents.css';

const KINDS=[['','All sources'],['file','Files'],['note','Notes'],['conversation','Conversations'],['observation','Observations'],['connector','Connectors']] as const;
interface Query {kind:string;cursors:(number|undefined)[]}
const first=(kind=''):Query=>({kind,cursors:[undefined]});
const name=(source:SourceSummary)=>source.source||`${source.kind.charAt(0).toUpperCase()+source.kind.slice(1)} #${source.episode_id}`;
const failure=(error:unknown)=>error instanceof Error?error.message:'Could not load sources.';
function SourceMark(){return <svg viewBox="0 0 24 28" fill="none" aria-hidden="true"><path d="M5 1h9l8 8v17H2V1h3Z" stroke="currentColor" strokeWidth="1.4"/><path d="M14 1v8h8M7 15h10M7 20h7" stroke="currentColor" strokeWidth="1.4"/></svg>;}

export function DocumentsView({api,attachments}:{api:ApiClient;attachments:boolean}) {
  const [snapshot,setSnapshot]=useState<{api:ApiClient;query:Query;data:SourcePage}|null>(null);
  const [work,setWork]=useState<{query:Query;loading:boolean;error?:string}>({query:first(),loading:true});
  const [selection,setSelection]=useState<SourceSummary|null>(null);
  const request=useRef<AbortController|null>(null);
  const selectedButton=useRef<HTMLButtonElement|null>(null);
  const run=useCallback(async(query:Query)=>{
    request.current?.abort();const controller=new AbortController();request.current=controller;
    setWork({query,loading:true});setSelection(null);
    const before=query.cursors.at(-1);
    const params=new URLSearchParams({limit:'25'});
    if(query.kind)params.set('kind',query.kind);
    if(before!==undefined)params.set('before',String(before));
    try{
      const value=await api.request<unknown>('/v1/sources?'+params,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(15000)])});
      const data=parseSourcePage(value,{kind:query.kind||undefined,before});
      if(!controller.signal.aborted){setSelection(null);setSnapshot({api,query,data});setWork({query,loading:false});}
    }catch(error){if(!controller.signal.aborted)setWork({query,loading:false,error:failure(error)});}
  },[api]);
  useEffect(()=>{setSnapshot(null);void run(first());return()=>request.current?.abort();},[run]);
  const current=snapshot?.api===api?snapshot:null;
  const query=current?.query??work.query;
  const data=current?.data;
  return <section className="documents" aria-label="Source library">
    {attachments&&<SourceComposer api={api} onSaved={()=>void run(first())}/>}
    <div className="documents-toolbar"><div><span className="eyebrow">Stored sources</span><p>Explore the material behind your memory.</p></div><button className="btn quiet small" disabled={work.loading} onClick={()=>void run(first(query.kind))}>Refresh sources</button></div>
    <div className="documents-filters" role="group" aria-label="Filter sources by type">{KINDS.map(([kind,label])=><button key={kind} aria-pressed={query.kind===kind} onClick={()=>void run(first(kind))}>{label}</button>)}</div>
    <div className="documents-caption"><span>{data?`${data.items.length} sources on this page`:'Source inventory'}</span><span>Newest stored IDs first · not search ranking</span></div>
    {work.loading&&<p className="documents-notice" role="status">Loading {work.query.kind||'all'} sources… {data?'The previous page stays visible until this request finishes.':''}</p>}
    {work.error&&<div className="documents-notice err" role="alert"><p>{work.error} {data?'The last confirmed page is still shown.':''}</p><button className="btn quiet small" onClick={()=>void run(work.query)}>Retry sources</button></div>}
    <div className={'documents-workspace'+(selection&&current?' has-selection':'')}>
      <div>
        {data?.items.length?<ul className="documents-list" aria-label="Stored sources">{data.items.map(source=><li key={source.episode_id}><button className="document-card" aria-label={'Open '+name(source)} aria-pressed={selection?.episode_id===source.episode_id} onClick={event=>{selectedButton.current=event.currentTarget;setSelection(source);}}>
          <span className="document-mark"><SourceMark/></span><span className="document-copy"><span className="document-meta"><span>{source.kind}</span><span>#{source.episode_id}</span></span><strong>{name(source)}</strong><span className="document-preview">{source.preview||'No retained text.'}</span><span className="document-footer"><time dateTime={source.created_at}>{source.created_at.slice(0,10)||'Date unavailable'}</time><span>{source.byte_count.toLocaleString()} text bytes{source.preview_truncated?' · excerpt':''}</span></span></span><span className="document-open" aria-hidden="true">↗</span>
        </button></li>)}</ul>:data&&!work.loading?<WorkspaceState className="documents-empty" icon="documents" title={`No ${query.kind||'stored'} sources on this page.`} description={<p>{query.kind?'Try another source type or return to all sources.':query.cursors.length>1?'Older sources may have been removed. Refresh to return to the newest records.':'Sources appear here when content is saved to this memory space.'}</p>}/>:null}
        {data&&<nav className="documents-paging" aria-label="Source pages"><button className="btn quiet small" disabled={work.loading||query.cursors.length===1} onClick={()=>void run({...query,cursors:query.cursors.slice(0,-1)})}>Newer sources</button><span>Page {query.cursors.length}</span><button className="btn quiet small" disabled={work.loading||!data.has_more} onClick={()=>void run({...query,cursors:[...query.cursors,data.next_before!]})}>Older sources</button></nav>}
      </div>
      {selection&&current&&<SourceDetail key={selection.episode_id} source={selection} api={api} attachments={attachments} close={()=>{setSelection(null);requestAnimationFrame(()=>{if(selectedButton.current?.isConnected)selectedButton.current.focus();});}}/>}
    </div>
    <p className="documents-footnote">Each entry is one retained source episode, not an approved claim. This is a live inventory, not a frozen snapshot; refresh to see newly stored sources.</p>
  </section>;
}

function SourceDetail({source,api,attachments,close}:{source:SourceSummary;api:ApiClient;attachments:boolean;close:()=>void}) {
  const [content,setContent]=useState<string|null>(null),[error,setError]=useState(''),[attempt,setAttempt]=useState(0);
  const heading=useRef<HTMLHeadingElement>(null);
  useEffect(()=>{heading.current?.focus();},[]);
  useEffect(()=>{
    const controller=new AbortController();setContent(null);setError('');
    api.request<unknown>(`/v1/episodes/${source.episode_id}`,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(15000)])})
      .then(value=>parseRetainedSource(value,source.episode_id)).then(value=>{if(!controller.signal.aborted)setContent(value.content);})
      .catch(e=>{if(!controller.signal.aborted)setError(e instanceof ApiError&&e.status===404?'This source is no longer available in this memory space.':failure(e));});
    return()=>controller.abort();
  },[api,source.episode_id,attempt]);
  let href:string|null=null;
  try{const url=new URL(source.source??'');if(url.protocol==='https:'||url.protocol==='http:')href=url.href;}catch{/* Native paths and filenames are provenance, not links. */}
  return <section className="document-detail" aria-label="Retained source">
    <header><span className="eyebrow">Retained source · #{source.episode_id}</span><button className="btn quiet small" onClick={close}>Close source</button></header>
    <h2 tabIndex={-1} ref={heading}>{name(source)}</h2><p className="document-provenance">{source.kind} · {source.created_at||'Date unavailable'}</p>
    {href&&<a href={href} target="_blank" rel="noopener noreferrer">Open source link ↗</a>}
    <p className="document-detail-note">Text retained by Scone, shown as written. This is source material, not a verified belief or an original-file download.</p>
    {error?<div role="alert"><p>{error}</p><button className="btn quiet small" onClick={()=>setAttempt(n=>n+1)}>Retry source</button></div>:content===null?<p role="status">Loading source…</p>:<div aria-label="Retained text">{content===''&&<p>No retained text.</p>}<SourceContent text={content}/></div>}
    {attachments&&content!==null&&<SourceImages api={api} episodeId={source.episode_id}/>}
  </section>;
}
