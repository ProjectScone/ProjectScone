import {useEffect,useRef,useState} from 'react';
import {Link,useLocation,useParams} from 'react-router-dom';
import {ApiError,type ApiClient} from '../api';
import {parseCapabilities} from '../capabilities';
import {SourceContent} from '../components/SourceContent';
import {SourceImages} from '../components/SourceImages';
import {WorkspaceState} from '../components/WorkspaceState';
import {parseRetainedSource} from './source-inventory';
import {readSourceAddress,sourceAddress,verifiedSpace} from './source-address';
import './source-page.css';

interface Original {content:string;title:string;kind:string;date:string}
interface PageData {original?:Original;attachments?:boolean;issue?:string;detail?:string}
function original(value:unknown,id:number):Original {
  const text=parseRetainedSource(value,id);
  const data=value as Record<string,unknown>;
  if(typeof data.kind!=='string'||!data.kind||typeof data.created_at!=='string'||!(data.source==null||typeof data.source==='string'))throw Error('Invalid source metadata');
  return {content:text.content,title:typeof data.source==='string'&&data.source?data.source:`Source episode #${id}`,kind:data.kind,date:data.created_at};
}

export function SourcePage({api,enabled}:{api:ApiClient;enabled:boolean}){
  const {id}=useParams(),location=useLocation(),address=readSourceAddress(id,location.search);
  const episodeId=address?.episodeId,space=address?.space;
  const [attempt,setAttempt]=useState(0),[copied,setCopied]=useState('');
  const [snapshot,setSnapshot]=useState<{api:ApiClient;episodeId:number;space:string;attempt:number;data:PageData}|null>(null);
  const heading=useRef<HTMLHeadingElement>(null);
  const current=snapshot?.api===api&&snapshot.episodeId===episodeId&&snapshot.space===space&&snapshot.attempt===attempt?snapshot.data:null;
  useEffect(()=>{
    setSnapshot(null);setCopied('');
    if(!enabled||episodeId===undefined||space===undefined)return;
    const controller=new AbortController(),signal=AbortSignal.any([controller.signal,AbortSignal.timeout(15000)]);
    const save=(data:PageData)=>{if(!controller.signal.aborted)setSnapshot({api,episodeId,space,attempt,data});};
    void (async()=>{
      try{
        const actual=verifiedSpace(await api.request<unknown>('/v1/status',{cache:'no-store',signal}));
        if(actual!==space){save({issue:'This source belongs to a different space',detail:`This link names “${space}”. Your key is connected to “${actual}”. Switch to the intended space using Memory connection; no source was read.`});return;}
        const caps=parseCapabilities(await api.request<unknown>('/v1/capabilities',{signal}));
        if(!caps.features['episodes.read']){save({issue:'Source pages are unavailable on this server',detail:'The server does not advertise source reads.'});return;}
        const record=await api.request<unknown>(`/v1/episodes/${episodeId}`,{cache:'no-store',signal});
        save({original:original(record,episodeId),attachments:caps.features['episodes.attachments']});
      }catch(error){
        save(error instanceof ApiError&&error.status===410?{issue:'This source was forgotten',detail:'Its retained text is no longer available.'}
          :error instanceof ApiError&&error.status===404?{issue:'Source unavailable',detail:'This source is not available in the linked memory space.'}
          :{issue:'Source could not be verified',detail:'The original could not be read safely. Retry checks the same space and episode again.'});
      }
    })();
    return()=>controller.abort();
  },[api,enabled,episodeId,space,attempt]);
  useEffect(()=>{heading.current?.focus();},[current?.original]);
  const retry=()=>{setSnapshot(null);setAttempt(n=>n+1);};
  return <main id="main" className="source-page">
    <nav aria-label="Source navigation"><Link to="/memory#documents">← Source library</Link><span>Retained source</span></nav>
    {!address?<WorkspaceState icon="documents" title="Invalid source link" description="A source link needs one positive episode ID and one exact memory space. Reopen the source from your library."/>
      :!enabled?<WorkspaceState icon="memory" title="Connect to read this source" description={`Use Connect memory to provide a key for “${address.space}”. This link contains no credentials and grants no access.`}/>
      :current?.issue?<WorkspaceState icon="documents" role="alert" title={current.issue} description={current.detail} actions={<button className="btn quiet" onClick={retry}>Retry source</button>}/>
      :!current?.original?<WorkspaceState icon="documents" role="status" busy title="Opening the original…" description="Checking the linked memory space before reading its source."/>
      :<article><header className="source-page-heading"><div><span className="eyebrow">{space} / Episode #{episodeId}</span><h1 ref={heading} tabIndex={-1}>{current.original.title}</h1><p>{current.original.kind} · {current.original.date||'Date unavailable'}</p></div>
        <div><button className="btn quiet small" onClick={async()=>{try{await navigator.clipboard.writeText(new URL(sourceAddress(address.episodeId,address.space),window.location.origin).href);setCopied('Link copied.');}catch{setCopied('Copy unavailable. Copy the source link below.');}}}>Copy source link</button><span role="status">{copied}</span></div></header>
        {copied.startsWith('Copy unavailable')&&<input aria-label="Source permalink" readOnly value={new URL(sourceAddress(address.episodeId,address.space),window.location.origin).href} onFocus={e=>e.currentTarget.select()}/>}
        <p className="source-page-note">The retained original, not an approved claim. Links require access to this memory space; they do not share your key. Text is not an original-file download.</p>
        <section className="source-page-original" aria-label="Source original"><SourceContent text={current.original.content}/>{current.original.content===''&&<p>No retained text.</p>}</section>
        {current.attachments&&<SourceImages api={api} episodeId={address.episodeId}/>}<footer><span>Episode #{episodeId} · {space}</span><button className="btn quiet small" onClick={retry}>Refresh source</button></footer>
      </article>}
  </main>;
}
