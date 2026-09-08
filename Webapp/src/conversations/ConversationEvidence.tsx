import {useEffect,useState} from 'react';
import {ApiError,type ApiClient} from '../api';
import {SourceImages} from '../components/SourceImages';
import {SourceContent} from '../components/SourceContent';
import {episode,type Episode,type TurnResult} from './contracts';
import {QueryEvidenceGraph} from '../memory/QueryEvidenceGraph';
import {useConversationFade} from './useConversationFade';

export function ConversationEvidence({api,result,selected,onSelect}:{api:ApiClient;result?:TurnResult;selected:number|null;onSelect:(id:number)=>void}){
  const [attempt,setAttempt]=useState(0);
  const [snapshot,setSnapshot]=useState<{api:ApiClient;selected:number;attempt:number;source?:Episode;error?:string}|null>(null);
  const current=snapshot?.api===api&&snapshot.selected===selected&&snapshot.attempt===attempt?snapshot:null;
  const source=current?.source,error=current?.error;
  const sourceFade=useConversationFade<HTMLElement>(`${selected}:${attempt}:${source?'ready':error?'error':'pending'}`);
  useEffect(()=>{
    const controller=new AbortController();setSnapshot(null);
    if(selected!==null)api.request(`/v1/episodes/${selected}`,{cache:'no-store',signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])})
      .then(value=>{
        const source=episode(value);
        if(source.episode_id!==selected)throw Error('Source identity mismatch');
        if(!controller.signal.aborted)setSnapshot({api,selected,attempt,source});
      })
      .catch(reason=>{
        const error=reason instanceof ApiError&&reason.status===410?'This source was forgotten. Its retained text is no longer available.'
          :reason instanceof ApiError&&reason.status===404?'Source unavailable in this memory space.'
          :'Source could not be loaded in this memory space.';
        if(!controller.signal.aborted)setSnapshot({api,selected,attempt,error});
      });
    return()=>controller.abort();
  },[api,selected,attempt]);
  const context=result?.memory_context;
  return <aside className="conversation-evidence" aria-label="Conversation evidence">
    <span className="eyebrow">Behind the conversation</span><h2>Memory & sources</h2>
    <p>See the recorded context behind a reply, and the messages saved to memory.</p>
    <section><h3>Latest reply context</h3>{context?<><span className="conversation-state">{context.status}</span>
      {context.references.length?<div className="conversation-source-list">{context.references.map((ref,i)=><button key={`${ref.episode_id}:${i}`} onClick={()=>onSelect(ref.episode_id)}>Source episode {ref.episode_id}</button>)}</div>:<p>No source references recorded for this reply.</p>}
      <p className="conversation-caption">Prepared context is not proof that the model used it. Provider completion remains unverified.</p></>:<p>Context receipts appear after a reply in this visit. Reopened messages do not reconstruct missing receipts.</p>}</section>
    {context?.evidence_graph!=null&&<QueryEvidenceGraph value={context.evidence_graph} api={api} title="Evidence supplied to this reply" context="reply"/>}
    <section ref={sourceFade} className="conversation-source-transition"><h3>{selected===null?'Inspect a saved message':`Episode ${selected}`}</h3>
      {selected===null?<p>Select a message’s source record to read the saved original.</p>:error?<div role="alert"><p>{error}</p><button className="btn quiet small" onClick={()=>setAttempt(n=>n+1)}>Retry source</button></div>:source?<><div className="conversation-source-text"><SourceContent text={source.content}/></div><SourceImages key={source.episode_id} episodeId={source.episode_id} api={api}/></>:<p role="status">Loading source…</p>}
    </section>
    <div className="conversation-evidence-note">Public messages only. No hidden reasoning is captured. Missing observations remain unknown.</div>
  </aside>;
}
