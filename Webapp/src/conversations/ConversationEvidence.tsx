import {useEffect,useState} from 'react';
import type {ApiClient} from '../api';
import {SourceImages} from '../components/SourceImages';
import {SourceContent} from '../components/SourceContent';
import {episode,type Episode,type TurnResult} from './contracts';

export function ConversationEvidence({api,result,selected,onSelect}:{api:ApiClient;result?:TurnResult;selected:number|null;onSelect:(id:number)=>void}){
  const [source,setSource]=useState<Episode|null>(null),[error,setError]=useState('');
  useEffect(()=>{
    const controller=new AbortController();setSource(null);setError('');
    if(selected!==null)api.request(`/v1/episodes/${selected}`,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])})
      .then(value=>{if(!controller.signal.aborted)setSource(episode(value));})
      .catch(()=>{if(!controller.signal.aborted)setError('Source could not be loaded in this memory space.');});
    return()=>controller.abort();
  },[api,selected]);
  const context=result?.memory_context;
  return <aside className="conversation-evidence" aria-label="Conversation evidence">
    <span className="eyebrow">Behind the conversation</span><h2>Memory & sources</h2>
    <p>See the recorded context behind a reply, and the messages saved to memory.</p>
    <section><h3>Latest reply context</h3>{context?<><span className="conversation-state">{context.status}</span>
      {context.references.length?<div className="conversation-source-list">{context.references.map((ref,i)=><button key={`${ref.episode_id}:${i}`} onClick={()=>onSelect(ref.episode_id)}>Source episode {ref.episode_id}</button>)}</div>:<p>No source references recorded for this reply.</p>}
      <p className="conversation-caption">Prepared context is not proof that the model used it. Provider completion remains unverified.</p></>:<p>Context receipts appear after a reply in this visit. Reopened messages do not reconstruct missing receipts.</p>}</section>
    <section><h3>{selected===null?'Inspect a saved message':`Episode ${selected}`}</h3>
      {selected===null?<p>Select a message’s source record to read the saved original.</p>:error?<p role="alert">{error}</p>:source?<><div className="conversation-source-text"><SourceContent text={source.content}/></div><SourceImages key={source.episode_id} episodeId={source.episode_id} api={api}/></>:<p role="status">Loading source…</p>}
    </section>
    <div className="conversation-evidence-note">Public messages only. No hidden reasoning is captured. Missing observations remain unknown.</div>
  </aside>;
}
