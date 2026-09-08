import {useEffect,useState} from 'react';
import type {ApiClient} from '../api';
import {readPersonaCatalog,type ModelChoice,type PersonaOption} from './personas';

export function usePersonaCatalog(api:ApiClient,count:number){
  const [items,setItems]=useState<PersonaOption[]>([]),[state,setState]=useState<'loading'|'ready'|'failed'>('loading'),[attempt,setAttempt]=useState(0);
  useEffect(()=>{
    const controller=new AbortController();setItems([]);
    if(!count){setState('ready');return()=>controller.abort();}
    setState('loading');
    void(async()=>{try{
      const choices=readPersonaCatalog(await api.request('/v1/conversations/personas',{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])}));
      if(!controller.signal.aborted){setItems(choices);setState('ready');}
    }catch{if(!controller.signal.aborted)setState('failed');}})();
    return()=>controller.abort();
  },[api,count,attempt]);
  return {items,state,retry:()=>setAttempt(n=>n+1)};
}

const describe=(choice:ModelChoice|null)=>choice?`${choice.provider} / ${choice.model}`:'Not configured';
export function PersonaPicker({catalog,selected,disabled,onChange,mode='text'}:{catalog:ReturnType<typeof usePersonaCatalog>;selected:string;disabled:boolean;onChange:(id:string)=>void;mode?:'text'|'voice'}){
  const option=catalog.items.find(p=>p.id===selected);
  const ready=(p:PersonaOption)=>mode==='voice'?p.voice_ready&&Boolean(p.fingerprint):p.text_ready;
  return <fieldset className="persona-picker" disabled={disabled}>
    <legend>Choose a persona</legend>
    <p className="conversation-caption">A named instruction set and provider configuration. Your choice stays with this conversation.</p>
    {catalog.state==='loading'?<p role="status">Loading personas…</p>:catalog.state==='failed'?<div role="alert" className="conversation-notice">Personas could not be loaded. No provider has been selected.<button type="button" onClick={catalog.retry}>Retry personas</button></div>:!catalog.items.length?<div className="conversation-notice">No personas are available in this catalog.<button type="button" onClick={catalog.retry}>Retry personas</button></div>:<>
      <div className="persona-options">{catalog.items.map(p=><label key={p.id} className={`persona-option${selected===p.id?' is-selected':''}${!ready(p)?' is-unavailable':''}`}>
        <input type="radio" name="conversation-persona" aria-label={p.name} checked={selected===p.id} disabled={!ready(p)} onChange={()=>onChange(p.id)}/>
        <span><strong>{p.name}</strong><small>{describe(p.reply)}</small><small>{mode==='voice'?(ready(p)?'Voice configured':'Voice unavailable'):(p.text_ready?'Text configured':'Text unavailable')}</small></span>
      </label>)}</div>
      {option&&<div className="persona-detail" aria-label="Selected persona configuration"><dl>
        <div><dt>Replies</dt><dd>{describe(option.reply)}</dd></div>
        <div><dt>Transcription</dt><dd>{describe(option.transcription)}</dd></div>
        <div><dt>Voice configuration</dt><dd>{option.speech?`${describe(option.speech)} · ${option.speech.voice}`:'Not configured'}</dd></div>
        <div><dt>Speech activity</dt><dd>{option.activity?describe(option.activity):'Recognizer events'}</dd></div>
        <div><dt>Provider version</dt><dd>{option.fingerprint?<code>{option.fingerprint}</code>:'Not supplied by this server'}</dd></div>
      </dl><p className="conversation-caption">{mode==='voice'?'You will enable your microphone separately after starting. Audio goes to the providers configured in this persona.':'This starts a text conversation. Selecting a voice configuration does not activate audio.'}</p></div>}
    </>}
  </fieldset>;
}
