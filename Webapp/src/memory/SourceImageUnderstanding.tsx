import {useEffect,useRef,useState} from 'react';
import {Link} from 'react-router-dom';
import type {ApiClient,ImageAttachment} from '../api';
import {SourceContent} from '../components/SourceContent';
import {imageMemoryBody,parseImageUnderstanding,type ImageUnderstandingReceipt} from './image-understanding';
import {sourceAddress} from './source-address';
import './image-understanding.css';

export function SourceImageUnderstanding({api,episodeId,space,images,available,canSetup}:{api:ApiClient;episodeId:number;space:string;images:ImageAttachment[];available:boolean;canSetup:boolean}){
  const [selected,setSelected]=useState(images[0]?.attachment_id??'');
  const [prompt,setPrompt]=useState('Describe the visible image and transcribe readable text. State any uncertainty.');
  const [result,setResult]=useState<{receipt:ImageUnderstandingReceipt;prompt:string;requestId:string}|null>(null);
  const [busy,setBusy]=useState<'analysis'|'save'|null>(null),[error,setError]=useState(''),[saved,setSaved]=useState<number|null>(null);
  const active=useRef<AbortController|null>(null),serial=useRef(0),owner=useRef(api);owner.current=api;
  const cancel=()=>{active.current?.abort();serial.current++;};
  const clear=()=>{cancel();setResult(null);setSaved(null);setError('');setBusy(null);};
  useEffect(()=>{clear();setSelected(images[0]?.attachment_id??'');return cancel;},[api,episodeId,space,images,available]);
  const begin=(kind:'analysis'|'save')=>{cancel();const controller=new AbortController();active.current=controller;const id=serial.current;setBusy(kind);setError('');return {signal:AbortSignal.any([controller.signal,AbortSignal.timeout(kind==='analysis'?610000:30000)]),current:()=>!controller.signal.aborted&&serial.current===id&&owner.current===api};};
  const analyze=async()=>{
    if(!available||!images.some(image=>image.attachment_id===selected)||!prompt.trim())return;
    const operation=begin('analysis');setResult(null);setSaved(null);
    try{
      const value=await api.request<unknown>(`/v1/episodes/${episodeId}/attachments/${selected}/understand`,{method:'POST',body:JSON.stringify({prompt}),signal:operation.signal,cache:'no-store',redirect:'error'});
      const receipt=parseImageUnderstanding(value,episodeId,selected);
      if(operation.current())setResult({receipt,prompt,requestId:crypto.randomUUID()});
    }catch(failure){if(operation.current())setError(failure instanceof Error?failure.message:'Image understanding failed.');}
    finally{if(operation.current())setBusy(null);}
  };
  const save=async()=>{
    if(!result||saved!==null)return;
    const operation=begin('save');
    try{
      const value=await api.request<unknown>('/v1/episodes',{method:'POST',body:JSON.stringify(imageMemoryBody(result.receipt,result.prompt,result.requestId)),signal:operation.signal,redirect:'error'});
      if(!value||typeof value!=='object'||!('episode_id' in value)||typeof value.episode_id!=='number'||!Number.isSafeInteger(value.episode_id)||value.episode_id<1)throw Error('The memory save returned an invalid receipt. Refresh before retrying.');
      if(operation.current())setSaved(value.episode_id);
    }catch(failure){if(operation.current())setError(failure instanceof Error?failure.message:'The interpretation could not be saved.');}
    finally{if(operation.current())setBusy(null);}
  };
  if(!images.length||(!available&&!canSetup))return null;
  return <section className="source-understanding" aria-label="Image understanding"><header><span className="eyebrow">From image to evidence</span><h2>Understand this image</h2><p>Ask your self-hosted vision model about a retained image.</p></header>
    {!available?<p><Link to="/memory#models">Configure an image model in Models</Link>, then refresh this source to enable understanding.</p>:<>
      <form onSubmit={event=>{event.preventDefault();void analyze();}}><label>Source image<select aria-label="Image to understand" value={selected} disabled={busy==='save'} onChange={event=>{clear();setSelected(event.target.value);}}>{images.map(image=><option key={image.attachment_id} value={image.attachment_id}>{image.filename||`Image ${image.attachment_id.slice(0,12)}`} · {image.media_type}</option>)}</select></label>
        <label>Task<textarea aria-label="Image understanding task" required maxLength={16000} rows={3} value={prompt} disabled={busy==='save'} onChange={event=>{clear();setPrompt(event.target.value);}}/></label><div><button className="btn primary" type="submit" disabled={busy!==null||!prompt.trim()}>{busy==='analysis'?'Understanding image…':'Understand image'}</button>{busy==='analysis'&&<button className="btn quiet" type="button" onClick={clear}>Cancel</button>}</div></form>
      {error&&<p className="source-understanding-error" role="alert">{error}</p>}
      {result&&<div className="source-understanding-result"><header><strong>{saved===null?'Unsaved model interpretation':'Saved model interpretation'}</strong><span>{result.receipt.understanding.model} · {result.receipt.understanding.width} × {result.receipt.understanding.height}</span></header><SourceContent text={result.receipt.understanding.text}/><p>Generated from episode #{episodeId} and image {selected.slice(0,12)}. This interpretation may be mistaken; saving retains it as source material.</p>
        {saved===null?<button className="btn primary" type="button" disabled={busy!==null} onClick={()=>void save()}>{busy==='save'?'Saving interpretation…':'Save interpretation as memory'}</button>:<p role="status">Saved as <Link to={sourceAddress(saved,space)}>source episode #{saved}</Link>. No facts were approved by this action.</p>}
      </div>}
    </>}
  </section>;
}
