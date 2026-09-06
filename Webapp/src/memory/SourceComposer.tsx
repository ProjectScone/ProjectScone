import {useEffect, useRef, useState, type FormEvent} from 'react';
import {PREVIEW_TYPES, type ApiClient} from '../api';
import {SourceImages} from '../components/SourceImages';
import './source-composer.css';

interface SavedSource {episode_id:number; deduplicated:boolean; attachmentId?:string}

export function SourceComposer({api,onSaved}: {api:ApiClient;onSaved:()=>void}) {
  const [open,setOpen]=useState(false),[note,setNote]=useState(''),[file,setFile]=useState<File|null>(null);
  const [error,setError]=useState(''),[phase,setPhase]=useState(''),[uncertain,setUncertain]=useState(false);
  const [saved,setSaved]=useState<SavedSource|null>(null),[inspect,setInspect]=useState(false);
  const request=useRef<AbortController|null>(null),busy=useRef(false);
  useEffect(()=>()=>request.current?.abort(),[api]);

  async function save(event:FormEvent) {
    event.preventDefault();if(busy.current||uncertain||!note.trim())return;
    if(file&&(!PREVIEW_TYPES.has(file.type)||!file.size||file.size>25*1024*1024)){
      setError('Choose a nonempty PNG, JPEG, GIF or WebP image up to 25 MB.');return;
    }
    busy.current=true;const controller=new AbortController();request.current=controller;
    setError('');setSaved(null);setInspect(false);let writing=false;
    try {
      setPhase(file?'Uploading original image…':'Saving source…');
      writing=true;
      const attachment=file?await api.uploadImage(file,controller.signal):null;
      if(controller.signal.aborted)return;
      setPhase('Saving source…');
      const result=await api.request<{episode_id:number;deduplicated:boolean}>('/v1/episodes',{
        method:'POST',signal:AbortSignal.any([controller.signal,AbortSignal.timeout(15000)]),
        body:JSON.stringify({content:note,kind:'note',attachment_ids:attachment?[attachment.attachment_id]:[]}),
      });
      if(!Number.isSafeInteger(result.episode_id)||result.episode_id<1||typeof result.deduplicated!=='boolean')throw Error('The save receipt could not be verified');
      if(controller.signal.aborted)return;
      setPhase('Verifying saved source…');
      const source=await api.request<{episode_id:number;attachments?:{attachment_id:string}[]}>(`/v1/episodes/${result.episode_id}`,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])});
      if(source.episode_id!==result.episode_id||(attachment&&!source.attachments?.some(a=>a.attachment_id===attachment.attachment_id)))throw Error('The saved attachment link could not be verified');
      if(!controller.signal.aborted){setSaved({...result,attachmentId:attachment?.attachment_id});onSaved();}
    } catch(e) {
      if(!controller.signal.aborted){setError(e instanceof Error?e.message:'Could not confirm the save');setUncertain(writing);}
    } finally {busy.current=false;if(!controller.signal.aborted)setPhase('');}
  }

  if(!open)return <div className="source-entry"><button className="btn quiet" onClick={()=>setOpen(true)}>Add source</button><span>Save a note with its original image.</span></div>;
  return <section className="source-composer" aria-label="Add a source">
    <header><div><span className="eyebrow">Your knowledge, at the source</span><h2>Add a source</h2></div><button className="btn quiet small" disabled={!!phase} onClick={()=>setOpen(false)}>Close</button></header>
    <p>Write the context you want to find later. An optional image is kept as original evidence; it is not read or described by a model.</p>
    {saved?<div className="source-saved" role="status"><h3>Source saved · episode #{saved.episode_id}</h3><p>{saved.deduplicated?'Identical note text already existed. That episode was reused.':'A new source episode was created.'} {saved.attachmentId?'The image link was verified.':''}</p>
      {saved.attachmentId&&<button className="btn quiet" onClick={()=>setInspect(!inspect)}>{inspect?'Close saved preview':'Inspect saved source'}</button>}
      {inspect&&<SourceImages api={api} episodeId={saved.episode_id} initiallyOpen/>}
      <button className="btn quiet" onClick={()=>{setSaved(null);setNote('');setFile(null);setError('');setInspect(false);}}>Add another source</button>
    </div>:<form onSubmit={save}>
      <label>Source note<textarea value={note} onChange={e=>setNote(e.target.value)} required maxLength={100000} disabled={!!phase||uncertain} placeholder="What does this source show, and why does it matter?"/></label>
      <div className="source-file"><label>Original image<input type="file" accept="image/png,image/jpeg,image/gif,image/webp" disabled={!!phase||uncertain} onChange={e=>{setFile(e.target.files?.[0]??null);setError('');}}/></label><small>Optional · PNG, JPEG, GIF or WebP · up to 25 MB</small></div>
      {file&&<p className="source-selection">{file.name} · {(file.size/1024).toFixed(1)} KB</p>}
      {error&&<div role="alert" className="source-save-error"><p>{error}</p>{uncertain&&<p>The save is unconfirmed. An image or note may already be stored. Check Memory Search before trying again; this form will not repeat the write.</p>}</div>}
      <footer><span>{phase||'Will save to the connected memory space. This does not approve a claim.'}</span><button className="btn" disabled={!!phase||uncertain||!note.trim()}>{phase?'Saving…':'Save source'}</button></footer>
      {phase&&<p role="status">{phase} Keep this page open until the result is confirmed.</p>}
    </form>}
  </section>;
}
