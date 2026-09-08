import {useEffect, useRef, useState, type FormEvent} from 'react';
import {ApiError, PREVIEW_TYPES, type ApiClient} from '../api';
import {SourceImages} from '../components/SourceImages';
import {readTextSource,TEXT_FILE_ACCEPT,type TextSource} from './text-import';
import './source-composer.css';

interface SavedSource {episode_id:number; deduplicated:boolean; attachmentId?:string; text?:string; source?:string|null}

export function SourceComposer({api,onSaved}: {api:ApiClient;onSaved:()=>void}) {
  const [open,setOpen]=useState(false),[note,setNote]=useState(''),[file,setFile]=useState<File|null>(null);
  const [mode,setMode]=useState<'note'|'file'>('note'),[document,setDocument]=useState<TextSource|null>(null);
  const [error,setError]=useState(''),[phase,setPhase]=useState(''),[uncertain,setUncertain]=useState(false);
  const [saved,setSaved]=useState<SavedSource|null>(null),[inspect,setInspect]=useState(false);
  const request=useRef<AbortController|null>(null),busy=useRef(false);
  const reading=useRef(0);
  useEffect(()=>()=>{request.current?.abort();reading.current++;},[api]);
  const content=mode==='file'?document?.content??'':note;

  async function selectDocument(selected?:File){
    const generation=++reading.current;setDocument(null);setError('');
    if(!selected)return;
    busy.current=true;setPhase('Reading selected file locally…');
    try{const next=await readTextSource(selected);if(reading.current===generation)setDocument(next);}
    catch(e){if(reading.current===generation)setError(e instanceof Error?e.message:'Could not read the selected file.');}
    finally{if(reading.current===generation){busy.current=false;setPhase('');}}
  }

  async function save(event:FormEvent) {
    event.preventDefault();if(busy.current||uncertain||!content.trim())return;
    const image=mode==='note'?file:null;
    if(image&&(!PREVIEW_TYPES.has(image.type)||!image.size||image.size>25*1024*1024)){
      setError('Choose a nonempty PNG, JPEG, GIF or WebP image up to 25 MB.');return;
    }
    busy.current=true;const controller=new AbortController();request.current=controller;
    setError('');setSaved(null);setInspect(false);let writing=false,acknowledgedWrite=false;
    try {
      setPhase(image?'Uploading original image…':'Saving source…');
      writing=true;
      const attachment=image?await api.uploadImage(image,controller.signal):null;
      acknowledgedWrite=Boolean(attachment);
      if(controller.signal.aborted)return;
      setPhase('Saving source…');
      const result=await api.request<{episode_id:number;deduplicated:boolean}>('/v1/episodes',{
        method:'POST',signal:AbortSignal.any([controller.signal,AbortSignal.timeout(15000)]),
        body:JSON.stringify({content,kind:mode==='file'?'file':'note',...(mode==='file'?{source:document!.name}:{}),attachment_ids:attachment?[attachment.attachment_id]:[]}),
      });
      acknowledgedWrite=true;
      if(!Number.isSafeInteger(result.episode_id)||result.episode_id<1||typeof result.deduplicated!=='boolean')throw Error('The save receipt could not be verified');
      if(controller.signal.aborted)return;
      setPhase('Verifying saved source…');
      const source=await api.request<{episode_id:number;content?:string;source?:string|null;attachments?:{attachment_id:string}[]}>(`/v1/episodes/${result.episode_id}`,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])});
      if(source.episode_id!==result.episode_id||(attachment&&!source.attachments?.some(a=>a.attachment_id===attachment.attachment_id)))throw Error('The saved attachment link could not be verified');
      if(mode==='file'&&source.content!==content)throw Error('The retained text differs from the selected file. Inspect memory before retrying.');
      if(!controller.signal.aborted){setSaved({...result,attachmentId:attachment?.attachment_id,text:typeof source.content==='string'?source.content:undefined,source:typeof source.source==='string'?source.source:null});onSaved();}
    } catch(e) {
      if(!controller.signal.aborted){
        const deniedBeforeWrite=e instanceof ApiError&&e.status===403&&!acknowledgedWrite;
        setError(deniedBeforeWrite?'This key cannot save sources. Your draft is kept. Use a key with write access.':e instanceof Error?e.message:'Could not confirm the save');
        setUncertain(writing&&!deniedBeforeWrite);
      }
    } finally {busy.current=false;if(!controller.signal.aborted)setPhase('');}
  }

  if(!open)return <div className="source-entry"><button className="btn quiet" onClick={()=>setOpen(true)}>Add source</button><span>Bring in notes, text files and original images.</span></div>;
  return <section className="source-composer" aria-label="Add a source">
    <header><div><span className="eyebrow">Your knowledge, at the source</span><h2>Add a source</h2></div><button className="btn quiet small" disabled={!!phase} onClick={()=>setOpen(false)}>Close</button></header>
    <p>Keep the material you want to find later, with its source intact. Select a file or write a note; nothing is sent until you save.</p>
    {saved?<div className="source-saved" role="status"><h3>Source saved · episode #{saved.episode_id}</h3><p>{saved.deduplicated?'Matching text already existed. That episode was reused; its original source metadata is unchanged.':'A new source episode was created.'} {saved.attachmentId?'The image link was verified.':''}</p>
      {saved.text!==undefined&&<details><summary>Read saved text</summary><p>Verified at save · recorded source: {saved.source||'not specified'}</p><pre>{saved.text}</pre></details>}
      {saved.attachmentId&&<button className="btn quiet" onClick={()=>setInspect(!inspect)}>{inspect?'Close saved preview':'Inspect saved source'}</button>}
      {inspect&&<SourceImages api={api} episodeId={saved.episode_id} initiallyOpen/>}
      <button className="btn quiet" onClick={()=>{setSaved(null);setNote('');setFile(null);setDocument(null);setError('');setInspect(false);}}>Add another source</button>
    </div>:<form onSubmit={save}>
      <div className="source-modes" role="group" aria-label="Source type"><button type="button" aria-pressed={mode==='note'} disabled={!!phase||uncertain} onClick={()=>{setMode('note');setError('');}}>Write a note</button><button type="button" aria-pressed={mode==='file'} disabled={!!phase||uncertain} onClick={()=>{setMode('file');setError('');}}>Import text file</button></div>
      {mode==='file'?<>
        <div className="source-file"><label>Text file<input type="file" accept={TEXT_FILE_ACCEPT} disabled={!!phase||uncertain} onChange={e=>void selectDocument(e.target.files?.[0])}/></label><small>UTF-8 · text, Markdown or code · up to 1 MB. PDF and binary formats are not supported.</small></div>
        {document&&<section className="source-document" aria-label="File preview"><header><strong>{document.name}</strong><span>{document.bytes.toLocaleString()} bytes</span></header><pre>{document.content}</pre><p>Local preview only. Saving lets the server index and process this text using its configured models. Code is not executed; a separate file download is not created.</p></section>}
      </>:<><label>Source note<textarea value={note} onChange={e=>setNote(e.target.value)} required maxLength={100000} disabled={!!phase||uncertain} placeholder="What does this source show, and why does it matter?"/></label>
      <div className="source-file"><label>Original image<input type="file" accept="image/png,image/jpeg,image/gif,image/webp" disabled={!!phase||uncertain} onChange={e=>{setFile(e.target.files?.[0]??null);setError('');}}/></label><small>Optional · PNG, JPEG, GIF or WebP · up to 25 MB</small></div>
      {file&&<p className="source-selection">{file.name} · {(file.size/1024).toFixed(1)} KB</p>}<small>Images are retained as original evidence, not read or described by a model.</small></>}
      {error&&<div role="alert" className="source-save-error"><p>{error}</p>{uncertain&&<p>The save is unconfirmed. A source or image may already be stored. Check Memory Search before trying again; this form will not repeat the write.</p>}</div>}
      <footer><span>{phase||'Will save to the connected memory space. This does not approve a claim.'}</span><button className="btn" disabled={!!phase||uncertain||!content.trim()}>{phase?'Saving…':'Save source'}</button></footer>
      {phase&&<p role="status">{phase} Keep this page open until the result is confirmed.</p>}
    </form>}
  </section>;
}
