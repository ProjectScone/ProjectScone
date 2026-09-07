import {useEffect, useState} from 'react';
import {ApiError, PREVIEW_TYPES, type ApiClient, type ImageAttachment} from '../api';
import './source-images.css';

export function SourceImages({episodeId, api, initiallyOpen=false}: {episodeId:number|string; api:ApiClient; initiallyOpen?:boolean}) {
  const [open,setOpen]=useState(initiallyOpen);
  if(!/^\d+$/.test(String(episodeId)))return null;
  return <section className="source-images" aria-label="Source images">
    <button type="button" className="source-images-toggle" aria-expanded={open} onClick={()=>setOpen(!open)}>{open?'Hide source images':'View source images'}</button>
    {open&&<SourceImageList key={episodeId} episodeId={episodeId} api={api}/>}
  </section>;
}

function SourceImageList({episodeId,api}:{episodeId:number|string;api:ApiClient}) {
  const [items,setItems]=useState<ImageAttachment[]|null>(null),[error,setError]=useState(''),[attempt,setAttempt]=useState(0),[page,setPage]=useState(0);
  useEffect(()=>{
    const controller=new AbortController();setItems(null);setError('');setPage(0);
    api.request<{attachments?:unknown}>(`/v1/episodes/${episodeId}`,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(10000)])})
      .then(episode=>{
        if(controller.signal.aborted)return;
        if(!Array.isArray(episode.attachments))throw Error('This server did not return attachment metadata. Image availability is unknown.');
        const valid=episode.attachments.every(a=>a&&typeof a==='object'&&/^[a-f0-9]{64}$/.test(a.attachment_id)&&typeof a.media_type==='string'&&Number.isSafeInteger(a.bytes)&&a.bytes>0&&(a.filename==null||typeof a.filename==='string'));
        if(!valid)throw Error('The source returned invalid attachment metadata.');
        setItems(episode.attachments as ImageAttachment[]);
      }).catch(e=>{if(!controller.signal.aborted)setError(e instanceof ApiError&&e.status===404?'Source unavailable in this memory space.':e instanceof Error?e.message:'Could not load source images.');});
    return()=>controller.abort();
  },[api,episodeId,attempt]);
  if(error)return <div className="source-media-state" role="alert"><p>{error}</p><button onClick={()=>setAttempt(n=>n+1)}>Retry source images</button></div>;
  if(items===null)return <p className="source-media-state" role="status">Checking saved attachments…</p>;
  if(!items.length)return <p className="source-media-state">No image attachments were saved with this source.</p>;
  return <><p className="source-media-state">Original attachments · {items.length} saved · previews do not describe or interpret the image.</p>
    <div className="source-image-grid">{items.slice(page*6,page*6+6).map(item=><RasterImage key={item.attachment_id} item={item} api={api}/>)}</div>
    {items.length>6&&<div className="source-image-pages"><button disabled={!page} onClick={()=>setPage(n=>n-1)}>Previous images</button><span>{page+1} / {Math.ceil(items.length/6)}</span><button disabled={(page+1)*6>=items.length} onClick={()=>setPage(n=>n+1)}>Next images</button></div>}
  </>;
}

function RasterImage({item,api}:{item:ImageAttachment;api:ApiClient}) {
  const [url,setUrl]=useState(''),[error,setError]=useState(''),[attempt,setAttempt]=useState(0);
  const supported=PREVIEW_TYPES.has(item.media_type);
  useEffect(()=>{
    if(!supported)return;
    const controller=new AbortController();let objectUrl='';setUrl('');setError('');
    api.image(item,controller.signal).then(blob=>{
      if(controller.signal.aborted)return;
      objectUrl=URL.createObjectURL(blob);setUrl(objectUrl);
    }).catch(e=>{if(!controller.signal.aborted)setError(e instanceof ApiError&&e.status===404?'Image bytes are unavailable in this memory space.':e instanceof Error?e.message:'Image download failed.');});
    return()=>{controller.abort();if(objectUrl)URL.revokeObjectURL(objectUrl);};
  },[api,item,supported,attempt]);
  const name=item.filename||`Image ${item.attachment_id.slice(0,8)}`;
  return <figure className="source-image">
    {!supported?<p>Preview unavailable for {item.media_type}. The attachment is recorded, but is not rendered inline.</p>:error?<div role="alert"><p>{error}</p><button onClick={()=>setAttempt(n=>n+1)}>Retry image</button></div>:url?<img src={url} alt={name} onError={()=>setError('Saved bytes could not be decoded as an image.')} />:<p role="status">Loading original image…</p>}
    <figcaption><strong>{name}</strong><span>{item.media_type} · {(item.bytes/1024).toFixed(1)} KB</span>{url&&!error&&<a href={url} download={name.replace(/[^a-zA-Z0-9._-]/g,'_').slice(0,120)}>Save original image</a>}</figcaption>
  </figure>;
}
