import {useEffect,useRef,useState} from 'react';
import type {ApiClient} from '../api';
import {WorkspaceIcon} from '../components/WorkspaceIcon';
import {MODEL_ROLES,connectionDraft,connectionFromDraft,readModelConnections,saveModelConnection,probeModelConnection,
  type ModelRole,type ModelConnections,type ConnectionDraft,type ModelProbe} from './model-connections';
import './model-connections.css';

const PURPOSES:Record<ModelRole,{label:string;description:string;hint:string}>={
  chat:{label:'Conversation',description:'The model that responds in your conversations.',hint:'Use the exact chat model identifier exposed by your server.'},
  extraction:{label:'Memory extraction',description:'Turn retained source text into reviewable memory proposals.',hint:'Choose a text model that can follow structured extraction instructions.'},
  vision:{label:'Image understanding',description:'Describe retained images and read visible text.',hint:'Choose an image-capable model. A model listing does not verify image support.'},
  transcription:{label:'Speech to text',description:'Transcribe incoming speech with your own audio model.',hint:'Your server must provide an OpenAI-compatible audio transcription endpoint.'},
  speech:{label:'Text to speech',description:'Read responses aloud through your own voice model.',hint:'Your server must provide compatible PCM speech output for the selected voice and sample rate.'},
};
const emptyDrafts=()=>Object.fromEntries(MODEL_ROLES.map(role=>[role,connectionDraft(null)])) as Record<ModelRole,ConnectionDraft>;
const errorMessage=(error:unknown)=>error instanceof Error?error.message:'The model settings request failed.';
const isConflict=(error:unknown)=>typeof error==='object'&&error!==null&&'status' in error&&error.status===409;

export function ModelConnectionsView({api}:{api:ApiClient}){
  const [role,setRole]=useState<ModelRole>('chat');
  const [stored,setStored]=useState<{api:ApiClient;data:ModelConnections}|null>(null);
  const [drafts,setDrafts]=useState(emptyDrafts);
  const [busy,setBusy]=useState<'load'|'save'|'probe'|null>('load');
  const [error,setError]=useState<string|null>(null);
  const [notice,setNotice]=useState<string|null>(null);
  const [conflict,setConflict]=useState(false);
  const [probe,setProbe]=useState<ModelProbe|null>(null);
  const active=useRef<AbortController|null>(null),sequence=useRef(0),apiRef=useRef(api);
  apiRef.current=api;
  const snapshot=stored?.api===api?stored.data:null;
  const cancel=()=>{active.current?.abort();active.current=null;sequence.current++;};
  const begin=(kind:'load'|'save'|'probe')=>{
    cancel();const controller=new AbortController();active.current=controller;const serial=sequence.current;
    setBusy(kind);setError(null);setNotice(null);
    return {signal:AbortSignal.any([controller.signal,AbortSignal.timeout(30000)]),current:()=>!controller.signal.aborted&&serial===sequence.current&&apiRef.current===api};
  };
  const load=async(preserve:boolean)=>{
    const operation=begin('load');
    try{
      const data=await readModelConnections(api,operation.signal);if(!operation.current())return;
      setStored({api,data});setConflict(false);setProbe(null);
      if(!preserve)setDrafts(Object.fromEntries(MODEL_ROLES.map(key=>[key,connectionDraft(data.connections[key])])) as Record<ModelRole,ConnectionDraft>);
      else setNotice('Latest saved settings loaded. Your edits are intact; compare them with the saved configuration before saving.');
    }catch(failure){if(operation.current())setError(errorMessage(failure));}
    finally{if(operation.current())setBusy(null);}
  };
  useEffect(()=>{
    setStored(null);setDrafts(emptyDrafts());setConflict(false);setProbe(null);setRole('chat');
    void load(false);return cancel;
  },[api]);
  const draft=drafts[role],saved=snapshot?.connections[role]??null;
  const changed=JSON.stringify(draft)!==JSON.stringify(connectionDraft(saved));
  const locked=busy==='save'||busy==='load';
  const choose=(next:ModelRole)=>{cancel();setBusy(null);setRole(next);setProbe(null);setError(null);setNotice(null);};
  const edit=(field:keyof ConnectionDraft,value:string)=>{
    cancel();setBusy(null);setDrafts(previous=>({...previous,[role]:{...previous[role],[field]:value}}));setProbe(null);setError(null);setNotice(null);
  };
  const save=async(disconnect=false)=>{
    if(!snapshot||conflict)return;
    let connection;
    try{connection=disconnect?null:connectionFromDraft(draft,role);}catch(failure){setError(errorMessage(failure));return;}
    const operation=begin('save');setProbe(null);
    try{
      const data=await saveModelConnection(api,role,connection,snapshot.revision,operation.signal);if(!operation.current())return;
      setStored({api,data});
      if(!disconnect)setDrafts(previous=>({...previous,[role]:connectionDraft(data.connections[role])}));
      setNotice(disconnect?'Saved connection removed. Your draft remains available.':'Configuration saved. No inference was run.');
    }catch(failure){if(operation.current()){setError(errorMessage(failure));if(isConflict(failure))setConflict(true);}}
    finally{if(operation.current())setBusy(null);}
  };
  const discover=async()=>{
    let connection;
    try{connection=connectionFromDraft(draft,role);}catch(failure){setError(errorMessage(failure));return;}
    const operation=begin('probe');setProbe(null);
    try{const result=await probeModelConnection(api,connection,operation.signal);if(operation.current())setProbe(result);}
    catch(failure){if(operation.current())setError(errorMessage(failure));}
    finally{if(operation.current())setBusy(null);}
  };
  return <div className="model-connections">
    <header className="model-connections-header"><div className="model-connections-heading"><span className="model-connections-icon"><WorkspaceIcon name="scopes"/></span><div><span className="eyebrow">Your infrastructure</span><h2>Self-hosted models</h2><p>Connect the models you run, one purpose at a time.</p></div></div><button className="btn quiet" disabled={busy!==null} onClick={()=>void load(Boolean(snapshot))}>{busy==='load'?'Reading settings…':'Refresh saved settings'}</button></header>
    {error&&<div className="model-message model-error" role="alert">{error}</div>}
    {conflict&&<div className="model-message model-conflict" role="alert"><strong>Settings changed elsewhere.</strong><p>Your edits have been kept. Load the latest saved settings to compare before saving again.</p><button className="btn quiet" disabled={locked} onClick={()=>void load(true)}>Load latest saved settings</button></div>}
    {notice&&<div className="model-message" role="status">{notice}</div>}
    {!snapshot?<div className="model-connections-empty" role="status">{busy==='load'?'Reading your saved model connections…':'Model settings are unavailable. Refresh to try again.'}</div>:<>
      <nav className="model-purpose-list" aria-label="Model purposes">{MODEL_ROLES.map(key=><button type="button" key={key} aria-pressed={role===key} disabled={locked} onClick={()=>choose(key)}><span>{PURPOSES[key].label}</span><small><i className={snapshot.connections[key]?'configured':''}/>{snapshot.connections[key]?'Saved':'Not configured'}</small></button>)}</nav>
      <section className="model-connection-card" aria-labelledby="model-purpose-heading">
        <header><div><span className="eyebrow">{String(MODEL_ROLES.indexOf(role)+1).padStart(2,'0')} · Model purpose</span><h3 id="model-purpose-heading">{PURPOSES[role].label}</h3><p>{PURPOSES[role].description}</p></div><span className={`model-saved-badge${saved?' configured':''}`}>{saved?'Configuration saved':'Not configured'}</span></header>
        <form onSubmit={event=>{event.preventDefault();void save();}}>
          <fieldset disabled={locked}>
            <div className="model-connection-fields"><label>Self-hosted endpoint URL<input aria-label="Self-hosted endpoint URL" type="url" required value={draft.base_url} maxLength={2048} placeholder="http://inference.home.arpa:8080/v1" autoComplete="off" spellCheck={false} onChange={event=>edit('base_url',event.target.value)}/><small>The OpenAI-compatible base URL on your machine, edge server, or private network.</small></label>
            <label>Model identifier<input aria-label="Model identifier" required value={draft.model} list={`model-inventory-${role}`} maxLength={160} placeholder="Your server’s model name" autoComplete="off" spellCheck={false} onChange={event=>edit('model',event.target.value)}/><small>{PURPOSES[role].hint}</small></label></div>
            {role==='speech'&&<div className="model-connection-fields"><label>Voice identifier<input aria-label="Voice identifier" required value={draft.voice} maxLength={120} placeholder="Voice served by your model" onChange={event=>edit('voice',event.target.value)}/></label><label>PCM sample rate <span className="model-field-unit">Hz</span><input aria-label="PCM sample rate" type="number" min={8000} max={48000} step={1} required value={draft.sample_rate} onChange={event=>edit('sample_rate',event.target.value)}/></label></div>}
            <details className="model-advanced"><summary>Advanced connection settings</summary><div className="model-connection-fields"><label>Request timeout <span className="model-field-unit">seconds</span><input aria-label="Request timeout" type="number" min={1} max={600} step="any" required value={draft.timeout_s} onChange={event=>edit('timeout_s',event.target.value)}/></label><label>Server token environment variable <span className="model-field-unit">optional</span><input aria-label="Server token environment variable" value={draft.api_key_env} maxLength={128} placeholder="SELF_HOSTED_MODEL_TOKEN" autoComplete="off" spellCheck={false} onChange={event=>edit('api_key_env',event.target.value)}/><small>Name of an existing server environment variable. Enter the variable name here; keep the token on your server.</small></label></div></details>
          </fieldset>
          <div className="model-check-row"><button className="btn quiet" type="button" disabled={busy!==null} onClick={()=>void discover()}>{busy==='probe'?'Checking endpoint…':'Discover models'}</button><p>Reads the endpoint’s model list. Does not run inference or verify modality support.</p></div>
          {probe&&<div className="model-probe-result" role="status"><strong>Endpoint answered · {probe.models.length} model{probe.models.length===1?'':'s'} listed</strong><p>{probe.model_available?'The entered model is listed.':'The entered model was not listed. Some self-hosted services do not list every usable model.'}</p>{probe.models.length>0&&<><datalist id={`model-inventory-${role}`}>{probe.models.map(model=><option key={model} value={model}/>)}</datalist><details><summary>Available model identifiers</summary><ul>{probe.models.map(model=><li key={model}><button type="button" onClick={()=>edit('model',model)}>{model}</button></li>)}</ul></details></>}</div>}
          {saved&&<details className="model-current"><summary>Saved configuration{changed?' · differs from your draft':''}</summary><dl><div><dt>Endpoint</dt><dd>{saved.base_url}</dd></div><div><dt>Model</dt><dd>{saved.model}</dd></div><div><dt>Timeout</dt><dd>{saved.timeout_s} seconds</dd></div>{saved.api_key_env&&<div><dt>Token variable</dt><dd>{saved.api_key_env}</dd></div>}{role==='speech'&&<><div><dt>Voice</dt><dd>{saved.voice??'Not set'}</dd></div><div><dt>Sample rate</dt><dd>{saved.sample_rate} Hz</dd></div></>}</dl></details>}
          <footer className="model-save-row"><div><button className="btn primary" type="submit" disabled={busy!==null||conflict}>{busy==='save'?'Saving…':'Save configuration'}</button>{changed&&<button type="button" className="btn quiet" disabled={locked} onClick={()=>{cancel();setBusy(null);setDrafts(previous=>({...previous,[role]:connectionDraft(saved)}));setProbe(null);setError(null);setNotice(null);}}>Use saved values</button>}</div>{saved&&<button type="button" className="btn quiet model-disconnect" disabled={busy!==null||conflict} onClick={()=>void save(true)}>Disconnect saved model</button>}</footer>
        </form>
      </section>
      <p className="model-connections-footnote">Settings are shared by this server. Saving selects a connection; actual model work starts through the corresponding conversation or processing action.</p>
    </>}
  </div>;
}
