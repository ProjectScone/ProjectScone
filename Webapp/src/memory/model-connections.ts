import type {ApiClient} from '../api';

export const MODEL_ROLES=['chat','extraction','vision','transcription','speech'] as const;
export type ModelRole=typeof MODEL_ROLES[number];
export interface ModelConnection {base_url:string;model:string;timeout_s:number;api_key_env:string|null;voice:string|null;sample_rate:number}
export interface ModelConnections {schema_version:1;revision:number;connections:Record<ModelRole,ModelConnection|null>}
export interface ModelProbe {models:string[];model_available:boolean}
export interface ConnectionDraft {base_url:string;model:string;timeout_s:string;api_key_env:string;voice:string;sample_rate:string}
type Client=Pick<ApiClient,'request'>;

function object(value:unknown):Record<string,unknown>{
  if(!value||typeof value!=='object'||Array.isArray(value))throw Error('Invalid model settings response.');
  return value as Record<string,unknown>;
}
function text(value:unknown,name:string,max:number):string{
  if(typeof value!=='string'||!value.trim()||value.length>max||/[\x00-\x1f\x7f]/.test(value))throw Error(`Enter a valid ${name}.`);
  return value.trim();
}
function endpoint(value:unknown):string{
  const raw=text(value,'self-hosted endpoint URL',2048);
  let url:URL;
  try{url=new URL(raw);}catch{throw Error('Enter a complete self-hosted endpoint URL, including http:// or https://.');}
  if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash||raw.includes('?')||raw.includes('#'))throw Error('Use an HTTP(S) endpoint without credentials, query, or fragment.');
  return raw;
}
function parseConnection(value:unknown):ModelConnection{
  const data=object(value);
  const allowed=new Set(['base_url','model','timeout_s','api_key_env','voice','sample_rate']);
  if(Object.keys(data).some(key=>!allowed.has(key)))throw Error('Invalid model connection fields.');
  const timeout=data.timeout_s??180,rate=data.sample_rate??24000;
  if(typeof timeout!=='number'||!Number.isFinite(timeout)||timeout<1||timeout>600)throw Error('Timeout must be between 1 and 600 seconds.');
  if(typeof rate!=='number'||!Number.isInteger(rate)||rate<8000||rate>48000)throw Error('Sample rate must be an integer between 8000 and 48000 Hz.');
  let environment:string|null=null;
  if(data.api_key_env!=null){environment=text(data.api_key_env,'server token environment variable',128);if(!/^[A-Za-z_][A-Za-z0-9_]*$/.test(environment))throw Error('Enter an environment variable name, not a token.');}
  return {base_url:endpoint(data.base_url),model:text(data.model,'model identifier',160),timeout_s:timeout,
    api_key_env:environment,voice:data.voice==null?null:text(data.voice,'voice identifier',120),sample_rate:rate};
}
export function parseModelConnections(value:unknown):ModelConnections{
  const data=object(value),connections=object(data.connections);
  if(data.schema_version!==1||typeof data.revision!=='number'||!Number.isSafeInteger(data.revision)||data.revision<0)throw Error('Invalid model settings version.');
  const parsed={} as Record<ModelRole,ModelConnection|null>;
  for(const role of MODEL_ROLES){if(!(role in connections))throw Error('Model settings are missing a purpose.');parsed[role]=connections[role]===null?null:parseConnection(connections[role]);}
  return {schema_version:1,revision:data.revision,connections:parsed};
}
export function parseModelProbe(value:unknown):ModelProbe{
  const data=object(value);
  if(!Array.isArray(data.models)||data.models.length>1000||typeof data.model_available!=='boolean')throw Error('Invalid model discovery response.');
  return {models:data.models.map(model=>text(model,'discovered model identifier',256)),model_available:data.model_available};
}
export function connectionDraft(value:ModelConnection|null):ConnectionDraft{
  return {base_url:value?.base_url??'',model:value?.model??'',timeout_s:String(value?.timeout_s??180),
    api_key_env:value?.api_key_env??'',voice:value?.voice??'',sample_rate:String(value?.sample_rate??24000)};
}
export function connectionFromDraft(draft:ConnectionDraft,role:ModelRole):ModelConnection{
  const connection=parseConnection({...draft,timeout_s:Number(draft.timeout_s),sample_rate:Number(draft.sample_rate),api_key_env:draft.api_key_env.trim()||null,voice:draft.voice.trim()||null});
  if(role==='speech'&&!connection.voice)throw Error('Enter the voice identifier served by your speech model.');
  return connection;
}
const options=(signal:AbortSignal):RequestInit=>({signal,cache:'no-store',redirect:'error',credentials:'omit'});
export async function readModelConnections(api:Client,signal:AbortSignal):Promise<ModelConnections>{
  return parseModelConnections(await api.request<unknown>('/v1/model-connections',options(signal)));
}
export async function saveModelConnection(api:Client,role:ModelRole,connection:ModelConnection|null,revision:number,signal:AbortSignal):Promise<ModelConnections>{
  if(!MODEL_ROLES.includes(role)||!Number.isSafeInteger(revision)||revision<0)throw Error('Invalid model settings update.');
  const validated=connection===null?null:connectionFromDraft(connectionDraft(connection),role);
  return parseModelConnections(await api.request<unknown>(`/v1/model-connections/${role}`,{...options(signal),method:'PUT',body:JSON.stringify({expected_revision:revision,connection:validated})}));
}
export async function probeModelConnection(api:Client,connection:ModelConnection,signal:AbortSignal):Promise<ModelProbe>{
  return parseModelProbe(await api.request<unknown>('/v1/model-connections/probe',{...options(signal),method:'POST',body:JSON.stringify({connection:parseConnection(connection)})}));
}
