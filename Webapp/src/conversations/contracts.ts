import {readScope,type RecallScope} from './recall-scope.ts';
export type SessionState = 'created'|'running'|'stopping'|'ended'|'failed'|'interrupted';
export interface ConversationSession {session_id:string;space:string;state:SessionState;revision:number;created_at:string;active_request_id?:string|null;latest_request_id?:string|null;recall_scope?:RecallScope}
export interface Capabilities {text_configured:boolean;session_deletion:boolean;turn_cancellation:boolean;transcript_pagination:boolean;recall_scope:boolean}
export interface Episode {episode_id:number;content:string;metadata:Record<string,unknown>;created_at?:string}
export interface Transcript {episodes:Episode[];has_more:boolean;next_before:string|null}
export interface TurnResult {text:string;user_episode_id?:number;assistant_episode_id?:number;memory_context?:{status:string;references:{episode_id:number;chunk_id?:number}[]}}
export interface TurnReceipt {request_id:string;status:'pending'|'completed'|'failed'|'interrupted'|'cancelled';result_state:'available'|'forgotten'|'unavailable'|'unreadable';result?:TurnResult}
export const idPattern=/^[A-Za-z0-9._:-]{1,128}$/;
const states=['created','running','stopping','ended','failed','interrupted'];
function record(value:unknown):Record<string,unknown>{if(!value||typeof value!=='object'||Array.isArray(value))throw Error('Invalid conversation response');return value as Record<string,unknown>;}
function identifier(value:unknown):string{if(typeof value!=='string'||!idPattern.test(value))throw Error('Invalid conversation identifier');return value;}
function integer(value:unknown):number{if(typeof value!=='number'||!Number.isSafeInteger(value)||value<1)throw Error('Invalid conversation record number');return value;}
function text(value:unknown):string{if(typeof value!=='string')throw Error('Invalid conversation text');return value;}
export function capabilities(value:unknown):Capabilities{
  const v=record(value);
  if(v.schema_version!==1||typeof v.text_configured!=='boolean'||v.reply_transport!=='poll'||(v.reply_replay!=='process_lifetime'&&v.reply_replay!=='durable_receipts'))throw Error('This conversation service has an unsupported capability contract.');
  return {text_configured:v.text_configured,session_deletion:v.session_deletion===true,turn_cancellation:v.turn_cancellation===true,transcript_pagination:v.transcript_pagination===true,recall_scope:v.recall_scope===true};
}
export function session(value:unknown):ConversationSession{
  const v=record(value);
  if(!states.includes(String(v.state)))throw Error('Unknown conversation state');
  return {session_id:identifier(v.session_id),space:text(v.space),state:v.state as SessionState,revision:integer(v.revision),created_at:text(v.created_at),active_request_id:v.active_request_id==null?null:identifier(v.active_request_id),latest_request_id:v.latest_request_id==null?null:identifier(v.latest_request_id),recall_scope:v.recall_scope===undefined?undefined:readScope(v.recall_scope)};
}
export function sessionPage(value:unknown):{items:ConversationSession[];next_after:string|null;has_more:boolean}{
  const v=record(value);if(!Array.isArray(v.items)||v.items.length>200||typeof v.has_more!=='boolean')throw Error('Invalid session list');
  return {items:v.items.map(session),has_more:v.has_more,next_after:v.has_more?identifier(v.next_after):null};
}
export function episode(value:unknown):Episode{
  const v=record(value);return {episode_id:integer(v.episode_id),content:text(v.content),metadata:v.metadata==null?{}:record(v.metadata),created_at:typeof v.created_at==='string'?v.created_at:undefined};
}
export function transcript(value:unknown):Transcript{
  const v=record(value);if(!Array.isArray(v.episodes)||v.episodes.length>200||typeof v.has_more!=='boolean')throw Error('Invalid saved transcript');
  let next_before:string|null=null;
  if(v.next_before!=null){
    if(!v.has_more||!v.episodes.length||typeof v.next_before!=='string'||!/^[A-Za-z0-9_-]{1,1024}$/.test(v.next_before))throw Error('Invalid transcript cursor');
    next_before=v.next_before;
  }
  return {episodes:v.episodes.map(episode),has_more:v.has_more,next_before};
}
export function turnReceipt(value:unknown):TurnReceipt{
  const v=record(value);if(!['pending','completed','failed','interrupted','cancelled'].includes(String(v.status)))throw Error('Invalid reply status');
  const availability=v.result_state===undefined||(v.status==='pending'&&v.result_state===null)?(v.status==='completed'?'available':'unavailable'):v.result_state;
  if(typeof availability!=='string'||!['available','forgotten','unavailable','unreadable'].includes(availability))throw Error('Invalid reply availability');
  if(availability!=='unavailable'&&v.status!=='completed')throw Error('Reply availability contradicts its status');
  if(availability!=='available'&&v.result!=null)throw Error('Unavailable reply contains stale content');
  const result:TurnReceipt={request_id:identifier(v.request_id),status:v.status as TurnReceipt['status'],result_state:availability as TurnReceipt['result_state']};
  if(availability==='available'){
    const r=record(v.result);result.result={text:text(r.text)};
    if(v.result_state==='available')result.result.assistant_episode_id=integer(r.assistant_episode_id);
    if(r.user_episode_id!=null)result.result.user_episode_id=integer(r.user_episode_id);
    if(r.assistant_episode_id!=null)result.result.assistant_episode_id=integer(r.assistant_episode_id);
    if(r.memory_context!=null){
      const c=record(r.memory_context);if(!Array.isArray(c.references)||c.references.length>200)throw Error('Invalid context references');
      result.result.memory_context={status:text(c.status),references:c.references.map(ref=>{const e=record(ref);return {episode_id:integer(e.episode_id),chunk_id:e.chunk_id==null?undefined:integer(e.chunk_id)};})};
    }
  }
  return result;
}
