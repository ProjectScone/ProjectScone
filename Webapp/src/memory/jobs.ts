export type JobState='searchable'|'consolidated'|'failed'|'cancelled';
export interface JobItem {index:number;episode_id:number;outcome:'accepted'|'duplicate'|'updated';state:JobState;searchable_at:string|null;consolidated_at:string|null;error:string|null;attempts:number}
export interface IngestJob {job_id:string;space:string;created_at:string;request_id:string|null;cancelled_at:string|null;items:JobItem[];searchable:number;consolidated:number;state:JobState}
export interface JobPage {jobs:IngestJob[];next:string|null}
const invalid=()=>Error('The server returned an invalid batch receipt. Refresh to read it again.');
function object(value:unknown):Record<string,unknown>{if(!value||typeof value!=='object'||Array.isArray(value))throw invalid();return value as Record<string,unknown>;}
const count=(value:unknown):value is number=>typeof value==='number'&&Number.isSafeInteger(value)&&value>=0;
const identifier=(value:unknown):value is string=>typeof value==='string'&&/^[A-Za-z0-9_-]{1,128}$/.test(value);
function timestamp(value:unknown):value is string {
  if(typeof value!=='string'||!/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)$/.test(value)||!Number.isFinite(Date.parse(value)))return false;
  const day=new Date(value.slice(0,10)+'T00:00:00Z');
  return Number.isFinite(day.getTime())&&day.toISOString().slice(0,10)===value.slice(0,10);
}
function parseItem(value:unknown,index:number):JobItem {
  const v=object(value);
  if(v.index!==index||!count(v.episode_id)||v.episode_id===0||!count(v.attempts)
    ||typeof v.outcome!=='string'||!['accepted','duplicate','updated'].includes(v.outcome)
    ||typeof v.state!=='string'||!['searchable','consolidated','failed','cancelled'].includes(v.state)
    ||!(v.error===null||typeof v.error==='string')
    ||!(v.searchable_at===null||timestamp(v.searchable_at))
    ||!(v.consolidated_at===null||timestamp(v.consolidated_at)))throw invalid();
  if(v.state==='searchable'&&!v.searchable_at
    ||(v.state==='consolidated')!==Boolean(v.consolidated_at)
    ||v.consolidated_at&&!v.searchable_at)throw invalid();
  return v as unknown as JobItem;
}
function parseJob(value:unknown,space:string):IngestJob {
  const v=object(value);
  if(!identifier(v.job_id)||v.space!==space||!timestamp(v.created_at)
    ||!(v.request_id===null||typeof v.request_id==='string')
    ||!(v.cancelled_at===null||timestamp(v.cancelled_at))
    ||!Array.isArray(v.items)||v.items.length>500||!count(v.searchable)||!count(v.consolidated))throw invalid();
  const items=v.items.map(parseItem);
  if(v.searchable!==items.filter(i=>i.searchable_at!==null).length
    ||v.consolidated!==items.filter(i=>i.consolidated_at!==null).length)throw invalid();
  const state=v.cancelled_at?'cancelled':items.some(i=>i.state==='failed')?'failed':items.length&&v.consolidated===items.length?'consolidated':'searchable';
  if(v.state!==state)throw invalid();
  return {...v,items} as unknown as IngestJob;
}
export function parseJobPage(value:unknown,space:string,visited:readonly string[]=[]):JobPage {
  const v=object(value);
  if(!Array.isArray(v.jobs)||v.jobs.length>20)throw invalid();
  const jobs=v.jobs.map(j=>parseJob(j,space)),ids=jobs.map(j=>j.job_id);
  if(new Set(ids).size!==ids.length||ids.some(id=>visited.includes(id)))throw invalid();
  if(v.next!==null&&(!identifier(v.next)||v.next!==ids.at(-1)))throw invalid();
  return {jobs,next:v.next as string|null};
}
