export const sourceKinds=['note','file','conversation','observation','connector'] as const;
export interface RecallScope {kind?:string;source_prefix?:string;since?:string;until?:string;where?:Record<string,string>}
export interface ScopeDraft {kind:string;source_prefix:string;since:string;until:string;metadata:{key:string;value:string}[]}
export const emptyScopeDraft=():ScopeDraft=>({kind:'',source_prefix:'',since:'',until:'',metadata:[]});

function timestamp(value:unknown):string {
  if(typeof value!=='string'||!/^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))?$/.test(value))throw Error('Use a date or an RFC3339 timestamp with a timezone.');
  if(Number(value.slice(0,4))<1||(value.length>10&&(Number(value.slice(11,13))>23||Number(value.slice(14,16))>59||Number(value.slice(17,19))>59)))throw Error('Enter a valid date and time without rollover.');
  const day=value.slice(0,10),date=new Date(day+'T00:00:00Z'),time=new Date(value);
  if(!Number.isFinite(date.getTime())||date.toISOString().slice(0,10)!==day||!Number.isFinite(time.getTime()))throw Error('Enter a valid calendar date.');
  return time.toISOString();
}
export function readScope(value:unknown):RecallScope {
  if(!value||typeof value!=='object'||Array.isArray(value))throw Error('Invalid recall scope.');
  const v=value as Record<string,unknown>,out:RecallScope={};
  if(Object.keys(v).some(key=>!['kind','source_prefix','since','until','where'].includes(key)))throw Error('Unrecognized recall filter.');
  if(v.kind!=null){if(typeof v.kind!=='string'||!sourceKinds.includes(v.kind as typeof sourceKinds[number]))throw Error('Choose a supported source type.');out.kind=v.kind;}
  if(v.source_prefix!=null){if(typeof v.source_prefix!=='string'||[...v.source_prefix].length>1000)throw Error('Source prefix must be at most 1,000 characters.');out.source_prefix=v.source_prefix;}
  if(v.since!=null)out.since=timestamp(v.since);
  if(v.until!=null)out.until=timestamp(v.until);
  if(out.since&&out.until&&out.since>out.until)throw Error('The end date must be on or after the start date.');
  if(v.where!=null){
    if(typeof v.where!=='object'||Array.isArray(v.where))throw Error('Invalid metadata filters.');
    const pairs=Object.entries(v.where);
    if(pairs.length>16||pairs.some(([key,val])=>!/^[a-z][a-z0-9_]{0,31}$/.test(key)||typeof val!=='string'||!val||[...val].length>256))throw Error('Use up to 16 metadata filters: lowercase keys and values of 1–256 characters.');
    if(pairs.length)out.where=Object.fromEntries(pairs.sort(([a],[b])=>a.localeCompare(b))) as Record<string,string>;
  }
  return out;
}
export function scopeFromDraft(draft:ScopeDraft):RecallScope {
  const pairs=draft.metadata.filter(row=>row.key||row.value);
  if(new Set(pairs.map(row=>row.key)).size!==pairs.length)throw Error('Each metadata key can appear only once.');
  return readScope({...(draft.kind?{kind:draft.kind}:{}),...(draft.source_prefix?{source_prefix:draft.source_prefix}:{}),
    ...(draft.since?{since:draft.since}:{}),...(draft.until?{until:draft.until}:{}),where:Object.fromEntries(pairs.map(row=>[row.key,row.value]))});
}
export function sameScope(expected:RecallScope,actual:RecallScope|undefined):boolean {
  return actual!==undefined&&JSON.stringify(readScope(expected))===JSON.stringify(readScope(actual));
}
