import type {Fact} from './types';
export type RelationKind = 'extends' | 'derived_from' | 'contradicts' | 'supports';
export interface ClaimLink {link_id:number;from_fact:number;to_fact:number;kind:RelationKind;created_at:string;source_episode_id:number|null;quote:string|null}
export interface ClaimDetail {fact:Fact;links:ClaimLink[];sources:number[]}
const object=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);
const id=(value:unknown):value is number=>Number.isSafeInteger(value)&&Number(value)>0;
const timestamp=(value:unknown):value is string=>typeof value==='string'&&Number.isFinite(Date.parse(value));
const optionalText=(value:unknown)=>value==null||typeof value==='string';
const optionalId=(value:unknown)=>value==null||id(value);
const labels:Record<RelationKind,[string,string]>={extends:['Extends','Extended by'],derived_from:['Derived from','Used to derive'],contradicts:['Contradicts','Contradicted by'],supports:['Supports','Supported by']};

export function parseClaimDetail(value:unknown,expectedId:number):ClaimDetail {
  const invalid=()=>Error('Invalid claim relationship response. No detail was confirmed.');
  if(!id(expectedId)||!object(value)||!object(value.fact)||value.fact.fact_id!==expectedId
    ||!Array.isArray(value.links)||!Array.isArray(value.sources))throw invalid();
  const f=value.fact;
  if(!['subject','predicate','object'].every(key=>typeof f[key]==='string')
    ||!['active','closed','proposed','declined'].includes(String(f.status))
    ||!(f.origin===undefined||['stated','extracted','inferred'].includes(String(f.origin)))
    ||!timestamp(f.valid_from)||!(f.valid_until===null||timestamp(f.valid_until))
    ||typeof f.confidence!=='number'||!Number.isFinite(f.confidence)||f.confidence<0||f.confidence>1
    ||!optionalId(f.source_episode_id)||!optionalId(f.superseded_by)
    ||!['quote','excluded_reason','closed_reason'].every(key=>optionalText(f[key])))throw invalid();
  const links:ClaimLink[]=[],seen=new Set<number>();
  for(const link of value.links){
    if(!object(link)||!id(link.link_id)||seen.has(link.link_id)||!id(link.from_fact)||!id(link.to_fact)
      ||link.from_fact===link.to_fact||link.from_fact!==expectedId&&link.to_fact!==expectedId
      ||typeof link.kind!=='string'||!Object.hasOwn(labels,link.kind)||!timestamp(link.created_at)
      ||!optionalId(link.source_episode_id)||!optionalText(link.quote))throw invalid();
    seen.add(link.link_id);
    links.push({link_id:link.link_id,from_fact:link.from_fact,to_fact:link.to_fact,kind:link.kind as RelationKind,
      created_at:link.created_at,source_episode_id:link.source_episode_id as number|null??null,quote:link.quote as string|null??null});
  }
  if(!value.sources.every(id)||new Set(value.sources).size!==value.sources.length)throw invalid();
  return {fact:{fact_id:expectedId,subject:f.subject as string,predicate:f.predicate as string,object:f.object as string,
    status:f.status as Fact['status'],origin:f.origin as Fact['origin'],valid_from:f.valid_from,valid_until:f.valid_until as string|null,
    confidence:f.confidence,source_episode_id:f.source_episode_id as number|null|undefined,superseded_by:f.superseded_by as number|null|undefined,
    quote:f.quote as string|null|undefined,excluded_reason:f.excluded_reason as string|null|undefined,closed_reason:f.closed_reason as string|null|undefined},
    links,sources:value.sources as number[]};
}
export function relationTarget(link:ClaimLink,current:number):number {
  if(link.from_fact===current)return link.to_fact;
  if(link.to_fact===current)return link.from_fact;
  throw Error('Relationship does not belong to this claim.');
}
export function relationLabel(link:ClaimLink,current:number):string {
  relationTarget(link,current);
  return labels[link.kind][link.from_fact===current?0:1];
}
