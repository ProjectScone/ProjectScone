export interface ProfileClaim {fact_id:number;subject:string;predicate:string;object:string;confidence:number;source_episode_id?:number|null;sources?:number[]}
export interface ProfileSource {episode_id:number;excerpt:string;created_at:string}
export interface ProfileSnapshot {claims:ProfileClaim[];recent:ProfileSource[]}
const object=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==='object'&&!Array.isArray(v);
const id=(v:unknown):v is number=>Number.isSafeInteger(v)&&Number(v)>0;
/** Preserve server selection/order. A profile is context, not a truth verdict. */
export function parseProfile(value:unknown):ProfileSnapshot {
  const invalid=()=>Error('Invalid profile response. Source-linked context could not be verified.');
  if(!object(value)||!Array.isArray(value.static_facts)||!Array.isArray(value.dynamic)||!Array.isArray(value.recent)
    ||value.static_facts.length>50||value.recent.length>50||value.dynamic.length!==value.recent.length)throw invalid();
  const claims:ProfileClaim[]=[],recent:ProfileSource[]=[],facts=new Set<number>(),sources=new Set<number>();
  for(const f of value.static_facts){
    if(!object(f)||!id(f.fact_id)||facts.has(f.fact_id)||!['subject','predicate','object'].every(k=>typeof f[k]==='string')
      ||typeof f.confidence!=='number'||!Number.isFinite(f.confidence)||f.confidence<0||f.confidence>1
      ||!(f.source_episode_id==null||id(f.source_episode_id)))throw invalid();
    if(f.sources!==undefined&&(!Array.isArray(f.sources)||f.sources.length>1000||!f.sources.every(id)||new Set(f.sources).size!==f.sources.length))throw invalid();
    facts.add(f.fact_id);
    claims.push({fact_id:f.fact_id,subject:f.subject as string,predicate:f.predicate as string,object:f.object as string,confidence:f.confidence,
      ...(f.source_episode_id!==undefined?{source_episode_id:f.source_episode_id as number|null}:{}),
      ...(f.sources!==undefined?{sources:[...f.sources as number[]]}:{})});
  }
  for(const [i,r] of value.recent.entries()){
    if(!object(r)||!id(r.episode_id)||sources.has(r.episode_id)||typeof r.excerpt!=='string'||Array.from(r.excerpt).length>200
      ||r.excerpt!==value.dynamic[i]||typeof r.created_at!=='string'||!Number.isFinite(Date.parse(r.created_at)))throw invalid();
    sources.add(r.episode_id);recent.push({episode_id:r.episode_id,excerpt:r.excerpt,created_at:r.created_at});
  }
  return {claims,recent};
}
