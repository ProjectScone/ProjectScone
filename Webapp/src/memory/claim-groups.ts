import type {Fact} from './types';

export type ClaimFilter = 'all' | 'current' | 'excluded' | 'no-quote';
export type ClaimGroup = {key:string; subject:string; predicate:string; versions:Fact[]; current?:Fact};
export type DisplayClaimGroup = ClaimGroup & {displayed:Fact};

/** Presentation groups retain every ledger version; grouping never changes facts. */
export function claimGroups(facts:Fact[],now:number):ClaimGroup[]{
  const groups=new Map<string,ClaimGroup>();
  for(const fact of facts){
    if(fact.status!=='active'&&fact.status!=='closed')continue;
    const key=JSON.stringify([fact.subject,fact.predicate]);
    let group=groups.get(key);
    if(!group){group={key,subject:fact.subject,predicate:fact.predicate,versions:[]};groups.set(key,group);}
    group.versions.push(fact);
  }
  for(const group of groups.values()){
    group.versions.sort((a,b)=>a.valid_from.localeCompare(b.valid_from)||a.fact_id-b.fact_id);
    group.current=[...group.versions].reverse().find(f=>!f.excluded_reason
      &&Date.parse(f.valid_from)<=now&&(!f.valid_until||Date.parse(f.valid_until)>now));
  }
  return [...groups.values()].sort((a,b)=>a.subject.localeCompare(b.subject)||a.predicate.localeCompare(b.predicate));
}

export function filterClaimGroups(groups:ClaimGroup[],query:string,filter:ClaimFilter):DisplayClaimGroup[]{
  const search=query.trim().toLocaleLowerCase();
  return groups.flatMap(group=>{
    const candidates=group.versions.filter(f=>(filter==='all'||filter==='current'&&f.fact_id===group.current?.fact_id
      ||filter==='excluded'&&Boolean(f.excluded_reason)||filter==='no-quote'&&!f.quote?.trim())
      &&(!search||[f.subject,f.predicate.replaceAll('_',' '),f.object,f.quote,String(f.fact_id),String(f.source_episode_id??'')].join(' ').toLocaleLowerCase().includes(search)));
    if(!candidates.length)return [];
    return [{...group,displayed:candidates.find(f=>f.fact_id===group.current?.fact_id)||candidates[candidates.length-1]}];
  });
}
