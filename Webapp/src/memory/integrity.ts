export const findingKeys=['chunks_without_episode','vectors_without_chunk','facts_citing_forgotten','facts_citing_unknown','links_with_missing_ends','attachments_unlinked'] as const;
export type FindingKey=typeof findingKeys[number];
export interface IntegrityReport {
  space:string; episodes:number; chunks:number; facts:number; links:number;
  tombstones:number|null; healthy:boolean; not_inspected:string[];
  chunks_without_episode:number[]|null; vectors_without_chunk:number[]|null;
  facts_citing_forgotten:number[]; facts_citing_unknown:number[];
  links_with_missing_ends:number[]; attachments_unlinked:string[]|null;
}

export function parseIntegrityReport(value:unknown,space:string):IntegrityReport {
  const invalid=()=>{throw Error('Invalid integrity report. Its scope, findings or coverage could not be verified.');};
  if(!value||typeof value!=='object'||Array.isArray(value))return invalid();
  const r=value as Record<string,unknown>;
  const count=(v:unknown)=>typeof v==='number'&&Number.isSafeInteger(v)&&v>=0;
  if(!space||r.space!==space||!['episodes','chunks','facts','links'].every(k=>count(r[k]))
    ||!(r.tombstones===null||count(r.tombstones))||typeof r.healthy!=='boolean'
    ||!Array.isArray(r.not_inspected)||!r.not_inspected.every(v=>typeof v==='string'&&v.trim()&&v.length<=100)
    ||new Set(r.not_inspected).size!==r.not_inspected.length)return invalid();
  const optional:Partial<Record<FindingKey,string>>={chunks_without_episode:'chunks',vectors_without_chunk:'vectors',attachments_unlinked:'attachments'};
  for(const key of findingKeys){
    const items=r[key],store=optional[key];
    if(items===null){if(!store||!r.not_inspected.includes(store))return invalid();continue;}
    if(store&&r.not_inspected.includes(store))return invalid();
    if(!Array.isArray(items)||new Set(items).size!==items.length||!items.every(id=>key==='attachments_unlinked'
      ?typeof id==='string'&&/^[a-f0-9]{64}$/.test(id):count(id)&&(id as number)>0))return invalid();
  }
  if(r.healthy!==!findingKeys.some(k=>Array.isArray(r[k])&&r[k].length>0))return invalid();
  return r as unknown as IntegrityReport;
}
