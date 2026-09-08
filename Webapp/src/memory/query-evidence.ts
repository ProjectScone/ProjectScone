export type QueryNodeKind = 'query' | 'chunk' | 'episode' | 'claim' | 'concept';
export type QueryEdgeKind = 'returned' | 'chunked_into' | 'source_of' | 'extends' | 'derived_from' | 'contradicts' | 'supports' | 'superseded_by' | 'mentions' | 'asserts' | 'relation';
export interface QueryEvidenceNode {id:string;kind:QueryNodeKind;label:string;ts?:string;data:Record<string,unknown>}
export interface QueryEvidenceEdge {id:string;source:string;target:string;kind:QueryEdgeKind;data:Record<string,unknown>}
export interface QueryEvidence {
  nodes:QueryEvidenceNode[];edges:QueryEvidenceEdge[];truncated:boolean;
  provenanceMissing:number;provenanceOmitted:number;notices:string[];discarded:number;
}
export interface EvidenceSelection {nodeId:string|null;edgeId:string|null;expanded:boolean}
export interface ScopedEvidenceSelection extends EvidenceSelection {graph:QueryEvidence;api:object}

const object=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);
const validId=(value:unknown):value is number=>typeof value==='number'&&Number.isSafeInteger(value)&&value>0;
const count=(value:unknown):number=>typeof value==='number'&&Number.isSafeInteger(value)&&value>=0?value:0;
const nodeKinds:QueryNodeKind[]=['query','chunk','episode','claim','concept'];
export const edgeLabels:Record<QueryEdgeKind,string>={returned:'Returned by search',chunked_into:'Contains passage',source_of:'Recorded source of',extends:'Extends',derived_from:'Derived from',contradicts:'Contradicts',supports:'Supports',superseded_by:'Superseded by',mentions:'Mentions exact name',asserts:'Asserts concept',relation:'Recorded relation'};
export const evidenceEdgeLabel=(edge:QueryEvidenceEdge):string=>edge.kind==='relation'&&typeof edge.data.predicate==='string'?edge.data.predicate.replaceAll('_',' '):edge.kind==='asserts'&&typeof edge.data.role==='string'?`Asserts ${edge.data.role}`:edgeLabels[edge.kind];

export function edgeCategory(edge:QueryEvidenceEdge):string {
  if(edge.kind==='returned')return 'Retrieval';
  if(edge.kind==='chunked_into')return 'Source membership';
  if(edge.kind==='source_of')return 'Source provenance';
  if(edge.kind==='mentions')return 'Exact text mention';
  if(edge.kind==='asserts')return 'Claim assertion';
  if(edge.kind==='relation')return 'Recorded concept relation';
  return 'Recorded claim relationship';
}

function validEndpoints(kind:QueryEdgeKind,source:QueryEvidenceNode,target:QueryEvidenceNode):boolean {
  if(kind==='returned')return source.kind==='query'&&(target.kind==='chunk'||target.kind==='claim');
  if(kind==='chunked_into')return source.kind==='episode'&&target.kind==='chunk';
  if(kind==='source_of')return source.kind==='episode'&&target.kind==='claim';
  if(kind==='mentions')return source.kind==='chunk'&&target.kind==='concept';
  if(kind==='asserts')return source.kind==='claim'&&target.kind==='concept';
  if(kind==='relation')return source.kind==='concept'&&target.kind==='concept';
  return source.kind==='claim'&&target.kind==='claim';
}

/** Only display supported recorded edges. Scores never create relationships. */
export function parseQueryEvidence(value:unknown):QueryEvidence|null {
  if(!object(value)||!Array.isArray(value.nodes)||!Array.isArray(value.edges))return null;
  const nodes:QueryEvidenceNode[]=[],edges:QueryEvidenceEdge[]=[],seen=new Set<string>();
  let discarded=Math.max(0,value.nodes.length-256)+Math.max(0,value.edges.length-512);
  for(const node of value.nodes.slice(0,256)){
    if(!object(node)||typeof node.id!=='string'||!node.id||node.id.length>200||seen.has(node.id)
      ||!nodeKinds.includes(node.kind as QueryNodeKind)||typeof node.label!=='string') {discarded++;continue;}
    seen.add(node.id);
    nodes.push({id:node.id,kind:node.kind as QueryNodeKind,label:node.label,ts:typeof node.ts==='string'?node.ts:undefined,data:object(node.data)?node.data:{}});
  }
  const byId=new Map(nodes.map(node=>[node.id,node]));
  seen.clear();
  for(const edge of value.edges.slice(0,512)){
    if(!object(edge)||typeof edge.source!=='string'||typeof edge.target!=='string'||typeof edge.kind!=='string'
      ||!Object.hasOwn(edgeLabels,edge.kind)||edge.source===edge.target&&edge.kind!=='relation'){discarded++;continue;}
    const source=byId.get(edge.source),target=byId.get(edge.target),kind=edge.kind as QueryEdgeKind;
    if(!source||!target||!validEndpoints(kind,source,target)){discarded++;continue;}
    const data=object(edge.data)?edge.data:{};
    if(kind==='relation'&&(typeof data.predicate!=='string'||!data.predicate.trim()||!validId(data.fact_id))
      ||kind==='asserts'&&!['subject','object'].includes(String(data.role))){discarded++;continue;}
    const id=JSON.stringify([edge.source,edge.target,kind,validId(data.link_id)?data.link_id:null,validId(data.fact_id)?data.fact_id:null,kind==='relation'?data.predicate:kind==='asserts'?data.role:null]);
    if(seen.has(id)){discarded++;continue;}
    seen.add(id);edges.push({id,source:edge.source,target:edge.target,kind,data});
  }
  return {nodes,edges,truncated:value.truncated===true||discarded>0,
    provenanceMissing:count(value.provenance_missing),provenanceOmitted:count(value.provenance_omitted),
    notices:Array.isArray(value.notices)?value.notices.filter((notice):notice is string=>typeof notice==='string').slice(0,20):[],discarded};
}

export function connectionsFor(graph:QueryEvidence,nodeId:string):QueryEvidenceEdge[] {
  return graph.edges.filter(edge=>edge.source===nodeId||edge.target===nodeId);
}

export function evidenceSourceId(item:QueryEvidenceNode|QueryEvidenceEdge):number|null {
  if(['missing','omitted','out_of_scope'].includes(String(item.data.provenance_status)))return null;
  const id=item.data.episode_id??item.data.source_episode_id;
  return validId(id)?id:null;
}

export function evidenceSelection(graph:QueryEvidence,api:object,selected:ScopedEvidenceSelection|null):EvidenceSelection {
  const current=selected?.graph===graph&&selected.api===api?selected:null;
  const nodeId=current?.nodeId&&graph.nodes.some(node=>node.id===current.nodeId)?current.nodeId:graph.nodes.find(node=>node.kind==='query')?.id??graph.nodes[0]?.id??null;
  return {nodeId,edgeId:current?.edgeId&&graph.edges.some(edge=>edge.id===current.edgeId&&(edge.source===nodeId||edge.target===nodeId))?current.edgeId:null,expanded:current?.expanded??false};
}
