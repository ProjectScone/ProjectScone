import type { EvidenceNode, EvidenceEdge } from '../types.ts';

export const DEPTH_PAGE_SIZE=1000;
export const DEPTH_EDGE_LIMIT=4000;
export function depthPage(nodes:EvidenceNode[],edges:EvidenceEdge[],requested:number,focus?:string|null) {
  const ordered=[...nodes].sort((a,b)=>a.id.localeCompare(b.id));
  const anchor=focus?ordered.find(n=>n.id===focus):undefined;
  const neighbors=new Set<string>();
  if(anchor)for(const edge of edges){
    if(edge.source===anchor.id)neighbors.add(edge.target);
    if(edge.target===anchor.id)neighbors.add(edge.source);
  }
  const candidates=anchor?ordered.filter(n=>n.id!==anchor.id&&neighbors.has(n.id)):ordered;
  const pageSize=DEPTH_PAGE_SIZE-(anchor?1:0);
  const pages=Math.max(1,Math.ceil(candidates.length/pageSize));
  const page=Math.max(0,Math.min(requested,pages-1));
  const slice=candidates.slice(page*pageSize,(page+1)*pageSize);
  const shown=anchor?[anchor,...slice]:slice;
  const ids=new Set(shown.map(n=>n.id));
  const scopeIds=new Set(anchor?[anchor.id,...candidates.map(n=>n.id)]:ordered.map(n=>n.id));
  const scopedEdges=anchor?edges.filter(e=>e.source===anchor.id||e.target===anchor.id||(scopeIds.has(e.source)&&scopeIds.has(e.target))):edges;
  if(anchor)scopedEdges.sort((a,b)=>Number(b.source===anchor.id||b.target===anchor.id)-Number(a.source===anchor.id||a.target===anchor.id));
  const links=scopedEdges.filter(e=>ids.has(e.source)&&ids.has(e.target)).slice(0,DEPTH_EDGE_LIMIT);
  return {nodes:shown,edges:links,page,pages,total:candidates.length+(anchor?1:0),omittedLinks:scopedEdges.length-links.length,focus:anchor?.id??null};
}
