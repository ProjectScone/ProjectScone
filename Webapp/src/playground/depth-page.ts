import type { EvidenceNode, EvidenceEdge } from '../types.ts';

export const DEPTH_PAGE_SIZE=1000;
export const DEPTH_EDGE_LIMIT=4000;
export function depthPage(nodes:EvidenceNode[],edges:EvidenceEdge[],requested:number) {
  const ordered=[...nodes].sort((a,b)=>a.id.localeCompare(b.id));
  const pages=Math.max(1,Math.ceil(ordered.length/DEPTH_PAGE_SIZE));
  const page=Math.max(0,Math.min(requested,pages-1));
  const shown=ordered.slice(page*DEPTH_PAGE_SIZE,(page+1)*DEPTH_PAGE_SIZE);
  const ids=new Set(shown.map(n=>n.id));
  const links=edges.filter(e=>ids.has(e.source)&&ids.has(e.target)).slice(0,DEPTH_EDGE_LIMIT);
  return {nodes:shown,edges:links,page,pages,total:ordered.length,omittedLinks:edges.length-links.length};
}
