import type { EvidenceNode, EvidenceEdge, NodeKind } from '../types.ts';

export interface ViewNode extends EvidenceNode {
  owner: string;
  members?: string[];
  category: number;
  sequence?: number;
}
export interface ViewEdge extends EvidenceEdge { count: number; grouped: boolean }
export interface GraphView { nodes: ViewNode[]; edges: ViewEdge[]; total: number; pages: number; page: number; grouped: boolean; mode:'overview'|'focus'|'group' }
export interface GraphScope { group?: string; page?: number; focus?: string }
export const PAGE_SIZE = 8;
export const CARD_WIDTH = 224;
export const CARD_HEIGHT = 108;
export type NetworkLayout = 'constellation' | 'radial';

// Positions are presentation only. No inferred edges or simulated activity.
// A deterministic seed and fixed iteration budget avoid perpetual motion.
export function layoutNetwork(nodes: ViewNode[], edges: ViewEdge[], mode: NetworkLayout) {
  const ordered=[...nodes].sort((a,b)=>a.id.localeCompare(b.id));
  const points=new Map<string,{x:number;y:number}>();
  const lanes=[...new Set(ordered.map(n=>n.owner))].sort();
  const centers=new Map<string,{x:number;y:number}>();
  const ringRadius=(count:number)=>Math.max(190,count*40);
  const span=Math.max(640,...lanes.map(owner=>2*ringRadius(ordered.filter(n=>n.owner===owner&&n.kind!=='session').length)+260));
  for(const [index,owner] of lanes.entries()) {
    const center={x:(index%2)*span,y:Math.floor(index/2)*span};
    centers.set(owner,center);
    const members=ordered.filter(n=>n.owner===owner), anchor=members.find(n=>n.kind==='session');
    const satellites=members.filter(n=>n!==anchor);
    if(anchor)points.set(anchor.id,{...center});
    const radius=ringRadius(satellites.length);
    satellites.forEach((n,i)=>{const angle=-Math.PI/2+(mode==='constellation'?.7:0)+i*2*Math.PI/Math.max(1,satellites.length);
      points.set(n.id,{x:center.x+Math.cos(angle)*radius,y:center.y+Math.sin(angle)*radius});});
  }
  if(mode==='constellation') {
    const links=[...edges].sort((a,b)=>`${a.source}:${a.target}:${a.kind}`.localeCompare(`${b.source}:${b.target}:${b.kind}`));
    for(let tick=0;tick<220;tick++) {
      const forces=new Map(ordered.map(n=>[n.id,{x:0,y:0}]));
      for(let i=0;i<ordered.length;i++)for(let j=i+1;j<ordered.length;j++) {
        const a=points.get(ordered[i].id)!,b=points.get(ordered[j].id)!;
        const dx=a.x-b.x,dy=a.y-b.y,d=Math.max(1,Math.hypot(dx,dy));
        const strength=Math.min(22,16000/(d*d)+Math.max(0,175-d)*.18);
        const fa=forces.get(ordered[i].id)!,fb=forces.get(ordered[j].id)!;
        fa.x+=dx/d*strength;fa.y+=dy/d*strength;fb.x-=dx/d*strength;fb.y-=dy/d*strength;
      }
      for(const edge of links) {
        const a=points.get(edge.source),b=points.get(edge.target);if(!a||!b)continue;
        const dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)),pull=(d-190)*.025;
        const fa=forces.get(edge.source)!,fb=forces.get(edge.target)!;
        fa.x+=dx/d*pull;fa.y+=dy/d*pull;fb.x-=dx/d*pull;fb.y-=dy/d*pull;
      }
      for(const n of ordered) {
        const p=points.get(n.id)!,f=forces.get(n.id)!,center=centers.get(n.owner)!;
        p.x+=Math.max(-12,Math.min(12,f.x+(center.x-p.x)*.004));
        p.y+=Math.max(-12,Math.min(12,f.y+(center.y-p.y)*.004));
      }
    }
  }
  // Return stable order regardless of API record ordering.
  return new Map(ordered.map(n=>[n.id,points.get(n.id)!]));
}
const category: Record<NodeKind, number> = {session:0,turn:1,tool_call:1,episode:2,chunk:2,claim:3,recall:3,feedback:3};
const labels = ['', 'Interactions', 'Memories', 'Claims & recall'];

export function projectGrowth(nodes:EvidenceNode[],edges:EvidenceEdge[],requestedPage=0):GraphView & {undated:number} {
  const ownership=owners(nodes,edges);
  const activity=nodes.filter(n=>['turn','tool_call'].includes(n.kind));
  const dated=activity.filter(n=>n.ts&&Number.isFinite(Date.parse(n.ts))).sort((a,b)=>Date.parse(a.ts!)-Date.parse(b.ts!)||a.id.localeCompare(b.id));
  const counts=new Map<string,number>(),sequence=new Map<string,number>();
  for(const n of dated){const owner=ownership.get(n.id)||'unattributed',index=counts.get(owner)||0;sequence.set(n.id,index);counts.set(owner,index+1);}
  const ownersInOrder=[...counts.keys()].sort(),slices:EvidenceNode[][]=[];
  for(let i=0;i<ownersInOrder.length;i+=4){
    const batch=new Set(ownersInOrder.slice(i,i+4)),events=dated.filter(n=>batch.has(ownership.get(n.id)||'unattributed'));
    for(let start=0;start<events.length;start+=120)slices.push(events.slice(start,start+120));
  }
  const pages=Math.max(1,slices.length),page=Math.max(0,Math.min(requestedPage,pages-1));
  const slice=slices[page]||[],sessionIds=new Set(slice.map(n=>ownership.get(n.id)));
  const shown=[...nodes.filter(n=>n.kind==='session'&&sessionIds.has(n.id)),...slice];
  const ids=new Set(shown.map(n=>n.id));
  return {nodes:shown.map(n=>({...n,owner:ownership.get(n.id)||'unattributed',category:category[n.kind],sequence:sequence.get(n.id)})),
    edges:edges.filter(e=>ids.has(e.source)&&ids.has(e.target)).map(e=>({...e,count:1,grouped:false})),
    total:dated.length,undated:activity.length-dated.length,pages,page,grouped:false,mode:'overview'};
}

export function layoutGrowth(nodes:ViewNode[]) {
  const points=new Map<string,{x:number;y:number}>(),lanes=[...new Set(nodes.map(n=>n.owner))].sort();
  const goldenAngle=Math.PI*(3-Math.sqrt(5));
  const largest=Math.max(1,...nodes.map(n=>(n.sequence??0)+1));
  const span=2*(80+28*Math.sqrt(largest))+160;
  for(const [lane,owner] of lanes.entries()) {
    const cx=lane%2*span,cy=Math.floor(lane/2)*span;
    const records=nodes.filter(n=>n.owner===owner&&n.kind!=='session').sort((a,b)=>Date.parse(a.ts!)-Date.parse(b.ts!)||a.id.localeCompare(b.id));
    const anchor=nodes.find(n=>n.id===owner);if(anchor)points.set(anchor.id,{x:cx,y:cy});
    records.forEach((n,index)=>{const ordinal=n.sequence??index,radius=80+28*Math.sqrt(ordinal+1),angle=ordinal*goldenAngle-Math.PI/2;
      points.set(n.id,{x:cx+radius*Math.cos(angle),y:cy+radius*Math.sin(angle)});});
  }
  return points;
}

// Ownership is a presentation grouping, not a new evidence relationship.
// Multi-source records get one deterministic home; their real links remain.
export function owners(nodes: EvidenceNode[], edges: EvidenceEdge[]) {
  const known = new Set(nodes.map(n => n.id));
  const result = new Map<string, string>();
  const adjacency = new Map<string, string[]>();
  for (const e of edges) {
    if (!known.has(e.source) || !known.has(e.target)) continue;
    adjacency.set(e.source, [...(adjacency.get(e.source) || []), e.target]);
    if (['returned', 'held', 'judged'].includes(e.kind)) adjacency.set(e.target, [...(adjacency.get(e.target) || []), e.source]);
  }
  const queue = nodes.filter(n => n.kind === 'session').map(n => n.id).sort();
  for (const id of queue) result.set(id, id);
  for (let i = 0; i < queue.length; i++) {
    const id = queue[i];
    for (const next of (adjacency.get(id) || []).sort()) if (!result.has(next)) {
      result.set(next, result.get(id)!); queue.push(next);
    }
  }
  return result;
}

export function projectGraph(nodes: EvidenceNode[], edges: EvidenceEdge[], scope: GraphScope = {}): GraphView {
  const ownership = owners(nodes, edges);
  const originals: ViewNode[] = nodes.map(n => ({...n, owner: ownership.get(n.id) || 'unattributed', category:category[n.kind]}));
  const buckets = new Map<string, ViewNode[]>();
  for (const n of originals) if (n.kind !== 'session') {
    const key = `view-group:${JSON.stringify([n.owner, n.category])}`;
    buckets.set(key, [...(buckets.get(key) || []), n]);
  }
  let shown = originals, total = nodes.length, page = 0, pages = 1;
  let mode:GraphView['mode']='overview';
  const replacements = new Map<string, string>();
  if (scope.focus && nodes.some(n => n.id === scope.focus && n.kind !== 'session')) {
    mode='focus';
    const ids = new Set<string>();
    for (const e of edges) {
      if (e.source === scope.focus) ids.add(e.target);
      if (e.target === scope.focus) ids.add(e.source);
    }
    ids.delete(scope.focus);
    const neighbors=originals.filter(n=>ids.has(n.id)).sort((a,b)=>String(b.ts||'').localeCompare(String(a.ts||''))||a.id.localeCompare(b.id));
    total=neighbors.length;pages=Math.max(1,Math.ceil(total/PAGE_SIZE));page=Math.max(0,Math.min(scope.page||0,pages-1));
    shown=[originals.find(n=>n.id===scope.focus)!,...neighbors.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE)];
  } else if (scope.group && buckets.has(scope.group)) {
    mode='group';
    const members = [...buckets.get(scope.group)!].sort((a,b) => String(b.ts || '').localeCompare(String(a.ts || '')) || a.id.localeCompare(b.id));
    total = members.length; pages = Math.ceil(total / PAGE_SIZE); page = Math.max(0, Math.min(scope.page || 0, pages - 1));
    const chosen = members.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
    const owner = originals.find(n => n.id === members[0].owner);
    shown = owner ? [owner, ...chosen] : chosen;
  } else if (nodes.length > 24) {
    shown = originals.filter(n => n.kind === 'session');
    for (const [id, members] of buckets) {
      if (members.length === 1) { shown.push(members[0]); continue; }
      const first = members[0];
      shown.push({ id, kind:first.kind, label:`${labels[first.category]} · ${members.length}`, owner:first.owner, category:first.category, members:members.map(n => n.id) });
      for (const n of members) replacements.set(n.id, id);
    }
  }
  if(mode==='overview') {
    const lanes=[...new Set(shown.map(n=>n.owner))].sort();
    pages=Math.max(1,Math.ceil(lanes.length/4));page=Math.max(0,Math.min(scope.page||0,pages-1));
    const visibleLanes=new Set(lanes.slice(page*4,(page+1)*4));
    shown=shown.filter(n=>visibleLanes.has(n.owner));
  }
  const visible = new Set(shown.map(n => n.id));
  const links = new Map<string, ViewEdge>();
  for (const e of edges) {
    const source = replacements.get(e.source) || e.source, target = replacements.get(e.target) || e.target;
    if (source === target || !visible.has(source) || !visible.has(target)) continue;
    const grouped = replacements.has(e.source) || replacements.has(e.target);
    const key = JSON.stringify([source,target,e.kind]);
    const previous = links.get(key);
    if (previous) previous.count++;
    else links.set(key, {...e,source,target,count:1,grouped});
  }
  return {nodes:shown, edges:[...links.values()], total, page, pages, grouped:replacements.size > 0,mode};
}

export function layoutGraph(nodes: ViewNode[]) {
  const points = new Map<string, {x:number;y:number}>();
  const lanes = [...new Set(nodes.map(n => n.owner))].sort();
  let top = 0;
  for (const lane of lanes) {
    const members = nodes.filter(n => n.owner === lane);
    const counts = [0,1,2,3].map(c=>members.filter(n=>n.category===c).length);
    const widths = counts.map(count=>count > 3 ? Math.min(4,count) * 264 : 272);
    const offsets = widths.map((_,i)=>widths.slice(0,i).reduce((sum,w)=>sum+w,0));
    const rows = new Map<number, number>();
    for (const n of members) {
      const row = rows.get(n.category) || 0;
      rows.set(n.category,row + 1);
      // Wrap long expanded groups across a four-card grid, rather than
      // accumulating an unbounded vertical column or persistent cache gaps.
      const expanded = counts[n.category] > 3;
      const x = offsets[n.category] + (expanded ? row % 4 * 264 : 0);
      const y = top + (expanded ? Math.floor(row / 4) : row) * 140;
      points.set(n.id, {x,y});
    }
    const bottom = Math.max(top, ...members.map(n => points.get(n.id)!.y));
    top = bottom + 172;
  }
  return points;
}
