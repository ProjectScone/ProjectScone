import type { EvidenceNode, EvidenceEdge, NodeKind } from '../types.ts';
import { owners } from './graph-layout.ts';

export interface Vec3 { x: number; y: number; z: number }
export interface Orbit { yaw: number; pitch: number }
export interface ProjectedPoint { x: number; y: number; z: number; scale: number }
export type DepthLayout = 'constellation' | 'radial' | 'flow';
export const nodeCategory: Record<NodeKind, number> = {session:0,turn:1,tool_call:1,episode:2,chunk:2,claim:3,recall:3,feedback:3};
const goldenAngle = Math.PI * (3 - Math.sqrt(5));

export function rotatePoint(point: Vec3, orbit: Orbit): Vec3 {
  const cy=Math.cos(orbit.yaw), sy=Math.sin(orbit.yaw), cp=Math.cos(orbit.pitch), sp=Math.sin(orbit.pitch);
  // R_x(pitch) × R_y(yaw), row-major. Camera rotation never edits evidence.
  const matrix=[cy,0,sy, sp*sy,cp,-sp*cy, -cp*sy,sp,cp*cy];
  return {x:matrix[0]*point.x+matrix[1]*point.y+matrix[2]*point.z,
    y:matrix[3]*point.x+matrix[4]*point.y+matrix[5]*point.z,
    z:matrix[6]*point.x+matrix[7]*point.y+matrix[8]*point.z};
}

export function projectPoint(point: Vec3, distance: number): ProjectedPoint | null {
  const denominator=distance-point.z;
  if(denominator<=20)return null;
  const scale=distance/denominator;
  return {...point,x:point.x*scale,y:point.y*scale,scale};
}

function sphere(index: number, count: number, radius: number): Vec3 {
  const y=1-2*(index+.5)/Math.max(1,count), ring=Math.sqrt(1-y*y), angle=index*goldenAngle;
  return {x:Math.cos(angle)*ring*radius,y:y*radius,z:Math.sin(angle)*ring*radius};
}

export function layoutDepth(nodes: EvidenceNode[], edges: EvidenceEdge[], mode: DepthLayout, ownership: ReadonlyMap<string,string> = owners(nodes,edges)): Map<string,Vec3> {
  const ordered=[...nodes].sort((a,b)=>a.id.localeCompare(b.id));
  const lanes=[...new Set(ordered.map(n=>ownership.get(n.id)||'unattributed'))].sort();
  const positions=new Map<string,Vec3>();
  for(const [lane,owner] of lanes.entries()){
    const members=ordered.filter(n=>(ownership.get(n.id)||'unattributed')===owner);
    const center=lanes.length===1?{x:0,y:0,z:0}:sphere(lane,lanes.length,550);
    const radius=Math.max(170,50*Math.cbrt(members.length));
    members.forEach((node,index)=>{
      let offset=sphere(index,members.length,radius);
      if(mode==='constellation'){
        const spread=.4+.6*Math.cbrt((index+.5)/members.length);
        offset={x:offset.x*spread,y:offset.y*spread,z:offset.z*spread};
      } else if(mode==='flow'){
        offset={x:(nodeCategory[node.kind]-1.5)*130+offset.x*.22,y:offset.y,z:offset.z};
      }
      if(node.kind==='session')offset={x:0,y:0,z:0};
      positions.set(node.id,{x:center.x+offset.x,y:center.y+offset.y,z:center.z+offset.z});
    });
  }
  // Fit one stable world volume; camera changes only project that volume.
  const values=[...positions.values()];
  if(!values.length)return positions;
  const center={x:values.reduce((s,p)=>s+p.x,0)/values.length,y:values.reduce((s,p)=>s+p.y,0)/values.length,z:values.reduce((s,p)=>s+p.z,0)/values.length};
  const radius=Math.max(1,...values.map(p=>Math.hypot(p.x-center.x,p.y-center.y,p.z-center.z)));
  for(const [id,p] of positions)positions.set(id,{x:(p.x-center.x)*400/radius,y:(p.y-center.y)*400/radius,z:(p.z-center.z)*400/radius});
  return positions;
}
