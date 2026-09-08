import type {QueryEvidence,QueryEvidenceNode,QueryEvidenceEdge} from './query-evidence.ts';

export interface EvidenceGroup {id:string;label:string;nodeIds:string[];isolated:boolean}
export interface EvidenceGroups {method:'label_propagation'|'components'|'unlinked';groups:EvidenceGroup[]}
export interface EvidencePoint {x:number;y:number}
export interface EvidenceCamera extends EvidencePoint {zoom:number}
export interface EvidenceSize {width:number;height:number}
export type EvidenceView='concepts'|'provenance';
const order=(a:string,b:string)=>a<b?-1:a>b?1:0;
const clamp=(value:number,min:number,max:number)=>Math.min(max,Math.max(min,value));

/** Original deterministic label propagation on recorded concept relations only.
 * Mentions and retrieval proximity do not enter this adjacency matrix. */
export function detectEvidenceGroups(graph:QueryEvidence):EvidenceGroups {
  const ids=graph.nodes.filter(node=>node.kind==='concept').map(node=>node.id).sort(order);
  const neighbors=new Map(ids.map(id=>[id,new Set<string>()]));
  for(const edge of graph.edges){
    if(edge.kind!=='relation'||edge.source===edge.target||!neighbors.has(edge.source)||!neighbors.has(edge.target))continue;
    neighbors.get(edge.source)!.add(edge.target);neighbors.get(edge.target)!.add(edge.source);
  }
  const visited=new Set<string>(),components:string[][]=[],isolated:string[]=[];
  for(const id of ids){
    if(visited.has(id))continue;
    if(!neighbors.get(id)!.size){isolated.push(id);visited.add(id);continue;}
    const component:string[]=[],queue=[id];visited.add(id);
    for(let cursor=0;cursor<queue.length;cursor++){
      const current=queue[cursor];component.push(current);
      for(const next of [...neighbors.get(current)!].sort(order))if(!visited.has(next)){visited.add(next);queue.push(next);}
    }
    components.push(component.sort(order));
  }
  let split=false;const partition:string[][]=[];
  for(const component of components){
    const edgeCount=component.reduce((sum,id)=>sum+neighbors.get(id)!.size,0)/2;
    // Trees and pairs are honestly described as components, not communities.
    if(edgeCount<component.length){partition.push(component);continue;}
    const labels=new Map(component.map(id=>[id,id]));
    const sequence=[...component].sort((a,b)=>neighbors.get(a)!.size-neighbors.get(b)!.size||order(a,b));
    for(let pass=0;pass<30;pass++){
      let changed=false;
      for(const id of sequence){
        const votes=new Map<string,number>();
        for(const next of neighbors.get(id)!){const label=labels.get(next)!;votes.set(label,(votes.get(label)??0)+1);}
        const best=Math.max(...votes.values()),current=labels.get(id)!;
        const winner=votes.get(current)===best?current:[...votes].filter(([,weight])=>weight===best).map(([label])=>label).sort(order)[0];
        if(winner!==current){labels.set(id,winner);changed=true;}
      }
      if(!changed)break;
    }
    const found=new Map<string,string[]>();
    for(const id of component){const label=labels.get(id)!;found.set(label,[...(found.get(label)??[]),id]);}
    split=split||found.size>1;partition.push(...found.values());
  }
  partition.sort((a,b)=>order(a[0],b[0]));
  const groups:EvidenceGroup[]=partition.map((nodeIds,index)=>({id:`group:${nodeIds[0]}`,label:`Group ${index+1}`,nodeIds,isolated:false}));
  if(isolated.length)groups.push({id:'unlinked',label:'Unlinked concepts',nodeIds:isolated,isolated:true});
  return {method:split?'label_propagation':components.length?'components':'unlinked',groups};
}

export function projectEvidenceNetwork(graph:QueryEvidence,view:EvidenceView,groups:EvidenceGroups,groupId:string|null){
  const group=groups.groups.find(group=>group.id===groupId);
  const allowed=group?new Set(group.nodeIds):null;
  const nodes=graph.nodes.filter(node=>view==='concepts'?node.kind==='concept'&&(!allowed||allowed.has(node.id)):true);
  const ids=new Set(nodes.map(node=>node.id));
  return {nodes,edges:graph.edges.filter(edge=>ids.has(edge.source)&&ids.has(edge.target))};
}

/** Bounded spring layout with deterministic initial positions and hard separation. */
export function layoutEvidenceNetwork(nodes:QueryEvidenceNode[],edges:QueryEvidenceEdge[],groups?:EvidenceGroups):Map<string,EvidencePoint> {
  const sorted=[...nodes].sort((a,b)=>order(a.id,b.id)),ids=new Set(sorted.map(node=>node.id));
  const links=[...edges].filter(edge=>ids.has(edge.source)&&ids.has(edge.target)&&edge.source!==edge.target).sort((a,b)=>order(a.id,b.id));
  const memberships=new Map<string,string>();
  if(groups)for(const group of groups.groups)for(const id of group.nodeIds)memberships.set(id,group.id);
  const groupIds=[...new Set(sorted.map(node=>memberships.get(node.id)??'all'))].sort(order);
  const biggest=Math.max(1,...groupIds.map(group=>sorted.filter(node=>(memberships.get(node.id)??'all')===group).length));
  const stride=Math.max(300,Math.sqrt(biggest)*105),columns=Math.ceil(Math.sqrt(groupIds.length));
  const centers=new Map(groupIds.map((group,index)=>[group,{x:(index%columns)*stride,y:Math.floor(index/columns)*stride}]));
  const points=new Map<string,EvidencePoint>();
  for(const group of groupIds){
    const members=sorted.filter(node=>(memberships.get(node.id)??'all')===group),center=centers.get(group)!;
    members.forEach((node,index)=>{const angle=index*2.399963229728653,radius=members.length===1?0:Math.sqrt(index+.5)*48;
      points.set(node.id,{x:center.x+Math.cos(angle)*radius,y:center.y+Math.sin(angle)*radius});});
  }
  for(let tick=0;tick<160;tick++){
    const forces=new Map(sorted.map(node=>[node.id,{x:0,y:0}]));
    for(let i=0;i<sorted.length;i++)for(let j=i+1;j<sorted.length;j++){
      const a=points.get(sorted[i].id)!,b=points.get(sorted[j].id)!,dx=a.x-b.x,dy=a.y-b.y,distance=Math.max(.01,Math.hypot(dx,dy));
      const strength=Math.min(12,1100/(distance*distance)+Math.max(0,62-distance)*.2);
      const ax=dx/distance*strength,ay=dy/distance*strength,af=forces.get(sorted[i].id)!,bf=forces.get(sorted[j].id)!;
      af.x+=ax;af.y+=ay;bf.x-=ax;bf.y-=ay;
    }
    for(const edge of links){
      const a=points.get(edge.source)!,b=points.get(edge.target)!,dx=b.x-a.x,dy=b.y-a.y,distance=Math.max(.01,Math.hypot(dx,dy));
      const across=memberships.get(edge.source)!==memberships.get(edge.target),strength=(distance-(across?stride*.65:95))*(across?.004:.018);
      const af=forces.get(edge.source)!,bf=forces.get(edge.target)!;
      af.x+=dx/distance*strength;af.y+=dy/distance*strength;bf.x-=dx/distance*strength;bf.y-=dy/distance*strength;
    }
    for(const node of sorted){const point=points.get(node.id)!,force=forces.get(node.id)!,center=centers.get(memberships.get(node.id)??'all')!;
      point.x+=clamp(force.x+(center.x-point.x)*.007,-9,9);point.y+=clamp(force.y+(center.y-point.y)*.007,-9,9);}
  }
  // A finite collision pass reserves circle hit targets without shrinking records.
  for(let pass=0;pass<120;pass++){
    let moved=false;
    for(let i=0;i<sorted.length;i++)for(let j=i+1;j<sorted.length;j++){
      const a=points.get(sorted[i].id)!,b=points.get(sorted[j].id)!,dx=b.x-a.x,dy=b.y-a.y,distance=Math.hypot(dx,dy);
      if(distance>=38)continue;
      const x=distance>0?dx/distance:1,y=distance>0?dy/distance:0,shift=(38.1-distance)/2;
      a.x-=x*shift;a.y-=y*shift;b.x+=x*shift;b.y+=y*shift;moved=true;
    }
    if(!moved)break;
  }
  return points;
}

export function fitEvidenceCamera(points:Map<string,EvidencePoint>,size:EvidenceSize):EvidenceCamera {
  if(!points.size)return {x:size.width/2,y:size.height/2,zoom:1};
  const values=[...points.values()],left=Math.min(...values.map(point=>point.x))-45,right=Math.max(...values.map(point=>point.x))+45;
  const top=Math.min(...values.map(point=>point.y))-45,bottom=Math.max(...values.map(point=>point.y))+45;
  const zoom=clamp(Math.min((size.width-70)/(right-left),(size.height-80)/(bottom-top),1.5),.08,2);
  return {x:size.width/2-(left+right)/2*zoom,y:size.height/2-(top+bottom)/2*zoom,zoom};
}
export function zoomEvidenceCamera(camera:EvidenceCamera,factor:number,focus:EvidencePoint):EvidenceCamera {
  const zoom=clamp(camera.zoom*factor,.08,5);
  return {zoom,x:focus.x-(focus.x-camera.x)*zoom/camera.zoom,y:focus.y-(focus.y-camera.y)*zoom/camera.zoom};
}

export interface EvidenceLabel extends EvidencePoint {id:string;text:string;width:number;height:number}
export function placeEvidenceLabels(nodes:QueryEvidenceNode[],points:Map<string,EvidencePoint>,camera:EvidenceCamera,size:EvidenceSize,selected:string|null):EvidenceLabel[] {
  const labels:EvidenceLabel[]=[];
  const ordered=[...nodes].sort((a,b)=>Number(b.id===selected)-Number(a.id===selected)||order(a.id,b.id));
  for(const node of ordered){
    if(camera.zoom<.55&&node.id!==selected)continue;
    const point=points.get(node.id);if(!point)continue;
    const px=point.x*camera.zoom+camera.x,py=point.y*camera.zoom+camera.y;
    if(px<0||px>size.width||py<0||py>size.height)continue;
    const text=node.label.length>23?node.label.slice(0,22)+'…':node.label,width=Math.min(size.width-8,text.length*6.1+12),height=22;
    const candidates=[{x:px+12,y:py-11},{x:px-width-12,y:py-11},{x:px-width/2,y:py+13},{x:px-width/2,y:py-height-13}];
    for(const candidate of candidates){
      const box={id:node.id,text,width,height,x:clamp(candidate.x,4,size.width-width-4),y:clamp(candidate.y,4,size.height-height-4)};
      if(labels.some(other=>!(box.x+box.width+5<=other.x||other.x+other.width+5<=box.x||box.y+box.height+3<=other.y||other.y+other.height+3<=box.y)))continue;
      labels.push(box);break;
    }
    if(labels.length>=Math.max(8,Math.min(28,Math.floor(size.width/30))))break;
  }
  return labels;
}
