import { test } from 'node:test';
import assert from 'node:assert/strict';
import { layoutDepth, projectPoint, rotatePoint, type Vec3 } from '../src/playground/depth-layout.ts';
import type { EvidenceNode, EvidenceEdge } from '../src/types.ts';
import { depthPage } from '../src/playground/depth-page.ts';
import { owners } from '../src/playground/graph-layout.ts';

test('later depth pages retain session clusters when their source paths are off-page',()=>{
  const sessions:EvidenceNode[]=['a','b'].map(id=>({id,kind:'session',label:id}));
  const records:EvidenceNode[]=Array.from({length:1250},(_,i)=>({id:`t${String(i).padStart(4,'0')}`,kind:'turn',label:'Turn'}));
  const nodes=[...sessions,...records];
  const edges:EvidenceEdge[]=records.map((n,i)=>({source:i%2?'a':'b',target:n.id,kind:'has'}));
  const page=depthPage(nodes,edges,1),ownership=owners(nodes,edges);
  assert.equal(page.edges.length,0);
  assert.ok(page.nodes.every(n=>n.kind!=='session'));
  const points=layoutDepth(page.nodes,page.edges,'constellation',ownership);
  const center=(owner:string)=>{
    const members=page.nodes.filter(n=>ownership.get(n.id)===owner).map(n=>points.get(n.id)!);
    return members.reduce((sum,p)=>({x:sum.x+p.x/members.length,y:sum.y+p.y/members.length,z:sum.z+p.z/members.length}),{x:0,y:0,z:0});
  };
  const a=center('a'),b=center('b');
  assert.ok(Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z)>200,'Known session clusters must remain separated.');
});

test('large depth pages bound nodes and links and disclose every omitted link',()=>{
  const nodes:EvidenceNode[]=Array.from({length:1250},(_,i)=>({id:String(i).padStart(4,'0'),kind:'claim',label:'Claim'}));
  const edges:EvidenceEdge[]=Array.from({length:5000},(_,i)=>({source:'0000',target:String(1+i%999).padStart(4,'0'),kind:'derived_from'}));
  const first=depthPage(nodes,edges,0),last=depthPage(nodes,edges,99);
  assert.equal(first.nodes.length,1000);assert.equal(first.edges.length,4000);
  assert.equal(first.omittedLinks,1000);
  assert.equal(last.nodes.length,250);assert.equal(last.page,1);assert.equal(last.edges.length,0);
  assert.equal(last.omittedLinks,5000);
  assert.equal(new Set([...first.nodes,...last.nodes].map(n=>n.id)).size,1250);
  assert.ok(first.edges.every(e=>Number(e.target)<1000));
});

test('camera rotation changes Z and preserves distance from the origin',()=>{
  const point={x:100,y:0,z:0};
  const rotated=rotatePoint(point,{yaw:Math.PI/2,pitch:0});
  assert.ok(Math.abs(rotated.x)<1e-10);
  assert.ok(Math.abs(rotated.z+100)<1e-10);
  assert.ok(Math.abs(Math.hypot(rotated.x,rotated.y,rotated.z)-100)<1e-10);
});

test('perspective produces parallax and clips points behind the camera',()=>{
  const near=projectPoint({x:50,y:20,z:100},400);
  const far=projectPoint({x:50,y:20,z:-100},400);
  assert.ok(near && far);
  assert.ok(near.x>far.x && near.scale>far.scale);
  assert.equal(projectPoint({x:0,y:0,z:400},400),null);
});

test('depth keeps every record and finite XYZ positions independent of input order',()=>{
  const nodes:EvidenceNode[]=[{id:'s',kind:'session',label:'Session'},...Array.from({length:60},(_,i)=>({id:`t${i}`,kind:'turn' as const,label:'Turn'})),{id:'claim',kind:'claim',label:'No source in snapshot'}];
  const edges:EvidenceEdge[]=nodes.slice(1,-1).map(n=>({source:'s',target:n.id,kind:'has'}));
  const before=JSON.stringify({nodes,edges});
  for(const mode of ['constellation','radial','flow'] as const){
    const points=layoutDepth(nodes,edges,mode);
    assert.equal(points.size,62);
    assert.deepEqual(points,layoutDepth([...nodes].reverse(),[...edges].reverse(),mode));
    assert.ok([...points.values()].every(p=>Object.values(p).every(Number.isFinite)));
    assert.ok(new Set([...points.values()].map(p=>p.z)).size>3);
    const ps:Vec3[]=[...points.values()];
    const [a,b,c,d]=ps;
    const ab={x:b.x-a.x,y:b.y-a.y,z:b.z-a.z},ac={x:c.x-a.x,y:c.y-a.y,z:c.z-a.z},ad={x:d.x-a.x,y:d.y-a.y,z:d.z-a.z};
    assert.ok(Math.abs(ab.x*(ac.y*ad.z-ac.z*ad.y)-ab.y*(ac.x*ad.z-ac.z*ad.x)+ab.z*(ac.x*ad.y-ac.y*ad.x))>1e-5,'Non-coplanar geometry, not a tilted 2D plane.');
  }
  assert.notDeepEqual(layoutDepth(nodes,edges,'constellation'),layoutDepth(nodes,edges,'radial'));
  assert.equal(JSON.stringify({nodes,edges}),before);
  assert.equal(layoutDepth([],[],'constellation').size,0);
});
