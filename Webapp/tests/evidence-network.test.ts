import {test} from 'node:test';
import assert from 'node:assert/strict';
import {detectEvidenceGroups,layoutEvidenceNetwork,projectEvidenceNetwork,fitEvidenceCamera,zoomEvidenceCamera,placeEvidenceLabels} from '../src/memory/evidence-network.ts';
import {parseQueryEvidence} from '../src/memory/query-evidence.ts';

function clusters(){
  const nodes=Array.from({length:7},(_,index)=>({id:`concept:${index}`,kind:'concept',label:`Concept ${index}`,data:{name:`Concept ${index}`}}));
  const pairs=[[0,1],[0,2],[1,2],[3,4],[3,5],[4,5],[2,3]];
  return parseQueryEvidence({nodes,edges:pairs.map(([source,target],index)=>({source:`concept:${source}`,target:`concept:${target}`,kind:'relation',data:{predicate:'uses',fact_id:index+1}}))})!;
}
test('communities find dense recorded concept groups and ignore retrieval/mention proximity',()=>{
  const graph=clusters(),groups=detectEvidenceGroups(graph);
  assert.equal(groups.method,'label_propagation');
  assert.deepEqual(groups.groups.filter(group=>!group.isolated).map(group=>group.nodeIds),[['concept:0','concept:1','concept:2'],['concept:3','concept:4','concept:5']]);
  assert.deepEqual(groups.groups.find(group=>group.isolated)?.nodeIds,['concept:6']);
  assert.deepEqual(detectEvidenceGroups({...graph,nodes:[...graph.nodes].reverse(),edges:[...graph.edges].reverse()}),groups);
  const sparse=detectEvidenceGroups({...graph,edges:graph.edges.slice(0,1)});
  assert.equal(sparse.method,'components');
  assert.equal(sparse.groups.filter(group=>!group.isolated).length,1);
  const noFacts=detectEvidenceGroups({...graph,edges:[]});
  assert.equal(noFacts.method,'unlinked');
  assert.equal(noFacts.groups.length,1);
  assert.equal(noFacts.groups[0].isolated,true);
});

test('network projection preserves only real endpoints and edges in the selected view/group',()=>{
  const graph=clusters();graph.nodes.push({id:'query',kind:'query',label:'Query',data:{}});
  const groups=detectEvidenceGroups(graph);
  const concepts=projectEvidenceNetwork(graph,'concepts',groups,null);
  assert.equal(concepts.nodes.length,7);assert.equal(concepts.edges.length,7);
  const first=projectEvidenceNetwork(graph,'concepts',groups,groups.groups[0].id);
  assert.equal(first.nodes.length,3);assert.equal(first.edges.length,3);
  assert.ok(first.edges.every(edge=>first.nodes.some(node=>node.id===edge.source)&&first.nodes.some(node=>node.id===edge.target)));
  assert.equal(projectEvidenceNetwork(graph,'provenance',groups,null).nodes.length,8);
});

test('100-node layouts are deterministic, finite and keep selectable nodes apart',()=>{
  const graph=parseQueryEvidence({nodes:Array.from({length:100},(_,i)=>({id:`concept:${i}`,kind:'concept',label:`Name ${i}`,data:{}})),edges:Array.from({length:99},(_,i)=>({source:`concept:${i}`,target:`concept:${i+1}`,kind:'relation',data:{fact_id:i+1,predicate:'next'}}))})!;
  const points=layoutEvidenceNetwork(graph.nodes,graph.edges),reversed=layoutEvidenceNetwork([...graph.nodes].reverse(),[...graph.edges].reverse());
  assert.deepEqual(points,reversed);assert.equal(points.size,100);
  const positions=[...points.values()];
  assert.ok(positions.every(point=>Number.isFinite(point.x)&&Number.isFinite(point.y)));
  for(let i=0;i<positions.length;i++)for(let j=i+1;j<positions.length;j++)assert.ok(Math.hypot(positions[i].x-positions[j].x,positions[i].y-positions[j].y)>=37.9,'selectable circles overlap');
});

test('fit and zoom preserve focal points and labels do not collide in mobile canvas',()=>{
  const graph=clusters(),points=layoutEvidenceNetwork(graph.nodes,graph.edges),size={width:358,height:340};
  const camera=fitEvidenceCamera(points,size),zoomed=zoomEvidenceCamera(camera,2,{x:100,y:130});
  assert.equal((100-camera.x)/camera.zoom,(100-zoomed.x)/zoomed.zoom);
  assert.equal((130-camera.y)/camera.zoom,(130-zoomed.y)/zoomed.zoom);
  const labels=placeEvidenceLabels(graph.nodes,points,camera,size,'concept:2');
  assert.ok(labels.some(label=>label.id==='concept:2'));
  for(const label of labels){assert.ok(label.x>=0&&label.y>=0&&label.x+label.width<=size.width&&label.y+label.height<=size.height);}
  for(let i=0;i<labels.length;i++)for(let j=i+1;j<labels.length;j++){
    const a=labels[i],b=labels[j];assert.ok(a.x+a.width<=b.x||b.x+b.width<=a.x||a.y+a.height<=b.y||b.y+b.height<=a.y);
  }
  assert.deepEqual(fitEvidenceCamera(new Map(),size),{x:179,y:170,zoom:1});
});
