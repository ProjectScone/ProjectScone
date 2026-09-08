import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parseQueryEvidence, evidenceSelection, evidenceSourceId, connectionsFor, edgeCategory} from '../src/memory/query-evidence.ts';

const nodes=[
  {id:'query:4',kind:'query',label:'Where did the decision come from?',data:{event_id:4}},
  {id:'chunk:2',kind:'chunk',label:'A passage',data:{chunk_id:2,episode_id:3,text:'We chose Cedar.',score:.7}},
  {id:'episode:3',kind:'episode',label:'Meeting note',data:{episode_id:3,preview:'We chose Cedar.'}},
  {id:'claim:5',kind:'claim',label:'Team uses Cedar',data:{fact_id:5,source_episode_id:3,quote:'We chose Cedar.',provenance_status:'retained'}},
  {id:'claim:6',kind:'claim',label:'Team uses Maple',data:{fact_id:6,provenance_status:'missing'}},
];
const edges=[
  {source:'query:4',target:'chunk:2',kind:'returned',data:{category:'retrieval'}},
  {source:'episode:3',target:'chunk:2',kind:'chunked_into',data:{category:'membership'}},
  {source:'episode:3',target:'claim:5',kind:'source_of',data:{category:'provenance',quote:'We chose Cedar.',verified:true}},
  {source:'claim:5',target:'claim:6',kind:'contradicts',data:{category:'fact_relation',source_episode_id:3,quote:'We chose Cedar.',provenance_status:'retained'}},
];
const raw={nodes,edges,truncated:false,provenance_missing:1,provenance_omitted:0,notices:['One source was forgotten.'],counts:{chunks:1,claims:2,sources:1,links:1}};

test('query evidence preserves recorded direction, source quotes and partial provenance',()=>{
  const graph=parseQueryEvidence(raw)!;
  assert.equal(graph.nodes.length,5);
  assert.equal(graph.edges.length,4);
  assert.equal(graph.provenanceMissing,1);
  assert.deepEqual(graph.notices,['One source was forgotten.']);
  const connections=connectionsFor(graph,'claim:5');
  assert.deepEqual(connections.map(edge=>[edge.source,edge.target,edge.kind]),[
    ['episode:3','claim:5','source_of'],['claim:5','claim:6','contradicts'],
  ]);
  assert.equal(connections[1].data.quote,'We chose Cedar.');
  assert.equal(edgeCategory(graph.edges[0]),'Retrieval');
  assert.equal(edgeCategory(graph.edges[1]),'Source membership');
  assert.equal(edgeCategory(connections[1]),'Recorded claim relationship');
});

test('malformed or invented relationships never become graph evidence',()=>{
  assert.equal(parseQueryEvidence(undefined),null);
  assert.equal(parseQueryEvidence({nodes:[],edges:'bad'}),null);
  const graph=parseQueryEvidence({...raw,nodes:[...nodes,nodes[0],{id:'bad',kind:'similar',label:'False claim'}],edges:[...edges,edges[0],
    {source:'chunk:2',target:'claim:5',kind:'supports'},
    {source:'claim:5',target:'claim:6',kind:'causes'},
    {source:'missing',target:'claim:5',kind:'source_of'},
    {source:'claim:5',target:'claim:5',kind:'contradicts'},
  ]})!;
  assert.equal(graph.nodes.length,5);
  assert.equal(graph.edges.length,4);
  assert.ok(graph.discarded>0);
  assert.equal(graph.truncated,true);
  assert.ok(connectionsFor(graph,'not-in-graph').length===0);
});

test('graph source navigation rejects malformed ids and withheld provenance',()=>{
  const graph=parseQueryEvidence(raw)!;
  assert.equal(evidenceSourceId(graph.nodes[1]),3);
  assert.equal(evidenceSourceId(graph.nodes[2]),3);
  assert.equal(evidenceSourceId(graph.nodes[3]),3);
  assert.equal(evidenceSourceId(graph.nodes[4]),null);
  for(const value of ['../../secrets',0,-1,Number.MAX_SAFE_INTEGER+1]){
    assert.equal(evidenceSourceId({...graph.nodes[3],data:{source_episode_id:value}}),null);
  }
  for(const status of ['out_of_scope','omitted','missing']){
    assert.equal(evidenceSourceId({...graph.nodes[3],data:{source_episode_id:3,provenance_status:status}}),null);
  }
});

test('selection and expanded lanes reset for new results or a new credential client',()=>{
  const graph=parseQueryEvidence(raw)!,api={},otherApi={};
  const selected={graph,api,nodeId:'claim:5',edgeId:graph.edges[3].id,expanded:true};
  assert.deepEqual(evidenceSelection(graph,api,selected),{nodeId:'claim:5',edgeId:graph.edges[3].id,expanded:true});
  assert.deepEqual(evidenceSelection(graph,otherApi,selected),{nodeId:'query:4',edgeId:null,expanded:false});
  assert.deepEqual(evidenceSelection(parseQueryEvidence(raw)!,api,selected),{nodeId:'query:4',edgeId:null,expanded:false});
  assert.deepEqual(evidenceSelection(graph,api,{...selected,nodeId:'deleted',edgeId:'missing'}),{nodeId:'query:4',edgeId:null,expanded:true});
});

test('concepts preserve exact mentions, asserted roles and parallel recorded predicates without upgrading mentions to facts',()=>{
  const concepts=[{id:'concept:team',kind:'concept',label:'Team',data:{name:'Team'}},{id:'concept:cedar',kind:'concept',label:'Cedar',data:{name:'Cedar'}}];
  const graph=parseQueryEvidence({...raw,nodes:[...nodes,...concepts],edges:[...edges,
    {source:'chunk:2',target:'concept:cedar',kind:'mentions',data:{quote:'Cedar'}},
    {source:'claim:5',target:'concept:team',kind:'asserts',data:{role:'subject'}},
    {source:'concept:team',target:'concept:cedar',kind:'relation',label:'uses',data:{predicate:'uses',fact_id:5,quote:'We chose Cedar.',source_episode_id:3}},
    {source:'concept:team',target:'concept:cedar',kind:'relation',label:'evaluated',data:{predicate:'evaluated',fact_id:8}},
    {source:'chunk:2',target:'concept:cedar',kind:'relation',data:{predicate:'supports',fact_id:9}},
  ]})!;
  assert.equal(graph.nodes.length,7);
  assert.equal(graph.edges.length,8);
  assert.equal(graph.edges.filter(edge=>edge.kind==='relation').length,2);
  assert.equal(edgeCategory(graph.edges[4]),'Exact text mention');
  assert.equal(edgeCategory(graph.edges[5]),'Claim assertion');
  assert.equal(edgeCategory(graph.edges[6]),'Recorded concept relation');
});
