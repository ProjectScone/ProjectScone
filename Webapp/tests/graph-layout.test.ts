import { test } from 'node:test';
import assert from 'node:assert/strict';
import { projectGraph, layoutGraph, layoutNetwork, CARD_WIDTH, CARD_HEIGHT } from '../src/playground/graph-layout.ts';
import type { EvidenceNode, EvidenceEdge } from '../src/types.ts';

function crowded() {
  const nodes: EvidenceNode[] = [], edges: EvidenceEdge[] = [];
  for (let s = 0; s < 4; s++) {
    const session = `session:${s}`;
    nodes.push({ id: session, kind: 'session', label: `Session ${s}` });
    for (let i = 0; i < 60; i++) {
      const id = `tool:${s}:${i}`;
      nodes.push({ id, kind: 'tool_call', label: 'Bash', ts: `2026-09-06T00:${String(i).padStart(2, '0')}:00Z` });
      edges.push({ source: session, target: id, kind: 'invoked' });
    }
  }
  return { nodes, edges };
}

test('network layouts are deterministic, distinct, finite and preserve the evidence', () => {
  const raw=crowded(), view=projectGraph(raw.nodes,raw.edges), before=JSON.stringify(view);
  for(const mode of ['constellation','radial'] as const) {
    const points=layoutNetwork(view.nodes,view.edges,mode);
    assert.equal(points.size,view.nodes.length);
    assert.deepEqual([...points], [...layoutNetwork([...view.nodes].reverse(),[...view.edges].reverse(),mode)]);
    assert.ok([...points.values()].every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)));
    const ps=[...points.values()];
    for(let i=0;i<ps.length;i++)for(let j=i+1;j<ps.length;j++)assert.ok(Math.hypot(ps[i].x-ps[j].x,ps[i].y-ps[j].y)>110);
  }
  assert.notDeepEqual([...layoutNetwork(view.nodes,view.edges,'radial')],[...layoutNetwork(view.nodes,view.edges,'constellation')]);
  assert.equal(JSON.stringify(view),before);
  assert.equal(layoutNetwork([],[],'radial').size,0);
});

test('adjacent radial clusters reserve clearance for node circles and labels',()=>{
  const nodes:EvidenceNode[]=[],edges:EvidenceEdge[]=[];
  for(let s=0;s<2;s++){
    nodes.push({id:`s${s}`,kind:'session',label:'Session'});
    for(let i=0;i<11;i++){nodes.push({id:`t${s}-${i}`,kind:'turn',label:'Recorded interaction'});edges.push({source:`s${s}`,target:`t${s}-${i}`,kind:'has'});}
  }
  const view=projectGraph(nodes,edges),points=layoutNetwork(view.nodes,view.edges,'radial');
  for(const depth of [false,true])for(let i=0;i<view.nodes.length;i++)for(let j=i+1;j<view.nodes.length;j++){
    const a=view.nodes[i],b=view.nodes[j],pa=points.get(a.id)!,pb=points.get(b.id)!;
    const dx=pa.x-pb.x+(depth?(a.category-b.category)*20:0);
    const dy=(pa.y-pb.y)*(depth?.75:1)-(depth?(a.category-b.category)*24:0);
    assert.ok(Math.abs(dx)>=165||Math.abs(dy)>=115,'node label envelopes overlap');
  }
});

test('two-node constellation and radial views differ after camera fitting',()=>{
  const view=projectGraph([{id:'s',kind:'session',label:'Session'},{id:'t',kind:'turn',label:'Turn'}],[{source:'s',target:'t',kind:'has'}]);
  const angles=['constellation','radial'].map(mode=>{const p=layoutNetwork(view.nodes,view.edges,mode as 'constellation'|'radial'),a=p.get('s')!,b=p.get('t')!;return Math.atan2(b.y-a.y,b.x-a.x);});
  assert.ok(Math.abs(angles[0]-angles[1])>.3,'fitting must not make both layouts look identical');
});

// Removing grouping brings back the unreadable 240-card column.
test('crowded graph aggregates by session without losing records or inventing links', () => {
  const raw = crowded(), view = projectGraph(raw.nodes, raw.edges);
  assert.equal(view.nodes.length, 8);
  const groups = view.nodes.filter(n => n.members);
  assert.deepEqual(groups.map(g => g.members!.length), [60, 60, 60, 60]);
  assert.equal(new Set(groups.flatMap(g => g.members!)).size, 240);
  assert.equal(view.edges.length, 4);
  assert.ok(view.edges.every(e => e.count === 60 && e.grouped));
  assert.equal(raw.nodes.length, 244);
  const points = [...layoutGraph(view.nodes, false).values()];
  assert.ok(Math.max(...points.map(p => p.y)) < 600);
});

test('group pages expose real records and retain their real incident edges', () => {
  const raw = crowded(), overview = projectGraph(raw.nodes, raw.edges);
  const group = overview.nodes.find(n => n.members)!;
  const first = projectGraph(raw.nodes, raw.edges, { group: group.id, page: 0 });
  const second = projectGraph(raw.nodes, raw.edges, { group: group.id, page: 1 });
  assert.equal(first.total, 60);
  assert.equal(first.pages, 8);
  assert.equal(first.nodes.filter(n => n.kind === 'tool_call').length, 8);
  assert.ok(first.nodes.some(n => n.id === 'tool:0:59'));
  assert.ok(!second.nodes.some(n => n.id === 'tool:0:59'));
  assert.ok(first.edges.every(e => raw.edges.some(r => r.source === e.source && r.target === e.target)));
});

test('small graphs keep exact links; depth changes positions, never evidence', () => {
  const nodes: EvidenceNode[] = [{id:'s',kind:'session',label:'S'}, {id:'t',kind:'turn',label:'Prompt'}, {id:'e',kind:'episode',label:'Evidence'}];
  const edges = [{source:'s',target:'t',kind:'has'}, {source:'t',target:'e',kind:'captured_as'}];
  const view = projectGraph(nodes, edges);
  assert.equal(view.nodes.length, 3);
  assert.deepEqual(view.edges.map(e => [e.source,e.target]), [['s','t'],['t','e']]);
  assert.notDeepEqual([...layoutGraph(view.nodes, false)], [...layoutGraph(view.nodes, true)]);
});

test('focus retains the selected evidence and only its real neighborhood', () => {
  const raw = crowded();
  const view = projectGraph(raw.nodes, raw.edges, { focus: 'tool:1:5' });
  assert.deepEqual(view.nodes.map(n => n.id).sort(), ['session:1','tool:1:5']);
  assert.equal(view.edges.length, 1);
});

test('expanded columns cannot overlap neighboring record types', () => {
  const nodes: EvidenceNode[] = [{id:'s',kind:'session',label:'S'},...Array.from({length:6},(_,i)=>({id:`t${i}`,kind:'turn' as const,label:'Turn'})),{id:'e',kind:'episode',label:'Source'}];
  const edges = nodes.slice(1).map(n=>({source:'s',target:n.id,kind:'has'}));
  const view=projectGraph(nodes,edges);
  for(const depth of [false,true]) {
    const boxes=[...layoutGraph(view.nodes,depth).values()];
    for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++)assert.ok(Math.abs(boxes[i].x-boxes[j].x)>=CARD_WIDTH||Math.abs(boxes[i].y-boxes[j].y)>=CARD_HEIGHT,'record cards overlap');
  }
});

test('high-degree focus pages neighbors instead of shrinking hundreds of chunks', () => {
  const nodes: EvidenceNode[] = [{id:'e',kind:'episode',label:'Document'},...Array.from({length:200},(_,i)=>({id:`c${i}`,kind:'chunk' as const,label:'Chunk'}))];
  const edges = nodes.slice(1).map(n=>({source:'e',target:n.id,kind:'chunked_into'}));
  const first=projectGraph(nodes,edges,{focus:'e',page:0});
  const next=projectGraph(nodes,edges,{focus:'e',page:1});
  assert.equal(first.nodes.length,9);assert.equal(first.pages,25);assert.equal(first.edges.length,8);
  assert.ok(first.nodes.some(n=>n.id==='e'));assert.ok(next.nodes.some(n=>n.id==='e'));
  assert.notDeepEqual(first.nodes.map(n=>n.id),next.nodes.map(n=>n.id));
});

test('many-session overview is paged and every record remains reachable', () => {
  const nodes: EvidenceNode[] = [],edges:EvidenceEdge[]=[];
  for(let i=0;i<30;i++){nodes.push({id:`s${i}`,kind:'session',label:'Session'});for(let j=0;j<6;j++){const id=`t${i}:${j}`;nodes.push({id,kind:'tool_call',label:'Tool'});edges.push({source:`s${i}`,target:id,kind:'invoked'});}}
  const view=projectGraph(nodes,edges);
  assert.equal(view.pages,8);assert.equal(view.nodes.length,8);
  const reached=new Set<string>();
  for(let page=0;page<view.pages;page++)for(const n of projectGraph(nodes,edges,{page}).nodes)for(const id of n.members || [n.id])reached.add(id);
  assert.equal(reached.size,210);
});
