import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { EvidenceEdge, EvidenceNode } from '../types';
import { projectGraph, projectGrowth, layoutGrowth, layoutGraph, layoutNetwork, CARD_WIDTH, CARD_HEIGHT } from './graph-layout';
import { LayoutPicker, type GraphLayout } from './LayoutPicker';

type Point = { x: number; y: number };
const categories = ['Session', 'Interactions', 'Memories', 'Claims & recall'];
const clip = (label: string, length = 25) => label.length > length ? label.slice(0, length - 1) + '…' : label;
function RecordGlyph({ category }: { category: number }) {
  const paths = [
    'M3 5h18v14H3z M7 10l3 3-3 3 M13 16h4',
    'M4 5h16v11H9l-5 4z M8 9h8 M8 12h5',
    'M4 7l8-4 8 4-8 4z M4 12l8 4 8-4 M4 17l8 4 8-4',
    'M12 3l9 9-9 9-9-9z M8 12l3 3 5-6',
  ];
  return <path d={paths[category] || paths[3]} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />;
}

export function GraphCanvas({ nodes, edges, selected, select, depth, layout, setLayout }: { nodes: EvidenceNode[]; edges: EvidenceEdge[]; selected: string | null; select: (id: string) => void; depth: boolean; layout:GraphLayout; setLayout:(value:GraphLayout)=>void }) {
  const svg = useRef<SVGSVGElement>(null);
  const [group, setGroup] = useState<string | undefined>();
  const [page, setPage] = useState(0);
  const [focus, setFocus] = useState<string | undefined>();
  const [hovered, setHovered] = useState<string | null>(null);
  const network = layout !== 'flow';
  const drag = useRef<Point | null>(null);
  const [size, setSize] = useState({ width: 700, height: 450 });
  const [camera, setCamera] = useState({ x: 20, y: 100, zoom: 1 });
  const fitted = useRef(false);
  const view = useMemo(() => layout==='growth'?projectGrowth(nodes,edges,page):projectGraph(nodes, edges, {group,page,focus}), [nodes,edges,group,page,focus,layout]);
  const points = useMemo(() => {
    if(layout==='growth')return layoutGrowth(view.nodes);
    if(layout==='flow')return layoutGraph(view.nodes,depth);
    const result=layoutNetwork(view.nodes,view.edges,layout);
    if(depth)for(const n of view.nodes){const p=result.get(n.id)!;p.x+=n.category*20;p.y=p.y*.75-n.category*24;}
    return result;
  }, [view.nodes,view.edges,depth,layout]);
  const topology = [...points].map(([id,p])=>`${id}:${p.x}:${p.y}`).sort().join('|');
  const active = hovered || selected;
  const lanes = [...new Set(view.nodes.map(n => n.owner))].map((owner, index) => {
    const members = view.nodes.filter(n => n.owner === owner), ps = members.map(n => points.get(n.id)!);
    const anchor = nodes.find(n => n.id === owner);
    return { owner, index, x: Math.min(...ps.map(p=>p.x)) - 18, y: Math.min(...ps.map(p=>p.y)) - 34,
      width: Math.max(...ps.map(p=>p.x)) - Math.min(...ps.map(p=>p.x)) + CARD_WIDTH + 36,
      height: Math.max(...ps.map(p=>p.y)) - Math.min(...ps.map(p=>p.y)) + CARD_HEIGHT + 50,
      label: owner === 'unattributed' ? 'No session attribution' : String(anchor?.data?.project || 'Session context'),
      count: members.reduce((sum,n)=>sum+(n.members?.length || 1),0) };
  });
  const fit = useCallback(() => {
    const ps = [...points.values()]; if (!ps.length) return;
    const left = Math.min(...ps.map(p => p.x)) - 18, top = Math.min(...ps.map(p => p.y)) - 34;
    const width = Math.max(...ps.map(p => p.x)) + CARD_WIDTH + 18 - left, height = Math.max(...ps.map(p => p.y)) + CARD_HEIGHT + 16 - top;
    const zoom = Math.max(.05, Math.min(1.25, (size.width - 64) / width, (size.height - 162) / height));
    setCamera({ zoom, x: (size.width - width * zoom) / 2 - left * zoom, y: 100 + (size.height - 162 - height * zoom) / 2 - top * zoom });
    fitted.current = true;
  }, [points, size]);
  useLayoutEffect(() => {
    const measure=(width:number,height:number)=>{if(width&&height){setSize(previous=>previous.width===width&&previous.height===height?previous:{width,height});fitted.current=false;}};
    const rect=svg.current?.getBoundingClientRect();if(rect)measure(rect.width,rect.height);
    const observer=new ResizeObserver(entries=>{const r=entries[0]?.contentRect;if(r)measure(r.width,r.height);});
    if(svg.current)observer.observe(svg.current);return()=>observer.disconnect();
  }, []);
  useLayoutEffect(() => { fitted.current = false; }, [depth,layout,group,page,focus,topology]);
  useEffect(() => { if (!nodes.length) {setGroup(undefined);setFocus(undefined);setPage(0);} }, [nodes.length]);
  useLayoutEffect(() => { if (!fitted.current) fit(); }, [fit]);
  function zoomBy(factor: number) { setCamera(c => { const zoom = Math.max(.12, Math.min(2.5, c.zoom * factor)); return { zoom, x: size.width / 2 - (size.width / 2 - c.x) * zoom / c.zoom, y: size.height / 2 - (size.height / 2 - c.y) * zoom / c.zoom }; }); }
  return <>
    <div className="graph-navigation" aria-label="Graph scope">
      <LayoutPicker value={layout} onChange={value=>{setLayout(value);setHovered(null);setGroup(undefined);setFocus(undefined);setPage(0);}}/>
      {layout==='growth'?<span>{view.total} dated agent events · {'undated' in view?String(view.undated):0} undated not plotted · 120 per page</span>:view.mode !== 'overview' ? <><button onClick={() => {setGroup(undefined);setFocus(undefined);setPage(0);}}>← Overview</button><span>{view.mode==='focus' ? `${view.total} direct neighbors · paged` : `${view.total} records · newest first`}</span></> : <span>{view.grouped ? 'Grouped by session · select a group to explore' : 'Recorded connections · select a record to inspect'}</span>}
      {view.pages > 1 && <div className="graph-pagination"><button aria-label={view.mode==='group' ? 'Previous group page' : 'Previous graph page'} disabled={view.page === 0} onClick={() => setPage(view.page - 1)}>←</button><span>{view.page + 1} / {view.pages}</span><button aria-label={view.mode==='group' ? 'Next group page' : 'Next graph page'} disabled={view.page + 1 >= view.pages} onClick={() => setPage(view.page + 1)}>→</button></div>}
      {layout!=='growth'&&selected && nodes.some(n => n.id === selected && n.kind !== 'session') && focus !== selected && <button onClick={() => {setFocus(selected);setPage(0);}}>Focus selected</button>}
    </div>
    <svg ref={svg} id="graph" className={network ? 'network-map' : 'flow-map'} data-layout={layout} aria-label="Memory evidence graph" onWheel={e => zoomBy(e.deltaY < 0 ? 1.08 : 1 / 1.08)}
      onPointerDown={e => { if ((e.target as Element).closest('[data-node], [data-group]')) return; drag.current = { x: e.clientX - camera.x, y: e.clientY - camera.y }; e.currentTarget.setPointerCapture(e.pointerId); }}
      onPointerMove={e => { if (drag.current) { const p = drag.current; setCamera(c => ({ ...c, x: e.clientX - p.x, y: e.clientY - p.y })); } }} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
      <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" /></marker></defs>
      <g transform={`translate(${camera.x} ${camera.y}) scale(${camera.zoom})`}>
        {layout==='radial'&&<g className="radial-guides" aria-hidden="true">{lanes.map(lane=>{
          const members=view.nodes.filter(n=>n.owner===lane.owner),anchor=members.find(n=>n.kind==='session');
          if(!anchor||members.length<3)return null;
          const center=points.get(anchor.id)!,radius=Math.max(...members.map(n=>{const p=points.get(n.id)!;return Math.hypot(p.x-center.x,(p.y-center.y)/(depth?.75:1));}));
          return <g key={lane.owner}><ellipse cx={center.x+112} cy={center.y+54} rx={radius} ry={radius*(depth?.75:1)}/><text x={center.x+112} y={center.y+54-radius*(depth?.75:1)-25} textAnchor="middle">{clip(lane.label,24)} · layout guide</text></g>;
        })}</g>}
        {!network && <g className="graph-lanes" aria-hidden="true">{lanes.map(lane=><g key={lane.owner}><rect x={lane.x} y={lane.y} width={lane.width} height={lane.height} rx={16} /><text x={lane.x+14} y={lane.y+19}>{String(lane.index+1).padStart(2,'0')} / {clip(lane.label,26)} · {lane.count} records in view</text></g>)}</g>}
        <g id="edges">{view.edges.map((edge, i) => {
          const a=points.get(edge.source),b=points.get(edge.target);if(!a||!b)return null;
          const forward=b.x>a.x,dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy));
          const radius=(id:string)=>{const n=view.nodes.find(n=>n.id===id);return n?.members ? 36 : n?.kind==='session' ? 29 : layout==='growth'?6:20;};
          const x1=network?a.x+112+dx/d*radius(edge.source):a.x+(forward?CARD_WIDTH:0),x2=network?b.x+112-dx/d*radius(edge.target):b.x+(forward?0:CARD_WIDTH);
          const y1=a.y+54+(network?dy/d*radius(edge.source):0),y2=b.y+54-(network?dy/d*radius(edge.target):0),mid=(x1+x2)/2;
          const bend=network?Math.min(46,d*.12):0,cx=mid-dy/d*bend,cy=(y1+y2)/2+dx/d*bend;
          const highlight=active===edge.source||active===edge.target;
          return <g key={`${edge.source}:${edge.target}:${i}`}><path data-edge={edge.grouped?undefined:i} data-bundle={edge.grouped?edge.count:undefined} data-source={edge.source} data-target={edge.target} className={highlight?'highlight':active?'context-edge':''} style={{strokeWidth:edge.grouped?Math.min(4,1.4+Math.log2(edge.count+1)*.3):1.5}} d={network?`M${x1},${y1} Q${cx},${cy} ${x2},${y2}`:`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`} markerEnd="url(#arrow)"><title>{edge.grouped?`${edge.count} recorded ${edge.kind} links (grouped)`:edge.label||edge.kind}</title></path>{highlight&&network&&<text className="edge-caption" x={(x1+2*cx+x2)/4} y={(y1+2*cy+y2)/4-7} textAnchor="middle">{edge.kind.replaceAll('_',' ')}{edge.grouped?` · ${edge.count}`:''}</text>}</g>;
        })}</g>
        <g id="nodes">{view.nodes.map(node => { const p = points.get(node.id)!; const session = node.kind === 'session'; const agent = node.data?.agent === 'claude-code' ? 'Claude Code' : node.data?.agent === 'codex' ? 'Codex' : node.label; const activate = () => {setHovered(null);if(node.members){setGroup(node.id);setPage(0);setFocus(undefined);}else{select(node.id);}}; return <g key={node.id} className={`atlas-card category-${node.category} ${session ? 'session-anchor' : ''}`} data-node={node.members ? undefined : node.id} data-group={node.members ? node.id : undefined} transform={`translate(${p.x} ${p.y})`} role="button" tabIndex={0} aria-label={`${node.members ? 'Expand' : 'Inspect'} ${node.label}`} aria-pressed={selected === node.id} onMouseEnter={()=>setHovered(node.id)} onMouseLeave={()=>setHovered(null)} onFocus={()=>setHovered(node.id)} onBlur={()=>setHovered(null)} onClick={activate} onKeyDown={e => { if (['Enter', ' '].includes(e.key)) { e.preventDefault(); activate(); } if (['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp'].includes(e.key)) { e.preventDefault(); const list = [...(svg.current?.querySelectorAll<SVGGElement>('[data-node], [data-group]') || [])], index = list.indexOf(e.currentTarget), delta = ['ArrowRight', 'ArrowDown'].includes(e.key) ? 1 : -1; list[(index + delta + list.length) % list.length]?.focus(); } }}>
          <title>{node.label}{node.members ? ' — presentation group, not an evidence record' : ''}</title>
          {layout==='growth'&&!session?<>
            <circle className={`growth-event ${['session_start','session_end'].includes(String(node.data?.event))?'lifecycle-boundary':''}`} cx={112} cy={54} r={active===node.id?10:['session_start','session_end'].includes(String(node.data?.event))?9:6}/>
            {active===node.id&&<><text className="growth-label" x={128} y={51}>{clip(node.label,38)}</text><text className="growth-time" x={128} y={69}>{node.ts}</text></>}
          </>:network ? <>
            <circle className="node-halo" cx={112} cy={54} r={node.members?44:session?37:28} />
            <circle className="orb-surface" cx={112} cy={54} r={node.members?36:session?29:20} />
            {node.members?<><text className="orb-count" x={112} y={60} textAnchor="middle">{node.members.length}</text><circle className="expand-disc" cx={139} cy={28} r={9}/><path className="expand-plus" d="M135 28h8m-4-4v8"/></>:<g className="orb-glyph" transform="translate(100 42)"><RecordGlyph category={node.category}/></g>}
            <text className="orb-label" x={112} y={node.members?111:101} textAnchor="middle">{clip(String(node.members?categories[node.category]:session?agent:node.label),23)}</text>
            <text className="orb-subtitle" x={112} y={node.members?127:117} textAnchor="middle">{node.members?'Expand recorded items':session?clip(String(node.data?.session_id||node.id),24):node.kind.replaceAll('_',' ')}</text>
          </> : <><rect className="card-surface" width={CARD_WIDTH} height={CARD_HEIGHT} rx={12} />
          <g className="card-glyph" transform="translate(16 14)"><RecordGlyph category={node.category} /></g>
          <text className="kind" x={48} y={30}>{session ? 'AGENT SESSION' : node.members ? categories[node.category].toUpperCase() : node.kind.replaceAll('_', ' ').toUpperCase()}</text>
          {node.members ? <><text className="group-count" x={18} y={72}>{node.members.length.toLocaleString()}</text><text className="card-subtitle" x={18} y={92}>recorded {node.members.length === 1 ? 'item' : 'items'}</text><path className="card-open" d="M186 81h20m-7-7 7 7-7 7" /></> : <><text className="card-label" x={18} y={65}>{clip(String(session ? agent : node.label),24)}</text><text className="card-subtitle" x={18} y={90}>{session ? clip(String(node.data?.session_id || node.id),28) : node.ts ? new Date(node.ts).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : clip(node.id,30)}</text></>}</>}
        </g>; })}</g>
      </g>
    </svg>
    {(!nodes.length||layout==='growth'&&!view.nodes.length) && <div id="empty"><div className="empty-mark" aria-hidden>◇</div><h2>{layout==='growth'?'No dated agent events in this snapshot':'No recorded interactions yet'}</h2><p>{layout==='growth'?'This view needs timestamped agent activity. Missing dates and session boundaries are not inferred.':'Connect an explicitly selected Claude Code or Codex session. Its captured activity will appear here.'}</p></div>}
    {layout==='growth'&&<div className="growth-explanation">Golden-angle sequence within this snapshot · outward means later, not elapsed time.<br/>A larger event mark identifies a captured session start or end. Silence is not an end event.</div>}
    <div className="graph-bottom"><div className="atlas-legend"><span className="legend-session">Sessions</span><span className="legend-interaction">Interactions</span><span className="legend-memory">Memory</span><small>{view.grouped ? 'Dashed links bundle stored references' : 'Only recorded relationships'}</small></div><div className="zoom"><button aria-label="Zoom out" onClick={() => zoomBy(1 / 1.2)}>−</button><output>{Math.round(camera.zoom * 100)}%</output><button aria-label="Zoom in" onClick={() => zoomBy(1.2)}>+</button><button aria-label="Fit graph" onClick={fit}>Fit</button></div></div>
  </>;
}
