import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { EvidenceNode, EvidenceEdge } from '../types';
import { LayoutPicker, type GraphLayout } from './LayoutPicker';
import { layoutDepth, nodeCategory, projectPoint, rotatePoint, type Orbit } from './depth-layout';
import { owners } from './graph-layout';
import { depthPage, DEPTH_PAGE_SIZE } from './depth-page';

const initialOrbit:Orbit={yaw:-.35,pitch:.22};
type Props={nodes:EvidenceNode[];edges:EvidenceEdge[];selected:string|null;select:(id:string)=>void;layout:GraphLayout;setLayout:(layout:GraphLayout)=>void};

export function DepthCanvas({nodes,edges,selected,select,layout,setLayout}:Props){
  const [page,setPage]=useState(0);
  const view=useMemo(()=>depthPage(nodes,edges,page),[nodes,edges,page]);
  const ownership=useMemo(()=>owners(nodes,edges),[nodes,edges]);
  useEffect(()=>{
    if(!selected)return;
    const index=[...nodes].sort((a,b)=>a.id.localeCompare(b.id)).findIndex(n=>n.id===selected);
    if(index>=0)setPage(Math.floor(index/DEPTH_PAGE_SIZE));
  },[selected]);
  return <DepthScene nodes={view.nodes} edges={view.edges} selected={selected} select={select} layout={layout} setLayout={setLayout} view={view} setPage={setPage} ownership={ownership}/>;
}

function DepthScene({nodes,edges,selected,select,layout,setLayout,view,setPage,ownership}:Props&{view:ReturnType<typeof depthPage>;setPage:(page:number)=>void;ownership:Map<string,string>}){
  const svg=useRef<SVGSVGElement>(null);
  const drag=useRef<{x:number;y:number;pan:boolean}|null>(null);
  const motion=useRef({yaw:0,pitch:0,x:0,y:0});
  const frame=useRef<number|null>(null);
  const [size,setSize]=useState({width:700,height:600});
  const [orbit,setOrbit]=useState<Orbit>(initialOrbit);
  const [zoom,setZoom]=useState(1);
  const [pan,setPan]=useState({x:0,y:0});
  const [hover,setHover]=useState<string|null>(null);
  const [focused,setFocused]=useState<string|null>(null);
  const known=useMemo(()=>new Set(nodes.map(n=>n.id)),[nodes]);
  const active=[focused,hover,selected].find(id=>id&&known.has(id))||null;
  const mode=layout==='growth'?'constellation':layout;
  // Ignore content-only refreshes: identical topology must not restart layout.
  const topology=useMemo(()=>JSON.stringify([nodes.map(n=>[n.id,n.kind]).sort(),edges.map(e=>[e.source,e.target,e.kind]).sort()]),[nodes,edges]);
  const ownershipKey=JSON.stringify(nodes.map(n=>[n.id,ownership.get(n.id)]).sort());
  const world=useMemo(()=>layoutDepth(nodes,edges,mode,ownership),[topology,mode,ownershipKey]);
  const unassigned=nodes.filter(n=>n.kind!=='session'&&!ownership.has(n.id)).length;
  const reset=()=>{motion.current={yaw:0,pitch:0,x:0,y:0};setOrbit(initialOrbit);setZoom(1);setPan({x:0,y:0});};
  const rotate=(yaw:number,pitch:number)=>setOrbit(o=>({yaw:o.yaw+yaw,pitch:Math.max(-1.45,Math.min(1.45,o.pitch+pitch))}));
  const zoomBy=(factor:number)=>setZoom(z=>Math.max(.35,Math.min(3,z*factor)));
  useEffect(()=>()=>{if(frame.current!==null)cancelAnimationFrame(frame.current);},[]);

  useLayoutEffect(()=>{
    const canvas=svg.current;if(!canvas)return;
    const measure=()=>{const r=canvas.getBoundingClientRect();if(r.width&&r.height)setSize({width:r.width,height:r.height});};
    measure();const observer=new ResizeObserver(measure);observer.observe(canvas);
    return()=>observer.disconnect();
  },[]);
  useEffect(()=>{
    const canvas=svg.current;if(!canvas)return;
    const wheel=(event:WheelEvent)=>{
      if(!event.ctrlKey||!event.deltaY)return;
      event.preventDefault();event.stopPropagation();
      const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?size.height:1);
      zoomBy(Math.exp(-Math.max(-100,Math.min(100,delta))*.008));
    };
    canvas.addEventListener('wheel',wheel,{passive:false});
    return()=>canvas.removeEventListener('wheel',wheel);
  },[size.height]);

  const scale=Math.max(.1,Math.min(size.width-70,size.height-210)/880)*zoom;
  const center={x:size.width/2+pan.x,y:120+(size.height-210)/2+pan.y};
  const projected=useMemo(()=>new Map([...world].flatMap(([id,point])=>{
    const p=projectPoint(rotatePoint(point,orbit),1100);
    return p?[[id,{...p,x:center.x+p.x*scale,y:center.y+p.y*scale}] as const]:[];
  })),[world,orbit,scale,center.x,center.y]);
  const sorted=useMemo(()=>[...nodes].filter(n=>projected.has(n.id)).sort((a,b)=>projected.get(a.id)!.z-projected.get(b.id)!.z||a.id.localeCompare(b.id)),[nodes,projected]);
  const radiusFor=(node:EvidenceNode,depthScale:number)=>(node.kind==='session'?12:nodes.length>100?3.8:7)*depthScale*Math.sqrt(zoom);
  const nodeById=useMemo(()=>new Map(nodes.map(n=>[n.id,n])),[nodes]);
  const adjacent=new Set<string>();
  for(const edge of edges){if(edge.source===active)adjacent.add(edge.target);if(edge.target===active)adjacent.add(edge.source);}
  const axisOrigin={x:size.width-58,y:size.height-120};

  return <>
    <div className="graph-navigation depth-navigation" aria-label="Graph scope">
      <LayoutPicker value={layout} onChange={setLayout}/>
      <span>{view.pages>1?`${view.page*DEPTH_PAGE_SIZE+1}–${view.page*DEPTH_PAGE_SIZE+nodes.length} of ${view.total} records`:`${nodes.length} individual records`} · {edges.length} stored links</span>
      {view.pages>1&&<div className="graph-pagination"><button aria-label="Previous 3D page" disabled={!view.page} onClick={()=>setPage(view.page-1)}>←</button><span>{view.page+1} / {view.pages}</span><button aria-label="Next 3D page" disabled={view.page+1===view.pages} onClick={()=>setPage(view.page+1)}>→</button></div>}
    </div>
    <svg ref={svg} id="graph" className="depth-map" data-layout={layout} data-dimensions="3" aria-label="Memory evidence graph" tabIndex={0}
      aria-describedby="depth-help"
      onKeyDown={event=>{
        if(event.target!==event.currentTarget)return;
        if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','0'].includes(event.key))event.preventDefault();
        if(event.key==='ArrowLeft')rotate(-.12,0);if(event.key==='ArrowRight')rotate(.12,0);
        if(event.key==='ArrowUp')rotate(0,-.12);if(event.key==='ArrowDown')rotate(0,.12);
        if(['+','='].includes(event.key))zoomBy(1.2);if(event.key==='-')zoomBy(1/1.2);if(event.key==='0')reset();
      }}
      onPointerDown={event=>{
        if((event.target as Element).closest('[data-node]'))return;
        if(event.button!==0&&event.button!==1)return;
        event.preventDefault();event.currentTarget.focus();
        drag.current={x:event.clientX,y:event.clientY,pan:event.shiftKey||event.button===1};
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={event=>{
        const previous=drag.current;if(!previous)return;
        const dx=event.clientX-previous.x,dy=event.clientY-previous.y;
        if(previous.pan){motion.current.x+=dx;motion.current.y+=dy;}else{motion.current.yaw+=dx*.007;motion.current.pitch+=dy*.007;}
        if(frame.current===null)frame.current=requestAnimationFrame(()=>{
          const delta=motion.current;motion.current={yaw:0,pitch:0,x:0,y:0};frame.current=null;
          rotate(delta.yaw,delta.pitch);setPan(p=>({x:p.x+delta.x,y:p.y+delta.y}));
        });
        drag.current={...previous,x:event.clientX,y:event.clientY};
      }}
      onPointerUp={()=>{drag.current=null;}} onPointerCancel={()=>{drag.current=null;}} onLostPointerCapture={()=>{drag.current=null;}}>
      <defs><marker id="depth-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="context-stroke"/></marker><radialGradient id="depth-node-light" cx="30%" cy="25%"><stop offset="0" stopColor="white" stopOpacity=".65"/><stop offset=".45" stopColor="white" stopOpacity=".05"/><stop offset="1" stopColor="black" stopOpacity=".22"/></radialGradient></defs>
      <g className="depth-volume" aria-hidden="true">{[-260,0,260].map(z=>{
        const corners=[[-300,-300],[300,-300],[300,300],[-300,300]].map(([x,y])=>projectPoint(rotatePoint({x,y,z},orbit),1100)!);
        return <polygon key={z} points={corners.map(p=>`${center.x+p.x*scale},${center.y+p.y*scale}`).join(' ')}/>;
      })}</g>
      <g id="edges">{edges.map((edge,index)=>{
        const a=projected.get(edge.source),b=projected.get(edge.target);if(!a||!b)return null;
        const lit=edge.source===active||edge.target===active;
        const dx=b.x-a.x,dy=b.y-a.y,length=Math.max(1,Math.hypot(dx,dy));
        const inset=Math.min(length*.45,radiusFor(nodeById.get(edge.target)!,b.scale)+2);
        return <g key={`${edge.source}:${edge.target}:${edge.kind}:${index}`}><line data-edge={index} data-source={edge.source} data-target={edge.target}
          x1={a.x} y1={a.y} x2={b.x-dx/length*inset} y2={b.y-dy/length*inset} markerEnd={lit?'url(#depth-arrow)':undefined} className={lit?'highlight':active?'context-edge':''}>
          <title>{`${edge.source} → ${edge.target}: ${edge.label||edge.kind}`}</title>
        </line>{lit&&<text className="depth-edge-caption" x={(a.x+b.x)/2} y={(a.y+b.y)/2-6} textAnchor="middle">{edge.kind.replaceAll('_',' ')}</text>}</g>;
      })}</g>
      <g id="nodes">{sorted.map(node=>{
        const p=projected.get(node.id)!,session=node.kind==='session',lit=node.id===active;
        const radius=radiusFor(node,p.scale);
        const label=session?String(node.data?.agent==='claude-code'?'Claude Code':node.data?.agent==='codex'?'Codex':node.label):node.label;
        const labelLeft=p.x>size.width*.55;
        const room=(labelLeft?p.x:size.width-p.x)-radius-20;
        const labelLimit=Math.max(5,Math.min(34,Math.floor(room/7)));
        const displayLabel=label.length>labelLimit?label.slice(0,labelLimit-1)+'…':label;
        const labeled=session||lit||nodes.length<=24;
        return <g key={node.id} data-node={node.id} data-z={world.get(node.id)!.z} data-camera-z={p.z} className={`depth-node category-${nodeCategory[node.kind]} ${lit?'is-active':''} ${active&&!lit&&!adjacent.has(node.id)?'is-dimmed':''}`}
          transform={`translate(${p.x} ${p.y})`} role="button" tabIndex={0} aria-label={`Inspect ${node.label}`} aria-pressed={selected===node.id}
          onMouseEnter={()=>setHover(node.id)} onMouseLeave={()=>setHover(null)} onFocus={()=>setFocused(node.id)} onBlur={()=>setFocused(null)} onClick={()=>select(node.id)}
          onKeyDown={event=>{
            if(['Enter',' '].includes(event.key)){event.preventDefault();select(node.id);}
            if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){
              event.preventDefault();const list=[...(svg.current?.querySelectorAll<SVGGElement>('[data-node]')||[])],index=list.indexOf(event.currentTarget);
              list[(index+(['ArrowLeft','ArrowUp'].includes(event.key)?-1:1)+list.length)%list.length]?.focus();
            }
          }}>
          <title>{`${node.label} · ${node.id}${!ownership.has(node.id)?' · No session path in this snapshot':''}`}</title>
          {labeled?<rect className="depth-hit" x={labelLeft?-radius-7-displayLabel.length*7:-radius-5} y={-radius-5} width={radius*2+12+displayLabel.length*7} height={radius*2+10}/>:<circle className="depth-hit" r={Math.max(7,radius+3)}/>}
          {(session||lit)&&<circle className="depth-halo" r={radius+5}/>}
          <circle className="depth-sphere" r={radius}/><circle r={radius} fill="url(#depth-node-light)" pointerEvents="none"/>
          {labeled&&<text className="depth-label" textAnchor={labelLeft?'end':'start'} x={labelLeft?-radius-7:radius+7} y={4}>{displayLabel}</text>}
        </g>;
      })}</g>
      <g className="depth-axes" aria-label="Camera orientation, spatial axes">{[{name:'X',point:{x:26,y:0,z:0}},{name:'Y',point:{x:0,y:26,z:0}},{name:'Z',point:{x:0,y:0,z:26}}].map(axis=>{
        const p=rotatePoint(axis.point,orbit);return <g key={axis.name}><line x1={axisOrigin.x} y1={axisOrigin.y} x2={axisOrigin.x+p.x} y2={axisOrigin.y+p.y}/><text x={axisOrigin.x+p.x*1.3} y={axisOrigin.y+p.y*1.3}>{axis.name}</text></g>;
      })}</g>
    </svg>
    {!nodes.length&&<div id="empty"><h2>No recorded interactions yet</h2><p>Connect an agent session to explore its evidence in three dimensions.</p></div>}
    <div className="depth-help" id="depth-help">Drag to orbit · Shift-drag to pan · Pinch / Ctrl-scroll to zoom · Arrow keys rotate the focused canvas
      {unassigned>0&&<span>{unassigned} records have no session path in this snapshot. Missing references are not invented.</span>}
      {view.omittedLinks>0&&<span>{view.omittedLinks} links are outside this page or its 4,000-link drawing limit. Inspect a record for all its loaded connections.</span>}
    </div>
    <div className="graph-bottom"><div className="atlas-legend"><span className="legend-session">Sessions</span><span className="legend-interaction">Interactions</span><span className="legend-memory">Memory</span><span className="legend-claim">Claims / recall</span></div>
      <div className="zoom"><button aria-label="Zoom out" onClick={()=>zoomBy(1/1.2)}>−</button><output>{Math.round(zoom*100)}%</output><button aria-label="Zoom in" onClick={()=>zoomBy(1.2)}>+</button><button aria-label="Fit graph" onClick={reset}>Fit</button><button aria-label="Reset 3D camera" onClick={reset}>Reset</button></div>
    </div>
  </>;
}
