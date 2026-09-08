import {useEffect,useLayoutEffect,useMemo,useRef,useState,type KeyboardEvent} from 'react';
import {evidenceEdgeLabel,type QueryEvidenceNode,type QueryEvidenceEdge} from './query-evidence';
import {fitEvidenceCamera,layoutEvidenceNetwork,placeEvidenceLabels,zoomEvidenceCamera,type EvidenceCamera,type EvidenceGroups,type EvidencePoint} from './evidence-network';

export const GROUP_COLORS=['#3979ca','#c5772d','#8d65b7','#26917d','#c25b79','#77923a','#546bb2','#9c704f'];
const KIND_COLORS={query:'#335987',chunk:'#628baf',episode:'#32907a',claim:'#8b62b1',concept:'#4389c9'};

export function EvidenceNetworkCanvas({nodes,edges,groups,groupColors,selectedNode,selectedEdge,selectNode,selectEdge,reset,labels=true}:{
  nodes:QueryEvidenceNode[];edges:QueryEvidenceEdge[];groups?:EvidenceGroups;groupColors:boolean;selectedNode:string|null;selectedEdge:string|null;
  selectNode:(id:string)=>void;selectEdge:(id:string)=>void;reset:()=>void;labels?:boolean;
}){
  const svg=useRef<SVGSVGElement>(null),drag=useRef<{x:number;y:number;camera:EvidenceCamera}|null>(null);
  const [size,setSize]=useState({width:700,height:500}),[camera,setCamera]=useState<EvidenceCamera>({x:0,y:0,zoom:1});
  const [hovered,setHovered]=useState<string|null>(null);
  const points=useMemo(()=>layoutEvidenceNetwork(nodes,edges,groups),[nodes,edges,groups]);
  const colors=useMemo(()=>{
    const result=new Map<string,string>();
    groups?.groups.forEach((group,index)=>group.nodeIds.forEach(id=>result.set(id,group.isolated?'#8995a7':GROUP_COLORS[index%GROUP_COLORS.length])));
    return result;
  },[groups]);
  const degrees=useMemo(()=>{const result=new Map<string,number>();for(const edge of edges){result.set(edge.source,(result.get(edge.source)??0)+1);result.set(edge.target,(result.get(edge.target)??0)+1);}return result;},[edges]);
  useLayoutEffect(()=>{
    if(!svg.current)return;
    const measure=()=>{const rect=svg.current!.getBoundingClientRect();if(rect.width&&rect.height)setSize(previous=>previous.width===rect.width&&previous.height===rect.height?previous:{width:rect.width,height:rect.height});};
    measure();const observer=new ResizeObserver(measure);observer.observe(svg.current);return()=>observer.disconnect();
  },[]);
  useLayoutEffect(()=>setCamera(fitEvidenceCamera(points,size)),[points,size]);
  useEffect(()=>{
    const element=svg.current;if(!element)return;
    const wheel=(event:WheelEvent)=>{event.preventDefault();const rect=element.getBoundingClientRect();setCamera(previous=>zoomEvidenceCamera(previous,Math.exp(-Math.max(-100,Math.min(100,event.deltaY))*.008),{x:event.clientX-rect.left,y:event.clientY-rect.top}));};
    element.addEventListener('wheel',wheel,{passive:false});return()=>element.removeEventListener('wheel',wheel);
  },[]);
  const zoom=(factor:number)=>setCamera(previous=>zoomEvidenceCamera(previous,factor,{x:size.width/2,y:size.height/2}));
  const active=hovered??selectedNode;
  const neighbors=new Set(edges.filter(edge=>edge.source===active||edge.target===active||edge.id===selectedEdge).flatMap(edge=>[edge.source,edge.target]));
  const placed=useMemo(()=>placeEvidenceLabels(nodes,points,camera,size,active),[nodes,points,camera,size,active]);
  const nodeRadius=(id:string)=>Math.max(6/camera.zoom,Math.min(12,6+Math.sqrt(degrees.get(id)??0)));
  const color=(node:QueryEvidenceNode)=>groupColors&&node.kind==='concept'?colors.get(node.id)??KIND_COLORS.concept:KIND_COLORS[node.kind];
  const paths=useMemo(()=>{
    const pairs=new Map<string,QueryEvidenceEdge[]>();for(const edge of edges){const key=[edge.source,edge.target].sort().join('|');pairs.set(key,[...(pairs.get(key)??[]),edge]);}
    return edges.map(edge=>{
      const a=points.get(edge.source)!,b=points.get(edge.target)!,distance=Math.hypot(b.x-a.x,b.y-a.y);
      if(distance<.01)return {edge,path:`M${a.x-5},${a.y-8} C${a.x-45},${a.y-70} ${a.x+45},${a.y-70} ${a.x+5},${a.y-8}`};
      const nx=(b.x-a.x)/distance,ny=(b.y-a.y)/distance;
      const family=pairs.get([edge.source,edge.target].sort().join('|'))!,index=family.findIndex(item=>item.id===edge.id),bend=(index-(family.length-1)/2)*27;
      const start={x:a.x+nx*nodeRadius(edge.source),y:a.y+ny*nodeRadius(edge.source)},end={x:b.x-nx*(nodeRadius(edge.target)+3/camera.zoom),y:b.y-ny*(nodeRadius(edge.target)+3/camera.zoom)};
      return {edge,path:`M${start.x},${start.y} Q${(a.x+b.x)/2-ny*bend},${(a.y+b.y)/2+nx*bend} ${end.x},${end.y}`};
    });
  },[edges,points,camera.zoom,degrees]);
  function moveFocus(event:KeyboardEvent<SVGGElement>,index:number){
    if(event.key==='Enter'||event.key===' '){event.preventDefault();selectNode(nodes[index].id);return;}
    if(!['ArrowRight','ArrowDown','ArrowLeft','ArrowUp'].includes(event.key))return;
    event.preventDefault();const next=(index+(['ArrowRight','ArrowDown'].includes(event.key)?1:-1)+nodes.length)%nodes.length;
    svg.current?.querySelectorAll<SVGGElement>('[data-evidence-node]')[next]?.focus();selectNode(nodes[next].id);
  }
  const anchor:EvidencePoint={x:size.width/2,y:size.height/2};
  return <div className="en-canvas">
    <svg ref={svg} className="en-svg" role="group" aria-label="Interactive evidence network" tabIndex={0}
      onKeyDown={event=>{if(event.target!==event.currentTarget)return;if(['+','=','-','0'].includes(event.key)){event.preventDefault();if(event.key==='0')setCamera(fitEvidenceCamera(points,size));else setCamera(previous=>zoomEvidenceCamera(previous,event.key==='-'?.8:1.25,anchor));}}}
      onPointerDown={event=>{if((event.target as Element).closest('[data-evidence-node],[data-evidence-edge]'))return;drag.current={x:event.clientX,y:event.clientY,camera};event.currentTarget.setPointerCapture(event.pointerId);}}
      onPointerMove={event=>{if(drag.current)setCamera({...drag.current.camera,x:drag.current.camera.x+event.clientX-drag.current.x,y:drag.current.camera.y+event.clientY-drag.current.y});}}
      onPointerUp={event=>{drag.current=null;if(event.currentTarget.hasPointerCapture(event.pointerId))event.currentTarget.releasePointerCapture(event.pointerId);}}
      onPointerCancel={()=>{drag.current=null;}}>
      <defs><marker id="en-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L8 4L0 8Z" fill="#8294ad"/></marker><marker id="en-arrow-active" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L8 4L0 8Z" fill="#315fd0"/></marker></defs>
      <g transform={`translate(${camera.x} ${camera.y}) scale(${camera.zoom})`}>
        {paths.map(({edge,path})=>{
          const chosen=selectedEdge===edge.id,incident=edge.source===active||edge.target===active;
          return <g key={edge.id} data-evidence-edge={edge.id} className="en-edge" data-selected={chosen} data-muted={!!active&&!incident&&!chosen}>
            <path d={path} className="en-edge-line" stroke={chosen?'#315fd0':edge.kind==='relation'&&groupColors?colors.get(edge.source)??'#8294ad':'#8294ad'} strokeWidth={(chosen?2.5:incident?1.7:1)/camera.zoom} strokeDasharray={['returned','chunked_into','mentions','asserts'].includes(edge.kind)?`${4/camera.zoom} ${4/camera.zoom}`:undefined} markerEnd={chosen?'url(#en-arrow-active)':'url(#en-arrow)'}/>
            <path d={path} className="en-edge-hit" strokeWidth={13/camera.zoom} role="button" tabIndex={-1} aria-label={`${evidenceEdgeLabel(edge)}: ${nodes.find(node=>node.id===edge.source)?.label} → ${nodes.find(node=>node.id===edge.target)?.label}`} onClick={event=>{event.stopPropagation();selectEdge(edge.id);}}><title>{evidenceEdgeLabel(edge)}</title></path>
          </g>;
        })}
        {nodes.map((node,index)=>{
          const point=points.get(node.id)!,radius=nodeRadius(node.id),selected=node.id===selectedNode;
          return <g key={node.id} transform={`translate(${point.x} ${point.y})`} className="en-node" data-evidence-node={node.id} data-selected={selected} data-muted={!!active&&active!==node.id&&!neighbors.has(node.id)} role="button" aria-label={`${node.kind}: ${node.label}`} aria-pressed={selected} tabIndex={selected||!selectedNode&&index===0?0:-1} onKeyDown={event=>moveFocus(event,index)} onClick={event=>{event.stopPropagation();selectNode(node.id);}} onPointerEnter={()=>setHovered(node.id)} onPointerLeave={()=>setHovered(null)}>
            <circle r={Math.max(radius+6/camera.zoom,12/camera.zoom)} fill="transparent"/>
            {selected&&<circle r={radius+5/camera.zoom} fill="none" stroke="#315fd0" strokeWidth={1.5/camera.zoom}/>}
            <circle r={radius} fill={color(node)} stroke="white" strokeWidth={1.8/camera.zoom}/><title>{node.label}</title>
          </g>;
        })}
      </g>
      {(labels?placed:placed.filter(label=>label.id===active)).map(label=><g key={label.id} className="en-node-label" data-active={label.id===active} pointerEvents="none"><rect x={label.x} y={label.y} width={label.width} height={label.height} rx="5"/><text x={label.x+6} y={label.y+15}>{label.text}</text></g>)}
    </svg>
    {!nodes.length&&<div className="en-empty"><strong>No records in this view</strong><p>Try the evidence view or a different group. Missing links are not inferred.</p></div>}
    <div className="en-canvas-note">{nodes.length} nodes · {edges.length} recorded links<span>Drag to move · scroll to zoom</span></div>
    <div className="en-camera" aria-label="Graph camera"><button type="button" onClick={()=>zoom(.8)} aria-label="Zoom out">−</button><output aria-label="Graph zoom">{Math.round(camera.zoom*100)}%</output><button type="button" onClick={()=>zoom(1.25)} aria-label="Zoom in">+</button><button type="button" onClick={()=>setCamera(fitEvidenceCamera(points,size))}>Fit graph</button><button type="button" onClick={()=>{reset();setCamera(fitEvidenceCamera(points,size));}}>Reset</button></div>
  </div>;
}
