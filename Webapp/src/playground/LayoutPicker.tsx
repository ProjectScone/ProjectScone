import { useEffect, useRef, useState } from 'react';
export type GraphLayout = 'constellation' | 'radial' | 'flow';
const options: {id:GraphLayout;name:string;description:string}[] = [
  {id:'constellation',name:'Constellation',description:'Explore the shape of recorded connections.'},
  {id:'radial',name:'Radial clusters',description:'Gather records around their session.'},
  {id:'flow',name:'Evidence flow',description:'Follow sessions, sources and recall.'},
];
function Preview({mode}:{mode:GraphLayout}) {
  const points=mode==='flow'?[[9,15],[31,15],[53,15],[31,35],[53,35]]:mode==='radial'?[[31,25],[31,5],[53,25],[31,45],[9,25]]:[[25,23],[12,7],[51,12],[48,40],[8,41]];
  return <svg viewBox="0 0 62 50" aria-hidden="true"><g className="layout-preview-links">{points.slice(1).map((p,i)=><path key={i} d={`M${points[0][0]} ${points[0][1]} Q31 25 ${p[0]} ${p[1]}`}/>)}</g>{points.map((p,i)=>mode==='flow'?<rect key={i} x={p[0]-5} y={p[1]-4} width={10} height={8} rx={2}/>:<circle key={i} cx={p[0]} cy={p[1]} r={i?3.5:6}/>)}</svg>;
}
export function LayoutPicker({value,onChange}:{value:GraphLayout;onChange:(value:GraphLayout)=>void}) {
  const [open,setOpen]=useState(false),root=useRef<HTMLDivElement>(null),trigger=useRef<HTMLButtonElement>(null);
  const close=(restore=false)=>{setOpen(false);if(restore)trigger.current?.focus();};
  useEffect(()=>{
    if(!open)return;
    root.current?.querySelector<HTMLButtonElement>('[aria-checked="true"]')?.focus();
    const outside=(event:PointerEvent)=>{if(!root.current?.contains(event.target as Node))setOpen(false);};
    document.addEventListener('pointerdown',outside);return()=>document.removeEventListener('pointerdown',outside);
  },[open]);
  return <div ref={root} className="layout-picker" onBlur={event=>{if(!event.currentTarget.contains(event.relatedTarget as Node))setOpen(false);}}>
    <span className="layout-picker-caption">Layout</span><button ref={trigger} className="layout-trigger" aria-label="Graph layout" aria-haspopup="menu" aria-expanded={open} onClick={()=>setOpen(!open)} onKeyDown={event=>{if(['ArrowDown','ArrowUp'].includes(event.key)){event.preventDefault();setOpen(true);}}}><Preview mode={value}/><span>{options.find(o=>o.id===value)!.name}</span><span aria-hidden="true">⌄</span></button>
    {open&&<div className="layout-menu" role="menu" aria-label="Graph layouts" onKeyDown={event=>{
      if(event.key==='Escape'){event.preventDefault();close(true);}
      if(['ArrowDown','ArrowUp','Home','End'].includes(event.key)){
        event.preventDefault();const buttons=[...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="menuitemradio"]')],index=buttons.indexOf(document.activeElement as HTMLButtonElement);
        buttons[event.key==='Home'?0:event.key==='End'?buttons.length-1:(index+(event.key==='ArrowDown'?1:-1)+buttons.length)%buttons.length]?.focus();
      }
    }}><div className="layout-menu-heading" role="presentation">A different perspective. The same evidence.</div>{options.map(option=><button key={option.id} role="menuitemradio" aria-label={option.name} aria-checked={value===option.id} className="layout-option" onClick={()=>{onChange(option.id);close(true);}}><Preview mode={option.id}/><span><b>{option.name}</b><small>{option.description}</small></span><span className="layout-check" aria-hidden="true">{value===option.id?'✓':''}</span></button>)}</div>}
  </div>;
}
