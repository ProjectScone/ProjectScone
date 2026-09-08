interface LabelCandidate {id:string;text:string;x:number;y:number;radius:number}
interface Bounds {left:number;top:number;right:number;bottom:number}
interface LabelBox {id:string;text:string;x:number;y:number;width:number;height:number;anchorX:number;anchorY:number}

/** Candidates arrive in priority order: active record, then session anchors. */
export function placeDepthLabels(candidates:LabelCandidate[],bounds:Bounds):LabelBox[]{
  const placed:LabelBox[]=[];
  for(const candidate of candidates){
    const limit=Math.max(1,Math.min(30,Math.floor((bounds.right-bounds.left-20)/6.7)));
    const text=candidate.text.length>limit?candidate.text.slice(0,limit-1)+'…':candidate.text;
    const width=Math.min(bounds.right-bounds.left,text.length*6.7+20),height=26,gap=candidate.radius+12;
    const positions=[
      {x:candidate.x+gap,y:candidate.y-height/2},
      {x:candidate.x-gap-width,y:candidate.y-height/2},
      {x:candidate.x-width/2,y:candidate.y-gap-height},
      {x:candidate.x-width/2,y:candidate.y+gap},
    ];
    for(const position of positions){
      const box={id:candidate.id,text,width,height,anchorX:candidate.x,anchorY:candidate.y,
        x:Math.max(bounds.left,Math.min(bounds.right-width,position.x)),
        y:Math.max(bounds.top,Math.min(bounds.bottom-height,position.y))};
      if(placed.some(p=>box.x<p.x+p.width+6&&box.x+width+6>p.x&&box.y<p.y+p.height+6&&box.y+height+6>p.y))continue;
      placed.push(box);break;
    }
  }
  return placed;
}
