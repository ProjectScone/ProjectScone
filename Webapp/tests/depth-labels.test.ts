import {test} from 'node:test';
import assert from 'node:assert/strict';
import {placeDepthLabels} from '../src/playground/depth-labels.ts';

test('Depth labels stay inside a narrow viewport and never overlap each other',()=>{
  const bounds={left:12,top:105,right:348,bottom:480};
  const labels=placeDepthLabels([
    {id:'active',text:'A very long selected source that needs to be shortened',x:335,y:112,radius:4},
    {id:'session:a',text:'Claude Code',x:330,y:115,radius:8},
    {id:'session:b',text:'Codex',x:335,y:118,radius:8},
  ],bounds);
  assert.equal(labels[0].id,'active');
  for(const box of labels){
    assert.ok(box.x>=bounds.left&&box.x+box.width<=bounds.right);
    assert.ok(box.y>=bounds.top&&box.y+box.height<=bounds.bottom);
    assert.ok(box.text.length<60);
  }
  for(let i=0;i<labels.length;i++)for(let j=i+1;j<labels.length;j++){
    const a=labels[i],b=labels[j];
    assert.ok(a.x+a.width<=b.x||b.x+b.width<=a.x||a.y+a.height<=b.y||b.y+b.height<=a.y);
  }
});
