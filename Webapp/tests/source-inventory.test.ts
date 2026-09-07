import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parseSourcePage,parseRetainedSource} from '../src/memory/source-inventory.ts';

const item={episode_id:8,kind:'file',source:'notes.md',created_at:'2026-09-06',byte_count:900,preview:'Café 🥐\u0000<script>literal</script>',preview_truncated:true};
test('inventory preserves literal source previews and a descending continuation',()=>{
  const result=parseSourcePage({items:[item,{...item,episode_id:5}],has_more:true,next_before:5},{kind:'file',before:10});
  assert.equal(result.items[0].preview,'Café 🥐\u0000<script>literal</script>');
  assert.equal(result.next_before,5);
  assert.deepEqual(parseSourcePage({items:[],has_more:false,next_before:null}),{items:[],has_more:false,next_before:null});
});
test('malformed, looping or cross-filter inventory pages cannot become navigation state',()=>{
  const base={items:[item],has_more:false,next_before:null};
  for(const bad of [null,{}, {...base,has_more:'false'}, {...base,next_before:8},
    {...base,has_more:true,next_before:null}, {...base,has_more:true,next_before:9},
    {...base,items:[item,item]}, {...base,items:[item,{...item,episode_id:9}]},
    ...[0,-1,Number.MAX_SAFE_INTEGER+1].map(episode_id=>({...base,items:[{...item,episode_id}]})),
    ...[{preview:5},{preview:'x'.repeat(501)},{byte_count:-1},{preview_truncated:0},{source:{}},{created_at:null}].map(patch=>({...base,items:[{...item,...patch}]}))]){
    assert.throws(()=>parseSourcePage(bad),/invalid source page/i);
  }
  assert.throws(()=>parseSourcePage(base,{before:8}),/invalid source page/i);
  assert.throws(()=>parseSourcePage(base,{kind:'note'}),/invalid source page/i);
});
test('retained text must belong to the requested episode and is not interpreted as markup',()=>{
  assert.equal(parseRetainedSource({episode_id:8,content:'<b>literal</b>'},8).content,'<b>literal</b>');
  for(const bad of [null,{episode_id:9,content:'wrong'},{episode_id:8,content:null}])assert.throws(()=>parseRetainedSource(bad,8),/source response/i);
});
