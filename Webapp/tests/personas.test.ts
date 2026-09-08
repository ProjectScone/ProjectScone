import {test} from 'node:test';
import assert from 'node:assert/strict';
import {capabilities,session} from '../src/conversations/contracts.ts';
import {readPersonaCatalog} from '../src/conversations/personas.ts';

const wire={schema_version:1,text_configured:true,reply_transport:'poll',reply_replay:'durable_receipts'};
test('persona discovery requires a valid explicit catalog count',()=>{
  assert.equal(capabilities(wire).personas,0);
  assert.equal(capabilities({...wire,personas:2}).personas,2);
  for(const personas of ['2',null,-1,1.5,true,Number.MAX_SAFE_INTEGER+1])assert.throws(()=>capabilities({...wire,personas}));
});

test('persona catalog validates choices and strips non-display fields',()=>{
  const item={id:'guide',name:'Guide',reply:{provider:'local',model:'atlas',key:'secret'},transcription:{provider:'deepgram',model:'nova'},speech:{provider:'elevenlabs',model:'voice',voice:'warm'},activity:null,text_ready:true,voice_ready:false,instructions:'private'};
  const read=readPersonaCatalog({schema_version:1,personas:[item]});
  assert.equal(read[0].speech?.voice,'warm');assert.equal(read[0].text_ready,true);
  assert.equal(JSON.stringify(read).includes('secret'),false);assert.equal(JSON.stringify(read).includes('private'),false);
  for(const bad of [{schema_version:2,personas:[item]},{schema_version:1,personas:[item,item]},
    ...[{text_ready:'true'},{voice_ready:null},{id:'../bad'},{reply:{provider:'http://evil',model:'x'}},{speech:{provider:'x',model:'y'}}].map(extra=>({schema_version:1,personas:[{...item,...extra}]}))])assert.throws(()=>readPersonaCatalog(bad));
});

test('session persona preserves unknown, default and removed identity distinctly',()=>{
  const saved={session_id:'one',space:'alpha',state:'running',revision:2,created_at:'2026-09-06'};
  assert.equal(session(saved).persona,undefined);
  assert.equal(session({...saved,persona:null}).persona,null);
  assert.deepEqual(session({...saved,persona:{id:'guide',name:'Research guide'}}).persona,{id:'guide',name:'Research guide'});
  assert.deepEqual(session({...saved,persona:{id:'removed',name:null}}).persona,{id:'removed',name:null});
  for(const persona of [{id:'../secret',name:'Bad'},{id:'guide',name:5},{id:'guide'},'guide',[],{id:'guide',name:' '}])assert.throws(()=>session({...saved,persona}));
});

test('catalog fingerprints survive parsing and a versioned catalog cannot silently lose one',()=>{
  const item={id:'guide',name:'Guide',reply:{provider:'local',model:'atlas'},transcription:null,speech:null,activity:null,text_ready:true,voice_ready:false,fingerprint:'0123456789abcdef'};
  const wire={schema_version:1,revision:'abcdef0123456789',personas:[item]};
  assert.equal(readPersonaCatalog(wire)[0].fingerprint,'0123456789abcdef');
  for(const fingerprint of [null,undefined,'nope',true,'ABCDEF0123456789'])assert.throws(()=>readPersonaCatalog({...wire,personas:[{...item,fingerprint}]}));
  for(const revision of [null,42,'bad'])assert.throws(()=>readPersonaCatalog({...wire,revision}));
});

test('session receipt retains historical fingerprint and validates current-state claims',()=>{
  const saved={session_id:'one',space:'alpha',state:'interrupted',revision:3,created_at:'2026-09-06'};
  const persona={id:'guide',name:'Renamed guide',fingerprint:'0123456789abcdef',current:false};
  assert.deepEqual(session({...saved,persona}).persona,persona);
  assert.deepEqual(session({...saved,persona:{...persona,name:null}}).persona,{...persona,name:null});
  for(const extra of [{fingerprint:'bad'},{current:'true'},{fingerprint:undefined},{current:undefined},{fingerprint:null,current:true},{name:null,current:true}])assert.throws(()=>session({...saved,persona:{...persona,...extra}}));
});
