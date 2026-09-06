const {test}=require('node:test');
const assert=require('node:assert/strict');
const {configure}=require('./install-capture-hooks.cjs');
const {prepare}=require('./observe-agent.cjs');
const root='/Users/me/ProjectScone';
const config={server:'http://127.0.0.1:7437',key:'test-only',space:'projectscone'};

test('capture install preserves compiler and permissions and is idempotent',()=>{
  const original={permissions:{deny:['Bash(rm *)']},hooks:{UserPromptSubmit:[{hooks:[{type:'command',command:'scone prompt-hook'}]}]}};
  const next=configure(original,'/usr/bin/node','/repo/observe-agent.cjs','claude-code');
  assert.deepEqual(next.permissions,original.permissions);
  assert.equal(next.hooks.UserPromptSubmit[0].hooks[0].command,'scone prompt-hook');
  assert.equal(next.hooks.UserPromptSubmit.length,2);
  assert.equal(next.hooks.Stop.length,1);
  assert.equal(next.hooks.Stop[0].hooks[0].async,false,'terminal capture must finish before the host tears down');
  assert.equal(next.hooks.PermissionRequest,undefined);
  assert.deepEqual(configure(next,'/usr/bin/node','/repo/observe-agent.cjs','claude-code'),next);
  assert.equal(original.hooks.Stop,undefined);
});
test('observer rejects unrelated roots and allows only an exact external session exception',()=>{
  const prompt={hook_event_name:'UserPromptSubmit',session_id:'s1',cwd:root+'/python',prompt:'Keep this decision'};
  assert.equal(prepare('codex',config,prompt,root).projectRoot,root);
  assert.equal(prepare('codex',config,{...prompt,cwd:root+'-private'},root),null);
  const scoped={...config,session_overrides:[{agent:'claude-code',session_id:'fable',cwd:'/work/examples'}]};
  assert.equal(prepare('claude-code',scoped,{...prompt,cwd:'/work/examples',session_id:'fable'},root).projectRoot,'/work/examples');
  assert.equal(prepare('claude-code',scoped,{...prompt,cwd:'/work/examples'},root),null);
  assert.equal(prepare('codex',scoped,{...prompt,cwd:'/work/examples',session_id:'fable'},root),null);
});
test('observer keeps prompt text but strips tool payloads and cannot send a key remotely',()=>{
  const payload={hook_event_name:'PostToolUse',session_id:'s1',cwd:root,tool_name:'Bash',tool_input:{command:'secret'},tool_response:'private output',transcript_path:'/unrelated/private'};
  const prepared=prepare('codex',config,payload,root);
  assert.equal(prepared.payload.tool_response,undefined);
  assert.equal(prepared.payload.tool_input,undefined);
  assert.equal(prepared.payload.transcript_path,undefined);
  assert.equal(prepared.feed,'metadata');
  assert.throws(()=>prepare('codex',{...config,server:'https://example.com'},payload,root));
  assert.equal(prepare('codex',config,{...payload,hook_event_name:'PermissionRequest'},root),null);
});
