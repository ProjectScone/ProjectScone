const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const file=path.join(__dirname,'install-prompt-hooks.cjs');
test('installation preserves unrelated hooks and is idempotent',()=>{
  assert.ok(fs.existsSync(file),'prompt hook installer is not implemented');
  const {configure}=require(file);
  const original={theme:'dark',hooks:{Stop:[{hooks:[{type:'command',command:'existing-stop'}]}],UserPromptSubmit:[{hooks:[{type:'command',command:'existing-prompt'}]}]}};
  const configured=configure(original,"/tmp/a'b/scone",'codex');
  assert.equal(configured.theme,'dark');assert.deepEqual(configured.hooks.Stop,original.hooks.Stop);
  assert.equal(configured.hooks.UserPromptSubmit[0].hooks[0].command,'existing-prompt');
  assert.equal(configured.hooks.UserPromptSubmit[1].hooks[0].async,false);
  assert.equal(configured.hooks.UserPromptSubmit[1].hooks[0].command,"'/tmp/a'\"'\"'b/scone' prompt-hook");
  assert.deepEqual(configure(configured,"/tmp/a'b/scone",'codex'),configured);
  assert.equal(original.hooks.UserPromptSubmit.length,1);
});
test('malformed existing hooks are refused instead of overwritten',()=>{
  assert.ok(fs.existsSync(file),'prompt hook installer is not implemented');
  const {configure}=require(file);
  assert.throws(()=>configure({hooks:{UserPromptSubmit:'broken'}},'/tmp/scone','claude'));
});
test('moving to an installed binary updates only the exact former handler',()=>{
  const {configure}=require(file);
  const original=configure({hooks:{UserPromptSubmit:[{hooks:[{type:'command',command:'other-helper'}]}]}},'/tmp/debug/scone','claude');
  const moved=configure(original,'/tmp/installed/scone','claude','/tmp/debug/scone');
  assert.equal(moved.hooks.UserPromptSubmit.length,2);
  assert.equal(moved.hooks.UserPromptSubmit[0].hooks[0].command,'other-helper');
  assert.equal(moved.hooks.UserPromptSubmit[1].hooks[0].command,"'/tmp/installed/scone' prompt-hook");
});
