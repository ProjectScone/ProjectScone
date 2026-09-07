// Local capture adapter. No transcript scanning, permission changes or remote sends.
const fs=require('node:fs'),path=require('node:path'),{spawnSync}=require('node:child_process');
const EVENTS=new Set(['SessionStart','UserPromptSubmit','PreToolUse','PostToolUse','PostToolUseFailure','Stop','SessionEnd']);
function prepare(agent,config,payload,root){
  if(!['codex','claude-code'].includes(agent))throw Error('Unknown host');
  const server=new URL(config.server);
  if(server.protocol!=='http:'||!['127.0.0.1','localhost','[::1]'].includes(server.hostname)||server.username||server.password||server.pathname!=='/'||server.search||server.hash)throw Error('Loopback server required');
  if(typeof config.key!=='string'||!config.key||config.space!=='projectscone')throw Error('ProjectScone connection required');
  if(!EVENTS.has(payload.hook_event_name))return null;
  const session=payload.session_id||payload.thread_id;
  if(typeof session!=='string'||!session||typeof payload.cwd!=='string'||!path.isAbsolute(payload.cwd))return null;
  const cwd=path.resolve(payload.cwd),project=path.resolve(root),relative=path.relative(project,cwd);
  let projectRoot=project;
  if(relative==='..'||relative.startsWith('..'+path.sep)||path.isAbsolute(relative)){
    const allowed=(config.session_overrides||[]).some(s=>s.agent===agent&&s.session_id===session&&path.resolve(s.cwd)===cwd);
    if(!allowed)return null;
    projectRoot=cwd;
  }
  const fields=['hook_event_name','session_id','thread_id','cwd','turn_id','prompt_id','tool_use_id','call_id','tool_name','ok','model'];
  const conversation=['UserPromptSubmit','Stop'].includes(payload.hook_event_name);
  if(conversation)fields.push('prompt','user_input','last_assistant_message');
  const selected=Object.fromEntries(fields.filter(k=>payload[k]!==undefined).map(k=>[k,payload[k]]));
  return {payload:selected,projectRoot,feed:conversation?'full':'metadata'};
}
function main(){
  try{
    const root=path.resolve(__dirname,'..');
    const file=path.join(root,'memory/runtime/live-connection.json');
    const info=fs.statSync(file);
    if((info.mode&0o077)!==0)throw Error('Connection file permissions must be private');
    const config=JSON.parse(fs.readFileSync(file,'utf8'));
    const raw=fs.readFileSync(0,'utf8');
    if(Buffer.byteLength(raw)>4*1024*1024)throw Error('Hook payload too large');
    const payload=JSON.parse(raw);
    // Resolve symlinks before applying project boundaries.
    if(typeof payload.cwd==='string')payload.cwd=fs.realpathSync(payload.cwd);
    const agent=process.argv[2],p=prepare(agent,config,payload,fs.realpathSync(root));
    if(!p)return;
    const env={...process.env,SCONE_CAPTURE_KEY:config.key,SCONE_HOOK_COMPILE:'0',SCONE_HOOK_DEBUG:'1'};
    const args=['-m','scone_memory.capture.agent_hook','--agent',agent,'--server',config.server,'--key-env','SCONE_CAPTURE_KEY','--feed',p.feed,'--projects','ProjectScone='+p.projectRoot];
    if(p.feed==='full')args.push('--capture');
    const child=spawnSync(path.join(root,'python/memory/.venv/bin/python'),args,{input:JSON.stringify(p.payload),env,encoding:'utf8',timeout:4500,maxBuffer:65536});
    if(child.error||child.status!==0||child.stderr.includes('agent-hook:'))process.stderr.write('Scone capture: event delivery not confirmed; inspect the local API.\n');
  }catch{process.stderr.write('Scone capture: unavailable or invalid local configuration.\n');}
  finally{process.stdout.write('{}\n');}
}
module.exports={prepare};
if(require.main===module)main();
