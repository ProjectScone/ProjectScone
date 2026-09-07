// Merge observers without replacing prompt compilation or granting hook trust.
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),{spawnSync}=require('node:child_process');
const quote=s=>"'"+s.replaceAll("'","'\"'\"'")+"'";
function configure(original,node,script,agent){
  if(!original||typeof original!=='object'||Array.isArray(original))throw Error('Invalid config');
  const next=structuredClone(original);next.hooks??={};
  if(typeof next.hooks!=='object'||Array.isArray(next.hooks))throw Error('Invalid hooks');
  const events=['SessionStart','UserPromptSubmit','PreToolUse','PostToolUse','Stop','SessionEnd'];
  if(agent==='claude-code')events.push('PostToolUseFailure');
  const command=[node,script,agent].map(quote).join(' ');
  for(const event of events){
    const groups=next.hooks[event]??=[];
    if(!Array.isArray(groups)||groups.some(g=>!Array.isArray(g.hooks)))throw Error('Malformed existing hook');
    const background=['PreToolUse','PostToolUse','PostToolUseFailure'].includes(event);
    const existing=groups.flatMap(g=>g.hooks).find(h=>h.type==='command'&&h.command===command);
    if(existing){existing.async=background;existing.timeout=5;continue;}
    groups.push({hooks:[{type:'command',command,timeout:5,async:background}]});
  }
  return next;
}
function main(){
  const script=path.resolve(__dirname,'observe-agent.cjs');
  for(const [agent,file] of [['claude-code',path.join(os.homedir(),'.claude/settings.json')],['codex',path.join(os.homedir(),'.codex/hooks.json')]]){
    const exists=fs.existsSync(file),before=exists?fs.readFileSync(file,'utf8'):'';
    const original=exists?JSON.parse(before):{},updated=configure(original,process.execPath,script,agent);
    if(JSON.stringify(original)===JSON.stringify(updated)){console.log(agent+': already configured');continue;}
    if(!process.argv.includes('--apply')){console.log(agent+': would add scoped capture hooks');continue;}
    const after=JSON.stringify(updated,null,2)+'\n',lines=s=>s.trimEnd().split('\n');
    const delta=exists?'*** Update File: '+file+'\n@@\n'+lines(before).map(s=>'-'+s).join('\n')+'\n':'*** Add File: '+file+'\n';
    const patch='*** Begin Patch\n'+delta+lines(after).map(s=>'+'+s).join('\n')+'\n*** End Patch\n';
    const result=spawnSync('apply_patch',[],{input:patch,encoding:'utf8'});
    if(result.status!==0)throw Error('Could not apply hook configuration');
    console.log(agent+': configured project-scoped capture; existing settings preserved');
  }
  console.log('No trust or permission settings changed. Codex requires review through /hooks.');
}
module.exports={configure};
if(require.main===module){try{main();}catch{process.stderr.write('Capture setup failed; check configuration without printing credentials.\n');process.exitCode=1;}}
