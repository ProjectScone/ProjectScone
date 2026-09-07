// Install local-only prompt shaping. Does not grant hook trust or alter permissions.
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),{spawnSync}=require('node:child_process');
function configure(original,binary,host,previousBinary=null){
  if(!original||Array.isArray(original)||typeof original!=='object')throw Error('Configuration must be an object');
  const next=structuredClone(original);next.hooks??={};
  if(!next.hooks||Array.isArray(next.hooks)||typeof next.hooks!=='object')throw Error('Existing hooks configuration is malformed');
  const groups=next.hooks.UserPromptSubmit??=[];
  if(!Array.isArray(groups)||groups.some(g=>!Array.isArray(g.hooks)))throw Error('Existing prompt hooks are malformed');
  const command="'"+binary.replaceAll("'","'\"'\"'")+"' prompt-hook";
  if(previousBinary){const previous="'"+previousBinary.replaceAll("'","'\"'\"'")+"' prompt-hook";for(const group of groups)for(const handler of group.hooks)if(handler.type==='command'&&handler.command===previous)handler.command=command;}
  if(groups.some(g=>g.hooks.some(h=>h.type==='command'&&h.command===command)))return next;
  const handler={type:'command',command,async:false,timeout:3};
  // Compiler bounds output; avoid silently spilling its JSON to an unread file.
  if(host==='codex'){handler.additionalContextLimit=0;handler.statusMessage='Structuring request with Scone';}
  groups.push({hooks:[handler]});return next;
}
function main(){
  const args=process.argv.slice(2),value=name=>{const i=args.indexOf(name);return i<0?null:args[i+1];};
  const binary=path.resolve(value('--binary')||path.join(__dirname,'../target/debug/scone'));
  fs.accessSync(binary,fs.constants.X_OK);
  const targets=[['codex',value('--codex-file')||path.join(os.homedir(),'.codex/hooks.json')],['claude',value('--claude-file')||path.join(os.homedir(),'.claude/settings.json')]];
  const plans=targets.map(([host,target])=>{
    target=path.resolve(target);const exists=fs.existsSync(target),before=exists?fs.readFileSync(target,'utf8'):'';
    const original=exists?JSON.parse(before):{},updated=configure(original,binary,host,value('--replace-binary'));
    return {host,target,exists,before,after:JSON.stringify(updated,null,2)+'\n',changed:JSON.stringify(updated)!==JSON.stringify(original)};
  });
  for(const p of plans){
    if(!p.changed){process.stdout.write(`${p.host}: already configured\n`);continue;}
    if(!args.includes('--apply')){process.stdout.write(`${p.host}: would add one synchronous prompt hook to ${p.target}\n`);continue;}
    // apply_patch receives the diff through stdin; no config values enter logs.
    const lines=text=>text.replace(/\n$/,'').split('\n');
    const patch=p.exists?`*** Update File: ${p.target}\n@@\n${lines(p.before).map(x=>'-'+x).join('\n')}\n${lines(p.after).map(x=>'+'+x).join('\n')}\n`:`*** Add File: ${p.target}\n${lines(p.after).map(x=>'+'+x).join('\n')}\n`;
    const result=spawnSync('apply_patch',[],{input:'*** Begin Patch\n'+patch+'*** End Patch\n',encoding:'utf8'});
    if(result.status!==0)throw Error(`Could not update ${p.host} configuration; no trust or permission settings were changed`);
    process.stdout.write(`${p.host}: prompt hook configured in ${p.target}\n`);
  }
  process.stdout.write('Codex requires review of the new hook through /hooks. Existing sessions may need to reload their hook configuration. No hook trust was granted by this installer.\n');
}
module.exports={configure};
if(require.main===module){try{main();}catch(e){process.stderr.write(`Prompt-hook setup failed: ${e instanceof SyntaxError?'existing configuration is not valid JSON':e.message}\n`);process.exitCode=1;}}
