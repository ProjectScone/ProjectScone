// Compatibility entry point. Webapp is the only playground source of truth.
const {spawnSync}=require('node:child_process');
const path=require('node:path');
const result=spawnSync('npm',['run',process.argv.includes('--check')?'check:assets':'build'],{
  cwd:path.resolve(__dirname,'../Webapp'),stdio:'inherit'
});
process.exitCode=result.status??1;
