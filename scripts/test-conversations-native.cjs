// Real React → HTTP service → Pipecat scheduler → native memory, no live provider.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const {once}=require('node:events');
const path=require('node:path');
const {chromium}=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'..');

test('conversation deep links, public capture and sources work through native Pipecat',{timeout:60000},async t=>{
  const html=process.env.SCONE_CONVERSATIONS_HTML;
  assert.ok(html,'SCONE_CONVERSATIONS_HTML must name the verified isolated webapp artifact');
  const server=spawn(process.env.SCONE_TEST_PYTHON||path.join(root,'python/scone-memory/.venv/bin/python'),
    ['-u',path.join(__dirname,'fixtures/conversation-server.py'),html],{cwd:root,stdio:['ignore','pipe','pipe']});
  let logs='';server.stderr.on('data',part=>{logs=(logs+part).slice(-6000);});
  t.after(async()=>{
    if(server.exitCode!==null||server.signalCode!==null)return;
    const exited=once(server,'exit');server.kill('SIGTERM');
    const timeout=setTimeout(()=>server.kill('SIGKILL'),5000);
    try{await exited;}finally{clearTimeout(timeout);}
  });
  const port=await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(Error('Fixture startup timed out: '+logs)),20000);
    let data='';
    server.stdout.on('data',part=>{data+=part;const m=data.match(/CONVERSATIONS_READY (\d+)/);if(m){clearTimeout(timer);resolve(Number(m[1]));}});
    server.once('error',error=>{clearTimeout(timer);reject(error);});
    server.once('exit',code=>{clearTimeout(timer);reject(Error(`Fixture exited ${code}: ${logs}`));});
  });
  const base=`http://127.0.0.1:${port}`;
  let ready=false;
  for(let i=0;i<100;i++){
    try{if((await fetch(base+'/healthz',{signal:AbortSignal.timeout(500)})).ok){ready=true;break;}}catch{}
    await new Promise(resolve=>setTimeout(resolve,25));
  }
  assert.ok(ready,'isolated API became ready: '+logs);
  const browser=await chromium.launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH,args:['--disable-gpu']});
  t.after(()=>browser.close());
  const page=await browser.newPage({viewport:{width:1320,height:940}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto(base+'/conversations');
  assert.equal(await page.getByRole('dialog',{name:'Memory connection'}).count(),1);
  assert.ok(!(await page.content()).includes('conversation-fixture-alpha'),'public page must not contain a space key');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await page.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.waitForURL(/\/conversations\/[^/]+$/);
  const sessionUrl=page.url(),sid=new URL(sessionUrl).pathname.split('/').at(-1);
  await page.getByLabel('Message',{exact:true}).fill('How is Juniper calibrated?');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Scripted answer: use Polaris.',{exact:true}).waitFor();
  const messages=page.getByLabel('Saved messages',{exact:true});
  assert.equal(await messages.locator('article').count(),2);
  await page.getByLabel('Message',{exact:true}).fill('What was my question?');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Scripted follow-up: you asked about Juniper calibration.',{exact:true}).waitFor();
  assert.equal(await messages.locator('article').count(),4);
  await messages.locator('article').first().getByRole('button').click();
  const evidence=page.getByRole('complementary',{name:'Conversation evidence'});
  await evidence.getByText('How is Juniper calibrated?',{exact:true}).waitFor();
  assert.ok(!(await page.innerText('body')).includes('private-fixture-thought'));
  const headers={authorization:'Bearer conversation-fixture-alpha'};
  const api='/v1/conversations/'+sid;
  const transcript=await(await fetch(base+api+'/transcript',{headers})).json();
  assert.deepEqual(transcript.episodes.map(e=>e.content),[
    'How is Juniper calibrated?','Scripted answer: use Polaris.',
    'What was my question?','Scripted follow-up: you asked about Juniper calibration.',
  ]);
  assert.equal((await fetch(base+api+'/transcript',{headers:{authorization:'Bearer conversation-fixture-beta'}})).status,404);
  await page.reload();
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await messages.locator('article').nth(3).waitFor();
  assert.equal(page.url(),sessionUrl,'direct session refresh preserves the address');
  await page.getByRole('button',{name:'End conversation',exact:true}).click();
  await page.getByRole('heading',{name:'Conversation ended',exact:true}).waitFor();
  assert.equal(await page.getByLabel('Message',{exact:true}).isDisabled(),true);
  assert.equal((await(await fetch(base+api+'/transcript',{headers})).json()).episodes.length,4,'refresh and stop do not replay a turn');
  assert.deepEqual(await page.evaluate(()=>({local:localStorage.length,session:sessionStorage.length})),{local:0,session:0});
  assert.deepEqual(errors,[]);
});
