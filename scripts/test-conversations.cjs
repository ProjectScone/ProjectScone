// Isolated HTTP fixtures exercise the built React app. No live Scone/model calls.
const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const http=require('node:http');
const {chromium}=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
let browser;
before(async()=>{browser=await chromium.launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH,args:['--disable-gpu']});});
after(async()=>{await browser?.close();});
async function fixture(t,{unavailable=false,mobile=false,uncertain=false,reject=false}={}){
  const html=fs.readFileSync(process.env.SCONE_CONVERSATIONS_HTML||path.resolve(__dirname,'../crates/scone/src/playground.html'),'utf8').replaceAll('__SCONE_TOKEN__','fixture-key');
  const sessions=[{session_id:'previous',space:'alpha',state:'ended',revision:4,created_at:'2026-09-06T10:00:00Z',active_request_id:null}];
  const saved={previous:[{episode_id:2,content:'Earlier conversation.',metadata:{role:'user'}}]};
  let turn=null,posts=0,checks=0;const requested=[];
  const server=http.createServer(async(req,res)=>{
    requested.push(req.url);res.setHeader('content-type','application/json');
    if(!req.url.startsWith('/v1/')){res.setHeader('content-type','text/html');return res.end(html);}
    assert.equal(req.headers.authorization,'Bearer fixture-key');
    if(req.url==='/v1/status')return res.end('{"space":"alpha"}');
    if(req.url==='/v1/conversations/capabilities')return res.end(JSON.stringify({schema_version:1,text_configured:!unavailable,reply_transport:'poll',reply_replay:'process_lifetime'}));
    let body='';for await(const chunk of req)body+=chunk;
    const data=body?JSON.parse(body):null;
    if(req.url==='/v1/conversations'&&req.method==='POST'){
      assert.equal(data.capture,true);const value={...sessions[0],session_id:'current',state:'running',revision:2};sessions.push(value);saved.current=[];return res.end(JSON.stringify(value));
    }
    if(req.url.startsWith('/v1/conversations?'))return res.end(JSON.stringify({items:sessions,has_more:false,next_after:null}));
    if(req.url==='/v1/episodes/7')return res.end(JSON.stringify({episode_id:7,content:'Juniper is calibrated with Polaris. <script>not executable</script>',metadata:{},attachments:[]}));
    const match=req.url.match(/^\/v1\/conversations\/([^/]+)(.*)$/);
    if(match){
      const value=sessions.find(s=>s.session_id===match[1]);if(!value){res.statusCode=404;return res.end('{}');}
      if(!match[2])return res.end(JSON.stringify(value));
      if(match[2]==='/transcript')return res.end(JSON.stringify({episodes:saved[value.session_id]||[],has_more:false}));
      if(match[2]==='/stop'){value.state='ended';value.revision=4;return res.end(JSON.stringify(value));}
      if(match[2]==='/turns'&&req.method==='POST'){
        posts++;if(reject){res.statusCode=429;return res.end('{"error":"capacity reached"}');}turn={request_id:data.request_id,status:'pending'};value.active_request_id=data.request_id;
        saved.current=[{episode_id:10,content:data.text,metadata:{role:'user'}}];
        res.statusCode=uncertain?503:202;return res.end(JSON.stringify(uncertain?{error:'Delivery uncertain'}:turn));
      }
      if(match[2].startsWith('/turns/')){
        if(!turn){res.statusCode=404;return res.end('{"error":"receipt unavailable"}');}
        checks++;if(checks>1){turn.status='completed';turn.result={text:'Use Polaris.',user_episode_id:10,assistant_episode_id:11,provider_completion:'unverified',memory_context:{status:'prepared',references:[{episode_id:7,chunk_id:8}]}};value.active_request_id=null;saved.current=[saved.current[0],{episode_id:11,content:'Use Polaris.',metadata:{role:'assistant'}}];}
        return res.end(JSON.stringify(turn));
      }
    }
    res.statusCode=404;res.end('{"error":"not found"}');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const page=await browser.newPage({viewport:mobile?{width:390,height:844}:{width:1440,height:1000},reducedMotion:'reduce'});
  page.setDefaultTimeout(4000);const errors=[];page.on('pageerror',e=>errors.push(e.message));
  t.after(async()=>{await page.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));assert.deepEqual(errors,[]);});
  await page.goto(`http://127.0.0.1:${server.address().port}/conversations`);
  return {page,requested,posts:()=>posts};
}
async function start(page){
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  const start=page.getByRole('button',{name:'Start text conversation',exact:true});
  assert.equal(await start.isDisabled(),true);
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await start.click();await page.getByLabel('Message',{exact:true}).waitFor();
}
test('unconfigured service explains setup and never probes sessions',async t=>{
  const {page,requested}=await fixture(t,{unavailable:true});
  await page.getByText('Text runtime not configured',{exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'New conversation',exact:true}).isDisabled(),true);
  assert.equal(requested.some(p=>p.startsWith('/v1/conversations?')),false);
});
test('capture consent, send, prepared sources and stop operate through the API',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.getByLabel('Message',{exact:true}).fill('How is Juniper calibrated?');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Use Polaris.',{exact:true}).waitFor();
  assert.equal(posts(),1);
  await page.getByRole('button',{name:'Source episode 7',exact:true}).click();
  await page.getByText('Juniper is calibrated with Polaris. <script>not executable</script>',{exact:true}).waitFor();
  await page.getByRole('button',{name:'End conversation',exact:true}).click();
  await page.getByText('Conversation ended',{exact:true}).waitFor();
  await page.waitForFunction(()=>document.querySelector('nav[aria-label="Saved conversations"] a.active')?.textContent.includes('ended'));
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'scone-conversation-desktop.png'),fullPage:true});
});
test('reopening a session reads saved messages without creating or sending',async t=>{
  const {page,posts}=await fixture(t);
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  await page.reload();await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  assert.equal(posts(),0);assert.match(page.url(),/\/conversations\/previous$/);
});
test('uncertain send polls the same request and never automatically submits twice',async t=>{
  const {page,posts}=await fixture(t,{uncertain:true});await start(page);
  await page.getByLabel('Message',{exact:true}).fill('Juniper');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Use Polaris.',{exact:true}).waitFor();assert.equal(posts(),1);
});
test('mobile conversation workspace fits the viewport',async t=>{
  const {page}=await fixture(t,{mobile:true});await start(page);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);
  assert.equal(await page.evaluate(()=>document.querySelector('.topbar nav').getBoundingClientRect().bottom<=document.querySelector('.server-strip').getBoundingClientRect().top+1),true,'navigation must not overlap connection strip');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'scone-conversation-mobile.png'),fullPage:true});
});
test('definitively rejected message preserves draft and does not wait for a nonexistent receipt',async t=>{
  const {page,posts}=await fixture(t,{reject:true});await start(page);
  await page.getByLabel('Message',{exact:true}).fill('Keep this draft');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Message was not accepted. Your draft is kept.',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Message',{exact:true}).inputValue(),'Keep this draft');
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isEnabled(),true);
  assert.equal(posts(),1);
});
test('late polling cannot reopen a conversation after an acknowledged end',async t=>{
  const {page}=await fixture(t);await start(page);
  await page.waitForFunction(()=>!document.querySelector('textarea')?.disabled);
  let release;const gate=new Promise(resolve=>{release=resolve;});
  let entered;const blocked=new Promise(resolve=>{entered=resolve;});
  await page.route('**/current/transcript',async route=>{entered();await gate;await route.continue();},{times:1});
  await blocked;
  await page.getByRole('button',{name:'End conversation',exact:true}).click();
  await page.getByText('Conversation ended',{exact:true}).waitFor();
  release();await page.waitForResponse(response=>response.url().endsWith('/current/transcript'));
  await page.waitForTimeout(150);
  assert.equal(await page.getByRole('button',{name:'End conversation',exact:true}).isDisabled(),true);
  assert.match(await page.locator('nav[aria-label="Saved conversations"] a.active').textContent(),/ended/);
});
