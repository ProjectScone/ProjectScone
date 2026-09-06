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
async function fixture(t,{unavailable=false,mobile=false,uncertain=false,reject=false,unknown=false,recovered=false,deletion=false,cancellation=false,pagination=false}={}){
  const html=fs.readFileSync(process.env.SCONE_CONVERSATIONS_HTML||path.resolve(__dirname,'../crates/scone/src/playground.html'),'utf8').replaceAll('__SCONE_TOKEN__','fixture-key');
  const sessions=[{session_id:'previous',space:'alpha',state:unavailable?'running':'ended',revision:4,created_at:'2026-09-06T10:00:00Z',active_request_id:null,...(recovered?{latest_request_id:'a-newer'}:{})}];
  const saved={previous:[{episode_id:2,content:'Earlier conversation.',metadata:{role:'user'}}]};
  if(pagination)saved.previous=Array.from({length:123},(_,i)=>({episode_id:i+1,content:`Saved message ${i+1}`,metadata:{role:i%2?'assistant':'user'}}));
  let turn=null,posts=0,checks=0;const requested=[];
  const server=http.createServer(async(req,res)=>{
    requested.push(req.url);res.setHeader('content-type','application/json');
    if(!req.url.startsWith('/v1/')){res.setHeader('content-type','text/html');return res.end(html);}
    assert.equal(req.headers.authorization,'Bearer fixture-key');
    if(req.url==='/v1/status')return res.end('{"space":"alpha"}');
    if(req.url==='/v1/conversations/capabilities')return res.end(JSON.stringify({schema_version:unknown?999:1,text_configured:!unavailable,reply_transport:'poll',reply_replay:'process_lifetime',session_deletion:deletion,turn_cancellation:cancellation,transcript_pagination:pagination}));
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
      if(!match[2]&&req.method==='DELETE'){sessions.splice(sessions.indexOf(value),1);delete saved[value.session_id];res.statusCode=204;return res.end();}
      if(!match[2])return res.end(JSON.stringify(value));
      if(match[2].split('?')[0]==='/transcript'){
        const query=new URL(req.url,'http://fixture').searchParams,limit=Number(query.get('limit')||200),before=query.get('before');
        const records=(saved[value.session_id]||[]).filter(e=>!before||e.episode_id<Number(before.replace('cursor-',''))),episodes=records.slice(-limit),has_more=records.length>limit;
        return res.end(JSON.stringify({episodes,has_more,...(pagination?{next_before:has_more?'cursor-'+episodes[0].episode_id:null}:{})}));
      }
      if(match[2]==='/stop'){value.state='ended';value.revision=4;return res.end(JSON.stringify(value));}
      if(match[2]==='/turns'&&req.method==='POST'){
        posts++;if(reject){res.statusCode=429;return res.end('{"error":"capacity reached"}');}turn={request_id:data.request_id,status:'pending'};value.active_request_id=data.request_id;
        saved.current=[{episode_id:10,content:data.text,metadata:{role:'user'}}];
        res.statusCode=uncertain?503:202;return res.end(JSON.stringify(uncertain?{error:'Delivery uncertain'}:turn));
      }
      if(match[2].startsWith('/turns/')){
        if(match[2].endsWith('/cancel')&&req.method==='POST'){turn.status='cancelled';turn.result=null;turn.result_state='unavailable';value.active_request_id=null;value.latest_request_id=turn.request_id;return res.end(JSON.stringify(turn));}
        if(turn?.status==='cancelled')return res.end(JSON.stringify(turn));
        if(recovered&&value.session_id==='previous')return res.end(JSON.stringify({request_id:'a-newer',status:'completed',result_state:'forgotten',result:null}));
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

for(const mobile of [false,true])test(`transcript navigation reaches older messages and keeps the page during polling, mobile=${mobile}`,async t=>{
  const {page,requested}=await fixture(t,{pagination:true,mobile});
  await page.getByRole('link',{name:/previous/}).click();
  const messages=page.locator('.conversation-messages');
  await page.getByText('Saved message 123',{exact:true}).waitFor();
  assert.equal(await messages.locator('article').count(),50);
  await page.getByRole('button',{name:'Older messages',exact:true}).click();
  await page.getByText('Saved message 24',{exact:true}).waitFor();
  assert.equal(await page.getByText('Saved message 123',{exact:true}).count(),0);
  const reads=()=>requested.filter(url=>url.includes('/transcript?')&&url.includes('before=cursor-74')).length;
  const prior=reads();await page.waitForResponse(response=>response.url().includes('before=cursor-74'));
  assert.ok(reads()>prior);assert.equal(await page.getByText('Saved message 24',{exact:true}).count(),1);
  await page.getByRole('button',{name:'Older messages',exact:true}).click();
  await page.getByText('Saved message 1',{exact:true}).waitFor();
  assert.equal(await messages.locator('article').count(),23);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`scone-transcript-${mobile?'mobile':'desktop'}.png`),fullPage:true});
  assert.equal(await page.getByRole('button',{name:'Older messages',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Newer messages',exact:true}).click();
  await page.getByText('Saved message 24',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Latest messages',exact:true}).click();
  await page.getByText('Saved message 123',{exact:true}).waitFor();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
});

test('a failed older page can be retried without resending or losing the selected boundary',async t=>{
  const {page,posts}=await fixture(t,{pagination:true});let fail=true;
  await page.route('**/previous/transcript?*before=*',route=>fail?route.fulfill({status:503,json:{error:'read unavailable'}}):route.continue());
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Saved message 123',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Older messages',exact:true}).click();
  await page.getByText(/Connection interrupted. Controls are paused/).waitFor();
  assert.equal(await page.getByText('Saved message 123',{exact:true}).count(),0);
  fail=false;await page.getByRole('button',{name:'Check connection',exact:true}).click();
  await page.getByText('Saved message 24',{exact:true}).waitFor();
  assert.equal(posts(),0);
});

test('late older-page responses cannot replace another selected session',async t=>{
  const {page}=await fixture(t,{pagination:true});let release;
  const gate=new Promise(resolve=>{release=resolve;});t.after(()=>release());
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Saved message 123',{exact:true}).waitFor();
  await page.route('**/previous/transcript?*before=*',async route=>{await gate;await route.fulfill({json:{episodes:[{episode_id:1,content:'Late older text',metadata:{}}],has_more:false,next_before:null}}).catch(()=>{});});
  const requested=page.waitForRequest(request=>request.url().includes('before=cursor-74'));
  await page.getByRole('button',{name:'Older messages',exact:true}).click();await requested;
  await start(page);release();
  await page.getByRole('heading',{name:'Start with a question.',exact:true}).waitFor();
  assert.match(page.url(),/\/current$/);assert.equal(await page.getByText('Late older text',{exact:true}).count(),0);
});

test('closed session deletion requires confirmation and removes the saved session',async t=>{
  const {page}=await fixture(t,{deletion:true,mobile:true});
  const commands=[];page.on('request',r=>{if(r.method()==='DELETE')commands.push(r.url());});
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Delete conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Delete this conversation?'});
  assert.equal(await dialog.getByRole('button',{name:'Permanently delete',exact:true}).isDisabled(),true);
  await dialog.getByLabel('I understand this cannot be undone').check();
  await dialog.getByRole('button',{name:'Keep conversation',exact:true}).click();
  assert.deepEqual(commands,[]);
  await page.getByRole('button',{name:'Delete conversation',exact:true}).click();
  assert.equal(await dialog.getByRole('button',{name:'Permanently delete',exact:true}).isDisabled(),true,'reopening needs fresh confirmation');
  await dialog.getByLabel('I understand this cannot be undone').check();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'scone-delete-mobile.png'),fullPage:true});
  await dialog.getByRole('button',{name:'Permanently delete',exact:true}).click();
  await page.waitForURL(/\/conversations$/);
  await page.getByText('Conversation deleted.',{exact:true}).waitFor();
  assert.equal(await page.getByRole('link',{name:/previous/}).count(),0);
  assert.equal(await page.getByText('Earlier conversation.',{exact:true}).count(),0);
  assert.equal(commands.length,1);
});

test('unsupported and active sessions expose no delete command',async t=>{
  const {page}=await fixture(t);
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Delete conversation',exact:true}).count(),0);
  await start(page);await page.getByLabel('Message',{exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Delete conversation',exact:true}).count(),0);
});

test('a supported server still cannot delete an active session',async t=>{
  const {page}=await fixture(t,{deletion:true});await start(page);
  await page.waitForFunction(()=>!document.querySelector('textarea')?.disabled);
  assert.equal(await page.getByRole('button',{name:'Delete conversation',exact:true}).count(),0);
});

test('a refused delete preserves the session and requires fresh confirmation',async t=>{
  const {page}=await fixture(t,{deletion:true});let deletes=0;
  await page.getByRole('link',{name:/previous/}).click();await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  await page.route('**/v1/conversations/previous',route=>{
    if(route.request().method()!=='DELETE')return route.continue();
    deletes++;return route.fulfill({status:409,json:{error:'stop first'}});
  });
  await page.getByRole('button',{name:'Delete conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Delete this conversation?'});
  await dialog.getByLabel('I understand this cannot be undone').check();
  await dialog.getByRole('button',{name:'Permanently delete',exact:true}).click();
  await dialog.getByText(/Deletion was refused/).waitFor();
  assert.equal(await dialog.getByRole('button',{name:'Permanently delete',exact:true}).isDisabled(),true);
  await dialog.getByRole('button',{name:'Keep conversation',exact:true}).click();
  assert.equal(await page.getByText('Earlier conversation.',{exact:true}).count(),1);
  assert.equal(deletes,1);
});

test('Escape cannot dismiss an in-flight deletion or permit a second command',async t=>{
  const {page}=await fixture(t,{deletion:true});let release;
  const gate=new Promise(resolve=>{release=resolve;});t.after(()=>release());let deletes=0;
  await page.getByRole('link',{name:/previous/}).click();await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  await page.route('**/v1/conversations/previous',async route=>{
    if(route.request().method()!=='DELETE')return route.continue();
    deletes++;await gate;await route.fulfill({status:204}).catch(()=>{});
  });
  await page.getByRole('button',{name:'Delete conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Delete this conversation?'});
  await dialog.getByLabel('I understand this cannot be undone').check();
  await dialog.getByRole('button',{name:'Permanently delete',exact:true}).click();
  await dialog.getByRole('button',{name:'Deleting…',exact:true}).waitFor();
  await page.keyboard.press('Escape');
  assert.equal(await dialog.isVisible(),true);
  assert.equal(await dialog.getByRole('button',{name:'Deleting…',exact:true}).isDisabled(),true);
  release();await page.waitForURL(/\/conversations$/);assert.equal(deletes,1);
});

for(const missing of [false,true])test(`uncertain deletion uses a read-only check; session missing=${missing}`,async t=>{
  const {page}=await fixture(t,{deletion:true});let deletes=0;
  await page.getByRole('link',{name:/previous/}).click();await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  await page.route('**/v1/conversations/previous',route=>{
    if(route.request().method()==='DELETE'){deletes++;return route.fulfill({status:503,json:{error:'unknown outcome'}});}
    if(deletes&&missing)return route.fulfill({status:404,json:{error:'not found'}});
    return route.continue();
  });
  await page.getByRole('button',{name:'Delete conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Delete this conversation?'});
  await dialog.getByLabel('I understand this cannot be undone').check();
  await dialog.getByRole('button',{name:'Permanently delete',exact:true}).click();
  await dialog.getByRole('button',{name:'Check deletion status',exact:true}).click();
  if(missing){await page.waitForURL(/\/conversations$/);await page.getByText('Conversation is no longer available.',{exact:true}).waitFor();}
  else{await dialog.getByText(/still exists/).waitFor();assert.equal(await dialog.getByRole('button',{name:'Permanently delete',exact:true}).isDisabled(),true);}
  assert.equal(deletes,1,'uncertain outcome must not automatically issue another DELETE');
});
test('unconfigured text keeps saved conversations readable without permitting new work',async t=>{
  const {page,posts}=await fixture(t,{unavailable:true});
  await page.getByText('Text runtime not configured',{exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'New conversation',exact:true}).isDisabled(),true);
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Message',{exact:true}).isDisabled(),true);
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
  await page.reload();await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  assert.equal(posts(),0);
});
test('an unknown conversation contract does not trigger session reads',async t=>{
  const {page,requested}=await fixture(t,{unknown:true});
  await page.getByRole('button',{name:'Retry service connection',exact:true}).waitFor();
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

test('cancel reply preserves the next draft and never resends the cancelled turn',async t=>{
  const {page,posts}=await fixture(t,{cancellation:true});await start(page);
  const cancels=[];page.on('request',r=>{if(r.url().endsWith('/cancel'))cancels.push(r.url());});
  await page.getByLabel('Message',{exact:true}).fill('First question');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByRole('button',{name:'Cancel reply',exact:true}).waitFor();
  await page.getByLabel('Message',{exact:true}).fill('Keep this next question');
  await page.getByRole('button',{name:'Cancel reply',exact:true}).click();
  await page.getByText(/Reply cancelled/).waitFor();
  await page.waitForFunction(()=>!document.querySelector('button[aria-label="Send message"]')?.disabled);
  assert.equal(await page.getByLabel('Message',{exact:true}).inputValue(),'Keep this next question');
  assert.equal(posts(),1);assert.equal(cancels.length,1);
  await page.getByRole('button',{name:'Send message',exact:true}).click();await page.getByText('Use Polaris.',{exact:true}).waitFor();
  assert.equal(posts(),2);
});

test('an uncertain cancellation checks the existing outcome without another POST',async t=>{
  const {page,posts}=await fixture(t,{cancellation:true});await start(page);let cancels=0;
  await page.route('**/turns/*/cancel',async route=>{cancels++;await route.fetch();await route.fulfill({status:503,json:{error:'acknowledgment lost'}});});
  await page.getByLabel('Message',{exact:true}).fill('Question');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByRole('button',{name:'Cancel reply',exact:true}).click();
  await page.getByText(/Cancellation was not confirmed/).waitFor();
  await page.getByText(/Reply cancelled locally/).waitFor();
  assert.equal(cancels,1);assert.equal(posts(),1);
});

test('an old cancellation response cannot replace a newly selected conversation',async t=>{
  const {page}=await fixture(t,{cancellation:true});await start(page);
  let release;const gate=new Promise(resolve=>{release=resolve;});t.after(()=>release());
  await page.route('**/turns/*/cancel',async route=>{
    await gate;await route.fulfill({json:{request_id:new URL(route.request().url()).pathname.split('/').at(-2),status:'cancelled',result_state:'unavailable',result:null}}).catch(()=>{});
  });
  await page.getByLabel('Message',{exact:true}).fill('Question');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByRole('button',{name:'Cancel reply',exact:true}).click();
  await page.getByRole('button',{name:'Cancelling…',exact:true}).waitFor();
  await page.getByRole('link',{name:/previous/}).click();await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  release();await page.waitForTimeout(150);
  assert.match(page.url(),/\/previous$/);assert.equal(await page.getByText(/Reply cancelled locally/).count(),0);
});

test('failed receipt reads keep Send paused while cancelled cleanup is active',async t=>{
  const {page,posts}=await fixture(t);await start(page);let reads=0;
  await page.route('**/current/turns/*',route=>{
    reads++;
    return reads===1?route.fulfill({json:{request_id:new URL(route.request().url()).pathname.split('/').at(-1),status:'cancelled',result_state:'unavailable',result:null}}):route.fulfill({status:503,json:{error:'temporarily unavailable'}});
  });
  await page.getByLabel('Message',{exact:true}).fill('Cancelled request');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText(/Reply cancelled/).waitFor();
  await page.getByLabel('Message',{exact:true}).fill('Keep this draft');
  await page.getByText('Reply receipt unavailable. No message was resent.',{exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
  assert.equal(posts(),1);
});

test('a cancelled receipt settles without replaying the message',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.route('**/current/turns/*',route=>route.fulfill({json:{request_id:new URL(route.request().url()).pathname.split('/').at(-1),status:'cancelled',result_state:'unavailable',result:null}}));
  await page.getByLabel('Message',{exact:true}).fill('Cancelled request');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText(/Reply cancelled/).waitFor();
  assert.equal(posts(),1);
});

for(const result_state of ['forgotten','unavailable'])test(`completed reply with ${result_state} content settles without resending`,async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.route('**/current/turns/*',route=>route.fulfill({json:{
    request_id:new URL(route.request().url()).pathname.split('/').at(-1),
    status:'completed',result_state,result:null,
  }}));
  await page.getByLabel('Message',{exact:true}).fill('An uncertain outcome');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText(result_state==='forgotten'?'Reply completed; its saved text was forgotten.':'Reply completed; its text is unavailable.',{exact:true}).waitFor();
  await page.getByLabel('Message',{exact:true}).fill('A different question');
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isEnabled(),true);
  assert.equal(posts(),1,'settling an outcome must not replay the original turn');
});

test('a forgotten settled reply clears its cached evidence on the next receipt check',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.getByLabel('Message',{exact:true}).fill('How is Juniper calibrated?');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Use Polaris.',{exact:true}).waitFor();
  const evidence=page.getByRole('complementary',{name:'Conversation evidence'});
  await evidence.getByRole('button',{name:'Source episode 7',exact:true}).click();
  await evidence.getByText('Juniper is calibrated with Polaris. <script>not executable</script>',{exact:true}).waitFor();
  await page.route('**/current/turns/*',route=>route.fulfill({json:{
    request_id:new URL(route.request().url()).pathname.split('/').at(-1),
    status:'completed',result_state:'forgotten',result:null,
  }}));
  await page.route('**/current/transcript',route=>route.fulfill({json:{episodes:[{episode_id:10,content:'How is Juniper calibrated?',metadata:{role:'user'}}],has_more:false}}));
  await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).waitFor();
  assert.equal(await page.getByText('Use Polaris.',{exact:true}).count(),0);
  assert.equal(await evidence.getByRole('button',{name:'Source episode 7',exact:true}).count(),0);
  assert.equal(await evidence.locator('.conversation-source-text').count(),0);
  assert.equal(posts(),1);
});

test('a late transcript from the previous session cannot replace the selected conversation',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.waitForFunction(()=>!document.querySelector('textarea')?.disabled);
  let release;const gate=new Promise(resolve=>{release=resolve;});
  let entered;const blocked=new Promise(resolve=>{entered=resolve;});
  await page.route('**/current/transcript',async route=>{
    entered();await gate;
    await route.fulfill({json:{episodes:[{episode_id:19,content:'Late response from old session',metadata:{role:'assistant'}}],has_more:false}}).catch(()=>{});
  },{times:1});
  await blocked;
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Earlier conversation.',{exact:true}).waitFor();
  release();await page.waitForTimeout(200);
  assert.equal(await page.getByText('Late response from old session',{exact:true}).count(),0);
  assert.match(page.url(),/\/conversations\/previous$/);
  assert.equal(posts(),0);
});

test('a delayed settled receipt cannot unlock sending while a newer turn is pending',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.getByLabel('Message',{exact:true}).fill('First question');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Use Polaris.',{exact:true}).waitFor();
  let release;const gate=new Promise(resolve=>{release=resolve;});t.after(()=>release());
  let entered;const blocked=new Promise(resolve=>{entered=resolve;});let oldId;
  await page.route('**/current/turns/*',async route=>{
    const request_id=new URL(route.request().url()).pathname.split('/').at(-1);
    if(!oldId){oldId=request_id;entered();await gate;
      await route.fulfill({json:{request_id,status:'completed',result:{text:'Old answer',assistant_episode_id:11}}});
    }else await route.fulfill({json:{request_id,status:'pending'}});
  });
  await blocked;
  await page.getByLabel('Message',{exact:true}).fill('Second question');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Reply in progress',{exact:true}).waitFor();
  await page.getByLabel('Message',{exact:true}).fill('Third question');
  const returned=page.waitForResponse(r=>r.url().endsWith('/turns/'+oldId));
  release();await returned;await page.waitForTimeout(200);
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
  assert.equal(await page.getByText('Reply in progress',{exact:true}).count(),1);
  assert.equal(posts(),2);
});

test('a forgotten reply does not repeatedly close a retained source the user inspects',async t=>{
  const {page}=await fixture(t);await start(page);
  await page.route('**/current/turns/*',route=>route.fulfill({json:{
    request_id:new URL(route.request().url()).pathname.split('/').at(-1),status:'completed',result_state:'forgotten',result:null,
  }}));
  await page.route('**/v1/episodes/10',route=>route.fulfill({json:{episode_id:10,content:'Retained user question',metadata:{role:'user'}}}));
  await page.getByLabel('Message',{exact:true}).fill('Retained user question');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Inspect message episode 10',exact:true}).click();
  const source=page.getByRole('complementary',{name:'Conversation evidence'}).locator('.conversation-source-text');
  await source.waitFor();
  await page.waitForResponse(r=>r.url().includes('/current/turns/'));await page.waitForTimeout(150);
  assert.equal(await source.textContent(),'Retained user question');
});

test('polling preserves a send that is still waiting for its HTTP acknowledgment',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  await page.getByLabel('Message',{exact:true}).fill('First question');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Use Polaris.',{exact:true}).waitFor();
  let release;const gate=new Promise(resolve=>{release=resolve;});t.after(()=>release());
  let entered;const blocked=new Promise(resolve=>{entered=resolve;});
  await page.route('**/current/turns',async route=>{entered();await gate;await route.continue().catch(()=>{});},{times:1});
  await page.getByLabel('Message',{exact:true}).fill('Second question');
  await page.getByRole('button',{name:'Send message',exact:true}).click();await blocked;
  // Cross a polling interval while the server still knows only the old turn.
  await page.waitForTimeout(2200);
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
  assert.equal(await page.getByText('Sending message…',{exact:true}).count(),1);
  assert.equal(posts(),1);
  release();
});

test('temporarily unreadable completed reply retries the read without declaring deletion or resending',async t=>{
  const {page,posts}=await fixture(t);await start(page);
  let available=false;
  await page.route('**/current/turns/*',route=>route.fulfill({json:{
    request_id:new URL(route.request().url()).pathname.split('/').at(-1),status:'completed',
    result_state:available?'available':'unreadable',result:available?{text:'Recovered saved answer',assistant_episode_id:11}:null,
  }}));
  await page.getByLabel('Message',{exact:true}).fill('A question worth preserving');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Reply completed; its saved text cannot be read right now. Checking again…',{exact:true}).waitFor();
  assert.equal(await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).count(),0);
  await page.getByLabel('Message',{exact:true}).fill('Keep my next question');
  available=true;await page.getByText('Reply saved',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Message',{exact:true}).inputValue(),'Keep my next question');
  assert.equal(posts(),1);
});

test('reopening and reloading discover the recorded latest outcome without submitting work',async t=>{
  const {page,requested,posts}=await fixture(t,{unavailable:true,recovered:true});
  await page.getByRole('link',{name:/previous/}).click();
  await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).waitFor();
  await page.reload();
  await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).waitFor();
  assert.ok(requested.filter(p=>p==='/v1/conversations/previous/turns/a-newer').length>=2);
  assert.equal(requested.some(p=>p.includes('/turns?')),false,'no UUID sorting or history scan to guess the newest turn');
  assert.equal(posts(),0);
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
});
