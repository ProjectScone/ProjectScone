// Real React → HTTP service → Pipecat scheduler → native memory, no live provider.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const {once}=require('node:events');
const path=require('node:path');
const {chromium}=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'..');

async function fixture(t,{reopened=false,scoped=false}={}){
  const html=process.env.SCONE_CONVERSATIONS_HTML;
  assert.ok(html,'SCONE_CONVERSATIONS_HTML must name the verified isolated webapp artifact');
  const server=spawn(process.env.SCONE_TEST_PYTHON||path.join(root,'python/scone-memory/.venv/bin/python'),
    ['-u',path.join(__dirname,'fixtures/conversation-server.py'),html,...(reopened?['--reopened']:[]),...(scoped?['--scoped']:[])],{cwd:root,stdio:['pipe','pipe','pipe']});
  let browser,logs='';server.stderr.on('data',part=>{logs=(logs+part).slice(-6000);});
  const closed=once(server,'close');
  let modelEntered;const modelWaiting=new Promise(resolve=>{modelEntered=resolve;});
  t.after(async()=>{
    if(server.exitCode===null&&server.signalCode===null){
      server.stdin.end('stop\n');
      const timeout=setTimeout(()=>server.kill('SIGKILL'),5000);
      try{await closed;}finally{clearTimeout(timeout);}
    }
    await closed; // stderr is drained before checking shutdown diagnostics
    await browser?.close(); // do not leak Chrome when a shutdown assertion fails
    assert.notEqual(server.signalCode,'SIGKILL','fixture required forced shutdown: '+logs);
    assert.equal(server.exitCode,0,'fixture did not exit cleanly: '+logs);
    assert.doesNotMatch(logs,/Traceback \(most recent call last\)|AttributeError/,'fixture shutdown failed');
  });
  const port=await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(Error('Fixture startup timed out: '+logs)),20000);
    let data='';
    server.stdout.on('data',part=>{data+=part;if(data.includes('CONVERSATIONS_MODEL_WAITING'))modelEntered();const m=data.match(/CONVERSATIONS_READY (\d+)/);if(m){clearTimeout(timer);resolve(Number(m[1]));}});
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
  browser=await chromium.launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH,args:['--disable-gpu']});
  const page=await browser.newPage({viewport:{width:1320,height:940}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  return {page,base,errors,modelWaiting};
}

test('Documents browse native inventory, import retained text, and inspect saved image evidence',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t);page.setDefaultTimeout(8000);
  const headers={authorization:'Bearer conversation-fixture-alpha','content-type':'application/json'};
  for(let i=0;i<28;i++){
    const response=await fetch(base+'/v1/episodes',{method:'POST',headers,body:JSON.stringify({kind:'file',source:`source-${i}.md`,content:`Native inventory document ${i}.`,created_at:i%2?'2020-01-01':'2026-09-06'})});
    assert.equal(response.status,200);
  }
  await page.goto(base+'/memory#documents');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'Files',exact:true}).click();
  await page.getByRole('button',{name:'Open source-27.md',exact:true}).waitFor();
  assert.equal(await page.locator('.document-card').count(),25);
  assert.equal(await page.locator('.document-card').first().getAttribute('aria-label'),'Open source-27.md','ID order is not a created-date or relevance sort');
  await page.getByRole('button',{name:'Older sources',exact:true}).click();
  await page.getByRole('button',{name:'Open source-2.md',exact:true}).waitFor();
  assert.equal(await page.locator('.document-card').count(),3);
  assert.equal(await page.getByRole('button',{name:'Older sources',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('button',{name:'Import text file',exact:true}).click();
  const content='\ufeff# Atlas guide\r\nA preserved document from the source library.\r\n';
  await page.getByLabel('Text file',{exact:true}).setInputFiles({name:'atlas.md',mimeType:'text/markdown',buffer:Buffer.from(content)});
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByRole('heading',{name:/Source saved · episode #/}).waitFor();
  await page.getByRole('button',{name:'Open atlas.md',exact:true}).click();
  const detail=page.getByRole('region',{name:'Retained source'});
  await page.waitForFunction(()=>document.querySelector('.document-detail pre')?.textContent.includes('Atlas guide'));
  assert.equal(await detail.locator('pre').textContent(),content);
  const alpha=await(await fetch(base+'/v1/sources?limit=25',{headers})).json();
  const id=alpha.items[0].episode_id;assert.equal(alpha.items[0].source,'atlas.md');
  const betaHeaders={authorization:'Bearer conversation-fixture-beta'};
  assert.deepEqual((await(await fetch(base+'/v1/sources',{headers:betaHeaders})).json()).items,[]);
  assert.equal((await fetch(base+'/v1/episodes/'+id,{headers:betaHeaders})).status,404);
  const bytes=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=','base64');
  const upload=await fetch(base+'/v1/attachments',{method:'POST',headers:{authorization:headers.authorization,'content-type':'image/png','x-filename':'atlas-original.png'},body:bytes});
  assert.equal(upload.status,200);const attachment=await upload.json();
  const attached=await fetch(base+'/v1/episodes',{method:'POST',headers,body:JSON.stringify({kind:'note',content:'Atlas reference image',source:'Atlas original',attachment_ids:[attachment.attachment_id]})});
  assert.equal(attached.status,200);
  await page.getByRole('button',{name:'Refresh sources',exact:true}).click();
  await page.getByRole('button',{name:'Open Atlas original',exact:true}).click();
  await detail.getByRole('button',{name:'View source images',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.document-detail img')?.naturalWidth===1);
  assert.match(await detail.getByRole('img',{name:'atlas-original.png'}).getAttribute('src'),/^blob:/);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'documents-native.png'),fullPage:true});
  assert.deepEqual(errors,[]);
});

for(const collection of ['manuals','missing'])test(`browser-selected scope reaches Pipecat and survives reload: ${collection}`,{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{scoped:true});page.setDefaultTimeout(8000);
  await page.goto(base+'/conversations');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Start a conversation'});
  await dialog.getByRole('button',{name:'Files',exact:true}).click();
  await dialog.getByLabel('Source prefix',{exact:true}).fill('docs/');
  await dialog.getByText('Dates & metadata',{exact:true}).click();
  await dialog.getByLabel('Created on or after (UTC)',{exact:true}).fill('2026-09-01');
  await dialog.getByLabel('Created on or before (UTC)',{exact:true}).fill('2026-09-03');
  await dialog.getByRole('button',{name:'Add metadata filter',exact:true}).click();
  await dialog.getByLabel('Metadata key 1',{exact:true}).fill('collection');
  await dialog.getByLabel('Metadata value 1',{exact:true}).fill(collection);
  await dialog.getByLabel('Save my public messages and replies to this memory space').check();
  await dialog.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByLabel('Message',{exact:true}).waitFor();
  const sid=new URL(page.url()).pathname.split('/').at(-1),headers={authorization:'Bearer conversation-fixture-alpha'};
  const expected={kind:'file',source_prefix:'docs/',where:{collection},since:'2026-09-01T00:00:00.000Z',until:'2026-09-03T00:00:00.000Z'};
  for(let turn=0;turn<2;turn++){
    await page.getByLabel('Message',{exact:true}).fill(`${collection==='missing'?'No matching':'Find'} Juniper guide ${turn}`);
    await page.getByRole('button',{name:'Send message',exact:true}).click();
    await page.getByText(collection==='missing'?'Scoped answer: no retrieved context.':'Scoped answer: the selected guide.',{exact:true}).nth(turn).waitFor();
    const current=await(await fetch(base+'/v1/conversations/'+sid,{headers})).json();
    assert.deepEqual(current.recall_scope,expected);
    const receipt=await(await fetch(base+'/v1/conversations/'+sid+'/turns/'+current.latest_request_id,{headers})).json();
    assert.equal(receipt.status,'completed');assert.equal(receipt.result.memory_context.references.length,collection==='missing'?0:1);
    for(const ref of receipt.result.memory_context.references){
      const source=await(await fetch(base+'/v1/episodes/'+ref.episode_id,{headers})).json();
      assert.equal(source.source,'docs/guide.md');assert.equal(source.content,'Juniper scope-eligible-guide.');
    }
  }
  await page.reload();await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByText('Memory selection',{exact:true}).click();
  await page.getByText(`collection = ${collection}`,{exact:true}).waitFor();
  assert.deepEqual(errors,[]);
});

test('a browser text-file import is retained and recalled by the native engine',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t);page.setDefaultTimeout(8000);
  await page.goto(base+'/memory');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'Add source',exact:true}).click();
  await page.getByRole('button',{name:'Import text file',exact:true}).click();
  const content='\ufeff# Nebulaforge\r\nUse the amber dial for calibration.\r\n';
  await page.getByLabel('Text file',{exact:true}).setInputFiles({name:'nebulaforge.md',mimeType:'text/markdown',buffer:Buffer.from(content)});
  await page.getByRole('region',{name:'File preview'}).waitFor();
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByRole('heading',{name:/Source saved · episode #/}).waitFor();
  const id=Number((await page.getByRole('heading',{name:/Source saved · episode #/}).textContent()).match(/#(\d+)/)[1]);
  const headers={authorization:'Bearer conversation-fixture-alpha'};
  const saved=await(await fetch(base+'/v1/episodes/'+id,{headers})).json();
  assert.equal(saved.content,content);assert.equal(saved.kind,'file');assert.equal(saved.source,'nebulaforge.md');
  assert.equal((await fetch(base+'/v1/episodes/'+id,{headers:{authorization:'Bearer conversation-fixture-beta'}})).status,404);
  await page.getByText('Read saved text',{exact:true}).click();
  assert.equal(await page.locator('.source-saved pre').textContent(),content);
  await page.locator('.source-composer').getByRole('button',{name:'Close',exact:true}).click();
  await page.getByPlaceholder('What have we decided about this project?').fill('Nebulaforge');
  await page.getByPlaceholder('What have we decided about this project?').press('Enter');
  await page.locator('.rows .row').filter({hasText:'amber dial'}).first().waitFor();
  assert.equal(await page.locator('.rows .row').getByRole('link',{name:'nebulaforge.md',exact:true}).count(),0,'a local filename is provenance, not a website URL');
  assert.deepEqual(errors,[]);
});

test('the browser pages a native transcript and opens an older original source',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t);page.setDefaultTimeout(8000);
  const headers={authorization:'Bearer conversation-fixture-alpha','content-type':'application/json'};
  const response=await fetch(base+'/v1/conversations',{method:'POST',headers,body:JSON.stringify({request_id:'history-fixture',capture:true})});
  assert.equal(response.status,200);const session=await response.json();
  const ids=[];
  // Explicit synthetic history belongs only to this disposable fixture server.
  // This tests navigation through real storage, not agent capture provenance.
  for(let index=1;index<=123;index++){
    const added=await fetch(base+'/v1/episodes',{method:'POST',headers,body:JSON.stringify({content:`Archived fixture message ${index}`,kind:'conversation',metadata:{session_id:session.session_id,role:index%2?'user':'assistant'}})});
    assert.equal(added.status,200);ids.push((await added.json()).episode_id);
  }
  let writes=0;page.on('request',request=>{if(request.method()==='POST')writes++;});
  await page.goto(base+'/conversations/'+session.session_id);
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByText('Archived fixture message 123',{exact:true}).waitFor();
  assert.equal(await page.getByRole('region',{name:'Saved messages'}).locator('article').count(),50);
  await page.getByRole('button',{name:'Older messages',exact:true}).click();
  await page.getByText('Archived fixture message 24',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Older messages',exact:true}).click();
  await page.getByText('Archived fixture message 1',{exact:true}).waitFor();
  await page.getByRole('button',{name:`Inspect message episode ${ids[0]}`,exact:true}).click();
  await page.getByRole('complementary',{name:'Conversation evidence'}).getByText('Archived fixture message 1',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Latest messages',exact:true}).click();
  await page.getByText('Archived fixture message 123',{exact:true}).waitFor();
  assert.equal(writes,0,'reading history must not create or submit work');
  assert.deepEqual(errors,[]);
});

test('the browser cancels a real Pipecat reply and completes the next question',{timeout:60000},async t=>{
  const {page,base,errors,modelWaiting}=await fixture(t);page.setDefaultTimeout(8000);
  const writes=[];page.on('request',request=>{if(request.method()==='POST')writes.push(new URL(request.url()).pathname);});
  await page.goto(base+'/conversations');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('button',{name:'New conversation',exact:true}).click();
  await page.getByLabel('Save my public messages and replies to this memory space').check();
  await page.getByRole('button',{name:'Start text conversation',exact:true}).click();
  await page.getByLabel('Message',{exact:true}).fill('Wait for cancellation');await page.getByRole('button',{name:'Send message',exact:true}).click();
  await modelWaiting;
  await page.getByRole('button',{name:'Cancel reply',exact:true}).waitFor();
  await page.getByLabel('Message',{exact:true}).fill('How is Juniper calibrated?');
  await page.getByRole('button',{name:'Cancel reply',exact:true}).click();
  await page.getByText(/Reply cancelled locally/).waitFor();
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Scripted answer: use Polaris.',{exact:true}).waitFor();
  const sid=new URL(page.url()).pathname.split('/').at(-1),headers={authorization:'Bearer conversation-fixture-alpha'};
  const transcript=await(await fetch(base+'/v1/conversations/'+sid+'/transcript',{headers})).json();
  assert.deepEqual(transcript.episodes.map(e=>e.content),['Wait for cancellation','How is Juniper calibrated?','Scripted answer: use Polaris.']);
  assert.equal(writes.filter(p=>p.endsWith('/turns')).length,2,'the cancelled turn was not resubmitted');
  assert.equal(writes.filter(p=>p.endsWith('/cancel')).length,1);
  assert.equal((await(await fetch(base+'/v1/conversations/'+sid,{headers})).json()).state,'running');
  assert.deepEqual(errors,[]);
});

test('conversation deep links, public capture and sources work through native Pipecat',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t);
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
  await page.getByRole('button',{name:'Delete conversation',exact:true}).click();
  const deletion=page.getByRole('dialog',{name:'Delete this conversation?'});
  await deletion.getByLabel('I understand this cannot be undone').check();
  await deletion.getByRole('button',{name:'Permanently delete',exact:true}).click();
  await page.waitForURL(base+'/conversations');
  await page.getByText('Conversation deleted.',{exact:true}).waitFor();
  assert.equal((await fetch(base+api,{headers})).status,404);
  for(const episode of transcript.episodes)assert.equal((await fetch(base+'/v1/episodes/'+episode.episode_id,{headers})).status,404);
  const original=await(await fetch(base+'/v1/episodes/1',{headers})).json();
  assert.equal(original.content,'Juniper is calibrated with Polaris.','deleting a conversation must not delete its imported context');
  assert.equal(await page.getByRole('navigation',{name:'Saved conversations'}).getByRole('link').count(),0);
  assert.deepEqual(await page.evaluate(()=>({local:localStorage.length,session:sessionStorage.length})),{local:0,session:0});
  assert.deepEqual(errors,[]);
});

test('recreated native service recovers the actual latest outcome without a model or browser cache',{timeout:60000},async t=>{
  const {page,base,errors}=await fixture(t,{reopened:true});
  page.setDefaultTimeout(7000);
  const writes=[];page.on('request',request=>{if(request.method()!=='GET')writes.push(request.method()+' '+new URL(request.url()).pathname);});
  await page.goto(base+'/conversations');
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByRole('navigation',{name:'Saved conversations'}).getByRole('link').click();
  await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).waitFor();
  const url=page.url(),sid=new URL(url).pathname.split('/').at(-1);
  assert.equal(await page.getByRole('button',{name:'New conversation',exact:true}).isDisabled(),true);
  assert.equal(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled(),true);
  const headers={authorization:'Bearer conversation-fixture-alpha'};
  const endpoint=base+'/v1/conversations/'+sid;
  const session=await(await fetch(endpoint,{headers})).json();
  assert.equal(session.latest_request_id,'a-newer','latest must come from chronology, not UUID order');
  assert.equal(session.active_request_id,null);
  assert.equal((await(await fetch(endpoint+'/transcript',{headers})).json()).episodes.length,3);
  assert.equal((await fetch(endpoint+'/turns/a-newer',{headers:{authorization:'Bearer conversation-fixture-beta'}})).status,404);
  await page.reload();
  await page.getByLabel('Scone space key',{exact:true}).fill('conversation-fixture-alpha');
  await page.getByRole('button',{name:'Connect',exact:true}).click();
  await page.getByText('Reply completed; its saved text was forgotten.',{exact:true}).waitFor();
  assert.equal(page.url(),url);
  assert.equal(await page.getByText('Scripted follow-up: you asked about Juniper calibration.',{exact:true}).count(),0);
  assert.deepEqual(writes,[],'reopening only reads; no create/send/stop commands');
  assert.deepEqual(await page.evaluate(()=>({local:localStorage.length,session:sessionStorage.length})),{local:0,session:0});
  assert.deepEqual(errors,[]);
});
