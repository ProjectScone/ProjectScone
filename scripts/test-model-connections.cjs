// Isolated real React screen; every model-settings request is fulfilled locally.
const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
let server,browser,base;
const config={base_url:'http://127.0.0.1:8080/v1/',model:'local-model',timeout_s:180,api_key_env:null,voice:null,sample_rate:24000};
const snapshot=(revision=3,model='local-model')=>({schema_version:1,revision,connections:{chat:{...config,model},extraction:null,vision:null,transcription:null,speech:null}});
before(async()=>{
  const root=path.resolve(__dirname,'../Webapp');
  const {createServer}=await import(pathToFileURL(path.join(root,'node_modules/vite/dist/node/index.js')).href);
  server=await createServer({root,server:{port:0,strictPort:false,proxy:{'/v1':undefined,'/healthz':undefined,'/__native_console':undefined}}});
  await server.listen();base=`http://127.0.0.1:${server.httpServer.address().port}`;
  browser=await chromium.launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH,args:['--disable-gpu']});
});
after(async()=>{await browser?.close();await server?.close();});
async function fixture(t,handle){
  const page=await browser.newPage();t.after(()=>page.close());
  page.on('pageerror',error=>console.error('Fixture page error:',error.message));
  await page.route('**/v1/**',handle);
  await page.goto(base+'/tests/fixtures/model-connections.html');
  await page.getByRole('heading',{name:'Conversation',exact:true}).waitFor({timeout:10000}).catch(async error=>{console.error(await page.locator('body').innerText());throw error;});
  return page;
}
test('conflict recovery preserves edits, displays current saved config, and submits the refreshed revision',async t=>{
  let current=snapshot(),writes=0,probes=0;
  const page=await fixture(t,async route=>{
    const request=route.request();
    if(request.method()==='GET')return route.fulfill({json:current});
    if(request.url().endsWith('/probe')){probes++;return route.fulfill({json:{models:['local-model'],model_available:true}});}
    writes++;
    if(writes===1){current=snapshot(4,'changed-elsewhere');return route.fulfill({status:409,json:{error:'Changed elsewhere'}});}
    const body=request.postDataJSON();assert.equal(body.expected_revision,4);assert.equal(body.connection.model,'my-draft');
    current=snapshot(5,'my-draft');return route.fulfill({json:current});
  });
  assert.equal(probes,0);assert.equal(writes,0);
  await page.getByLabel('Model identifier',{exact:true}).fill('my-draft');
  await page.getByRole('button',{name:'Save configuration',exact:true}).click();
  await page.getByText('Settings changed elsewhere.',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Model identifier',{exact:true}).inputValue(),'my-draft');
  assert.equal(await page.getByRole('button',{name:'Save configuration',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Load latest saved settings',exact:true}).click();
  await page.getByText('Latest saved settings loaded.',{exact:false}).waitFor();
  assert.equal(await page.getByLabel('Model identifier',{exact:true}).inputValue(),'my-draft');
  await page.locator('.model-current summary').click();
  assert.match(await page.locator('.model-current').innerText(),/changed-elsewhere/);
  await page.getByRole('button',{name:'Save configuration',exact:true}).click();
  await page.getByText('Configuration saved. No inference was run.',{exact:true}).waitFor();
  assert.equal(writes,2);assert.equal(probes,0);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'model-connections-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth),false);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'model-connections-mobile.png'),fullPage:true});
});
test('edits and purpose changes discard delayed discovery without losing purpose drafts',async t=>{
  let release,requested;
  const arrived=new Promise(resolve=>{requested=resolve;}),held=new Promise(resolve=>{release=resolve;});
  const page=await fixture(t,async route=>{
    if(route.request().method()==='GET')return route.fulfill({json:snapshot()});
    requested();await held;return route.fulfill({json:{models:['stale-result'],model_available:false}}).catch(()=>{});
  });
  await page.getByRole('button',{name:'Discover models',exact:true}).click();await arrived;
  await page.getByLabel('Model identifier',{exact:true}).fill('my-edited-model');
  await page.getByRole('button',{name:'Image understanding',exact:false}).click();
  release();
  await page.getByRole('heading',{name:'Image understanding',exact:true}).waitFor();
  assert.equal(await page.locator('.model-probe-result').count(),0);
  await page.getByRole('button',{name:'Conversation',exact:false}).click();
  assert.equal(await page.getByLabel('Model identifier',{exact:true}).inputValue(),'my-edited-model');
  assert.equal(await page.locator('.model-probe-result').count(),0);
});
test('switching API identities discards an outstanding read and clears old drafts',async t=>{
  let release,requested,reads=0;
  const arrived=new Promise(resolve=>{requested=resolve;}),held=new Promise(resolve=>{release=resolve;});
  const page=await fixture(t,async route=>{
    const auth=route.request().headers().authorization;
    if(auth==='Bearer fixture-b')return route.fulfill({json:snapshot(8,'new-server-model')});
    if(++reads===1)return route.fulfill({json:snapshot()});
    requested();await held;return route.fulfill({json:snapshot(9,'stale-server-model')}).catch(()=>{});
  });
  await page.getByLabel('Model identifier',{exact:true}).fill('old-draft');
  await page.getByRole('button',{name:'Refresh saved settings',exact:true}).click();await arrived;
  await page.getByRole('button',{name:'Switch fixture connection',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('input[list]')?.value==='new-server-model');
  release();
  assert.equal(await page.getByLabel('Model identifier',{exact:true}).inputValue(),'new-server-model');
});
test('Models navigation and deep link require explicit models.manage before settings reads',async t=>{
  const required=['recall','facts.read','facts.review','facts.close','facts.exclude','facts.include','events.read','metrics.read','scopes.read','status.read'];
  for(const enabled of [false,true]){
    let reads=0;
    const page=await browser.newPage();t.after(()=>page.close());
    await page.route('**/v1/**',route=>{
      if(route.request().url().endsWith('/capabilities'))return route.fulfill({json:{schema_version:1,implementation:'fixture',features:{...Object.fromEntries(required.map(key=>[key,false])),'models.manage':enabled}}});
      reads++;return route.fulfill({json:snapshot()});
    });
    await page.goto(base+'/tests/fixtures/model-connections.html?workspace=1#models');
    if(enabled){await page.getByRole('heading',{name:'Conversation',exact:true}).waitFor();assert.equal(reads,1);assert.equal(await page.getByRole('button',{name:'Models',exact:true}).count(),1);}
    else{await page.getByText('This page is not available on this server',{exact:true}).waitFor();assert.equal(reads,0);assert.equal(await page.getByRole('button',{name:'Models',exact:true}).count(),0);}
  }
});
test('retained image analysis is explicit, initially unsaved, and saves provenance only after review',async t=>{
  const attachmentId='a'.repeat(64);let analyses=0,saves=0;
  const page=await browser.newPage();t.after(()=>page.close());
  const required=['recall','facts.read','facts.review','facts.close','facts.exclude','facts.include','events.read','metrics.read','scopes.read','status.read'];
  await page.route('**/v1/**',route=>{
    const request=route.request(),pathname=new URL(request.url()).pathname;
    if(pathname==='/v1/status')return route.fulfill({json:{space:'alpha'}});
    if(pathname==='/v1/capabilities')return route.fulfill({json:{schema_version:1,implementation:'fixture',features:{...Object.fromEntries(required.map(key=>[key,false])),'episodes.read':true,'episodes.attachments':true,'images.understand':true,'models.manage':true}}});
    if(request.method()==='GET'&&pathname==='/v1/episodes/7')return route.fulfill({json:{episode_id:7,content:'Original retained caption',kind:'note',created_at:'2026-09-07',source:'slide.png',attachments:[{attachment_id:attachmentId,media_type:'image/png',bytes:40,filename:'slide.png'}]}});
    if(pathname.endsWith('/understand')){analyses++;assert.equal(pathname,`/v1/episodes/7/attachments/${attachmentId}/understand`);return route.fulfill({json:{schema_version:1,episode_id:7,attachment_id:attachmentId,persisted:false,understanding:{text:'A red rectangle. <script>window.imageInjected=true</script>',attachment_id:attachmentId,source:'slide.png',media_type:'image/png',model:'fixture-vision',width:3,height:2,origin:'model_generated'}}});}
    if(request.method()==='POST'&&pathname==='/v1/episodes'){saves++;const body=request.postDataJSON();assert.deepEqual(body.attachment_ids,[attachmentId]);assert.equal(body.metadata.origin,'model_generated');assert.equal(body.metadata.source_episode_id,'7');assert.equal(body.metadata.source_attachment_id,attachmentId);assert.match(body.dedup_key,/^image-analysis:/);return route.fulfill({json:{episode_id:9}});}
    throw Error(`Unexpected fixture request: ${request.method()} ${pathname}`);
  });
  await page.goto(base+'/tests/fixtures/model-connections.html?source=1');
  await page.getByRole('heading',{name:'Understand this image',exact:true}).waitFor();
  assert.equal(analyses,0);assert.equal(saves,0);
  await page.getByRole('button',{name:'Understand image',exact:true}).click();
  await page.getByText('Unsaved model interpretation',{exact:true}).waitFor();
  assert.equal(analyses,1);assert.equal(saves,0);
  assert.equal(await page.evaluate(()=>window.imageInjected),undefined);
  assert.match(await page.locator('.source-page-original').innerText(),/Original retained caption/);
  await page.getByRole('button',{name:'Save interpretation as memory',exact:true}).click();
  await page.getByText('Saved model interpretation',{exact:true}).waitFor();
  assert.equal(saves,1);assert.equal(await page.getByRole('link',{name:'source episode #9',exact:true}).getAttribute('href'),'/memory/sources/9?space=alpha');
  assert.match(await page.locator('.source-understanding').innerText(),/No facts were approved/);
});
test('an unconfigured vision model offers setup without running inference',async t=>{
  const page=await browser.newPage();t.after(()=>page.close());let actions=0;
  const required=['recall','facts.read','facts.review','facts.close','facts.exclude','facts.include','events.read','metrics.read','scopes.read','status.read'];
  await page.route('**/v1/**',route=>{
    const pathname=new URL(route.request().url()).pathname;
    if(pathname==='/v1/status')return route.fulfill({json:{space:'alpha'}});
    if(pathname==='/v1/capabilities')return route.fulfill({json:{schema_version:1,implementation:'fixture',features:{...Object.fromEntries(required.map(key=>[key,false])),'episodes.read':true,'images.understand':false,'models.manage':true}}});
    if(pathname==='/v1/episodes/7')return route.fulfill({json:{episode_id:7,content:'Image',kind:'note',created_at:'2026-09-07',source:'slide.png',attachments:[{attachment_id:'a'.repeat(64),media_type:'image/png',bytes:40}]}});
    actions++;return route.fulfill({status:500,json:{error:'Unexpected request'}});
  });
  await page.goto(base+'/tests/fixtures/model-connections.html?source=1');
  await page.getByRole('link',{name:'Configure an image model in Models',exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Understand image',exact:true}).count(),0);assert.equal(actions,0);
});
