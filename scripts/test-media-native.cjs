// A real Python API + real browser, isolated from every live Scone store.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const {once}=require('node:events');
const path=require('node:path');
const {chromium}=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'..');
const {pythonLayout}=require('./python-layout.cjs');

test('stored image bytes survive native recall and render in the real webapp',{timeout:40000},async t=>{
  const html=process.env.SCONE_PLAYGROUND_HTML;
  assert.ok(html,'SCONE_PLAYGROUND_HTML must name a freshly built isolated artifact');
  const server=spawn(process.env.SCONE_TEST_PYTHON||path.join(pythonLayout(root).project,'.venv/bin/python'),['-u','-c',String.raw`
import asyncio, socket, sys
from pathlib import Path
import uvicorn
from scone_memory import MemoryEngine, HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex
from scone_memory import InMemoryEventLog
import importlib
api_module = importlib.import_module('scone_memory.api.app')
api_module.CONSOLE = Path(sys.argv[1])
api_module.PLAYGROUND = Path(sys.argv[1])
async def run():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=InMemoryEventLog()).open()
    app = api_module.create_app(engine, {'media-fixture-key':'media-test', 'other-fixture-key':'other-test'}, console_key='media-fixture-key')
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    print('MEDIA_READY ' + str(sock.getsockname()[1]), flush=True)
    await uvicorn.Server(uvicorn.Config(app, log_level='error')).serve(sockets=[sock])
asyncio.run(run())
`,html],{cwd:root,stdio:['ignore','pipe','pipe']});
  let logs='';server.stderr.on('data',part=>{logs=(logs+part).slice(-8000);});
  t.after(async()=>{if(server.exitCode===null){server.kill('SIGTERM');await once(server,'exit');}});
  const port=await new Promise((resolve,reject)=>{
    let data='';server.stdout.on('data',part=>{data+=part;const m=data.match(/MEDIA_READY (\d+)/);if(m)resolve(Number(m[1]));});
    server.once('exit',code=>reject(Error(`Fixture server exited ${code}: ${logs}`)));
    server.once('error',reject);
  });
  const base=`http://127.0.0.1:${port}`;
  for(let i=0;i<100;i++){
    try{if((await fetch(base+'/healthz')).ok)break;}catch{}
    await new Promise(resolve=>setTimeout(resolve,25));
  }
  const headers={authorization:'Bearer media-fixture-key'};
  const bytes=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=','base64');
  const stored=await fetch(base+'/v1/attachments',{method:'POST',headers:{...headers,'content-type':'image/png','x-filename':'native-source.png'},body:bytes});
  assert.equal(stored.status,200);const attachment=await stored.json();
  assert.equal((await fetch(base+'/v1/attachments/'+attachment.attachment_id,{headers:{authorization:'Bearer other-fixture-key'}})).status,404);
  const episode=await fetch(base+'/v1/episodes',{method:'POST',headers:{...headers,'content-type':'application/json'},body:JSON.stringify({content:'Original screenshot of the memory workspace',attachment_ids:[attachment.attachment_id]})});
  assert.ok(episode.ok);const record=await episode.json();
  const browser=await chromium.launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH,args:['--disable-gpu']});
  t.after(()=>browser.close());
  const page=await browser.newPage({viewport:{width:1200,height:1000}});
  await page.goto(base+'/playground');
  await page.getByRole('textbox',{name:'Test memory recall'}).fill('memory workspace');
  await page.getByRole('button',{name:'Recall',exact:true}).click();
  await page.locator('.recall-card').filter({hasText:`Episode ${record.episode_id}`}).getByRole('button',{name:'View source images'}).click();
  const image=page.getByRole('img',{name:'native-source.png'});await image.waitFor();
  await page.waitForFunction(()=>document.querySelector('.source-image img')?.naturalWidth===1);
  assert.match(await image.getAttribute('src'),/^blob:/);
  assert.equal(await page.getByRole('link',{name:'Save original image'}).count(),1);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'native-image-recall.png'),fullPage:true});
  await page.getByRole('button',{name:'Records view',exact:true}).click();
  await page.locator(`[data-record="episode:${record.episode_id}"]`).click();
  const inspector=page.getByRole('complementary',{name:'Source inspector'});
  await inspector.getByRole('button',{name:'View source images',exact:true}).click({timeout:2000});
  await inspector.getByRole('img',{name:'native-source.png'}).waitFor();
  await page.waitForFunction(()=>document.querySelector('#inspector .source-image img')?.naturalWidth===1);
  assert.equal(await inspector.getByRole('link',{name:'Save original image'}).count(),1);
  if(process.env.SCONE_SCREENSHOT_DIR)await inspector.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'native-graph-source-image.png')});
  await page.locator('[data-record^="chunk:"]').first().click();
  assert.equal(await inspector.getByRole('img').count(),0,'a different record must not retain the prior source image');
  assert.equal(await inspector.getByRole('button',{name:'View source images',exact:true}).count(),0,'chunk IDs are not episode IDs');
  await page.locator(`[data-record="episode:${record.episode_id}"]`).click();
  assert.equal(await inspector.getByRole('img').count(),0,'returning to a source requires explicitly opening its originals');
  await page.getByRole('button',{name:'Close source inspector',exact:true}).click();
  await page.goto(base+'/memory?q=memory%20workspace#search');
  await page.locator('.rows .row').first().getByRole('button',{name:'View source images'}).click({timeout:2000});
  await page.getByRole('img',{name:'native-source.png'}).waitFor();
  await page.waitForFunction(()=>document.querySelector('.source-image img')?.naturalWidth===1);
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'Memory Search images fit a mobile viewport');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'native-image-search-mobile.png'),fullPage:true});
  await page.goto(base+'/memory#search');
  await page.locator('.rows .row').first().getByRole('button',{name:'View source images'}).click();
  await page.getByRole('img',{name:'native-source.png'}).waitFor();
  await page.waitForFunction(()=>document.querySelector('.source-image img')?.naturalWidth===1);
  await page.getByRole('button',{name:'Add source',exact:true}).click({timeout:2000});
  await page.getByRole('textbox',{name:'Source note',exact:true}).fill('A second screenshot documenting the upload workflow');
  await page.getByLabel('Original image', {exact:true}).setInputFiles({name:'uploaded-source.png',mimeType:'image/png',buffer:bytes});
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  await page.getByText(/Source saved.*episode/i).waitFor({timeout:5000});
  await page.locator('.rows .row').filter({hasText:'A second screenshot documenting the upload workflow'}).waitFor({timeout:1500});
  await page.getByRole('button',{name:'Inspect saved source',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.source-composer .source-image img')?.naturalWidth===1);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'the upload result fits mobile');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'native-source-upload-mobile.png'),fullPage:true});
  await page.locator('.search input').fill('second screenshot documenting');
  await page.locator('.search button').click();
  await page.locator('.rows .row').filter({hasText:'A second screenshot documenting the upload workflow'}).waitFor();
  const savedHeading=await page.locator('.source-saved h3').innerText();
  await page.getByRole('button',{name:'Add another source',exact:true}).click();
  await page.getByRole('textbox',{name:'Source note',exact:true}).fill('A second screenshot documenting the upload workflow');
  const refreshed=page.waitForRequest(req=>new URL(req.url()).pathname==='/v1/recall'&&new URL(req.url()).searchParams.get('q')==='second screenshot documenting');
  const duplicateReceipt=page.waitForResponse(response=>new URL(response.url()).pathname==='/v1/episodes'&&response.request().method()==='POST');
  await page.getByRole('button',{name:'Save source',exact:true}).click();
  const duplicateResponse=await duplicateReceipt;
  assert.equal(duplicateResponse.status(),200);
  assert.equal((await duplicateResponse.json()).deduplicated,true,'the native save receipt confirms reuse');
  await page.getByRole('status').getByText(/That episode was reused/).waitFor();
  await refreshed;
  assert.equal(await page.locator('.source-saved h3').innerText(),savedHeading,'note-only duplicate reuses the existing episode');
});
