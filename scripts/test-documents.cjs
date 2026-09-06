// Shipped React UI at a controlled HTTP boundary; native integration is separate.
const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const http=require('node:http');
const {chromium}=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
const contract=require('../tests/fixtures/http-capabilities.json');
let browser;
before(async()=>{browser=await chromium.launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH,args:['--disable-gpu']});});
after(async()=>{await browser?.close();});
const row=(id,kind='file')=>({episode_id:id,kind,source:kind==='file'?`guide-${id}.md`:null,created_at:'2026-09-06',byte_count:1600,preview:`Source ${id} <script>not markup</script>`,preview_truncated:true});
async function fixture(t,{supported=true,mobile=false,crowded=false}={}){
  const html=fs.readFileSync(process.env.SCONE_DOCUMENTS_HTML,'utf8').replaceAll('__SCONE_TOKEN__','documents-fixture');
  const requests=[];const state={fail:false,malformed:false,delay:null,detailDelay:null};
  const server=http.createServer(async(req,res)=>{
    const url=new URL(req.url,'http://fixture');
    if(url.pathname==='/memory'){res.setHeader('content-type','text/html');return res.end(html);}
    if(url.pathname==='/favicon.ico'){res.statusCode=204;return res.end();}
    requests.push({url,method:req.method});res.setHeader('content-type','application/json');
    assert.equal(req.headers.authorization,'Bearer documents-fixture');
    if(url.pathname==='/v1/status')return res.end(JSON.stringify({space:'library',episodes:45}));
    if(url.pathname==='/v1/capabilities'){
      const features={...contract.rust.features};if(!supported)delete features['episodes.list'];
      return res.end(JSON.stringify({...contract.rust,features}));
    }
    if(url.pathname==='/v1/sources'){
      if(state.fail){res.statusCode=503;return res.end('{"error":"Inventory temporarily unavailable"}');}
      if(state.delay)await state.delay;
      if(state.malformed)return res.end(JSON.stringify({items:[row(30)],has_more:true,next_before:999}));
      const kind=url.searchParams.get('kind');
      const before=url.searchParams.get('before');
      const items=kind==='note'?[row(6,'note')]:kind==='connector'?[]:before?[row(4)]:crowded?Array.from({length:25},(_,i)=>row(50-i)):[row(30),row(29)];
      return res.end(JSON.stringify({items,has_more:!kind&&!before,next_before:!kind&&!before?items.at(-1).episode_id:null}));
    }
    if(/^\/v1\/episodes\/\d+$/.test(url.pathname)){
      const id=Number(url.pathname.split('/').at(-1));if(id===30&&state.detailDelay)await state.detailDelay;
      return res.end(JSON.stringify({...row(id),content:`Full retained text ${id}\n<script>not executable</script>\nEND`,tags:[],metadata:{collection:'manuals'}}));
    }
    res.statusCode=404;res.end('{}');
  });
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const page=await browser.newPage({viewport:mobile?{width:390,height:844}:{width:1400,height:1000}});page.setDefaultTimeout(2500);
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  t.after(async()=>{await page.close();server.closeAllConnections();await new Promise(r=>server.close(r));assert.deepEqual(errors,[]);});
  await page.goto(`http://127.0.0.1:${server.address().port}/memory#documents`);
  return {page,requests,state};
}
test('Documents browse and inspect retained sources without search or unsupported writes',async t=>{
  const {page,requests}=await fixture(t);
  await page.getByRole('button',{name:'Open guide-30.md',exact:true}).waitFor();
  await page.getByRole('button',{name:'Open guide-30.md',exact:true}).click();
  const detail=page.getByRole('region',{name:'Retained source'});
  await detail.getByText(/Full retained text 30/).waitFor();
  assert.equal(await detail.locator('pre').textContent(),'Full retained text 30\n<script>not executable</script>\nEND');
  assert.equal(await detail.locator('script').count(),0);
  assert.equal(await detail.locator('a[href="guide-30.md"]').count(),0);
  assert.equal(await page.getByRole('button',{name:'Add source',exact:true}).count(),0);
  assert.equal(await page.getByRole('button',{name:'View source images',exact:true}).count(),0);
  assert.equal(requests.some(r=>r.url.pathname==='/v1/recall'||r.method!=='GET'),false);
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'documents-desktop.png'),fullPage:true});
});
test('source paging retains kind, navigates boundaries, and refresh returns to latest',async t=>{
  const {page,requests}=await fixture(t);
  await page.getByRole('button',{name:'Older sources',exact:true}).click();
  await page.getByRole('button',{name:'Open guide-4.md',exact:true}).waitFor();
  assert.equal(requests.filter(r=>r.url.pathname==='/v1/sources').at(-1).url.search,'?limit=25&before=29');
  await page.getByRole('button',{name:'Newer sources',exact:true}).click();
  await page.getByRole('button',{name:'Open guide-30.md',exact:true}).waitFor();
  await page.getByRole('button',{name:'Notes',exact:true}).click();
  await page.getByRole('button',{name:'Open Note #6',exact:true}).waitFor();
  assert.equal(requests.filter(r=>r.url.pathname==='/v1/sources').at(-1).url.search,'?limit=25&kind=note');
  assert.equal(await page.getByRole('button',{name:'Older sources',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Refresh sources',exact:true}).click();
  await page.getByRole('button',{name:'Open Note #6',exact:true}).waitFor();
  assert.equal(requests.filter(r=>r.url.pathname==='/v1/sources').at(-1).url.search,'?limit=25&kind=note');
});
test('failed or invalid next pages preserve the last confirmed page and can be retried',async t=>{
  const {page,state}=await fixture(t);
  await page.getByRole('button',{name:'Open guide-30.md',exact:true}).waitFor();state.fail=true;
  await page.getByRole('button',{name:'Older sources',exact:true}).click();
  await page.getByRole('alert').getByText(/temporarily unavailable/).waitFor();
  assert.equal(await page.getByRole('button',{name:'Open guide-30.md',exact:true}).count(),1);
  state.fail=false;state.malformed=true;
  await page.getByRole('button',{name:'Retry sources',exact:true}).click();
  await page.getByRole('alert').getByText(/invalid source page/i).waitFor();
  state.malformed=false;
  await page.getByRole('button',{name:'Retry sources',exact:true}).click();
  await page.getByRole('button',{name:'Open guide-4.md',exact:true}).waitFor();
});
test('late source inspection cannot replace a newer selection',async t=>{
  const {page,state,requests}=await fixture(t);
  let release;state.detailDelay=new Promise(r=>{release=r;});t.after(()=>release());
  await page.getByRole('button',{name:'Open guide-30.md',exact:true}).click();
  await page.getByRole('region',{name:'Retained source'}).getByText(/Loading source/).waitFor();
  await page.getByRole('button',{name:'Open guide-29.md',exact:true}).click();
  await page.getByText(/Full retained text 29/).waitFor();release();
  await page.getByRole('button',{name:'Notes',exact:true}).click();
  await page.getByRole('button',{name:'Open Note #6',exact:true}).waitFor();
  assert.equal(await page.getByRole('region',{name:'Retained source'}).count(),0);
  assert.ok(requests.some(r=>r.url.pathname==='/v1/episodes/30'));
});
test('a slow older-page response cannot overwrite a newly selected source filter',async t=>{
  const {page,state}=await fixture(t);
  await page.getByRole('button',{name:'Open guide-30.md',exact:true}).waitFor();
  let release;state.delay=new Promise(r=>{release=r;});t.after(()=>release());
  await page.getByRole('button',{name:'Older sources',exact:true}).click();
  await page.getByRole('status').filter({hasText:'Loading all sources'}).waitFor();
  state.delay=null;
  await page.getByRole('button',{name:'Notes',exact:true}).click();
  await page.getByRole('button',{name:'Open Note #6',exact:true}).waitFor();
  release();
  await page.getByRole('button',{name:'Open Note #6',exact:true}).click();
  await page.getByText(/Full retained text 6/).waitFor();
  assert.equal(await page.getByRole('button',{name:'Open guide-4.md',exact:true}).count(),0);
});
test('an inspection opened while paging is cleared when the new page arrives',async t=>{
  const {page,state}=await fixture(t);
  await page.getByRole('button',{name:'Open guide-30.md',exact:true}).waitFor();
  let release;state.delay=new Promise(r=>{release=r;});t.after(()=>release());
  await page.getByRole('button',{name:'Older sources',exact:true}).click();
  await page.getByRole('status').filter({hasText:'Loading all sources'}).waitFor();
  await page.getByRole('button',{name:'Open guide-30.md',exact:true}).click();
  await page.getByText(/Full retained text 30/).waitFor();release();
  await page.getByRole('button',{name:'Open guide-4.md',exact:true}).waitFor();
  assert.equal(await page.getByRole('region',{name:'Retained source'}).count(),0);
});
test('empty filtered inventory and mobile layout remain explicit and usable',async t=>{
  const {page}=await fixture(t,{mobile:true});
  await page.getByRole('button',{name:'Open guide-30.md',exact:true}).click();
  await page.getByText(/Full retained text 30/).waitFor();
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'library fits a phone');
  if(process.env.SCONE_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,'documents-mobile.png'),fullPage:true});
  await page.getByRole('button',{name:'Connectors',exact:true}).click();
  await page.getByText('No connector sources on this page.',{exact:true}).waitFor();
});
test('opening a later mobile source brings its inspector into view and closing restores its card',async t=>{
  const {page}=await fixture(t,{mobile:true,crowded:true});
  const card=page.getByRole('button',{name:'Open guide-26.md',exact:true});
  await card.click();
  const heading=page.getByRole('region',{name:'Retained source'}).getByRole('heading',{name:'guide-26.md',exact:true});
  await page.getByText(/Full retained text 26/).waitFor();
  assert.ok(await heading.evaluate(el=>{const box=el.getBoundingClientRect();return box.top>=0&&box.bottom<=innerHeight;}),'the selected source heading must be onscreen');
  await page.getByRole('button',{name:'Close source',exact:true}).click();
  await page.waitForFunction(()=>document.activeElement?.getAttribute('aria-label')==='Open guide-26.md');
  assert.ok(await card.evaluate(el=>{const box=el.getBoundingClientRect();return box.bottom>0&&box.top<innerHeight;}),'close returns to the selected card');
});
test('older servers do not expose or probe the unsupported inventory',async t=>{
  const {page,requests}=await fixture(t,{supported:false});
  await page.getByText(/This server does not support this page/).waitFor();
  assert.equal(await page.getByRole('button',{name:'Documents',exact:true}).count(),0);
  assert.equal(requests.some(r=>r.url.pathname==='/v1/sources'),false);
});
