// Real built UI against an isolated HTTP contract; no live memory or providers.
const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const http=require('node:http');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require(process.env.SCONE_PLAYWRIGHT_MODULE||'playwright');
const contract=require('../tests/fixtures/http-capabilities.json').python;
let browser;
before(async()=>{browser=await chromium.launch({headless:true,executablePath:process.env.SCONE_BROWSER_PATH});});
after(async()=>{await browser?.close();});
const at='2026-09-07T12:00:00.000Z';
function job(id){return {job_id:`batch-${id}`,space:'alpha',created_at:at,request_id:`import-${id}`,cancelled_at:null,state:'failed',searchable:2,consolidated:1,items:[
  {index:0,episode_id:id*2,outcome:'accepted',state:'consolidated',searchable_at:at,consolidated_at:at,attempts:0,error:null},
  {index:1,episode_id:id*2+1,outcome:'updated',state:'failed',searchable_at:at,consolidated_at:null,attempts:2,error:'Extractor temporarily unavailable'},
]};}
async function fixture(t,{supported=true,mobile=false,manyRecords=false}={}){
  const html=fs.readFileSync(path.resolve(__dirname,'../python/memory/src/scone_memory/api/playground.html'),'utf8').replaceAll('__SCONE_TOKEN__','jobs-fixture');
  const reads=[],writes=[];const state={fail:false,wrongSpace:false,hold:null};
  const jobs=Array.from({length:21},(_,i)=>job(21-i));
  if(manyRecords){jobs[0].items=Array.from({length:25},(_,i)=>({...jobs[0].items[1],index:i,episode_id:100+i}));jobs[0].searchable=25;jobs[0].consolidated=0;}
  const server=http.createServer(async(req,res)=>{
    const url=new URL(req.url,'http://fixture');
    if(url.pathname==='/memory'){res.setHeader('content-type','text/html');return res.end(html);}
    if(url.pathname==='/favicon.ico'){res.statusCode=204;return res.end();}
    res.setHeader('content-type','application/json');
    assert.equal(req.headers.authorization,'Bearer jobs-fixture');
    (req.method==='GET'?reads:writes).push(req.url);
    if(url.pathname==='/v1/capabilities')return res.end(JSON.stringify({...contract,features:{...contract.features,'jobs.read':supported}}));
    if(url.pathname==='/v1/status')return res.end(JSON.stringify({space:'alpha',episodes:42,chunks:42,pending_distill:21}));
    if(url.pathname==='/v1/jobs'){
      if(state.hold)await state.hold;
      if(state.fail){res.statusCode=503;return res.end('{"error":"private server diagnostic"}');}
      const before=url.searchParams.get('before');
      const found=jobs.slice(before?jobs.findIndex(j=>j.job_id===before)+1:0);
      const page=found.slice(0,20).map(j=>state.wrongSpace?{...j,space:'beta'}:j);
      return res.end(JSON.stringify({jobs:page,next:found.length>20?page.at(-1).job_id:null}));
    }
    if(url.pathname==='/v1/episodes/43')return res.end(JSON.stringify({episode_id:43,content:'**Original import** from this batch.',attachments:[]}));
    if(url.pathname==='/v1/episodes/100')return res.end(JSON.stringify({episode_id:100,content:'First source in the large batch.',attachments:[]}));
    res.statusCode=404;res.end('{}');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const page=await browser.newPage({viewport:mobile?{width:390,height:844}:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  t.after(async()=>{await page.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);});
  await page.goto(`http://127.0.0.1:${server.address().port}/memory#status`);
  return {page,reads,state};
}
for(const mobile of [false,true])test(`batch history separates receipts, inspects real source IDs and pages history, mobile=${mobile}`,async t=>{
  const {page,reads}=await fixture(t,{mobile});
  const panel=page.getByRole('region',{name:'Batch history',exact:true});
  await panel.getByRole('button',{name:'Inspect batch batch-21',exact:true}).click({timeout:2500});
  const detail=panel.getByRole('region',{name:'Batch details',exact:true});
  const heading=detail.getByRole('heading').first();
  assert.equal(await heading.evaluate(el=>{const r=el.getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight;}),true,'Opening a batch brings its details into view');
  assert.match(await detail.innerText(),/2 searchable/);assert.match(await detail.innerText(),/1 consolidated/);
  await detail.getByText('Last extraction error',{exact:true}).click();
  assert.match(await detail.innerText(),/Extractor temporarily unavailable/);
  assert.equal(reads.some(p=>p.startsWith('/v1/episodes/')),false,'Originals are lazy, not fetched for every batch');
  await detail.getByRole('button',{name:'Inspect source 43',exact:true}).click();
  await detail.locator('.source-markdown strong').getByText('Original import',{exact:true}).waitFor();
  await panel.getByRole('button',{name:'Older batches',exact:true}).click();
  await panel.getByRole('button',{name:'Inspect batch batch-1',exact:true}).waitFor();
  assert.equal(await panel.getByRole('button',{name:'Older batches',exact:true}).isDisabled(),true);
  assert.ok(reads.includes('/v1/jobs?limit=20&before=batch-2'));
  await panel.getByRole('button',{name:'Newer batches',exact:true}).click();
  await panel.getByRole('button',{name:'Inspect batch batch-21',exact:true}).waitFor();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  if(process.env.SCONE_SCREENSHOT_DIR){await panel.scrollIntoViewIfNeeded();await panel.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`batch-history-${mobile?'mobile':'desktop'}.png`)});}
});
test('unsupported batch reads do not probe the jobs API',async t=>{
  const {page,reads}=await fixture(t,{supported:false});
  await page.getByRole('heading',{name:'Memory processing',exact:true}).waitFor();
  assert.equal(await page.getByRole('region',{name:'Batch history',exact:true}).count(),0);
  assert.equal(reads.some(p=>p.startsWith('/v1/jobs')),false);
});
test('batch history retries failed reads and refuses cross-space receipts',async t=>{
  const {page,state}=await fixture(t);
  const panel=page.getByRole('region',{name:'Batch history',exact:true});
  await panel.getByRole('button',{name:'Inspect batch batch-21',exact:true}).waitFor({timeout:2500});
  state.fail=true;await panel.getByRole('button',{name:'Refresh batches',exact:true}).click();
  await panel.getByRole('alert').waitFor();assert.doesNotMatch(await panel.innerText(),/private server diagnostic/);
  state.fail=false;state.wrongSpace=true;
  await panel.getByRole('button',{name:'Retry batches',exact:true}).click();
  await panel.getByRole('alert').waitFor();assert.doesNotMatch(await panel.innerText(),/Space beta/);
  state.wrongSpace=false;await panel.getByRole('button',{name:'Retry batches',exact:true}).click();
  await panel.getByRole('button',{name:'Inspect batch batch-21',exact:true}).waitFor();
  await page.waitForFunction(()=>!document.querySelector('.jobs-panel [role="alert"]'));
});
test('large batch record pages reveal sources without leaving the user above the inspector',async t=>{
  const {page}=await fixture(t,{mobile:true,manyRecords:true});
  const panel=page.getByRole('region',{name:'Batch history',exact:true});
  await panel.getByRole('button',{name:'Inspect batch batch-21',exact:true}).click();
  const detail=panel.getByRole('region',{name:'Batch details',exact:true});
  assert.equal(await detail.locator('.job-items>li').count(),20);
  await detail.getByRole('button',{name:'Inspect source 100',exact:true}).click();
  const source=detail.getByRole('region',{name:'Batch source',exact:true});
  await source.locator('.source-markdown').getByText('First source in the large batch.',{exact:true}).waitFor();
  const heading=source.getByRole('heading');
  assert.equal(await heading.evaluate(el=>{const r=el.getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight;}),true);
  assert.equal(await heading.evaluate(el=>el===document.activeElement),true);
  await detail.getByRole('button',{name:'Next records',exact:true}).click();
  assert.equal(await detail.locator('.job-items>li').count(),5);
  assert.equal(await detail.getByRole('button',{name:'Inspect source 124',exact:true}).count(),1);
  assert.equal(await source.count(),0);
});
