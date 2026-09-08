// Exercise the built Analytics page against an isolated, read-only HTTP fixture.
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
const metric=(name,value,n=12,unit='items',denominator='successful recalls')=>({name,value,n,unit,denominator,definition:`Definition of ${name}`,caveat:'This observation does not establish answer quality.'});
const report={evidence:'fixture',coverage:{events_considered:1320,earliest_retained:'2026-09-01T09:00:00Z',latest:'2026-09-07T16:00:00Z',truncated:false,schema_versions:[1]},failures:{},embedders:['a-long-embedding-model-identifier/with-a-version-and-dimensions'],daily:{'2026-09-07':{recall:12},'2026-09-01':{remember:0},'2026-09-04':{recall:6}},metrics:[
  metric('recall.count',1234,1234),metric('recall.latency_ms.p50',2.54,12,'ms'),metric('recall.latency_ms.p95',8.91,12,'ms'),
  metric('recall.degraded_count',0),metric('recall.byte_context_reduction.median',.994,12,'share of bytes'),
  metric('ingest.calls',40),metric('ingest.fresh_episodes',650,680,'episodes','records offered'),metric('ingest.deduplicated',30),metric('ingest.dedup_fraction',.044,680,'share'),metric('ingest.stored_bytes',450312,680,'bytes'),
  ...['asserted','asserted_already_closed','restated','superseded','manual_closures'].map(name=>metric(`ledger.${name}`,0,0,'facts','assert_fact calls')),
  metric('feedback.judged_items',0,200),metric('feedback.useful_share',null,0,'share','judged items'),metric('feedback.coverage',0,200,'share'),
  ...['both_lanes','vector_only','text_only'].map(name=>metric(`recall.top_item.${name}_share`,.5,12,'share','successful recalls that returned at least one item')),
  metric('recall.top_similarity.model.with.a.long.identifier',{p10:null,median:.56,p90:.81},12,'cosine'),
  ...['embed','vector','text'].map(name=>metric(`recall.latency_ms.${name}.p50`,.12,12,'ms')),
]};
async function fixture(t,{mobile=false,data=report,error=false}={}){
  const html=fs.readFileSync(process.env.SCONE_PLAYGROUND_HTML||path.resolve(__dirname,'../python/memory/src/scone_memory/api/playground.html'),'utf8').replaceAll('__SCONE_TOKEN__','analytics-fixture');
  const writes=[],errors=[];
  const server=http.createServer((req,res)=>{
    const url=new URL(req.url,'http://fixture');
    if(url.pathname==='/memory'){res.setHeader('content-type','text/html');return res.end(html);}
    if(url.pathname==='/favicon.ico'){res.statusCode=204;return res.end();}
    res.setHeader('content-type','application/json');
    if(req.method!=='GET')writes.push(req.url);
    if(url.pathname==='/v1/capabilities')return res.end(JSON.stringify(contract));
    if(url.pathname==='/v1/status')return res.end(JSON.stringify({space:'alpha',episodes:650,chunks:900,pending_review:0}));
    if(url.pathname==='/v1/metrics'){if(error){res.statusCode=503;return res.end('{"error":"Metrics unavailable"}');}return res.end(JSON.stringify(data));}
    res.statusCode=404;res.end('{}');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const page=await browser.newPage({viewport:mobile?{width:390,height:844}:{width:1440,height:1000}});
  page.on('pageerror',e=>errors.push(e.message));
  t.after(async()=>{await page.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);});
  await page.goto(`http://127.0.0.1:${server.address().port}/memory#analytics`);
  return page;
}
for(const mobile of [false,true])test(`Analytics preserves evidence and readable responsive detail, mobile=${mobile}`,async t=>{
  const page=await fixture(t,{mobile});
  const workspace=page.locator('.analytics-workspace');
  await workspace.waitFor();
  assert.match(await workspace.innerText(),/1,320\s+events considered/);
  const useful=page.locator('.analytics-headline').filter({hasText:'Judged useful'});
  assert.match(await useful.innerText(),/No evidence yet/);
  assert.match(await useful.innerText(),/n = 0 judged items/);
  assert.doesNotMatch(await useful.innerText(),/0\.0%/);
  const latency=page.locator('.analytics-headline').filter({hasText:'Typical recall time'});
  assert.match(await latency.innerText(),/2\.5\s*ms/);
  await latency.locator('summary').focus();await page.keyboard.press('Enter');
  assert.equal(await latency.getAttribute('open'),'');
  assert.match(await latency.innerText(),/This observation does not establish answer quality/);
  const diagnostics=page.locator('.analytics-diagnostics');
  assert.equal(await diagnostics.getAttribute('open'),null);
  await diagnostics.locator(':scope > summary').click();
  assert.match(await diagnostics.innerText(),/Top similarity · model.with.a.long.identifier/);
  assert.match(await diagnostics.innerText(),/p10\s*Not reported/);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.equal(await workspace.locator('.analytics-metric-label,.analytics-evidence').evaluateAll(els=>els.every(el=>el.scrollWidth<=el.clientWidth+1)),true,'Metric labels and denominators fit without horizontal clipping');
  assert.equal(await workspace.locator('.analytics-chart-column').first().getAttribute('title'),'2026-09-01: 0 events');
  assert.equal(await workspace.locator('.analytics-chart-column i').first().evaluate(el=>el.getBoundingClientRect().height),0,'Zero event days are not drawn as nonzero bars');
  if(process.env.SCONE_SCREENSHOT_DIR){await diagnostics.locator(':scope > summary').click();await latency.locator('summary').click();await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(process.env.SCONE_SCREENSHOT_DIR,`analytics-${mobile?'mobile':'desktop'}.png`),fullPage:true});}
});
test('Analytics retains partial coverage and does not turn unreported evidence into none',async t=>{
  const data={...report,coverage:{...report.coverage,truncated:true,earliest_retained:null,latest:null},metrics:[metric('recall.count',0,0)],daily:undefined,failures:undefined,embedders:undefined};
  const page=await fixture(t,{data});
  const workspace=page.locator('.analytics-workspace');await workspace.waitFor();
  assert.match(await workspace.innerText(),/Partial coverage · read truncated/);
  assert.match(await workspace.innerText(),/Daily activity was not reported/);
  assert.equal(await workspace.locator('.analytics-headline').count(),1);
  assert.equal(await workspace.locator('.analytics-provenance dd').allTextContents().then(values=>values.every(value=>value==='Not reported')),true);
});
for(const [name,data,message] of [
  ['unattached',{...report,coverage:null},'No event log is attached'],
  ['empty',{...report,coverage:{...report.coverage,events_considered:0}},'No activity recorded yet'],
])test(`Analytics keeps its ${name} state`,async t=>{
  const page=await fixture(t,{data});await page.getByText(message,{exact:false}).waitFor();
  assert.equal(await page.locator('.analytics-headlines').count(),0);
});
test('Analytics renders read errors instead of fabricated metrics',async t=>{
  const page=await fixture(t,{error:true});await page.getByRole('alert').waitFor();
  assert.equal(await page.locator('.analytics-headlines').count(),0);
});
